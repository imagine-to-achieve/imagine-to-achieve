from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EEFAdapterConfig:
    state_dim: int = 7
    action_dim: int = 7
    horizon: int = 5
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    architecture: str = "gru"


class ActionToEEFPoseAdapter(nn.Module):
    """Predict future EEF deltas from the current state and LIBERO actions.

    Inputs use the Ctrl-World state convention:
    `[x, y, z, rx, ry, rz, gripper]`.
    Actions use the LIBERO policy convention:
    `[dx, dy, dz, drx, dry, drz, gripper_command]`.
    Outputs use the SE(3) delta-to-current-state convention:
    translation and gripper are linear deltas, while rotation is the
    relative axis-angle `R_future @ R_current^{-1}`.
    """

    def __init__(self, config: EEFAdapterConfig):
        super().__init__()
        self.config = config
        architecture = config.architecture.lower()
        if architecture == "gru":
            input_dim = config.state_dim + config.action_dim
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.SiLU(),
            )
            self.gru = nn.GRU(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
            )
            self.output_head = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.state_dim),
            )
        elif architecture == "mlp":
            input_dim = config.state_dim + config.horizon * config.action_dim
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.horizon * config.state_dim),
            )
        else:
            raise ValueError(
                f"Unsupported EEF adapter architecture={config.architecture!r}. "
                "Expected one of {'gru', 'mlp'}."
            )

    def forward(self, current_state: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        """Return predicted future EEF delta sequence.

        Args:
            current_state: `[B, state_dim]`.
            action_seq: `[B, H, action_dim]`.

        Returns:
            `[B, H, state_dim]` future EEF deltas relative to current_state.
        """
        if current_state.ndim != 2:
            raise ValueError(f"current_state must be [B,D], got {tuple(current_state.shape)}")
        if action_seq.ndim != 3:
            raise ValueError(f"action_seq must be [B,H,D], got {tuple(action_seq.shape)}")
        architecture = self.config.architecture.lower()
        if architecture == "gru":
            state_context = current_state[:, None, :].expand(-1, action_seq.shape[1], -1)
            x = torch.cat([state_context, action_seq], dim=-1)
            x = self.input_proj(x)
            y, _ = self.gru(x)
            return self.output_head(y)

        if action_seq.shape[1] != self.config.horizon:
            raise ValueError(
                f"MLP adapter expects horizon={self.config.horizon}, got {action_seq.shape[1]}"
            )
        x = torch.cat([current_state, action_seq.flatten(start_dim=1)], dim=-1)
        y = self.mlp(x)
        return y.view(current_state.shape[0], self.config.horizon, self.config.state_dim)


def _axis_angle_to_quat(axis_angle: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    scale = torch.where(
        angle > eps,
        torch.sin(half_angle) / torch.clamp(angle, min=eps),
        0.5 - angle.square() / 48.0,
    )
    xyz = axis_angle * scale
    w = torch.cos(half_angle)
    return torch.cat([w, xyz], dim=-1)


def _quat_to_axis_angle(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    quat = quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=eps)
    w = torch.clamp(quat[..., :1], -1.0, 1.0)
    xyz = quat[..., 1:]
    sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    # Keep the shortest axis-angle representation.
    angle = torch.where(angle > torch.pi, angle - 2.0 * torch.pi, angle)
    scale = torch.where(
        sin_half > eps,
        angle / torch.clamp(sin_half, min=eps),
        2.0 + angle.square() / 12.0,
    )
    return xyz * scale


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _quat_inv(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    quat = quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=eps)
    return torch.cat([quat[..., :1], -quat[..., 1:]], dim=-1)


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    q0 = q0 / torch.clamp(torch.linalg.norm(q0, dim=-1, keepdim=True), min=eps)
    q1 = q1 / torch.clamp(torch.linalg.norm(q1, dim=-1, keepdim=True), min=eps)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.clamp(dot.abs(), max=1.0)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    linear = (1.0 - t) * q0 + t * q1
    slerped = (
        torch.sin((1.0 - t) * theta) / torch.clamp(sin_theta, min=eps) * q0
        + torch.sin(t * theta) / torch.clamp(sin_theta, min=eps) * q1
    )
    quat = torch.where(sin_theta > eps, slerped, linear)
    return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=eps)


def resample_eef_state_sequence_by_fps(
    current_state: torch.Tensor,
    future_states: torch.Tensor,
    *,
    source_fps: float,
    target_fps: float,
    target_steps: int | None = None,
) -> torch.Tensor:
    """Resample absolute EEF states from policy/action fps to Ctrl-World fps.

    Args:
        current_state: Current absolute EEF state [B,D] at time 0.
        future_states: Future absolute EEF states [B,H,D], sampled at
            1/source_fps, ..., H/source_fps.
        source_fps: Sampling rate of future_states.
        target_fps: Desired output sampling rate.
        target_steps: Optional number of future states to return. Defaults to
            round(H * target_fps / source_fps).

    Returns:
        Future absolute EEF states [B,target_steps,D], sampled at
        1/target_fps, ..., target_steps/target_fps. xyz and gripper-like
        channels are linearly interpolated; rotation-vector channels 3:6 use
        quaternion slerp and are returned on the principal branch (norm <= pi),
        matching SciPy ``Rotation.as_rotvec`` and the Ctrl-World training data.
    """
    if current_state.ndim != 2:
        raise ValueError(f"current_state must be [B,D], got {tuple(current_state.shape)}")
    if future_states.ndim != 3:
        raise ValueError(f"future_states must be [B,H,D], got {tuple(future_states.shape)}")
    if current_state.shape[0] != future_states.shape[0]:
        raise ValueError(
            "current_state and future_states batch dimensions must match: "
            f"{tuple(current_state.shape)} vs {tuple(future_states.shape)}"
        )
    if current_state.shape[-1] < 7 or future_states.shape[-1] < 7:
        raise ValueError(
            "current_state and future_states must have at least 7 dimensions "
            f"got {tuple(current_state.shape)} and {tuple(future_states.shape)}"
        )
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError(f"source_fps and target_fps must be positive, got {source_fps}, {target_fps}")

    source_steps = int(future_states.shape[1])
    if target_steps is None:
        target_steps = int(round(source_steps * float(target_fps) / float(source_fps)))
    target_steps = int(target_steps)
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")

    dtype = future_states.dtype
    device = future_states.device
    current_state = current_state.to(device=device, dtype=dtype)
    if (
        target_steps == source_steps
        and abs(float(source_fps) - float(target_fps)) < 1e-8
    ):
        return future_states

    full_states = torch.cat([current_state[:, None, :], future_states], dim=1)
    target_times = (
        torch.arange(1, target_steps + 1, device=device, dtype=dtype)
        / float(target_fps)
    )
    source_pos = torch.clamp(
        target_times * float(source_fps),
        min=0.0,
        max=float(source_steps),
    )
    left = torch.floor(source_pos).to(torch.long)
    right = torch.clamp(left + 1, max=source_steps)
    alpha = (source_pos - left.to(dtype)).view(1, target_steps, 1)

    left_states = full_states.index_select(1, left)
    right_states = full_states.index_select(1, right)
    linear = left_states + (right_states - left_states) * alpha

    left_quat = _axis_angle_to_quat(left_states[..., 3:6])
    right_quat = _axis_angle_to_quat(right_states[..., 3:6])
    rot = _quat_to_axis_angle(_quat_slerp(left_quat, right_quat, alpha))

    output = linear.clone()
    output[..., 3:6] = rot
    return output


def resample_joint_state_sequence_by_fps(
    current_state: torch.Tensor,
    future_states: torch.Tensor,
    *,
    source_fps: float,
    target_fps: float,
    target_steps: int | None = None,
) -> torch.Tensor:
    """Linearly resample absolute joint states on the policy timeline."""

    if current_state.ndim != 2 or future_states.ndim != 3:
        raise ValueError(
            "current_state/future_states must be [B,D]/[B,H,D], got "
            f"{tuple(current_state.shape)} and {tuple(future_states.shape)}"
        )
    if current_state.shape[0] != future_states.shape[0]:
        raise ValueError("current_state and future_states batch sizes must match")
    if current_state.shape[-1] != future_states.shape[-1]:
        raise ValueError("current_state and future_states dimensions must match")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")

    source_steps = int(future_states.shape[1])
    if target_steps is None:
        target_steps = int(round(source_steps * float(target_fps) / float(source_fps)))
    target_steps = int(target_steps)
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    if target_steps == source_steps and abs(float(source_fps) - float(target_fps)) < 1e-8:
        return future_states

    dtype = future_states.dtype
    device = future_states.device
    full_states = torch.cat(
        [current_state.to(device=device, dtype=dtype)[:, None, :], future_states],
        dim=1,
    )
    target_times = (
        torch.arange(1, target_steps + 1, device=device, dtype=dtype)
        / float(target_fps)
    )
    source_pos = torch.clamp(
        target_times * float(source_fps), min=0.0, max=float(source_steps)
    )
    left = torch.floor(source_pos).to(torch.long)
    right = torch.clamp(left + 1, max=source_steps)
    alpha = (source_pos - left.to(dtype)).view(1, target_steps, 1)
    left_states = full_states.index_select(1, left)
    right_states = full_states.index_select(1, right)
    return left_states + (right_states - left_states) * alpha


def state_to_ctrl_world_eef_state(state: torch.Tensor) -> torch.Tensor:
    """Convert LIBERO/RLinf state cache to Ctrl-World 7D EEF state."""
    if state.shape[-1] < 7:
        raise ValueError(f"Expected state dim >= 7, got {tuple(state.shape)}")
    pose = state[..., :6]
    if state.shape[-1] >= 8:
        gripper = state[..., 6:8].abs().mean(dim=-1, keepdim=True)
    else:
        gripper = state[..., 6:7]
    return torch.cat([pose, gripper], dim=-1)


def eef_state_delta_from_current(
    current_state: torch.Tensor, future_state: torch.Tensor
) -> torch.Tensor:
    """Return SE(3) deltas from current EEF state to each future state.

    The state convention is `[x, y, z, rx, ry, rz, gripper]`, where
    `rx, ry, rz` are axis-angle rotation vectors. Translation and gripper are
    ordinary differences. Rotation uses the relative rotation vector
    corresponding to `R_future @ R_current^{-1}`.
    """
    if current_state.ndim != 2:
        raise ValueError(f"current_state must be [B,D], got {tuple(current_state.shape)}")
    if future_state.ndim != 3:
        raise ValueError(f"future_state must be [B,H,D], got {tuple(future_state.shape)}")
    if current_state.shape[-1] < 7 or future_state.shape[-1] < 7:
        raise ValueError(
            "current_state and future_state must have at least 7 dimensions "
            f"got {tuple(current_state.shape)} and {tuple(future_state.shape)}"
        )

    pos_delta = future_state[..., :3] - current_state[:, None, :3]
    current_quat = _axis_angle_to_quat(current_state[:, 3:6])[:, None, :]
    future_quat = _axis_angle_to_quat(future_state[..., 3:6])
    rot_delta = _quat_to_axis_angle(_quat_mul(future_quat, _quat_inv(current_quat)))
    gripper_delta = future_state[..., 6:7] - current_state[:, None, 6:7]
    return torch.cat([pos_delta, rot_delta, gripper_delta], dim=-1)


def _match_axis_angle_branch(rot: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Choose the equivalent axis-angle branch closest to a reference vector.

    Ctrl-World consumes the numeric rotation-vector state, so an equivalent
    SO(3) rotation on the opposite axis-angle branch can still be out of
    distribution. LIBERO trajectories are smooth, making the current state a
    good branch reference for the future pose sequence.
    """
    norm = torch.linalg.norm(rot, dim=-1, keepdim=True)
    unit = rot / torch.clamp(norm, min=1e-8)
    candidates = torch.stack(
        [rot, rot + 2.0 * torch.pi * unit, rot - 2.0 * torch.pi * unit],
        dim=0,
    )
    distances = (candidates - reference.unsqueeze(0)).square().sum(dim=-1)
    best = distances.argmin(dim=0)
    gather_index = best.unsqueeze(0).unsqueeze(-1).expand(1, *rot.shape)
    return torch.gather(candidates, dim=0, index=gather_index).squeeze(0)


def _match_axis_angle_sequence(rot: torch.Tensor, initial_reference: torch.Tensor) -> torch.Tensor:
    if rot.ndim != 3:
        raise ValueError(f"rot must be [B,H,3], got {tuple(rot.shape)}")
    reference = initial_reference
    fixed_steps = []
    for step in range(rot.shape[1]):
        fixed = _match_axis_angle_branch(rot[:, step, :], reference)
        fixed_steps.append(fixed)
        reference = fixed
    return torch.stack(fixed_steps, dim=1)


def _dh_transform(
    a: float,
    d: float,
    alpha: float,
    theta: torch.Tensor,
) -> torch.Tensor:
    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = torch.as_tensor(math.cos(alpha), device=theta.device, dtype=theta.dtype)
    sa = torch.as_tensor(math.sin(alpha), device=theta.device, dtype=theta.dtype)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    return torch.stack(
        [
            torch.stack([ct, -st * ca, st * sa, torch.as_tensor(a, device=theta.device, dtype=theta.dtype) * ct], dim=-1),
            torch.stack([st, ct * ca, -ct * sa, torch.as_tensor(a, device=theta.device, dtype=theta.dtype) * st], dim=-1),
            torch.stack([zeros, sa.expand_as(theta), ca.expand_as(theta), torch.as_tensor(d, device=theta.device, dtype=theta.dtype).expand_as(theta)], dim=-1),
            torch.stack([zeros, zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )


def _rotation_matrix_to_axis_angle(rot: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    quat = _rotation_matrix_to_quat(rot, eps=eps)
    return _quat_to_axis_angle(quat, eps=eps)


def _rotation_matrix_to_quat(rot: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r00 = rot[..., 0, 0]
    r01 = rot[..., 0, 1]
    r02 = rot[..., 0, 2]
    r10 = rot[..., 1, 0]
    r11 = rot[..., 1, 1]
    r12 = rot[..., 1, 2]
    r20 = rot[..., 2, 0]
    r21 = rot[..., 2, 1]
    r22 = rot[..., 2, 2]

    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + r00 + r11 + r22, min=eps))
    qx = torch.copysign(
        0.5 * torch.sqrt(torch.clamp(1.0 + r00 - r11 - r22, min=eps)),
        r21 - r12,
    )
    qy = torch.copysign(
        0.5 * torch.sqrt(torch.clamp(1.0 - r00 + r11 - r22, min=eps)),
        r02 - r20,
    )
    qz = torch.copysign(
        0.5 * torch.sqrt(torch.clamp(1.0 - r00 - r11 + r22, min=eps)),
        r10 - r01,
    )
    quat = torch.stack([qw, qx, qy, qz], dim=-1)
    return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=eps)


def ur5_joint_actions_to_eef_states(
    joint_action_seq: torch.Tensor,
    *,
    branch_reference: torch.Tensor | None = None,
    tcp_xyz: tuple[float, float, float] = (0.0, 0.0, 0.18),
    tcp_rotvec: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Convert absolute UR5 joint action chunks to absolute EEF pose chunks.

    Args:
        joint_action_seq: `[B,H,7]` with six joint angles and gripper.
        branch_reference: optional `[B,7]` or `[B,3]` EEF state/rotvec used to
            keep rotation-vector branches continuous from the current state.

    Returns:
        `[B,H,7]` absolute EEF pose `[x,y,z,rx,ry,rz,gripper]`.
    """
    if joint_action_seq.ndim != 3 or joint_action_seq.shape[-1] < 7:
        raise ValueError(
            f"joint_action_seq must be [B,H,>=7], got {tuple(joint_action_seq.shape)}"
        )

    dtype = joint_action_seq.dtype
    device = joint_action_seq.device
    q = joint_action_seq[..., :6]
    flat_q = q.reshape(-1, 6)
    transform = torch.eye(4, device=device, dtype=dtype).expand(flat_q.shape[0], 4, 4).clone()
    dh_params = [
        (0.0, 0.089159, torch.pi / 2),
        (-0.425, 0.0, 0.0),
        (-0.39225, 0.0, 0.0),
        (0.0, 0.10915, torch.pi / 2),
        (0.0, 0.09465, -torch.pi / 2),
        (0.0, 0.0823, 0.0),
    ]
    for joint_idx, (a, d, alpha) in enumerate(dh_params):
        transform = transform @ _dh_transform(a, d, float(alpha), flat_q[:, joint_idx])

    tcp = torch.eye(4, device=device, dtype=dtype)
    tcp[:3, :3] = _axis_angle_to_matrix(
        torch.as_tensor(tcp_rotvec, device=device, dtype=dtype)
    )
    tcp[:3, 3] = torch.as_tensor(tcp_xyz, device=device, dtype=dtype)
    transform = transform @ tcp

    pos = transform[:, :3, 3]
    rotvec = _rotation_matrix_to_axis_angle(transform[:, :3, :3])
    pose = torch.cat([pos, rotvec], dim=-1).reshape(*joint_action_seq.shape[:2], 6)

    if branch_reference is not None:
        if branch_reference.shape[-1] >= 6:
            reference = branch_reference[..., 3:6]
        else:
            reference = branch_reference[..., :3]
        pose = pose.clone()
        pose[:, :, 3:6] = _match_axis_angle_sequence(pose[:, :, 3:6], reference)

    gripper = joint_action_seq[..., 6:7]
    return torch.cat([pose, gripper], dim=-1)


def compute_eef_rigid_alignment(
    source_state: torch.Tensor,
    target_state: torch.Tensor,
) -> torch.Tensor:
    """Return ``T_target_from_source`` for batched 7D EEF states.

    The transform aligns an FK pose in the robot model's base frame to the
    recorded EEF pose used by Ctrl-World for the same physical instant.
    Gripper channels are intentionally excluded from the rigid transform.
    """

    if source_state.ndim != 2 or source_state.shape[-1] < 6:
        raise ValueError(f"source_state must be [B,>=6], got {tuple(source_state.shape)}")
    if target_state.ndim != 2 or target_state.shape[-1] < 6:
        raise ValueError(f"target_state must be [B,>=6], got {tuple(target_state.shape)}")
    if source_state.shape[0] != target_state.shape[0]:
        raise ValueError("source_state and target_state batch sizes must match")

    def state_matrix(state: torch.Tensor) -> torch.Tensor:
        matrix = torch.eye(4, device=state.device, dtype=state.dtype).expand(
            state.shape[0], 4, 4
        ).clone()
        matrix[:, :3, :3] = _axis_angle_to_matrix(state[:, 3:6])
        matrix[:, :3, 3] = state[:, :3]
        return matrix

    source_matrix = state_matrix(source_state)
    target_matrix = state_matrix(
        target_state.to(device=source_state.device, dtype=source_state.dtype)
    )
    return target_matrix @ torch.linalg.inv(source_matrix)


def apply_eef_rigid_alignment(
    states: torch.Tensor,
    alignment: torch.Tensor,
    *,
    branch_reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a batched rigid alignment to ``[B,H,7]`` EEF trajectories."""

    if states.ndim != 3 or states.shape[-1] < 7:
        raise ValueError(f"states must be [B,H,>=7], got {tuple(states.shape)}")
    if alignment.shape != (states.shape[0], 4, 4):
        raise ValueError(
            f"alignment must be [B,4,4], got {tuple(alignment.shape)} for batch {states.shape[0]}"
        )

    alignment = alignment.to(device=states.device, dtype=states.dtype)
    pose_matrix = torch.eye(4, device=states.device, dtype=states.dtype).expand(
        states.shape[0], states.shape[1], 4, 4
    ).clone()
    pose_matrix[..., :3, :3] = _axis_angle_to_matrix(states[..., 3:6])
    pose_matrix[..., :3, 3] = states[..., :3]
    aligned_matrix = alignment[:, None, :, :] @ pose_matrix

    position = aligned_matrix[..., :3, 3]
    rotation = _rotation_matrix_to_axis_angle(aligned_matrix[..., :3, :3])
    if branch_reference is not None:
        reference = (
            branch_reference[..., 3:6]
            if branch_reference.shape[-1] >= 6
            else branch_reference[..., :3]
        )
        rotation = _match_axis_angle_sequence(rotation, reference)
    return torch.cat([position, rotation, states[..., 6:7]], dim=-1)


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    quat = _axis_angle_to_quat(axis_angle)
    w, x, y, z = quat.unbind(dim=-1)
    two = torch.as_tensor(2.0, dtype=quat.dtype, device=quat.device)
    return torch.stack(
        [
            torch.stack([1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)], dim=-1),
            torch.stack([two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w)], dim=-1),
            torch.stack([two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )


def compose_learned_eef_delta_to_states(
    current_state: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    """Compose learned-adapter SE(3) deltas with current EEF state."""
    if current_state.ndim != 2:
        raise ValueError(f"current_state must be [B,D], got {tuple(current_state.shape)}")
    if delta.ndim != 3:
        raise ValueError(f"delta must be [B,H,D], got {tuple(delta.shape)}")
    if current_state.shape[-1] < 7 or delta.shape[-1] < 7:
        raise ValueError(
            "current_state and delta must have at least 7 dimensions "
            f"got {tuple(current_state.shape)} and {tuple(delta.shape)}"
        )

    pos = current_state[:, None, :3] + delta[..., :3]
    current_quat = _axis_angle_to_quat(current_state[:, 3:6])[:, None, :]
    delta_quat = _axis_angle_to_quat(delta[..., 3:6])
    rot = _quat_to_axis_angle(_quat_mul(delta_quat, current_quat))
    rot = _match_axis_angle_sequence(rot, current_state[:, 3:6])
    gripper = current_state[:, None, 6:7] + delta[..., 6:7]
    return torch.cat([pos, rot, gripper], dim=-1)


def analytic_libero_delta_actions_to_eef_states(
    current_state: torch.Tensor,
    action_seq: torch.Tensor,
    *,
    position_scale: float = 0.05,
    rotation_scale: float = 0.5,
    gripper_open: float = 0.04,
    gripper_close: float = 0.0,
) -> torch.Tensor:
    """Analytically map LIBERO delta actions to future absolute EEF states.

    This mirrors robosuite OSC_POSE's high-level target update:
    position is current position plus scaled delta, and orientation is
    left-multiplied by the scaled axis-angle delta. It is a kinematic target
    approximation, not a dynamics/contact simulator.
    """
    if action_seq.ndim != 3:
        raise ValueError(f"action_seq must be [B,H,D], got {tuple(action_seq.shape)}")
    state = state_to_ctrl_world_eef_state(current_state).to(action_seq.dtype)
    pos = state[:, :3]
    rot = state[:, 3:6]
    gripper = state[:, 6:7]
    outputs = []
    for step in range(action_seq.shape[1]):
        action = action_seq[:, step]
        pos = pos + action[:, :3] * position_scale

        current_quat = _axis_angle_to_quat(rot)
        delta_quat = _axis_angle_to_quat(action[:, 3:6] * rotation_scale)
        # robosuite set_goal_orientation uses R_delta @ R_current.
        rot = _quat_to_axis_angle(_quat_mul(delta_quat, current_quat))

        if action.shape[-1] >= 7:
            gripper_cmd = action[:, 6:7]
            gripper = torch.where(
                gripper_cmd < 0,
                torch.full_like(gripper, gripper_open),
                torch.where(
                    gripper_cmd > 0,
                    torch.full_like(gripper, gripper_close),
                    gripper,
                ),
            )

        outputs.append(torch.cat([pos, rot, gripper], dim=-1))
    return torch.stack(outputs, dim=1)


def _rot6d_to_matrix(rot6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert a 6D rotation representation to a proper rotation matrix.

    Matches cosmos-framework's ``pose_utils.convert_rotation`` rot6d
    convention: the vector stores the matrix's first two COLUMNS
    (``[col0(3), col1(3)]``); the third column is ``cross(col0, col1)``. Raw
    network output is not guaranteed orthonormal, so the result is projected
    onto SO(3) via SVD (same projection as cosmos-framework's
    ``_normalize_rotation_matrices``).
    """
    # SVD/det have no bfloat16 cuBLAS kernel (e.g. "lu_factor_cublas" backs
    # det). Callers commonly run under torch.amp.autocast(dtype=bfloat16),
    # which would silently re-downcast the matmuls below (`@`) even though
    # the inputs were cast to float32 -- so disable autocast explicitly for
    # this whole block, not just the input dtype.
    input_dtype = rot6d.dtype
    with torch.autocast(device_type=rot6d.device.type, enabled=False):
        rot6d_f32 = rot6d.to(torch.float32)
        col0 = rot6d_f32[..., 0:3]
        col1 = rot6d_f32[..., 3:6]
        col2 = torch.linalg.cross(col0, col1, dim=-1)
        matrix = torch.stack([col0, col1, col2], dim=-1)

        u, _, vh = torch.linalg.svd(matrix)
        normalized = u @ vh
        det = torch.linalg.det(normalized)
        reflect = (det < 0).unsqueeze(-1).unsqueeze(-1)
        u_reflect = u.clone()
        u_reflect[..., -1] *= -1
        result = torch.where(reflect, u_reflect @ vh, normalized)
    return result.to(input_dtype)


def analytic_ur5_rot6d_delta_actions_to_eef_states(
    current_state: torch.Tensor,
    action_seq: torch.Tensor,
    *,
    translation_gain: float = 0.15,
    rotation_gain: float = 1.0,
    translation_frame: str = "body",
    translation_sign: float = 1.0,
    rotation_mode: str = "body",
) -> torch.Tensor:
    """Map UR5EEFLeRobotDataset-convention 10D body-frame delta actions to
    future absolute EEF states.

    Actions are ``[pos_delta(3), rot6d_delta(6), gripper_abs(1)]``, matching
    the checkpoint training data: the default delta transform is defined in
    the CURRENT frame (``T_{i+1} = T_i @ delta_T``). Concretely, defaults are
    ``p_{i+1} = p_i + R_i @ pos_delta`` and ``R_{i+1} = R_i @ R_delta``.
    Gripper is the dataset raw absolute qpos passthrough, not a delta.

    ``translation_frame``, ``translation_sign``, and ``rotation_mode`` are
    diagnostic convention-ablation knobs. Their defaults preserve the verified
    dataset convention above.
    """
    if action_seq.ndim != 3 or action_seq.shape[-1] < 10:
        raise ValueError(f"action_seq must be [B,H,>=10], got {tuple(action_seq.shape)}")
    translation_frame = str(translation_frame).lower()
    rotation_mode = str(rotation_mode).lower()
    if translation_frame not in {"body", "world", "initial_body", "none"}:
        raise ValueError(
            f"Unsupported translation_frame={translation_frame!r}; "
            "expected one of {body, world, initial_body, none}."
        )
    if rotation_mode not in {"body", "body_inverse", "world", "world_inverse", "absolute", "none"}:
        raise ValueError(
            f"Unsupported rotation_mode={rotation_mode!r}; expected one of "
            "{body, body_inverse, world, world_inverse, absolute, none}."
        )

    state = state_to_ctrl_world_eef_state(current_state).to(action_seq.dtype)
    translation_gain_t = torch.as_tensor(translation_gain, device=action_seq.device, dtype=action_seq.dtype)
    translation_sign_t = torch.as_tensor(translation_sign, device=action_seq.device, dtype=action_seq.dtype)
    rotation_gain_f = float(rotation_gain)
    pos = state[:, :3]
    rot_matrix = _axis_angle_to_matrix(state[:, 3:6])
    initial_rot_matrix = rot_matrix.clone()
    outputs = []
    for step in range(action_seq.shape[1]):
        action = action_seq[:, step]
        pos_delta = action[:, 0:3] * translation_gain_t * translation_sign_t
        rot6d_delta = action[:, 3:9]
        gripper = action[:, 9:10]

        delta_matrix = _rot6d_to_matrix(rot6d_delta)
        if abs(rotation_gain_f - 1.0) > 1e-6:
            delta_rotvec = _rotation_matrix_to_axis_angle(delta_matrix) * rotation_gain_f
            delta_matrix = _axis_angle_to_matrix(delta_rotvec)

        if translation_frame == "body":
            pos = pos + torch.einsum("bij,bj->bi", rot_matrix, pos_delta)
        elif translation_frame == "world":
            pos = pos + pos_delta
        elif translation_frame == "initial_body":
            pos = pos + torch.einsum("bij,bj->bi", initial_rot_matrix, pos_delta)

        if rotation_mode == "body":
            rot_matrix = rot_matrix @ delta_matrix
        elif rotation_mode == "body_inverse":
            rot_matrix = rot_matrix @ delta_matrix.transpose(-1, -2)
        elif rotation_mode == "world":
            rot_matrix = delta_matrix @ rot_matrix
        elif rotation_mode == "world_inverse":
            rot_matrix = delta_matrix.transpose(-1, -2) @ rot_matrix
        elif rotation_mode == "absolute":
            rot_matrix = delta_matrix
        rot = _rotation_matrix_to_axis_angle(rot_matrix)

        outputs.append(torch.cat([pos, rot, gripper], dim=-1))
    return torch.stack(outputs, dim=1)
