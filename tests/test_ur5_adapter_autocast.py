"""SE(3) endpoint regression under the diffusion caller's autocast context."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
Rotation = pytest.importorskip("scipy.spatial.transform").Rotation

from cosmos_framework.data.vfm.action.pose_utils import build_abs_pose_from_components, pose_abs_to_rel
from rlinf.envs.world_model.world_model_ctrl_world_env import CtrlWorldEnv


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ur5_demo_endpoint_survives_ambient_bfloat16_autocast(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for production autocast regression")
    env = CtrlWorldEnv.__new__(CtrlWorldEnv)
    env.device = torch.device(device)
    env.adapter_translation_gain = env.adapter_rotation_gain = env.adapter_translation_sign = 1.0
    env.adapter_translation_frame = env.adapter_rotation_mode = "body"
    env.adapter_gripper_hard_clamp = False
    env.action_fps_resample = True
    env.cosmos_action_fps, env.ctrl_condition_fps = 15.0, 30.0
    env.ctrl_world_chunk = 64
    env.ctrl_world_internal_rollout = True

    # A metric trajectory with coupled translation and rotation; its endpoint
    # is independent of the adapter implementation and known from the poses.
    t = np.linspace(0, 2*np.pi, 129)
    xyz = np.column_stack([-.35+.07*np.sin(t), -.04+.03*np.sin(2*t), .3+.02*(np.cos(t)-1)])
    rotations = Rotation.from_rotvec([2.0, -1.8, .3]) * Rotation.from_rotvec(
        np.column_stack([.12*np.sin(t), .18*np.sin(2*t), .08*(np.cos(t)-1)]))
    absolute = np.column_stack([xyz, rotations.as_rotvec(), np.full(len(t), .8980392)]).astype(np.float32)
    env.current_states = torch.from_numpy(absolute[:1]).to(env.device)
    max_position_error = max_rotation_error = 0.0
    for chunk in range(4):
        waypoints = absolute[chunk*32:chunk*32+33]
        poses = build_abs_pose_from_components(waypoints[:, :3], waypoints[:, 3:6], "axisangle")
        delta = pose_abs_to_rel(poses, rotation_format="rot6d", pose_convention="backward_framewise")
        actions = np.concatenate([delta, waypoints[1:, 6:7]], axis=1)[None]
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            conditions, endpoint = env._convert_via_analytic_ur5_rot6d(actions)
        assert conditions.shape == (1, 64, 7)
        assert conditions.dtype == endpoint.dtype == np.float32
        position_error = np.linalg.norm(endpoint[0, :3]-waypoints[-1, :3])
        rotation_error = (Rotation.from_rotvec(endpoint[0, 3:6]).inv() * Rotation.from_rotvec(waypoints[-1, 3:6])).magnitude()
        max_position_error = max(max_position_error, float(position_error))
        max_rotation_error = max(max_rotation_error, float(rotation_error))
        env.current_states = torch.from_numpy(endpoint).to(env.device)
    assert max_position_error < 1e-4, max_position_error
    assert max_rotation_error < 1e-3, max_rotation_error
