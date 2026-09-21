"""Task prompt construction shared by rollout and checkpoint preflight."""

from __future__ import annotations


def _sentence(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        raise ValueError("base prompt must not be empty")
    return stripped if stripped.endswith((".", "!", "?")) else stripped + "."


def build_action_prompt(
    base_prompt: str,
    *,
    camera_labels: tuple[str, str, str],
    num_frames: int,
    fps: float,
    resolution: tuple[int, int],
    append_viewpoint: bool,
    append_duration_fps: bool,
    append_resolution: bool,
) -> str:
    """Build the same string-form metadata sequence used by Action SFT."""

    if num_frames <= 0 or fps <= 0:
        raise ValueError("num_frames and fps must be positive")
    pieces = [_sentence(base_prompt)]
    if append_viewpoint:
        top, left, right = camera_labels
        pieces.append("This video contains concatenated views from multiple camera perspectives.")
        pieces.append(f"The top row is the {top} view.")
        pieces.append(
            "The bottom row contains two horizontally concatenated views: "
            f"the {left} view on the left and the {right} view on the right."
        )
    if append_duration_fps:
        # Cosmos SFT uses int(num_frames / fps) before formatting with one decimal.
        duration = int(num_frames / fps)
        pieces.append(
            f"The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
        )
    if append_resolution:
        height, width = resolution
        pieces.append(f"This video is of {height}x{width} resolution.")
    return " ".join(pieces)

