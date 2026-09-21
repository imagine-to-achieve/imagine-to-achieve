"""Fail if model weights, datasets, logs, or external source symlinks enter this repository."""

from __future__ import annotations

from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".distcp", ".onnx", ".h5", ".npz"
}
FORBIDDEN_DIRS = {"logs", "wandb", "outputs", "checkpoints", "datasets"}
RUNTIME_ROOTS = {"outputs"}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        # Runtime outputs may contain newly trained checkpoints; they are not vendored assets.
        if relative.parts[0] in RUNTIME_ROOTS:
            continue
        if any(part in {".git", ".pytest_cache", "__pycache__"} for part in relative.parts):
            continue
        if path.is_symlink():
            violations.append(f"symlink: {relative}")
        elif path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"weight/data file: {relative}")
        elif path.is_dir() and path.name.lower() in FORBIDDEN_DIRS and relative.parts[0] != "third_party":
            violations.append(f"vendored generated directory: {relative}")
    if violations:
        raise SystemExit("forbidden repository contents:\n- " + "\n- ".join(sorted(violations)))
    print("no weights, datasets, logs, checkpoints, or external symlinks found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
