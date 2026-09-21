"""Command-line entry point for standalone synthetic and real training."""

from __future__ import annotations

import argparse
import json

from rlinf_modified.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to one fully explicit YAML profile")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate a real profile and its assets without launching Ray workers",
    )
    mode.add_argument(
        "--compose-only",
        action="store_true",
        help="Run preflight and resolve the full Hydra config without launching Ray",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if config.runtime.backend == "single":
        if args.preflight_only:
            print(json.dumps({"state": "valid", "backend": "single"}, sort_keys=True))
            return 0
        from rlinf_modified.engine.synthetic import SyntheticTrainer

        result = SyntheticTrainer(config).run()
    else:
        from rlinf_modified.engine.real import RealTrainer

        trainer = RealTrainer(config)
        if args.preflight_only:
            result = trainer.preflight()
        elif args.compose_only:
            result = trainer.validate_composition()
        else:
            result = trainer.run()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
