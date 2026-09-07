#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="General-purpose WSI patch extraction")
    parser.add_argument("inputs", nargs="*", help="One or more WSI paths")
    parser.add_argument("--input-list", type=Path, help="TXT containing one WSI path per line")
    parser.add_argument("--input-root", type=Path, help="Root for relative paths in TXT")
    parser.add_argument("--config", type=Path, help="YAML or JSON config")
    parser.add_argument("--output", type=Path, help="Override output root")
    parser.add_argument("--preview", action="store_true", help="Only save contour and sample patches")
    parser.add_argument("--n-patches", type=int, help="Random preview patches per WSI (default: 8)")
    parser.add_argument("--inspect", action="store_true", help="Resolve inputs without opening slides")
    parser.add_argument("--show-config", action="store_true", help="Print resolved config and exit")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one config value; repeatable",
    )
    return parser


def _input_value(args: argparse.Namespace) -> str | Path | Sequence[str | Path]:
    if args.input_list and args.inputs:
        raise ValueError("Use positional inputs or --input-list, not both")
    if args.input_list:
        return args.input_list
    if args.inputs:
        return args.inputs
    raise ValueError("At least one WSI path or --input-list is required")


def main(argv: Sequence[str] | None = None) -> int:
    from extract_patch.config import dump_config, load_config
    from extract_patch.inputs import parse_inputs

    args = build_parser().parse_args(argv)
    config_path = args.config
    if config_path is not None and not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config = load_config(config_path, args.set)
    if args.show_config:
        print(dump_config(config), end="")
        return 0

    try:
        specs = parse_inputs(_input_value(args), root=args.input_root)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Input error: {exc}") from exc

    if args.inspect:
        for spec in specs:
            sidecars = ",".join(str(path) for path in spec.sidecars) or "-"
            print(
                f"{spec.slide_id}\tsource={spec.source}\tentrypoint={spec.entrypoint}"
                f"\tsidecars={sidecars}"
            )
        return 0

    if args.preview:
        from extract_patch.preview import preview

        if args.output is not None:
            config.preview.root = str(args.output)
        if args.n_patches is not None:
            config.preview.n_patches = args.n_patches
        result = preview(specs, config, run_id=args.run_id)
        total_patches = sum(slide.patch_count for slide in result.slides)
        print(
            f"status={result.status} slides={len(result.slides)} "
            f"planned_patches={total_patches} "
            f"output={result.output_root} log={result.log_dir}"
        )
    else:
        from extract_patch.pipeline import extract

        if args.output is not None:
            config.output.root = str(args.output)
        result = extract(specs, config, run_id=args.run_id)
        print(
            f"status={result.status} slides={len(result.slides)} "
            f"output={result.output_root} log={result.log_dir}"
        )

    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
