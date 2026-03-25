"""CLI entrypoint for the HEROS to LLM experiment package."""

from __future__ import annotations

import argparse

from .env_utils import discover_default_env_files, load_env_file
from .runner import run_experiment_from_config_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the HEROS to LLM explanation experiment.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON config file under evaluation/experiments/heros_llm/configs.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Optional path to a .env file. If omitted, the CLI will auto-load .env from the cwd or repo root when present.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    for env_path in discover_default_env_files(args.env_file):
        load_env_file(env_path)
    output_dir = run_experiment_from_config_path(args.config)
    print("Experiment artifacts written to {0}".format(output_dir))


if __name__ == "__main__":
    main()
