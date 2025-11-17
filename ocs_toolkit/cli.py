"""Command-line runner for the AnswererWrapper adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from .answerer_wrapper import AnswererWrapperAdapter, load_config_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AnswererWrapper configs from Python.")
    parser.add_argument("--config", required=True, help="Path to the wrapper config JSON file.")
    parser.add_argument("--title", required=True, help="Question title to search for.")
    parser.add_argument("--question-type", default="single", help="Question type, e.g. single or multiple.")
    parser.add_argument("--options", default="", help="Question options separated by newlines (use literal \\n).")
    parser.add_argument(
        "--options-file",
        help="Load question options from a text file instead of --options.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional key=value pairs injected into the adapter environment.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output for downstream tooling.",
    )
    args = parser.parse_args()

    options = args.options
    if args.options_file:
        options = Path(args.options_file).read_text(encoding="utf-8")

    env = {"title": args.title, "type": args.question_type, "options": options}
    env.update(_parse_kv_pairs(args.env))

    configs = load_config_file(args.config)
    adapter = AnswererWrapperAdapter(configs)
    result = adapter.run(env)

    payload = {
        "matches": [match.__dict__ for match in result.matches],
        "errors": [error.__dict__ for error in result.errors],
    }

    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))

    return 0 if result.matches else 1 if result.errors else 0


def _parse_kv_pairs(pairs: list[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for entry in pairs:
        if "=" not in entry:
            raise ValueError(f"Invalid env override '{entry}', expected key=value format")
        key, value = entry.split("=", 1)
        parsed[key.strip()] = value
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
