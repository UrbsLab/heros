"""Post-hoc judge scoring for an existing HEROS to LLM run folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .cache import FileCache, hash_payload
from .config import JudgeConfig
from .env_utils import discover_default_env_files, load_env_file
from .judge import JUDGE_SYSTEM_PROMPTS, parse_judge_response
from .openai_client import OpenAIClientWrapper
from .results import (
    ResultsWriter,
    aggregate_serialized_records,
    flatten_dict,
    write_summary_tables,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load newline-delimited JSON records."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write newline-delimited JSON rows."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_judge_prompt_from_record(record: Dict[str, Any]) -> Dict[str, str]:
    """Build the judge prompt directly from a serialized record."""
    packet = record["packet"]
    model_context = packet["model_context"]
    generation = record["generation"]
    prompt = record["prompt"]
    return {
        "system_prompt": JUDGE_SYSTEM_PROMPTS["v2"],
        "user_prompt": """Audience: {audience}
Condition: {condition}
Prediction: {prediction}
Agreement status: {agreement_status}
Evidence strength: {evidence_strength}

Explanation:
{explanation}

Score audience understandability and audience technical fit for the intended audience, then return only JSON.""".format(
            audience=prompt["audience"],
            condition=prompt["condition"],
            prediction=model_context["prediction"],
            agreement_status=model_context["agreement_status"],
            evidence_strength=model_context["evidence_strength_label"],
            explanation=generation["raw_text"],
        ),
    }


def judge_cache_key(record: Dict[str, Any], judge_config: JudgeConfig, judge_prompt: Dict[str, str]) -> str:
    """Create a stable cache key for a judge-model request."""
    return hash_payload(
        {
            "model": judge_config.model,
            "temperature": judge_config.temperature,
            "prompt_version": judge_config.prompt_version,
            "system_prompt": judge_prompt["system_prompt"],
            "user_prompt": judge_prompt["user_prompt"],
            "explanation_text": record["generation"]["raw_text"],
        }
    )


def write_records_csv(run_dir: Path, records: List[Dict[str, Any]]) -> None:
    """Rewrite the flattened CSV for updated records."""
    rows = [flatten_dict(record) for record in records]
    ResultsWriter.write_csv(run_dir / "records.csv", rows)


def run_judge_scoring(run_dir: str, env_file: Optional[str], model: str, temperature: float) -> Path:
    """Apply judge-model scoring to an existing run directory."""
    for env_path in discover_default_env_files(env_file):
        load_env_file(env_path)

    target_dir = Path(run_dir).expanduser().resolve()
    records_path = target_dir / "records.jsonl"
    records = load_jsonl(records_path)
    if not records:
        raise RuntimeError("No records found at {0}".format(records_path))

    judge_config = JudgeConfig(
        enabled=True,
        model=model,
        temperature=temperature,
        prompt_version="v2",
    )
    client = OpenAIClientWrapper(judge_config)
    cache = FileCache(str(target_dir / "cache" / "judge"))

    judge_requests: List[Dict[str, Any]] = []
    judge_results: List[Dict[str, Any]] = []

    for record in records:
        judge_prompt = build_judge_prompt_from_record(record)
        judge_requests.append(
            {
                "dataset_name": record["packet"]["dataset_name"],
                "instance_id": record["packet"]["instance_id"],
                "condition": record["prompt"]["condition"],
                "audience": record["prompt"]["audience"],
                "system_prompt": judge_prompt["system_prompt"],
                "user_prompt": judge_prompt["user_prompt"],
                "prompt_version": judge_config.prompt_version,
            }
        )
        key = judge_cache_key(record, judge_config, judge_prompt)
        payload = cache.get(key)
        if payload is None:
            payload = client.generate_text(judge_prompt["system_prompt"], judge_prompt["user_prompt"])
            cache.set(key, payload)
        judge_metric = parse_judge_response(payload["text"], judge_config)
        record["judge_metrics"] = judge_metric.to_dict()
        judge_results.append(
            {
                "dataset_name": record["packet"]["dataset_name"],
                "instance_id": record["packet"]["instance_id"],
                "condition": record["prompt"]["condition"],
                "audience": record["prompt"]["audience"],
                "judge_key": key,
                "raw_text": payload["text"],
                "raw_response": payload["raw_response"],
            }
        )

    write_jsonl(target_dir / "judge_requests.jsonl", judge_requests)
    write_jsonl(target_dir / "judge_results.jsonl", judge_results)
    write_jsonl(records_path, records)
    write_records_csv(target_dir, records)
    with (target_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate_serialized_records(records), handle, indent=2, sort_keys=True)
    write_summary_tables(target_dir, records)
    return target_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run post-hoc judge scoring for an existing run.")
    parser.add_argument("--run-dir", required=True, help="Path to an existing run directory.")
    parser.add_argument(
        "--env-file",
        default=None,
        help="Optional path to a .env file. If omitted, .env from cwd or repo root is loaded when present.",
    )
    parser.add_argument("--model", default="gpt-4.1-mini", help="Judge model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Judge model temperature.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    updated_dir = run_judge_scoring(
        run_dir=args.run_dir,
        env_file=args.env_file,
        model=args.model,
        temperature=args.temperature,
    )
    print("Judge scoring written to {0}".format(updated_dir))


if __name__ == "__main__":
    main()
