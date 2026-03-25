"""Post-hoc judge scoring for an existing HEROS to LLM run folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

from .cache import FileCache, hash_payload
from .config import JudgeConfig
from .env_utils import discover_default_env_files, load_env_file
from .judge import JUDGE_SYSTEM_PROMPT, parse_judge_response
from .openai_client import OpenAIClientWrapper
from .results import flatten_dict, ResultsWriter


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
        "system_prompt": JUDGE_SYSTEM_PROMPT,
        "user_prompt": """Audience: {audience}
Condition: {condition}
Prediction: {prediction}
Agreement status: {agreement_status}
Evidence strength: {evidence_strength}

Explanation:
{explanation}

Score clarity and technical appropriateness for the intended audience, then return only JSON.""".format(
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


def aggregate_records_from_payloads(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics directly from serialized records."""
    if not records:
        return {
            "record_count": 0,
            "hallucination_rate": None,
            "mean_feature_grounding_score": None,
            "mean_key_feature_coverage": None,
            "mean_prediction_consistency": None,
            "mean_word_count": None,
            "uncertainty_ack_rate_on_required_cases": None,
            "mean_rule_mention_coverage": None,
            "mean_conflict_awareness_score": None,
            "confidence_calibration_pass_rate": None,
            "causal_overclaim_rate": None,
            "mean_clarity_score": None,
            "mean_technical_appropriateness_score": None,
        }

    def avg(values: List[float]) -> Optional[float]:
        return mean(values) if values else None

    faithfulness = [record["programmatic_metrics"] for record in records]
    judge_metrics = [record.get("judge_metrics", {}) for record in records]
    required_cases = [metrics for metrics in faithfulness if metrics.get("uncertainty_ack_required")]

    return {
        "record_count": len(records),
        "hallucination_rate": avg(
            [1.0 if metrics.get("hallucination_present") else 0.0 for metrics in faithfulness]
        ),
        "mean_feature_grounding_score": avg(
            [
                float(metrics["feature_grounding_score"])
                for metrics in faithfulness
                if metrics.get("feature_grounding_score") is not None
            ]
        ),
        "mean_key_feature_coverage": avg(
            [
                float(metrics["key_feature_coverage"])
                for metrics in faithfulness
                if metrics.get("key_feature_coverage") is not None
            ]
        ),
        "mean_prediction_consistency": avg(
            [
                float(metrics["prediction_consistency"])
                for metrics in faithfulness
                if metrics.get("prediction_consistency") is not None
            ]
        ),
        "mean_word_count": avg([float(metrics.get("word_count", 0)) for metrics in faithfulness]),
        "uncertainty_ack_rate_on_required_cases": avg(
            [1.0 if metrics.get("uncertainty_ack_present") else 0.0 for metrics in required_cases]
        ),
        "mean_rule_mention_coverage": avg(
            [
                float(metrics["rule_mention_coverage"])
                for metrics in faithfulness
                if metrics.get("rule_mention_coverage") is not None
            ]
        ),
        "mean_conflict_awareness_score": avg(
            [
                float(metrics["conflict_awareness_score"])
                for metrics in faithfulness
                if metrics.get("conflict_awareness_score") is not None
            ]
        ),
        "confidence_calibration_pass_rate": avg(
            [1.0 if metrics.get("confidence_calibration_pass") else 0.0 for metrics in faithfulness]
        ),
        "causal_overclaim_rate": avg(
            [1.0 if metrics.get("causal_overclaim_present") else 0.0 for metrics in faithfulness]
        ),
        "mean_clarity_score": avg(
            [
                float(metrics["clarity_score"])
                for metrics in judge_metrics
                if metrics.get("clarity_score") is not None
            ]
        ),
        "mean_technical_appropriateness_score": avg(
            [
                float(metrics["technical_appropriateness_score"])
                for metrics in judge_metrics
                if metrics.get("technical_appropriateness_score") is not None
            ]
        ),
    }


def write_records_csv(run_dir: Path, records: List[Dict[str, Any]]) -> None:
    """Rewrite the flattened CSV for updated records."""
    rows = [flatten_dict(record) for record in records]
    ResultsWriter.write_csv(run_dir / "records.csv", rows)


def write_summary_tables(run_dir: Path, records: List[Dict[str, Any]]) -> None:
    """Write grouped CSV and Markdown summaries including judge metrics."""
    import pandas as pd

    rows = []
    for record in records:
        pm = record["programmatic_metrics"]
        jm = record.get("judge_metrics", {})
        rows.append(
            {
                "Condition": "B" if record["prompt"]["condition"] == "condition_b" else "C",
                "Audience": record["prompt"]["audience"].title(),
                "FGS": pm.get("feature_grounding_score"),
                "HR": 1.0 if pm.get("hallucination_present") else 0.0,
                "KFC": pm.get("key_feature_coverage"),
                "PC": pm.get("prediction_consistency"),
                "UAR": 1.0 if pm.get("uncertainty_ack_present") else 0.0,
                "COR": 1.0 if pm.get("causal_overclaim_present") else 0.0,
                "RMC": pm.get("rule_mention_coverage"),
                "CAS": pm.get("conflict_awareness_score"),
                "CCP": 1.0 if pm.get("confidence_calibration_pass") else 0.0,
                "Clarity": jm.get("clarity_score"),
                "TAS": jm.get("technical_appropriateness_score"),
                "WordCount": pm.get("word_count"),
            }
        )

    df = pd.DataFrame(rows)
    agg_map = {
        "n": ("Condition", "size"),
        "FGS": ("FGS", "mean"),
        "HR": ("HR", "mean"),
        "KFC": ("KFC", "mean"),
        "PC": ("PC", "mean"),
        "UAR": ("UAR", "mean"),
        "COR": ("COR", "mean"),
        "RMC": ("RMC", "mean"),
        "CAS": ("CAS", "mean"),
        "CCP": ("CCP", "mean"),
        "Clarity": ("Clarity", "mean"),
        "TAS": ("TAS", "mean"),
        "WordCount": ("WordCount", "mean"),
    }

    condition = df.groupby(["Condition"], dropna=False).agg(**agg_map).reset_index()
    audience = df.groupby(["Audience"], dropna=False).agg(**agg_map).reset_index()
    condition_audience = df.groupby(["Condition", "Audience"], dropna=False).agg(**agg_map).reset_index()

    condition.to_csv(run_dir / "summary_by_condition.csv", index=False)
    audience.to_csv(run_dir / "summary_by_audience.csv", index=False)
    condition_audience.to_csv(run_dir / "summary_condition_by_audience.csv", index=False)

    def pct(value: Any) -> str:
        if value is None or pd.isna(value):
            return "NA"
        return "{0:.1f}".format(float(value) * 100.0)

    def score(value: Any) -> str:
        if value is None or pd.isna(value):
            return "NA"
        return "{0:.3f}".format(float(value))

    def count(value: Any) -> str:
        return str(int(value))

    def wc(value: Any) -> str:
        return "{0:.1f}".format(float(value))

    def display_df(dataframe: "pd.DataFrame") -> "pd.DataFrame":
        output = dataframe.copy()
        for column in ["FGS", "HR", "KFC", "PC", "UAR", "COR", "RMC", "CAS", "CCP"]:
            output[column] = output[column].map(pct)
        for column in ["Clarity", "TAS"]:
            output[column] = output[column].map(score)
        output["n"] = output["n"].map(count)
        output["WordCount"] = output["WordCount"].map(wc)
        return output

    def to_markdown_table(dataframe: "pd.DataFrame") -> str:
        columns = list(dataframe.columns)
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = [
            "| " + " | ".join(str(row[column]) for column in columns) + " |"
            for _, row in dataframe.iterrows()
        ]
        return "\n".join([header, separator] + body)

    markdown = "\n".join(
        [
            "# MUX6 HEROS-LLM Summary Tables",
            "",
            "Source run: `{0}`".format(run_dir.name),
            "",
            "Metrics reported as means. Percentage-style metrics are shown on a 0-100 scale.",
            "",
            "## Table 1. Condition Comparison",
            to_markdown_table(display_df(condition)),
            "",
            "## Table 2. Audience Comparison",
            to_markdown_table(display_df(audience)),
            "",
            "## Table 3. Condition x Audience",
            to_markdown_table(display_df(condition_audience)),
        ]
    )
    (run_dir / "summary_tables.md").write_text(markdown, encoding="utf-8")


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
        prompt_version="v1",
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
        json.dump(aggregate_records_from_payloads(records), handle, indent=2, sort_keys=True)
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
