"""End-to-end orchestration for the HEROS to LLM experiment."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import random
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .cache import FileCache, hash_payload
from .config import ExperimentConfig, load_experiment_config
from .data_models import (
    ExplanationRecord,
    ExperimentMetadata,
    GeneratedExplanation,
    InstanceExplanationPacket,
    JudgeMetrics,
    to_serializable,
)
from .dataset_registry import get_dataset_definition
from .heros_adapter import build_packets_for_split, train_heros_model
from .judge import build_judge_prompt, parse_judge_response
from .metrics_programmatic import compute_programmatic_metrics
from .openai_client import OpenAIClientWrapper
from .prompt_builder import build_prompt_bundle
from .results import ResultsWriter


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_sha() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return output.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _rule_bucket(num_matching_rules: int) -> str:
    if num_matching_rules <= 1:
        return "1"
    if num_matching_rules <= 3:
        return "2-3"
    return "4+"


def _packet_manifest_row(packet: InstanceExplanationPacket) -> Dict[str, Any]:
    return {
        "dataset_name": packet.dataset_name,
        "split": packet.split,
        "instance_id": packet.instance_id,
        "prediction": packet.model_context.prediction,
        "num_matching_rules": packet.model_context.num_matching_rules,
        "rule_bucket": _rule_bucket(packet.model_context.num_matching_rules),
        "agreement_status": packet.model_context.agreement_status,
        "evidence_strength_label": packet.model_context.evidence_strength_label,
    }


def _select_packets(
    packets: List[InstanceExplanationPacket], sample_size: int, seed: int, use_full_test_set: bool
) -> List[InstanceExplanationPacket]:
    if use_full_test_set or sample_size >= len(packets):
        return list(packets)

    grouped: Dict[Tuple[str, str], List[InstanceExplanationPacket]] = {}
    for packet in packets:
        key = (str(packet.model_context.prediction), _rule_bucket(packet.model_context.num_matching_rules))
        grouped.setdefault(key, []).append(packet)

    rng = random.Random(seed)
    group_keys = sorted(grouped.keys())
    for key in group_keys:
        rng.shuffle(grouped[key])

    selected: List[InstanceExplanationPacket] = []
    while len(selected) < sample_size:
        progress_made = False
        for key in group_keys:
            if grouped[key] and len(selected) < sample_size:
                selected.append(grouped[key].pop())
                progress_made = True
        if not progress_made:
            break
    return selected


def _build_experiment_metadata(
    config: ExperimentConfig,
    dataset_name: str,
    sample_size: int,
    dataset_definition: Any,
) -> ExperimentMetadata:
    return ExperimentMetadata(
        run_id="",
        git_sha=_git_sha(),
        dataset_name=dataset_name,
        split=config.split,
        sample_size=sample_size,
        sample_seed=config.sampling.seed,
        sampling_strategy=config.sampling.strategy,
        target_model_selection=(
            "auto_select_top_model"
            if config.heros.target_model_index is None
            else "fixed_index_{0}".format(config.heros.target_model_index)
        ),
        llm_model=config.llm.model,
        judge_model=config.judge.model if config.judge.enabled else "",
        temperature=config.llm.temperature,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config_hash=config.config_hash(),
        train_path=str(dataset_definition.train_path),
        test_path=str(dataset_definition.test_path),
    )


def _cache_key_for_prompt(prompt_bundle: Any, config: ExperimentConfig) -> str:
    return hash_payload(
        {
            "model": config.llm.model,
            "temperature": config.llm.temperature,
            "prompt_version": prompt_bundle.prompt_version,
            "condition": prompt_bundle.condition,
            "audience": prompt_bundle.audience,
            "system_prompt": prompt_bundle.system_prompt,
            "user_prompt": prompt_bundle.user_prompt,
        }
    )


def _cache_key_for_judge(explanation_record: ExplanationRecord, judge_prompt: Any, config: ExperimentConfig) -> str:
    return hash_payload(
        {
            "model": config.judge.model,
            "temperature": config.judge.temperature,
            "prompt_version": config.judge.prompt_version,
            "system_prompt": judge_prompt.system_prompt,
            "user_prompt": judge_prompt.user_prompt,
            "explanation_text": explanation_record.generation.raw_text,
        }
    )


def run_experiment(config: ExperimentConfig) -> str:
    """Run the full HEROS to LLM experiment from a resolved config."""
    dataset_definition = get_dataset_definition(config.dataset_name)
    trained_context = train_heros_model(config, dataset_definition)
    all_packets = build_packets_for_split(
        context=trained_context, split=config.split, prompt_config=config.prompt
    )
    selected_packets = _select_packets(
        all_packets,
        sample_size=config.sampling.sample_size,
        seed=config.sampling.seed,
        use_full_test_set=config.sampling.use_full_test_set,
    )

    run_id = "{0}_{1}".format(config.run_name, _timestamp_utc())
    metadata_template = _build_experiment_metadata(
        config, dataset_definition.name, len(selected_packets), dataset_definition
    )
    metadata_template.run_id = run_id

    writer = ResultsWriter(config.output, run_id)
    writer.write_config_snapshot(config)
    writer.write_csv(
        writer.run_paths.sample_manifest_path,
        [_packet_manifest_row(packet) for packet in selected_packets],
    )
    if config.output.write_packets:
        writer.write_jsonl(
            writer.run_paths.packets_path,
            [to_serializable(packet) for packet in selected_packets],
        )

    prompt_rows: List[Dict[str, Any]] = []
    generation_rows: List[Dict[str, Any]] = []
    judge_request_rows: List[Dict[str, Any]] = []
    judge_result_rows: List[Dict[str, Any]] = []
    explanation_records: List[ExplanationRecord] = []

    generation_cache = FileCache(str(writer.run_paths.cache_dir / "generations"))
    judge_cache = FileCache(str(writer.run_paths.cache_dir / "judge"))
    llm_client = OpenAIClientWrapper(config.llm) if config.llm.enabled else None
    judge_client = OpenAIClientWrapper(config.judge) if config.judge.enabled else None

    for packet in selected_packets:
        for condition in config.prompt.conditions:
            for audience in config.prompt.audiences:
                packet_for_prompt = deepcopy(packet)
                packet_for_prompt.condition = condition
                packet_for_prompt.audience = audience
                prompt_bundle = build_prompt_bundle(packet_for_prompt, condition, audience, config.prompt)
                prompt_rows.append(
                    {
                        "dataset_name": packet.dataset_name,
                        "instance_id": packet.instance_id,
                        "condition": condition,
                        "audience": audience,
                        "prompt_version": prompt_bundle.prompt_version,
                        "system_prompt": prompt_bundle.system_prompt,
                        "user_prompt": prompt_bundle.user_prompt,
                    }
                )

                if not llm_client:
                    continue

                generation_key = _cache_key_for_prompt(prompt_bundle, config)
                generation_payload = generation_cache.get(generation_key)
                if generation_payload is None:
                    generation_payload = llm_client.generate_text(
                        prompt_bundle.system_prompt, prompt_bundle.user_prompt
                    )
                    generation_cache.set(generation_key, generation_payload)

                generated = GeneratedExplanation(
                    condition=condition,
                    audience=audience,
                    system_prompt=prompt_bundle.system_prompt,
                    user_prompt=prompt_bundle.user_prompt,
                    raw_text=generation_payload["text"],
                    model_name=generation_payload["model_name"],
                    temperature=float(generation_payload["temperature"]),
                    created_at=generation_payload["created_at"],
                )
                generation_rows.append(
                    {
                        "dataset_name": packet.dataset_name,
                        "instance_id": packet.instance_id,
                        "condition": condition,
                        "audience": audience,
                        "generation_key": generation_key,
                        "raw_text": generated.raw_text,
                        "raw_response": generation_payload["raw_response"],
                    }
                )

                programmatic_metrics = compute_programmatic_metrics(
                    packet_for_prompt,
                    generated.raw_text,
                    top_k_rules=config.prompt.key_rules_top_k,
                )
                record = ExplanationRecord(
                    experiment_metadata=deepcopy(metadata_template),
                    packet=packet_for_prompt,
                    prompt=prompt_bundle,
                    generation=generated,
                    programmatic_metrics=programmatic_metrics,
                    judge_metrics=JudgeMetrics(),
                )

                if judge_client:
                    judge_prompt = build_judge_prompt(record, config.judge)
                    judge_request_rows.append(
                        {
                            "dataset_name": packet.dataset_name,
                            "instance_id": packet.instance_id,
                            "condition": condition,
                            "audience": audience,
                            "system_prompt": judge_prompt.system_prompt,
                            "user_prompt": judge_prompt.user_prompt,
                            "prompt_version": judge_prompt.prompt_version,
                        }
                    )
                    judge_key = _cache_key_for_judge(record, judge_prompt, config)
                    judge_payload = judge_cache.get(judge_key)
                    if judge_payload is None:
                        judge_payload = judge_client.generate_text(
                            judge_prompt.system_prompt, judge_prompt.user_prompt
                        )
                        judge_cache.set(judge_key, judge_payload)
                    record.judge_metrics = parse_judge_response(judge_payload["text"], config.judge)
                    judge_result_rows.append(
                        {
                            "dataset_name": packet.dataset_name,
                            "instance_id": packet.instance_id,
                            "condition": condition,
                            "audience": audience,
                            "judge_key": judge_key,
                            "raw_text": judge_payload["text"],
                            "raw_response": judge_payload["raw_response"],
                        }
                    )

                explanation_records.append(record)

    if config.output.write_prompts:
        writer.write_jsonl(writer.run_paths.prompts_path, prompt_rows)
    if config.output.write_generations:
        writer.write_jsonl(writer.run_paths.generations_path, generation_rows)
        writer.write_jsonl(writer.run_paths.judge_requests_path, judge_request_rows)
        writer.write_jsonl(writer.run_paths.judge_results_path, judge_result_rows)
    if config.output.write_records:
        writer.write_jsonl(
            writer.run_paths.records_path,
            [to_serializable(record) for record in explanation_records],
        )
    if config.output.write_csv:
        writer.write_csv(
            writer.run_paths.records_csv_path,
            [to_serializable(record) for record in explanation_records],
        )
    writer.write_aggregate_metrics(explanation_records)
    return str(writer.run_paths.run_dir)


def run_experiment_from_config_path(config_path: str) -> str:
    """Load a config file and execute the experiment."""
    config = load_experiment_config(config_path)
    return run_experiment(config)
