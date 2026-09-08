from __future__ import annotations

import argparse
import ast
from dataclasses import fields, replace
from datetime import datetime
from typing import Iterable, Optional

from systems import CheckpointNames, TrainingConfig, VerificationConfig, get_system
from training_pipeline import (
    RUNS_ROOT,
    PipelineRun,
    TrainingPipeline,
)


def parse_value(raw: str):
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw


def parse_overrides(items: Iterable[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE override, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = parse_value(value)
    return overrides


def apply_dataclass_overrides(instance, overrides: dict[str, object], label: str):
    valid = {field.name for field in fields(instance)}
    unknown = sorted(set(overrides) - valid)
    if unknown:
        raise ValueError(f"Unknown {label} override(s): {', '.join(unknown)}")
    return replace(instance, **overrides)


def split_training_overrides(overrides: dict[str, object]):
    training_overrides: dict[str, object] = {}
    data_profile_overrides: dict[str, object] = {}
    for key, value in overrides.items():
        if key.startswith("data_profile."):
            data_profile_overrides[key.removeprefix("data_profile.")] = value
        else:
            training_overrides[key] = value
    return training_overrides, data_profile_overrides


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run Streamlined with explicit config overrides.")
    parser.add_argument("--system", default="2d_bench", help="System key from systems.py")
    parser.add_argument("--stage", default="plan", choices=TrainingPipeline.STAGES)
    parser.add_argument("--run-name", default=None, help="Run folder name under pipeline_runs/<system>")
    parser.add_argument(
        "--set",
        dest="training_overrides",
        action="append",
        default=[],
        help="TrainingConfig override as KEY=VALUE. Repeatable.",
    )
    parser.add_argument(
        "--verification-set",
        dest="verification_overrides",
        action="append",
        default=[],
        help="VerificationConfig override as KEY=VALUE. Repeatable.",
    )
    parser.add_argument(
        "--checkpoint-set",
        dest="checkpoint_overrides",
        action="append",
        default=[],
        help="CheckpointNames override as KEY=VALUE. Repeatable.",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args(argv)

    system = get_system(args.system)
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    run_dir = RUNS_ROOT / system.name / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}")

    raw_training_overrides = parse_overrides(args.training_overrides)
    training_overrides, data_profile_overrides = split_training_overrides(raw_training_overrides)
    verification_overrides = parse_overrides(args.verification_overrides)
    checkpoint_overrides = parse_overrides(args.checkpoint_overrides)

    if training_overrides or data_profile_overrides:
        data_profile = system.training.data_profile
        if data_profile_overrides:
            data_profile = apply_dataclass_overrides(
                data_profile,
                data_profile_overrides,
                "training.data_profile",
            )
            training_overrides = {**training_overrides, "data_profile": data_profile}
        system = replace(
            system,
            training=apply_dataclass_overrides(system.training, training_overrides, "training"),
        )
    if checkpoint_overrides:
        system = replace(
            system,
            checkpoints=apply_dataclass_overrides(system.checkpoints, checkpoint_overrides, "checkpoint"),
        )
    if verification_overrides:
        if system.verification is None:
            system = replace(system, verification=VerificationConfig(**verification_overrides))
        else:
            system = replace(
                system,
                verification=apply_dataclass_overrides(
                    system.verification,
                    verification_overrides,
                    "verification",
                ),
            )

    run = PipelineRun(system, run_name=run_name)
    run.write_config()
    print(f"Run directory: {run.run_dir}")
    if raw_training_overrides:
        print(f"Training overrides: {raw_training_overrides}")
    if verification_overrides:
        print(f"Verification overrides: {verification_overrides}")
    if checkpoint_overrides:
        print(f"Checkpoint overrides: {checkpoint_overrides}")
    TrainingPipeline(system, run=run, skip_preflight=args.skip_preflight).run(args.stage)


if __name__ == "__main__":
    main()
