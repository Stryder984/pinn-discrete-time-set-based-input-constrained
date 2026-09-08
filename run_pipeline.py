from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime


HEAVY_STAGES = {
    "setup_lqr_box",
    "diagnose_training_domain",
    "train_omega",
    "train_policy",
    "train_closed_loop_omega",
    "search_thresholds",
    "plot_threshold_diagnostics",
    "export",
    "all",
}


def early_preflight(stage: str, *, min_free_ram_gb: float = 2.0) -> bool:
    """Fast resource check before importing torch-heavy pipeline modules."""
    if stage not in HEAVY_STAGES:
        return True
    try:
        import psutil  # type: ignore
    except Exception as exc:
        print(f"Early preflight skipped: psutil unavailable ({exc}).")
        return True

    messages = []
    memory = psutil.virtual_memory()
    free_ram_gb = float(memory.available) / (1024**3)
    messages.append(f"available system RAM = {free_ram_gb:.2f} GB")

    stale = []
    current_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "create_time", "memory_info"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if proc.info["pid"] == current_pid:
                continue
            if name not in {"python.exe", "pythonw.exe", "pwsh.exe", "powershell.exe"}:
                continue
            age_hours = (datetime.now().timestamp() - float(proc.info["create_time"])) / 3600.0
            rss_gb = float(proc.info["memory_info"].rss) / (1024**3)
            if age_hours >= 2.0 or rss_gb >= 1.0:
                stale.append((proc.info["pid"], name, age_hours, rss_gb))
        except Exception:
            continue

    if stale:
        messages.append(
            "stale-looking Python/PowerShell processes: "
            + ", ".join(
                f"pid={pid} {name} age={age:.1f}h ram={rss:.2f}GB"
                for pid, name, age, rss in stale[:8]
            )
        )

    if free_ram_gb >= min_free_ram_gb:
        print(f"Early preflight passed: available system RAM = {free_ram_gb:.2f} GB", flush=True)
        return True

    print("Early preflight check failed. Recommended reset before importing torch/training code:", flush=True)
    for message in messages:
        print(f"  - {message}", flush=True)
    print("Close stale terminals/apps, restart failed Python/CUDA processes, then rerun.", flush=True)
    print("Override only if intentional: add --skip-preflight.", flush=True)
    return False


def parse_preflight_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", default="plan")
    parser.add_argument("--skip-preflight", action="store_true")
    args, _ = parser.parse_known_args(argv)
    return args


if __name__ == "__main__":
    preflight_args = parse_preflight_args(sys.argv[1:])
    if not preflight_args.skip_preflight and not early_preflight(preflight_args.stage):
        raise SystemExit(0)

    from training_pipeline import main

    main()
