from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from models import load_models
from systems import SystemSpec, get_system


RUNS_ROOT = Path(__file__).resolve().parent / "pipeline_runs"


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def required_checkpoint_paths(system: SystemSpec, artifact_dir: Path) -> tuple[Path, Path, Path]:
    return (
        artifact_dir / system.checkpoints.pi,
        artifact_dir / system.checkpoints.omega,
        artifact_dir / system.checkpoints.omega_closed_loop,
    )


def has_required_checkpoints(system: SystemSpec, artifact_dir: Path) -> bool:
    return all(path.is_file() for path in required_checkpoint_paths(system, artifact_dir))


def latest_artifact_dir(system: SystemSpec) -> Path:
    system_runs = RUNS_ROOT / system.name
    if not system_runs.is_dir():
        raise FileNotFoundError(f"No run directory found for system {system.name!r}: {system_runs}")

    candidates = sorted(
        (path for path in system_runs.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        artifact_dir = run_dir / "artifacts"
        if has_required_checkpoints(system, artifact_dir):
            return artifact_dir

    missing = ", ".join(path.name for path in required_checkpoint_paths(system, system_runs / "<run>" / "artifacts"))
    raise FileNotFoundError(
        f"No usable {system.name!r} run found under {system_runs}. "
        f"Need artifacts containing: {missing}"
    )


def resolve_artifact_dir(system: SystemSpec, args: argparse.Namespace) -> Path:
    if args.artifact_dir is not None:
        artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    elif args.run_dir is not None:
        artifact_dir = Path(args.run_dir).expanduser().resolve() / "artifacts"
    elif args.run_name is not None:
        artifact_dir = RUNS_ROOT / system.name / args.run_name / "artifacts"
    else:
        artifact_dir = latest_artifact_dir(system)

    if not has_required_checkpoints(system, artifact_dir):
        required = "\n".join(f"  - {path}" for path in required_checkpoint_paths(system, artifact_dir))
        raise FileNotFoundError(f"Artifact directory is missing required checkpoints:\n{required}")
    return artifact_dir


def parse_x0_list(raw_x0: Optional[Sequence[Sequence[str]]], x0_json: Optional[str], n_dim: int) -> list[list[float]]:
    points: list[list[float]] = []
    if x0_json is not None:
        payload = json.loads(Path(x0_json).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("--x0-json must contain a JSON list of initial-condition lists.")
        points.extend(payload)

    if raw_x0:
        points.extend([[float(value) for value in point] for point in raw_x0])

    if not points:
        points = [[0.5] * n_dim]

    parsed: list[list[float]] = []
    for idx, point in enumerate(points):
        if len(point) != n_dim:
            raise ValueError(f"x0 #{idx} has dimension {len(point)}; expected {n_dim}.")
        parsed.append([float(value) for value in point])
    return parsed


@torch.no_grad()
def simulate_trajectories(
    *,
    system: SystemSpec,
    models: dict[str, torch.nn.Module],
    x0_list: Sequence[Sequence[float]],
    steps: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    pi = models["pi"]
    omega_orig = models["omega"]
    omega_cl = models["omega_closed_loop"]
    p_matrix = pi.P_matrix.to(device=device, dtype=torch.float32)

    n_traj = len(x0_list)
    state = torch.tensor(x0_list, device=device, dtype=torch.float32)
    states = torch.empty((n_traj, steps + 1, system.n_dim), device=device)
    omega_orig_values = torch.empty((n_traj, steps + 1), device=device)
    omega_cl_values = torch.empty((n_traj, steps + 1), device=device)
    nu_values = torch.empty((n_traj, steps + 1), device=device)
    norm2_values = torch.empty((n_traj, steps + 1), device=device)

    for t in range(steps + 1):
        states[:, t, :] = state
        omega_orig_values[:, t] = omega_orig.query(state).reshape(-1)
        omega_cl_values[:, t] = omega_cl.query(state).reshape(-1)
        nu_values[:, t] = (state @ p_matrix * state).sum(dim=-1)
        norm2_values[:, t] = (state * state).sum(dim=-1)
        if t < steps:
            control = pi(state)
            state = system.dynamics(state, control)

    return {
        "states": states.cpu().numpy(),
        "omega_orig": omega_orig_values.cpu().numpy(),
        "omega_cl": omega_cl_values.cpu().numpy(),
        "nu": nu_values.cpu().numpy(),
        "norm2": norm2_values.cpu().numpy(),
    }


def label_for_point(point: Sequence[float]) -> str:
    return "x0=(" + ", ".join(f"{value:g}" for value in point) + ")"


def plot_values(
    *,
    values: dict[str, np.ndarray],
    x0_list: Sequence[Sequence[float]],
    output_path: Path,
    title: str,
) -> None:
    panels = (
        ("omega_orig", r"$\omega_{orig}(x_t)$"),
        ("omega_cl", r"$\omega_{cl}(x_t)$"),
        ("nu", r"$\nu(x_t)=x_t^T P x_t$"),
        ("norm2", r"$||x_t||^2$"),
    )
    t = np.arange(values["omega_orig"].shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(x0_list), 1)))

    for ax, (key, ylabel) in zip(axes.flat, panels):
        for idx, point in enumerate(x0_list):
            ax.plot(t, values[key][idx], color=colors[idx % len(colors)], linewidth=2.0, label=label_for_point(point))
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_csv(values: dict[str, np.ndarray], x0_list: Sequence[Sequence[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    states = values["states"]
    steps = states.shape[1]
    state_columns = [f"x{i}" for i in range(states.shape[2])]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trajectory", "step", *state_columns, "omega_orig", "omega_cl", "nu", "norm2"])
        for traj_idx, _point in enumerate(x0_list):
            for step in range(steps):
                writer.writerow(
                    [
                        traj_idx,
                        step,
                        *states[traj_idx, step, :].tolist(),
                        values["omega_orig"][traj_idx, step],
                        values["omega_cl"][traj_idx, step],
                        values["nu"][traj_idx, step],
                        values["norm2"][traj_idx, step],
                    ]
                )


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Plot omega, omega_cl, nu, and ||x||^2 along chosen trajectories.")
    parser.add_argument("--system", default="2d_bench", help="System key from systems.py")
    parser.add_argument("--run-name", default=None, help="Run folder name under Streamlined/pipeline_runs/<system>.")
    parser.add_argument("--run-dir", default=None, help="Explicit run directory containing an artifacts/ folder.")
    parser.add_argument("--artifact-dir", default=None, help="Explicit directory containing pi/omega/omega_cl checkpoints.")
    parser.add_argument("--x0", nargs="+", action="append", help="One initial condition. Repeat for more trajectories.")
    parser.add_argument("--x0-json", default=None, help="JSON file containing a list of initial-condition lists.")
    parser.add_argument("--steps", type=int, default=50, help="Number of closed-loop steps to simulate.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--output", default=None, help="PNG output path. Defaults to the run's plots/ directory.")
    parser.add_argument("--csv", default=None, help="Optional CSV output path for the sampled values.")
    args = parser.parse_args(argv)

    if args.steps < 0:
        raise ValueError("--steps must be nonnegative.")

    system = get_system(args.system)
    device = pick_device(args.device)
    artifact_dir = resolve_artifact_dir(system, args)
    run_dir = artifact_dir.parent
    x0_list = parse_x0_list(args.x0, args.x0_json, system.n_dim)

    models = load_models(system, artifact_dir, device=device)
    values = simulate_trajectories(
        system=system,
        models=models,
        x0_list=x0_list,
        steps=args.steps,
        device=device,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else run_dir / "plots" / f"trajectory_values_{timestamp}.png"
    )
    plot_values(
        values=values,
        x0_list=x0_list,
        output_path=output_path,
        title=f"{system.name}: closed-loop trajectories ({run_dir.name})",
    )

    print(f"Loaded artifacts: {artifact_dir}")
    print(f"Saved plot: {output_path}")
    if args.csv is not None:
        csv_path = Path(args.csv).expanduser().resolve()
        write_csv(values, x0_list, csv_path)
        print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
