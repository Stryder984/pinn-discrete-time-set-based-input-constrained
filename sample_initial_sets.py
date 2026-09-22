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
from plot_trajectories import required_checkpoint_paths
from systems import SystemSpec, get_system
from training_pipeline import heatmap_pair_suffix, normalize_plot_pairs, system_with_existing_run_domain


RUNS_ROOT = Path(__file__).resolve().parent / "pipeline_runs"


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_run_dir(system: SystemSpec, *, run_name: Optional[str], run_dir: Optional[str]) -> Path:
    if run_dir is not None:
        path = Path(run_dir).expanduser().resolve()
    elif run_name is not None:
        path = RUNS_ROOT / system.name / run_name
    else:
        raise ValueError("Pass --run-name or --run-dir so thresholds and artifacts come from one explicit run.")

    if not path.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {path}")
    return path


def require_artifacts(system: SystemSpec, run_dir: Path) -> Path:
    artifact_dir = run_dir / "artifacts"
    missing = [path for path in required_checkpoint_paths(system, artifact_dir) if not path.is_file()]
    if missing:
        detail = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Run is missing required model checkpoints:\n{detail}")
    return artifact_dir


def load_thresholds(run_dir: Path) -> dict[str, float]:
    path = run_dir / "verification_thresholds_rescaled.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing thresholds file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("w2") is None:
        raise ValueError(f"Threshold file has no usable w2: {path}")
    return payload


def verification_domain(system: SystemSpec, run_dir: Path) -> np.ndarray:
    expansion_path = run_dir / "reachability_domain_expansion.json"
    if expansion_path.is_file():
        payload = json.loads(expansion_path.read_text(encoding="utf-8"))
        final_bounds = payload.get("final_bounds")
        domain = np.asarray(final_bounds, dtype=np.float32)
        if domain.shape == (system.n_dim, 2):
            return domain

    config_path = run_dir / "config.json"
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        verification = payload.get("verification")
        if verification and verification.get("domain") is not None:
            domain = np.asarray(verification["domain"], dtype=np.float32)
            if domain.shape == (system.n_dim, 2):
                return domain

    if system.verification is not None:
        return np.asarray(system.verification.domain, dtype=np.float32)
    return np.asarray(list(zip(system.x_lb, system.x_hi)), dtype=np.float32)


def run_training_config(run_dir: Path) -> dict[str, object]:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    training = payload.get("training", {})
    return training if isinstance(training, dict) else {}


def uniform_samples(domain: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lows = domain[:, 0]
    highs = domain[:, 1]
    return rng.uniform(lows, highs, size=(n_samples, domain.shape[0])).astype(np.float32)


@torch.no_grad()
def omega_sublevel_rejection_samples(
    omega: torch.nn.Module,
    domain: np.ndarray,
    n_inside: int,
    seed: int,
    *,
    sublevel: float,
    device: torch.device,
    batch_size: int,
    max_candidates: int,
    strict: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    """Uniform rejection sample with matched outside-sublevel controls.

    `n_inside` is the target number of points satisfying omega(x) <= sublevel.
    The sampler also keeps `n_inside` points with omega(x) > sublevel from the
    rejected candidates, so downstream plots compare balanced classes while the
    inside class remains conditionally uniform over the proposed DOS.
    """
    if batch_size <= 0:
        raise ValueError("--rejection-batch-size must be positive.")
    if max_candidates <= 0:
        raise ValueError("--rejection-max-candidates must be positive.")

    rng = np.random.default_rng(seed)
    lows = domain[:, 0]
    highs = domain[:, 1]
    inside_chunks: list[np.ndarray] = []
    outside_chunks: list[np.ndarray] = []
    inside_value_chunks: list[np.ndarray] = []
    outside_value_chunks: list[np.ndarray] = []
    candidates_seen = 0

    while candidates_seen < max_candidates:
        inside_count = sum(chunk.shape[0] for chunk in inside_chunks)
        outside_count = sum(chunk.shape[0] for chunk in outside_chunks)
        if inside_count >= n_inside and outside_count >= n_inside:
            break

        remaining_budget = max_candidates - candidates_seen
        current_batch = min(batch_size, remaining_budget)
        x_np = rng.uniform(lows, highs, size=(current_batch, domain.shape[0])).astype(np.float32)
        x = torch.as_tensor(x_np, device=device, dtype=torch.float32)
        values = omega.query(x).reshape(-1).detach().cpu().numpy().astype(np.float32)
        inside = values <= float(sublevel)
        candidates_seen += current_batch

        if inside.any() and inside_count < n_inside:
            needed = n_inside - inside_count
            accepted_points = x_np[inside][:needed]
            accepted_values = values[inside][:needed]
            inside_chunks.append(accepted_points)
            inside_value_chunks.append(accepted_values)
        if (~inside).any() and outside_count < n_inside:
            needed = n_inside - outside_count
            rejected_points = x_np[~inside][:needed]
            rejected_values = values[~inside][:needed]
            outside_chunks.append(rejected_points)
            outside_value_chunks.append(rejected_values)

    inside_points = (
        np.concatenate(inside_chunks, axis=0)[:n_inside]
        if inside_chunks
        else np.empty((0, domain.shape[0]), dtype=np.float32)
    )
    outside_points = (
        np.concatenate(outside_chunks, axis=0)[:n_inside]
        if outside_chunks
        else np.empty((0, domain.shape[0]), dtype=np.float32)
    )
    inside_values = (
        np.concatenate(inside_value_chunks, axis=0)[:n_inside]
        if inside_value_chunks
        else np.empty((0,), dtype=np.float32)
    )
    outside_values = (
        np.concatenate(outside_value_chunks, axis=0)[:n_inside]
        if outside_value_chunks
        else np.empty((0,), dtype=np.float32)
    )

    if strict and (inside_points.shape[0] < n_inside or outside_points.shape[0] < n_inside):
        raise RuntimeError(
            "Unable to rejection-sample the requested balanced classes: "
            f"inside={inside_points.shape[0]}, outside={outside_points.shape[0]}, "
            f"target_each={n_inside}, candidates_seen={candidates_seen}."
        )
    points = np.concatenate([inside_points, outside_points], axis=0).astype(np.float32)
    stats = {
        "mode": "omega-sublevel-rejection",
        "sublevel": float(sublevel),
        "requested_inside_points": int(n_inside),
        "requested_outside_points": int(n_inside),
        "returned_inside_points": int(inside_points.shape[0]),
        "returned_outside_points": int(outside_points.shape[0]),
        "returned_points": int(points.shape[0]),
        "candidates_seen": int(candidates_seen),
        "max_candidates": int(max_candidates),
        "batch_size": int(batch_size),
        "strict": bool(strict),
        "inside_acceptance_rate": float(inside_points.shape[0] / max(candidates_seen, 1)),
        "inside_omega_min": None if inside_values.size == 0 else float(inside_values.min()),
        "inside_omega_mean": None if inside_values.size == 0 else float(inside_values.mean()),
        "inside_omega_max": None if inside_values.size == 0 else float(inside_values.max()),
        "outside_omega_min": None if outside_values.size == 0 else float(outside_values.min()),
        "outside_omega_mean": None if outside_values.size == 0 else float(outside_values.mean()),
        "outside_omega_max": None if outside_values.size == 0 else float(outside_values.max()),
    }
    return points, stats


def slice_uniform_samples(
    domain: np.ndarray,
    n_samples: int,
    axis_pair: tuple[int, int],
    seed: int,
    *,
    slice_half_width_fraction: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if not (0.0 <= float(slice_half_width_fraction) <= 1.0):
        raise ValueError("slice_half_width_fraction must lie in [0, 1].")
    rng = np.random.default_rng(seed)
    lows = domain[:, 0]
    highs = domain[:, 1]
    widths = highs - lows
    axis_pair = tuple(int(axis) for axis in axis_pair)
    points = np.zeros((n_samples, domain.shape[0]), dtype=np.float32)
    fixed_axes = [idx for idx in range(domain.shape[0]) if idx not in axis_pair]
    for axis in axis_pair:
        points[:, axis] = rng.uniform(lows[axis], highs[axis], size=n_samples)
    half_widths = np.maximum(widths * float(slice_half_width_fraction), 0.0)
    for axis in fixed_axes:
        lo = max(float(lows[axis]), -float(half_widths[axis]))
        hi = min(float(highs[axis]), float(half_widths[axis]))
        points[:, axis] = rng.uniform(lo, hi, size=n_samples)
    return points.astype(np.float32), {
        "mode": "thin_slice_uniform",
        "axis_pair": [int(axis_pair[0]), int(axis_pair[1])],
        "fixed_axes": [int(axis) for axis in fixed_axes],
        "slice_half_width_fraction": float(slice_half_width_fraction),
        "slice_half_widths": {
            str(axis): float(half_widths[axis])
            for axis in fixed_axes
        },
        "requested_points": int(n_samples),
        "returned_points": int(points.shape[0]),
        "seed": int(seed),
    }


def nu_value(pi: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    p_matrix = pi.P_matrix.to(device=x.device, dtype=x.dtype)
    return (x @ p_matrix * x).sum(dim=-1, keepdim=True)


def controller_output(
    pi: torch.nn.Module,
    x: torch.Tensor,
    *,
    mode: str,
    c2: Optional[float],
) -> torch.Tensor:
    if mode == "nn":
        if hasattr(pi, "nn_control"):
            return pi.nn_control(x)
        return pi(x)
    if mode == "stitched":
        if c2 is None:
            raise ValueError("stitched controller mode requires a c2 threshold.")
        if not hasattr(pi, "lqr_control") or not hasattr(pi, "nn_control"):
            raise ValueError("stitched controller mode requires PiNet lqr_control and nn_control methods.")
        u_lqr = pi.lqr_control(x)
        u_nn = pi.nn_control(x)
        use_lqr = nu_value(pi, x) <= float(c2)
        return torch.where(use_lqr, u_lqr, u_nn)
    raise ValueError(f"Unknown controller mode: {mode}")


def _project_to_domain_(x: torch.Tensor, domain: torch.Tensor) -> None:
    x.clamp_(domain[:, 0], domain[:, 1])


def omega_sublevel_gd_samples(
    omega: torch.nn.Module,
    domain: np.ndarray,
    n_samples: int,
    seed: int,
    *,
    sublevel: float,
    device: torch.device,
    batch_size: int,
    steps: int,
    lr: float,
    max_attempts: int,
    strict_sublevel: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    if batch_size <= 0:
        raise ValueError("--gd-batch-size must be positive.")
    if steps <= 0:
        raise ValueError("--gd-steps must be positive for omega-sublevel-gd sampling.")
    if lr <= 0.0:
        raise ValueError("--gd-lr must be positive.")
    if max_attempts <= 0:
        raise ValueError("--gd-max-attempts must be positive.")

    rng = np.random.default_rng(seed)
    domain_tensor = torch.as_tensor(domain, device=device, dtype=torch.float32)
    lows = domain[:, 0]
    highs = domain[:, 1]
    accepted: list[np.ndarray] = []
    rejected_points: list[np.ndarray] = []
    rejected_values: list[np.ndarray] = []
    total_optimized = 0
    total_inside = 0

    for attempt in range(max_attempts):
        remaining = n_samples - sum(chunk.shape[0] for chunk in accepted)
        if remaining <= 0:
            break
        current_batch = min(batch_size, max(remaining * 2, min(batch_size, n_samples)))
        x0 = rng.uniform(lows, highs, size=(current_batch, domain.shape[0])).astype(np.float32)
        x = torch.as_tensor(x0, device=device, dtype=torch.float32).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            values = omega.query(x).reshape(-1)
            loss = torch.relu(values - float(sublevel)).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                _project_to_domain_(x, domain_tensor)

        with torch.no_grad():
            final_x = x.detach()
            final_values = omega.query(final_x).reshape(-1)
            inside = final_values <= float(sublevel)
            inside_np = inside.detach().cpu().numpy().astype(bool)
            final_x_np = final_x.cpu().numpy().astype(np.float32)
            final_values_np = final_values.detach().cpu().numpy().astype(np.float32)

        total_optimized += current_batch
        total_inside += int(inside_np.sum())
        if inside_np.any():
            accepted.append(final_x_np[inside_np])
        if (~inside_np).any():
            rejected_points.append(final_x_np[~inside_np])
            rejected_values.append(final_values_np[~inside_np])

    if accepted:
        points = np.concatenate(accepted, axis=0)[:n_samples]
    else:
        points = np.empty((0, domain.shape[0]), dtype=np.float32)

    if points.shape[0] < n_samples:
        if strict_sublevel:
            raise RuntimeError(
                "Unable to generate enough omega-sublevel-gd points below the requested sublevel: "
                f"got {points.shape[0]} of {n_samples}."
            )
        if rejected_points:
            fallback_points = np.concatenate(rejected_points, axis=0)
            fallback_values = np.concatenate(rejected_values, axis=0)
            order = np.argsort(fallback_values)
            needed = n_samples - points.shape[0]
            points = np.concatenate([points, fallback_points[order[:needed]]], axis=0)

    if points.shape[0] != n_samples:
        raise RuntimeError(f"Generated {points.shape[0]} points, expected {n_samples}.")

    with torch.no_grad():
        point_values = []
        for start in range(0, points.shape[0], batch_size):
            stop = min(start + batch_size, points.shape[0])
            x_batch = torch.as_tensor(points[start:stop], device=device, dtype=torch.float32)
            point_values.append(omega.query(x_batch).reshape(-1).detach().cpu().numpy())
    omega_values = np.concatenate(point_values).astype(np.float32)
    stats = {
        "mode": "omega-sublevel-gd",
        "sublevel": float(sublevel),
        "requested_points": int(n_samples),
        "returned_points": int(points.shape[0]),
        "strict_sublevel": bool(strict_sublevel),
        "attempts": int(max_attempts),
        "attempts_used": int(attempt + 1 if "attempt" in locals() else 0),
        "steps": int(steps),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "optimized_candidate_points": int(total_optimized),
        "candidate_points_below_sublevel": int(total_inside),
        "returned_points_below_sublevel": int((omega_values <= float(sublevel)).sum()),
        "returned_fraction_below_sublevel": float((omega_values <= float(sublevel)).mean()),
        "returned_omega_min": float(omega_values.min()),
        "returned_omega_mean": float(omega_values.mean()),
        "returned_omega_max": float(omega_values.max()),
    }
    return points.astype(np.float32), stats


def make_control_grid(
    system: SystemSpec,
    points_per_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if points_per_dim <= 1:
        raise ValueError("control grid points per dim must be at least 2.")
    axes = [
        torch.linspace(lo, hi, points_per_dim, device=device, dtype=dtype)
        for lo, hi in zip(system.u_lb, system.u_hi)
    ]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([axis.reshape(-1) for axis in mesh], dim=-1)


@torch.no_grad()
def classify_by_omega(
    omega: torch.nn.Module,
    points: np.ndarray,
    *,
    sublevel: float,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.empty(points.shape[0], dtype=np.float32)
    for start in range(0, points.shape[0], batch_size):
        stop = min(start + batch_size, points.shape[0])
        x = torch.as_tensor(points[start:stop], device=device, dtype=torch.float32)
        values[start:stop] = omega.query(x).reshape(-1).detach().cpu().numpy()
    return values < float(sublevel), values


@torch.no_grad()
def convergence_mask(
    *,
    system: SystemSpec,
    pi: torch.nn.Module,
    points: np.ndarray,
    steps: int,
    norm2_tol: float,
    controller_mode: str,
    c2: Optional[float],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    converged = np.zeros(points.shape[0], dtype=bool)
    for start in range(0, points.shape[0], batch_size):
        stop = min(start + batch_size, points.shape[0])
        x = torch.as_tensor(points[start:stop], device=device, dtype=torch.float32)
        for _ in range(steps):
            x = system.dynamics(x, controller_output(pi, x, mode=controller_mode, c2=c2))
        final_norm2 = (x * x).sum(dim=-1).detach().cpu().numpy()
        converged[start:stop] = final_norm2 <= float(norm2_tol)
    return converged


@torch.no_grad()
def grid_optimal_controls(
    *,
    system: SystemSpec,
    omega_orig: torch.nn.Module,
    x: torch.Tensor,
    control_grid: torch.Tensor,
    control_chunk_size: int,
) -> torch.Tensor:
    if x.shape[0] == 0:
        return torch.empty((0, system.u_dim), device=x.device, dtype=x.dtype)
    best_values = torch.full((x.shape[0],), torch.inf, device=x.device, dtype=x.dtype)
    best_controls = torch.empty((x.shape[0], system.u_dim), device=x.device, dtype=x.dtype)
    for u_start in range(0, control_grid.shape[0], control_chunk_size):
        ug = control_grid[u_start : u_start + control_chunk_size].to(device=x.device, dtype=x.dtype)
        k = ug.shape[0]
        x_expanded = x.unsqueeze(1).expand(-1, k, -1).reshape(x.shape[0] * k, system.n_dim)
        u_expanded = ug.unsqueeze(0).expand(x.shape[0], -1, -1).reshape(x.shape[0] * k, system.u_dim)
        next_x = system.dynamics(x_expanded, u_expanded)
        values = omega_orig.query(next_x).reshape(x.shape[0], k)
        chunk_values, chunk_indices = values.min(dim=1)
        improved = chunk_values < best_values
        if improved.any():
            best_values[improved] = chunk_values[improved]
            best_controls[improved] = ug[chunk_indices[improved]]
    return best_controls


@torch.no_grad()
def convergence_mask_grid_optimal(
    *,
    system: SystemSpec,
    omega_orig: torch.nn.Module,
    points: np.ndarray,
    control_grid: torch.Tensor,
    steps: int,
    norm2_tol: float,
    device: torch.device,
    batch_size: int,
    control_chunk_size: int,
) -> np.ndarray:
    converged = np.zeros(points.shape[0], dtype=bool)
    for start in range(0, points.shape[0], batch_size):
        stop = min(start + batch_size, points.shape[0])
        x = torch.as_tensor(points[start:stop], device=device, dtype=torch.float32)
        for _ in range(steps):
            u = grid_optimal_controls(
                system=system,
                omega_orig=omega_orig,
                x=x,
                control_grid=control_grid,
                control_chunk_size=control_chunk_size,
            )
            x = system.dynamics(x, u)
        final_norm2 = (x * x).sum(dim=-1).detach().cpu().numpy()
        converged[start:stop] = final_norm2 <= float(norm2_tol)
    return converged


def select_for_plot(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[np.sort(indices)]


def plot_side_by_side(
    *,
    values_inside: dict[str, np.ndarray],
    values_outside: dict[str, np.ndarray],
    sublevel: float,
    sublevel_key: str,
    inside_label: str,
    outside_label: str,
    output_path: Path,
    title: str,
) -> None:
    rows = (
        ("omega_orig", r"$\omega_{orig}(x_t)$"),
        ("omega_cl", r"$\omega_{cl}(x_t)$"),
        ("nu", r"$\nu(x_t)=x_t^T P x_t$"),
        ("norm2", r"$||x_t||^2$"),
    )
    columns = ((inside_label, values_inside), (outside_label, values_outside))
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True, constrained_layout=True)
    for col, (column_title, values) in enumerate(columns):
        for row, (key, ylabel) in enumerate(rows):
            ax = axes[row, col]
            if values[key].shape[0] == 0:
                ax.text(0.5, 0.5, "no trajectories", ha="center", va="center", transform=ax.transAxes)
            else:
                t = np.arange(values[key].shape[1])
                ax.plot(t, values[key].T, alpha=0.35, linewidth=1.0)
            if key == sublevel_key:
                ax.axhline(float(sublevel), color="black", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(column_title)
            if row == len(rows) - 1:
                ax.set_xlabel("step")
    fig.suptitle(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_control_inputs_side_by_side(
    *,
    values_inside: dict[str, np.ndarray],
    values_outside: dict[str, np.ndarray],
    system: SystemSpec,
    inside_label: str,
    outside_label: str,
    output_path: Path,
    title: str,
) -> None:
    columns = ((inside_label, values_inside), (outside_label, values_outside))
    n_rows = max(1, system.u_dim)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 2.8 * n_rows), sharex=True, constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(n_rows, 2)

    for col, (column_title, values) in enumerate(columns):
        controls = values.get("controls")
        for row in range(n_rows):
            ax = axes_arr[row, col]
            if controls is None or controls.shape[0] == 0 or controls.shape[1] == 0:
                ax.text(0.5, 0.5, "no controls", ha="center", va="center", transform=ax.transAxes)
            else:
                t = np.arange(controls.shape[1])
                ax.plot(t, controls[:, :, row].T, alpha=0.35, linewidth=1.0)
                ax.axhline(float(system.u_lb[row]), color="black", linestyle="--", linewidth=1.0, alpha=0.7)
                ax.axhline(float(system.u_hi[row]), color="black", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(f"u{row}")
            if row == 0:
                ax.set_title(column_title)
            if row == n_rows - 1:
                ax.set_xlabel("step")

    fig.suptitle(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_initial_convergence_scatter(
    *,
    points: np.ndarray,
    converged: np.ndarray,
    system: SystemSpec,
    axis_pair: tuple[int, int],
    domain: np.ndarray,
    slice_stats: dict[str, object],
    output_path: Path,
    title: str,
    seed: int,
    max_points: int = 1000,
) -> dict[str, object]:
    """Plot sampled initial conditions near one 2D state slice."""
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("Initial-condition convergence plot requires at least two state dimensions.")
    if converged.shape[0] != points.shape[0]:
        raise ValueError("converged mask length must match points.")

    axis_pair = tuple(int(axis) for axis in axis_pair)
    slice_points = points
    slice_converged = converged
    n_plot = min(int(max_points), int(slice_points.shape[0]))
    rng = np.random.default_rng(seed)
    if slice_points.shape[0] > n_plot:
        indices = np.sort(rng.choice(slice_points.shape[0], size=n_plot, replace=False))
    else:
        indices = np.arange(slice_points.shape[0])

    sample = slice_points[indices]
    sample_converged = slice_converged[indices]

    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    x_axis, y_axis = axis_pair
    if sample.shape[0] == 0:
        ax.text(0.5, 0.5, "no points in slice", ha="center", va="center", transform=ax.transAxes)
    if np.any(sample_converged):
        ax.scatter(
            sample[sample_converged, x_axis],
            sample[sample_converged, y_axis],
            c="tab:blue",
            s=34,
            alpha=0.78,
            label="converged",
        )
    if np.any(~sample_converged):
        ax.scatter(
            sample[~sample_converged, x_axis],
            sample[~sample_converged, y_axis],
            c="tab:red",
            marker="x",
            s=46,
            linewidths=1.5,
            alpha=0.9,
            label="not converged",
        )
    labels = system.state_labels or tuple(f"x{i}" for i in range(system.n_dim))
    ax.set_xlabel(labels[x_axis])
    ax.set_ylabel(labels[y_axis])
    ax.set_xlim(float(domain[x_axis, 0]), float(domain[x_axis, 1]))
    ax.set_ylim(float(domain[y_axis, 0]), float(domain[y_axis, 1]))
    ax.set_title(f"{title} ({labels[x_axis]}, {labels[y_axis]})")
    ax.grid(True, alpha=0.3)
    if sample.shape[0] > 0:
        ax.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "axis_pair": [int(x_axis), int(y_axis)],
        "fixed_axes": list(slice_stats.get("fixed_axes", [])),
        "slice_half_width_fraction": slice_stats.get("slice_half_width_fraction"),
        "slice_half_widths": dict(slice_stats.get("slice_half_widths", {})),
        "candidate_points": int(points.shape[0]),
        "slice_points": int(slice_points.shape[0]),
        "plotted_points": int(sample.shape[0]),
        "converged_in_slice": int(slice_converged.sum()) if slice_converged.size else 0,
        "not_converged_in_slice": int((~slice_converged).sum()) if slice_converged.size else 0,
    }


@torch.no_grad()
def add_learned_control_inputs(
    *,
    system: SystemSpec,
    pi: torch.nn.Module,
    values: dict[str, np.ndarray],
    controller_mode: str,
    c2: Optional[float],
    device: torch.device,
) -> dict[str, np.ndarray]:
    states = values["states"]
    steps = max(0, states.shape[1] - 1)
    if states.shape[0] == 0:
        enriched = dict(values)
        enriched["controls"] = np.empty((0, steps, system.u_dim), dtype=np.float32)
        return enriched
    if steps == 0:
        enriched = dict(values)
        enriched["controls"] = np.empty((states.shape[0], 0, system.u_dim), dtype=np.float32)
        return enriched

    x = torch.as_tensor(states[:, :-1, :].reshape(-1, system.n_dim), device=device, dtype=torch.float32)
    u = (
        controller_output(pi, x, mode=controller_mode, c2=c2)
        .reshape(states.shape[0], steps, system.u_dim)
        .detach()
        .cpu()
        .numpy()
    )
    enriched = dict(values)
    enriched["controls"] = u
    return enriched


def trajectory_values_for_plot(
    *,
    system: SystemSpec,
    models: dict[str, torch.nn.Module],
    points: np.ndarray,
    steps: int,
    controller_mode: str,
    c2: Optional[float],
    device: torch.device,
) -> dict[str, np.ndarray]:
    if points.shape[0] == 0:
        return {
            "states": np.empty((0, steps + 1, system.n_dim), dtype=np.float32),
            "omega_orig": np.empty((0, steps + 1), dtype=np.float32),
            "omega_cl": np.empty((0, steps + 1), dtype=np.float32),
            "nu": np.empty((0, steps + 1), dtype=np.float32),
            "norm2": np.empty((0, steps + 1), dtype=np.float32),
            "controls": np.empty((0, steps, system.u_dim), dtype=np.float32),
        }
    pi = models["pi"]
    omega_orig = models["omega"]
    omega_cl = models["omega_closed_loop"]
    p_matrix = pi.P_matrix.to(device=device, dtype=torch.float32)
    state = torch.as_tensor(points, device=device, dtype=torch.float32)
    n_traj = state.shape[0]
    states = torch.empty((n_traj, steps + 1, system.n_dim), device=device)
    omega_orig_values = torch.empty((n_traj, steps + 1), device=device)
    omega_cl_values = torch.empty((n_traj, steps + 1), device=device)
    nu_values = torch.empty((n_traj, steps + 1), device=device)
    norm2_values = torch.empty((n_traj, steps + 1), device=device)
    controls = torch.empty((n_traj, steps, system.u_dim), device=device)

    for t in range(steps + 1):
        states[:, t, :] = state
        omega_orig_values[:, t] = omega_orig.query(state).reshape(-1)
        omega_cl_values[:, t] = omega_cl.query(state).reshape(-1)
        nu_values[:, t] = (state @ p_matrix * state).sum(dim=-1)
        norm2_values[:, t] = (state * state).sum(dim=-1)
        if t < steps:
            control = controller_output(pi, state, mode=controller_mode, c2=c2)
            controls[:, t, :] = control
            state = system.dynamics(state, control)

    return {
        "states": states.detach().cpu().numpy(),
        "omega_orig": omega_orig_values.detach().cpu().numpy(),
        "omega_cl": omega_cl_values.detach().cpu().numpy(),
        "nu": nu_values.detach().cpu().numpy(),
        "norm2": norm2_values.detach().cpu().numpy(),
        "controls": controls.detach().cpu().numpy(),
    }


@torch.no_grad()
def simulate_grid_optimal_trajectories(
    *,
    system: SystemSpec,
    models: dict[str, torch.nn.Module],
    points: np.ndarray,
    control_grid: torch.Tensor,
    steps: int,
    device: torch.device,
    control_chunk_size: int,
) -> dict[str, np.ndarray]:
    if points.shape[0] == 0:
        return {
            "states": np.empty((0, steps + 1, system.n_dim), dtype=np.float32),
            "omega_orig": np.empty((0, steps + 1), dtype=np.float32),
            "omega_cl": np.empty((0, steps + 1), dtype=np.float32),
            "nu": np.empty((0, steps + 1), dtype=np.float32),
            "norm2": np.empty((0, steps + 1), dtype=np.float32),
            "controls": np.empty((0, steps, system.u_dim), dtype=np.float32),
        }

    omega_orig = models["omega"]
    omega_cl = models["omega_closed_loop"]
    pi = models["pi"]
    p_matrix = pi.P_matrix.to(device=device, dtype=torch.float32)
    state = torch.as_tensor(points, device=device, dtype=torch.float32)
    n_traj = state.shape[0]
    states = torch.empty((n_traj, steps + 1, system.n_dim), device=device)
    omega_orig_values = torch.empty((n_traj, steps + 1), device=device)
    omega_cl_values = torch.empty((n_traj, steps + 1), device=device)
    nu_values = torch.empty((n_traj, steps + 1), device=device)
    norm2_values = torch.empty((n_traj, steps + 1), device=device)
    controls = torch.empty((n_traj, steps, system.u_dim), device=device)

    for t in range(steps + 1):
        states[:, t, :] = state
        omega_orig_values[:, t] = omega_orig.query(state).reshape(-1)
        omega_cl_values[:, t] = omega_cl.query(state).reshape(-1)
        nu_values[:, t] = (state @ p_matrix * state).sum(dim=-1)
        norm2_values[:, t] = (state * state).sum(dim=-1)
        if t < steps:
            control = grid_optimal_controls(
                system=system,
                omega_orig=omega_orig,
                x=state,
                control_grid=control_grid,
                control_chunk_size=control_chunk_size,
            )
            controls[:, t, :] = control
            state = system.dynamics(state, control)

    return {
        "states": states.cpu().numpy(),
        "omega_orig": omega_orig_values.cpu().numpy(),
        "omega_cl": omega_cl_values.cpu().numpy(),
        "nu": nu_values.cpu().numpy(),
        "norm2": norm2_values.cpu().numpy(),
        "controls": controls.cpu().numpy(),
    }


def write_points_csv(
    path: Path,
    points: np.ndarray,
    classification_values: np.ndarray,
    inside_mask: np.ndarray,
    converged: np.ndarray,
    *,
    classification_value_name: str,
    inside_column_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        state_columns = [f"x{i}" for i in range(points.shape[1])]
        writer.writerow([*state_columns, classification_value_name, inside_column_name, "converged"])
        for idx in range(points.shape[0]):
            writer.writerow([*points[idx].tolist(), classification_values[idx], bool(inside_mask[idx]), bool(converged[idx])])


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sample verification-domain initial points, classify by omega_cl<w2, and simulate both classes."
    )
    parser.add_argument("--system", default="2d_bench", help="System key from systems.py")
    parser.add_argument("--run-name", default=None, help="Run folder name under Streamlined/pipeline_runs/<system>.")
    parser.add_argument("--run-dir", default=None, help="Explicit run directory.")
    parser.add_argument("--samples", type=int, default=10_000, help="Number of verification-domain samples.")
    parser.add_argument("--steps", type=int, default=100, help="Closed-loop simulation length for convergence checks.")
    parser.add_argument("--convergence-norm2-tol", type=float, default=1e-3, help="Converged iff final ||x||^2 is at most this value.")
    parser.add_argument("--seed", type=int, default=43215, help="Random seed for uniform samples and plot subsampling.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for classification and convergence simulation.")
    parser.add_argument(
        "--sampling-mode",
        choices=("uniform", "omega-sublevel-gd", "omega-sublevel-rejection"),
        default="uniform",
        help="How to generate initial points before omega_cl<w2 classification.",
    )
    parser.add_argument(
        "--sampling-omega",
        choices=("closed-loop", "original"),
        default="closed-loop",
        help="Omega network optimized by omega-sublevel-gd.",
    )
    parser.add_argument(
        "--sublevel-value",
        type=float,
        default=None,
        help="Target omega sublevel for omega-sublevel-gd. Defaults to the run's scaled w2.",
    )
    parser.add_argument("--gd-steps", type=int, default=80, help="Adam steps for omega-sublevel-gd sample generation.")
    parser.add_argument("--gd-lr", type=float, default=0.05, help="Adam learning rate for omega-sublevel-gd sample generation.")
    parser.add_argument("--gd-max-attempts", type=int, default=6, help="Maximum candidate batches for omega-sublevel-gd sampling.")
    parser.add_argument(
        "--gd-batch-size",
        type=int,
        default=None,
        help="Candidate batch size for omega-sublevel-gd. Defaults to --batch-size.",
    )
    parser.add_argument(
        "--gd-strict-sublevel",
        action="store_true",
        help="Fail if omega-sublevel-gd cannot return all samples below the requested sublevel.",
    )
    parser.add_argument(
        "--rejection-batch-size",
        type=int,
        default=None,
        help="Candidate batch size for omega-sublevel-rejection. Defaults to --batch-size.",
    )
    parser.add_argument(
        "--rejection-max-candidates",
        type=int,
        default=5_000_000,
        help="Maximum uniform candidates evaluated by omega-sublevel-rejection.",
    )
    parser.add_argument(
        "--rejection-strict",
        action="store_true",
        help="Fail if omega-sublevel-rejection cannot return --samples inside and --samples outside points.",
    )
    parser.add_argument("--max-plot-trajectories", type=int, default=200, help="Maximum trajectories plotted per class; use 0 for all.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument(
        "--controller-mode",
        choices=("stitched", "nn"),
        default="stitched",
        help="Controller used for the main rollout: stitched uses clamped LQR for nu<=c2 and saturated NN otherwise.",
    )
    parser.add_argument("--include-grid-optimal", action="store_true", help="Also simulate the training grid-search argmin controller.")
    parser.add_argument("--control-grid-points", type=int, default=None, help="Override grid points per input dimension for grid-optimal rollouts.")
    parser.add_argument("--control-chunk-size", type=int, default=50_000, help="Control-grid chunk size for grid-optimal argmin evaluation.")
    parser.add_argument("--output", default=None, help="PNG output path. Defaults to the run's plots/ directory.")
    parser.add_argument("--control-output", default=None, help="PNG output path for learned-controller control inputs.")
    parser.add_argument("--grid-optimal-output", default=None, help="PNG output path for grid-optimal rollouts.")
    parser.add_argument("--grid-optimal-control-output", default=None, help="PNG output path for grid-optimal control inputs.")
    parser.add_argument(
        "--initial-convergence-output",
        default=None,
        help="PNG output path for initial convergence sample. With multiple plot_pairs, a suffix is added.",
    )
    parser.add_argument(
        "--initial-convergence-slice-fraction",
        type=float,
        default=0.025,
        help="Half-width of the fixed-coordinate slice as a fraction of each expanded-domain width.",
    )
    parser.add_argument(
        "--initial-convergence-samples",
        type=int,
        default=100,
        help="Thin-slice initial points per plot_pair for the initial convergence scatter.",
    )
    parser.add_argument("--stats-json", default=None, help="Statistics JSON output path. Defaults to the run directory.")
    parser.add_argument("--points-csv", default=None, help="Optional CSV output for sampled points and labels.")
    args = parser.parse_args(argv)

    if args.samples <= 0:
        raise ValueError("--samples must be positive.")
    if args.steps < 0:
        raise ValueError("--steps must be nonnegative.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.initial_convergence_samples <= 0:
        raise ValueError("--initial-convergence-samples must be positive.")
    if args.gd_batch_size is not None and args.gd_batch_size <= 0:
        raise ValueError("--gd-batch-size must be positive.")
    if args.rejection_batch_size is not None and args.rejection_batch_size <= 0:
        raise ValueError("--rejection-batch-size must be positive.")
    if args.rejection_max_candidates <= 0:
        raise ValueError("--rejection-max-candidates must be positive.")

    system = get_system(args.system)
    run_dir = resolve_run_dir(system, run_name=args.run_name, run_dir=args.run_dir)
    system = system_with_existing_run_domain(system, run_dir)
    artifact_dir = require_artifacts(system, run_dir)
    thresholds = load_thresholds(run_dir)
    w2 = float(thresholds["w2"])
    c2 = float(thresholds["c2"])
    domain = verification_domain(system, run_dir)
    training_config = run_training_config(run_dir)
    device = pick_device(args.device)

    models = load_models(system, artifact_dir, device=device)
    requested_omega_key = "omega_closed_loop" if args.sampling_omega == "closed-loop" else "omega"
    requested_omega_name = "omega_cl" if requested_omega_key == "omega_closed_loop" else "omega_orig"
    if args.sampling_mode in {"omega-sublevel-gd", "omega-sublevel-rejection"} or args.sublevel_value is not None:
        classification_omega_key = requested_omega_key
        classification_omega_name = requested_omega_name
        classification_sublevel = w2 if args.sublevel_value is None else float(args.sublevel_value)
    else:
        classification_omega_key = "omega_closed_loop"
        classification_omega_name = "omega_cl"
        classification_sublevel = w2
    inside_label = f"{classification_omega_name} <= {classification_sublevel:.6g}"
    outside_label = f"{classification_omega_name} > {classification_sublevel:.6g}"

    if args.sampling_mode == "uniform":
        points = uniform_samples(domain, args.samples, args.seed)
        sampling_stats = {
            "mode": "uniform",
            "requested_points": int(args.samples),
            "returned_points": int(points.shape[0]),
            "seed": int(args.seed),
        }
    elif args.sampling_mode == "omega-sublevel-gd":
        points, sampling_stats = omega_sublevel_gd_samples(
            models[requested_omega_key],
            domain,
            args.samples,
            args.seed,
            sublevel=classification_sublevel,
            device=device,
            batch_size=args.gd_batch_size or args.batch_size,
            steps=args.gd_steps,
            lr=args.gd_lr,
            max_attempts=args.gd_max_attempts,
            strict_sublevel=args.gd_strict_sublevel,
        )
        sampling_stats["omega_source"] = requested_omega_key
        sampling_stats["omega_name"] = requested_omega_name
        sampling_stats["seed"] = int(args.seed)
    else:
        points, sampling_stats = omega_sublevel_rejection_samples(
            models[requested_omega_key],
            domain,
            args.samples,
            args.seed,
            sublevel=classification_sublevel,
            device=device,
            batch_size=args.rejection_batch_size or args.batch_size,
            max_candidates=args.rejection_max_candidates,
            strict=args.rejection_strict,
        )
        sampling_stats["omega_source"] = requested_omega_key
        sampling_stats["omega_name"] = requested_omega_name
        sampling_stats["seed"] = int(args.seed)
    inside_mask, classification_values = classify_by_omega(
        models[classification_omega_key],
        points,
        sublevel=classification_sublevel,
        device=device,
        batch_size=args.batch_size,
    )

    inside_points = points[inside_mask]
    outside_points = points[~inside_mask]
    inside_converged = convergence_mask(
        system=system,
        pi=models["pi"],
        points=inside_points,
        steps=args.steps,
        norm2_tol=args.convergence_norm2_tol,
        controller_mode=args.controller_mode,
        c2=c2,
        device=device,
        batch_size=args.batch_size,
    )
    outside_converged = convergence_mask(
        system=system,
        pi=models["pi"],
        points=outside_points,
        steps=args.steps,
        norm2_tol=args.convergence_norm2_tol,
        controller_mode=args.controller_mode,
        c2=c2,
        device=device,
        batch_size=args.batch_size,
    )

    all_converged = np.zeros(points.shape[0], dtype=bool)
    all_converged[inside_mask] = inside_converged
    all_converged[~inside_mask] = outside_converged

    grid_stats = None
    control_grid = None
    if args.include_grid_optimal:
        grid_points = (
            int(args.control_grid_points)
            if args.control_grid_points is not None
            else int(training_config.get("control_grid_points", system.training.control_grid_points))
        )
        control_grid = make_control_grid(system, grid_points, device=device, dtype=torch.float32)
        inside_converged_grid = convergence_mask_grid_optimal(
            system=system,
            omega_orig=models["omega"],
            points=inside_points,
            control_grid=control_grid,
            steps=args.steps,
            norm2_tol=args.convergence_norm2_tol,
            device=device,
            batch_size=args.batch_size,
            control_chunk_size=args.control_chunk_size,
        )
        outside_converged_grid = convergence_mask_grid_optimal(
            system=system,
            omega_orig=models["omega"],
            points=outside_points,
            control_grid=control_grid,
            steps=args.steps,
            norm2_tol=args.convergence_norm2_tol,
            device=device,
            batch_size=args.batch_size,
            control_chunk_size=args.control_chunk_size,
        )
        grid_stats = {
            "controller": "training_grid_search_argmin_u_omega_orig_next",
            "control_grid_points_per_dim": int(grid_points),
            "control_grid_total_points": int(control_grid.shape[0]),
            "classification_omega": classification_omega_name,
            "classification_sublevel": float(classification_sublevel),
            "n_classification_sublevel_initial_conditions_convergent": int(inside_converged_grid.sum()),
            "n_initial_points_outside_sublevel_convergent": int(outside_converged_grid.sum()),
        }

    stats = {
        "system": system.name,
        "run_dir": str(run_dir),
        "artifact_dir": str(artifact_dir),
        "thresholds_file": str(run_dir / "verification_thresholds_rescaled.json"),
        "w2": w2,
        "c2": c2,
        "verification_domain": domain.tolist(),
        "sampling": sampling_stats,
        "classification": {
            "omega_source": classification_omega_key,
            "omega_name": classification_omega_name,
            "sublevel": float(classification_sublevel),
            "inside_relation": f"{classification_omega_name}(x) < {classification_sublevel:.12g}",
        },
        "n_verification_domain_samples": int(points.shape[0]),
        "n_initial_points_inside_classification_sublevel": int(inside_mask.sum()),
        "n_classification_sublevel_initial_conditions_convergent": int(inside_converged.sum()),
        "n_initial_points_outside_sublevel_convergent": int(outside_converged.sum()),
        "n_initial_points_outside_sublevel": int((~inside_mask).sum()),
        "convergence": {
            "steps": int(args.steps),
            "criterion": "final ||x||^2 <= convergence_norm2_tol",
            "convergence_norm2_tol": float(args.convergence_norm2_tol),
        },
        "seed": int(args.seed),
        "learned_controller": {
            "n_classification_sublevel_initial_conditions_convergent": int(inside_converged.sum()),
            "n_initial_points_outside_sublevel_convergent": int(outside_converged.sum()),
            "controller_mode": args.controller_mode,
            "controller_branch": (
                "clamped_lqr_inside_nu_c2_else_saturated_nn"
                if args.controller_mode == "stitched"
                else "saturated_nn"
            ),
        },
    }
    if classification_omega_key == "omega_closed_loop" and abs(float(classification_sublevel) - w2) <= 1e-12:
        stats["n_initial_points_with_omega_cl_lt_w2"] = int(inside_mask.sum())
        stats["n_sublevel_contained_initial_conditions_convergent"] = int(inside_converged.sum())
    if args.sampling_mode == "uniform":
        stats["n_verification_domain_uniform_samples"] = int(points.shape[0])
    elif args.sampling_mode == "omega-sublevel-gd":
        stats["n_verification_domain_gd_sublevel_samples"] = int(points.shape[0])
    else:
        stats["n_verification_domain_rejection_sublevel_samples"] = int(points.shape[0])
    if grid_stats is not None:
        stats["grid_optimal_controller"] = grid_stats

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else run_dir / "plots" / f"sampled_initial_sets_{timestamp}.png"
    )
    control_output_path = (
        Path(args.control_output).expanduser().resolve()
        if args.control_output is not None
        else run_dir / "plots" / f"sampled_initial_sets_controls_{timestamp}.png"
    )
    stats_path = (
        Path(args.stats_json).expanduser().resolve()
        if args.stats_json is not None
        else run_dir / f"sampled_initial_sets_stats_{timestamp}.json"
    )
    initial_convergence_output_path = (
        Path(args.initial_convergence_output).expanduser().resolve()
        if args.initial_convergence_output is not None
        else run_dir / "plots" / f"sampled_initial_sets_initial_convergence_{timestamp}.png"
    )

    inside_plot_points = select_for_plot(inside_points, args.max_plot_trajectories, args.seed + 1)
    outside_plot_points = select_for_plot(outside_points, args.max_plot_trajectories, args.seed + 2)
    values_inside = trajectory_values_for_plot(
        system=system,
        models=models,
        points=inside_plot_points,
        steps=args.steps,
        controller_mode=args.controller_mode,
        c2=c2,
        device=device,
    )
    values_outside = trajectory_values_for_plot(
        system=system,
        models=models,
        points=outside_plot_points,
        steps=args.steps,
        controller_mode=args.controller_mode,
        c2=c2,
        device=device,
    )
    plot_side_by_side(
        values_inside=values_inside,
        values_outside=values_outside,
        sublevel=classification_sublevel,
        sublevel_key=classification_omega_name,
        inside_label=inside_label,
        outside_label=outside_label,
        output_path=output_path,
        title=(
            f"{system.name}: sampled initial sets "
            f"({run_dir.name}, {classification_omega_name}<={classification_sublevel:.6g})"
        ),
    )
    plot_control_inputs_side_by_side(
        values_inside=values_inside,
        values_outside=values_outside,
        system=system,
        inside_label=inside_label,
        outside_label=outside_label,
        output_path=control_output_path,
        title=f"{system.name}: learned-controller inputs ({run_dir.name})",
    )
    plot_pairs = normalize_plot_pairs(system)
    initial_convergence_paths: list[Path] = []
    initial_convergence_summaries = []
    for pair_idx, axis_pair in enumerate(plot_pairs):
        suffix = heatmap_pair_suffix(axis_pair)
        if args.initial_convergence_output is not None and len(plot_pairs) == 1:
            pair_output_path = initial_convergence_output_path
        elif args.initial_convergence_output is not None:
            pair_output_path = initial_convergence_output_path.with_name(
                f"{initial_convergence_output_path.stem}_{suffix}{initial_convergence_output_path.suffix}"
            )
        else:
            pair_output_path = run_dir / "plots" / (
                f"sampled_initial_sets_initial_convergence_{suffix}_{timestamp}.png"
            )
        slice_points, slice_stats = slice_uniform_samples(
            domain,
            int(args.initial_convergence_samples),
            axis_pair,
            args.seed + 300 + pair_idx,
            slice_half_width_fraction=args.initial_convergence_slice_fraction,
        )
        slice_converged = convergence_mask(
            system=system,
            pi=models["pi"],
            points=slice_points,
            steps=args.steps,
            norm2_tol=args.convergence_norm2_tol,
            controller_mode=args.controller_mode,
            c2=c2,
            device=device,
            batch_size=args.batch_size,
        )
        initial_convergence_paths.append(pair_output_path)
        initial_convergence_summaries.append(
            plot_initial_convergence_scatter(
                points=slice_points,
                converged=slice_converged,
                system=system,
                axis_pair=axis_pair,
                domain=domain,
                slice_stats=slice_stats,
                output_path=pair_output_path,
                title=f"{system.name}: initial convergence sample",
                seed=args.seed + 3 + pair_idx,
                max_points=100,
            )
        )
    stats["initial_convergence_slice_plots"] = initial_convergence_summaries

    grid_output_path = None
    grid_control_output_path = None
    if args.include_grid_optimal:
        assert control_grid is not None
        grid_output_path = (
            Path(args.grid_optimal_output).expanduser().resolve()
            if args.grid_optimal_output is not None
            else run_dir / "plots" / f"sampled_initial_sets_grid_optimal_{timestamp}.png"
        )
        grid_control_output_path = (
            Path(args.grid_optimal_control_output).expanduser().resolve()
            if args.grid_optimal_control_output is not None
            else run_dir / "plots" / f"sampled_initial_sets_grid_optimal_controls_{timestamp}.png"
        )
        grid_values_inside = simulate_grid_optimal_trajectories(
            system=system,
            models=models,
            points=inside_plot_points,
            control_grid=control_grid,
            steps=args.steps,
            device=device,
            control_chunk_size=args.control_chunk_size,
        )
        grid_values_outside = simulate_grid_optimal_trajectories(
            system=system,
            models=models,
            points=outside_plot_points,
            control_grid=control_grid,
            steps=args.steps,
            device=device,
            control_chunk_size=args.control_chunk_size,
        )
        plot_side_by_side(
            values_inside=grid_values_inside,
            values_outside=grid_values_outside,
            sublevel=classification_sublevel,
            sublevel_key=classification_omega_name,
            inside_label=inside_label,
            outside_label=outside_label,
            output_path=grid_output_path,
            title=(
                f"{system.name}: grid-optimal rollouts "
                f"({run_dir.name}, {classification_omega_name}<={classification_sublevel:.6g})"
            ),
        )
        plot_control_inputs_side_by_side(
            values_inside=grid_values_inside,
            values_outside=grid_values_outside,
            system=system,
            inside_label=inside_label,
            outside_label=outside_label,
            output_path=grid_control_output_path,
            title=f"{system.name}: grid-optimal control inputs ({run_dir.name})",
        )

    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Saved stats: {stats_path}")
    print(f"Saved plot: {output_path}")
    print(f"Saved control plot: {control_output_path}")
    for path in initial_convergence_paths:
        print(f"Saved initial convergence plot: {path}")
    if grid_output_path is not None:
        print(f"Saved grid-optimal plot: {grid_output_path}")
    if grid_control_output_path is not None:
        print(f"Saved grid-optimal control plot: {grid_control_output_path}")

    if args.points_csv is not None:
        points_csv = Path(args.points_csv).expanduser().resolve()
        write_points_csv(
            points_csv,
            points,
            classification_values,
            inside_mask,
            all_converged,
            classification_value_name=classification_omega_name,
            inside_column_name=f"inside_{classification_omega_name}_sublevel",
        )
        print(f"Saved sampled point labels: {points_csv}")


if __name__ == "__main__":
    main()
