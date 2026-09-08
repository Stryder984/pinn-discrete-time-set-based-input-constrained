from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

STREAMLINED_ROOT = Path(__file__).resolve().parent
REPO_ROOT = STREAMLINED_ROOT.parent
if str(STREAMLINED_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMLINED_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RUNS_ROOT = STREAMLINED_ROOT / "pipeline_runs"
plt = None
np = None
torch = None
load_models = None
get_system = None
nu_from_p = None
system_with_existing_run_domain = None
zubov_alpha_from_nu = None
normalize_plot_pairs = None
normalize_axis_pair = None
heatmap_pair_suffix = None
points_in_bounds = None
make_control_grid = None
compute_optimal_next_omega_grid_labels = None


def ensure_runtime_deps() -> None:
    """Load Torch/Matplotlib only for commands that actually render graphs."""
    global plt, np, torch, load_models, get_system, nu_from_p
    global system_with_existing_run_domain, zubov_alpha_from_nu
    global normalize_plot_pairs, normalize_axis_pair, heatmap_pair_suffix
    global points_in_bounds, make_control_grid, compute_optimal_next_omega_grid_labels
    if torch is not None:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
        import numpy as numpy_module
        import torch as torch_module
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "plot_threshold_graphs.py requires the Streamlined plotting runtime "
            "(torch, numpy, matplotlib). Run it from the same Python environment "
            "used for Streamlined training."
        ) from exc

    from models import load_models as load_models_fn
    from systems import get_system as get_system_fn
    from training_pipeline import (
        compute_optimal_next_omega_grid_labels as compute_optimal_next_omega_grid_labels_fn,
        heatmap_pair_suffix as heatmap_pair_suffix_fn,
        make_control_grid as make_control_grid_fn,
        normalize_axis_pair as normalize_axis_pair_fn,
        normalize_plot_pairs as normalize_plot_pairs_fn,
        nu_from_p as nu_from_p_fn,
        points_in_bounds as points_in_bounds_fn,
        system_with_existing_run_domain as system_with_existing_run_domain_fn,
        zubov_alpha_from_nu as zubov_alpha_from_nu_fn,
    )

    plt = pyplot
    np = numpy_module
    torch = torch_module
    load_models = load_models_fn
    get_system = get_system_fn
    nu_from_p = nu_from_p_fn
    system_with_existing_run_domain = system_with_existing_run_domain_fn
    zubov_alpha_from_nu = zubov_alpha_from_nu_fn
    normalize_plot_pairs = normalize_plot_pairs_fn
    normalize_axis_pair = normalize_axis_pair_fn
    heatmap_pair_suffix = heatmap_pair_suffix_fn
    points_in_bounds = points_in_bounds_fn
    make_control_grid = make_control_grid_fn
    compute_optimal_next_omega_grid_labels = compute_optimal_next_omega_grid_labels_fn


def pick_device(requested: str) -> torch.device:
    ensure_runtime_deps()
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


def latest_run_dir(system: SystemSpec) -> Path:
    system_runs = RUNS_ROOT / system.name
    if not system_runs.is_dir():
        raise FileNotFoundError(f"No run directory found for {system.name!r}: {system_runs}")
    candidates = sorted(
        (path for path in system_runs.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        if has_required_checkpoints(system, run_dir / "artifacts"):
            return run_dir
    raise FileNotFoundError(f"No run with required checkpoints found under {system_runs}")


def resolve_run_dir(system: SystemSpec, args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        run_dir = Path(args.run_dir).expanduser().resolve()
    elif args.run_name is not None:
        run_dir = RUNS_ROOT / system.name / args.run_name
    else:
        run_dir = latest_run_dir(system)
    if not has_required_checkpoints(system, run_dir / "artifacts"):
        required = "\n".join(f"  - {path}" for path in required_checkpoint_paths(system, run_dir / "artifacts"))
        raise FileNotFoundError(f"Run is missing required checkpoints:\n{required}")
    return run_dir


def landscape_bounds(system: SystemSpec) -> tuple[tuple[float, float], tuple[float, float]]:
    if system.verification is not None:
        domain = tuple((float(lo), float(hi)) for lo, hi in system.verification.domain)
    else:
        domain = tuple(zip(system.x_lb, system.x_hi))
    return domain[0], domain[1]


def load_original_omega_thresholds(run_dir: Path, system: SystemSpec) -> dict[str, Optional[float]]:
    """Read verifier c2 and original-omega w1/w2 from threshold-search output."""
    threshold_path = run_dir / "verification_thresholds_rescaled.json"
    payload = json.loads(threshold_path.read_text(encoding="utf-8")) if threshold_path.is_file() else {}
    original = payload.get("original_omega") if isinstance(payload.get("original_omega"), dict) else {}
    raw = payload.get("raw") if isinstance(payload.get("raw"), dict) else {}

    def number(value, fallback=None):
        if value is None:
            value = fallback
        return None if value is None else float(value)

    vcfg = system.verification
    return {
        "eps": number(payload.get("eps"), None if vcfg is None else vcfg.eps),
        "c2": number(payload.get("c2"), None if vcfg is None else vcfg.c2),
        "w1": number(original.get("w1")),
        "w2": number(original.get("w2")),
        "c_analytic": number(raw.get("c_analytic"), raw.get("c1_search")),
        "c_nn_off": number(raw.get("C_NN_OFF")),
        "c_s": number(raw.get("C_S")),
    }


def safe_contour(ax, x0_np, x1_np, z_values, levels: Sequence[float], **kwargs) -> None:
    finite = z_values[np.isfinite(z_values)]
    if finite.size == 0:
        return
    z_min = float(np.min(finite))
    z_max = float(np.max(finite))
    usable = sorted(set(float(level) for level in levels if level is not None and z_min <= float(level) <= z_max))
    if not usable:
        return
    try:
        ax.contour(x0_np, x1_np, z_values, levels=usable, **kwargs)
    except Exception:
        return


def plot_original_omega_heatmaps(
    *,
    system: SystemSpec,
    models: dict[str, torch.nn.Module],
    run_dir: Path,
    device: torch.device,
    axis_pair: tuple[int, int],
    thresholds: dict[str, Optional[float]],
    grid_points: Optional[int] = None,
) -> Optional[Path]:
    """Port of the pipeline heatmap style using original-omega w1/w2 thresholds."""
    ensure_runtime_deps()
    if system.verification is None or system.n_dim < 2:
        return None
    if thresholds.get("c2") is None or thresholds.get("w1") is None or thresholds.get("w2") is None:
        raise ValueError(
            "Original-omega heatmaps need c2 plus original_omega.w1/w2 in "
            f"{run_dir / 'verification_thresholds_rescaled.json'}."
        )

    from matplotlib.lines import Line2D

    axis_pair = normalize_axis_pair(system, axis_pair)
    pi = models["pi"]
    omega_orig = models["omega"]
    p_matrix = pi.P_matrix.to(device=device, dtype=torch.float32)
    vcfg = system.verification
    grid_n = int(grid_points or system.training.diagnostic_plot_grid_points)
    x_axis, y_axis = axis_pair
    x0 = torch.linspace(float(system.x_lb[x_axis]), float(system.x_hi[x_axis]), grid_n, device=device)
    x1 = torch.linspace(float(system.x_lb[y_axis]), float(system.x_hi[y_axis]), grid_n, device=device)
    x0_grid, x1_grid = torch.meshgrid(x0, x1, indexing="xy")
    points = torch.zeros(x0_grid.numel(), system.n_dim, device=device, dtype=torch.float32)
    points[:, x_axis] = x0_grid.reshape(-1)
    points[:, y_axis] = x1_grid.reshape(-1)

    pi.eval()
    omega_orig.eval()
    with torch.no_grad():
        u_pi = pi(points)
        next_pi = system.dynamics(points, u_pi)
        next_pi_in_domain = points_in_bounds(next_pi, tuple(tuple(v) for v in vcfg.domain)).reshape(grid_n, grid_n)
        omega_orig_values = omega_orig.query(points).reshape(grid_n, grid_n)
        omega_orig_next_pi = omega_orig.query(next_pi).reshape(grid_n, grid_n)

        control_grid = make_control_grid(
            system,
            int(system.training.control_grid_points),
            device=device,
            dtype=points.dtype,
        )
        labels = compute_optimal_next_omega_grid_labels(
            system,
            omega_orig,
            points,
            control_grid,
            state_batch_size=max(1, int(system.training.control_grid_batch_size)),
        )
        next_opt = system.dynamics(points, labels.u_opt)
        next_opt_in_domain = points_in_bounds(next_opt, tuple(tuple(v) for v in vcfg.domain)).reshape(grid_n, grid_n)
        omega_opt_next = labels.omega_opt_next.reshape(grid_n, grid_n)
        nu_values = nu_from_p(points, p_matrix).reshape(grid_n, grid_n)

    eps = float(thresholds["eps"])
    c2 = float(thresholds["c2"])
    w1 = float(thresholds["w1"])
    w2 = float(thresholds["w2"])
    delta_pi = omega_orig_next_pi - omega_orig_values
    delta_opt = omega_opt_next - omega_orig_values
    pi_violation_mask = (delta_pi > -eps) | (~next_pi_in_domain)
    opt_violation_mask = (delta_opt > -eps) | (~next_opt_in_domain)

    x0_np = x0_grid.detach().cpu().numpy()
    x1_np = x1_grid.detach().cpu().numpy()
    omega_np = omega_orig_values.detach().cpu().numpy()
    delta_pi_np = delta_pi.detach().cpu().numpy()
    delta_opt_np = delta_opt.detach().cpu().numpy()
    nu_np = nu_values.detach().cpu().numpy()
    pi_mask_np = pi_violation_mask.detach().cpu().numpy()
    opt_mask_np = opt_violation_mask.detach().cpu().numpy()

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    suffix = heatmap_pair_suffix(axis_pair)
    summary = {
        "threshold_source": str(run_dir / "verification_thresholds_rescaled.json"),
        "thresholds": {
            "c2": c2,
            "w1_original": w1,
            "w2_original": w2,
            "eps": eps,
            "c_analytic": thresholds.get("c_analytic"),
            "C_NN_OFF": thresholds.get("c_nn_off"),
            "C_S": thresholds.get("c_s"),
        },
        "slice": {
            "axes": [int(axis_pair[0]), int(axis_pair[1])],
            "fixed_coordinates": {str(i): 0.0 for i in range(system.n_dim) if i not in axis_pair},
        },
        "grid_points": grid_n,
        "control_grid_points_per_dim": int(system.training.control_grid_points),
        "control_grid_total_points": int(control_grid.shape[0]),
        "masks": {
            "omega_original_under_pi_violation": {
                "count": int(pi_mask_np.sum()),
                "fraction": float(pi_mask_np.mean()),
            },
            "omega_original_optimal_label_violation": {
                "count": int(opt_mask_np.sum()),
                "fraction": float(opt_mask_np.mean()),
            },
        },
        "field_ranges": {
            "omega_original": [float(np.nanmin(omega_np)), float(np.nanmax(omega_np))],
            "delta_omega_original_pi": [float(np.nanmin(delta_pi_np)), float(np.nanmax(delta_pi_np))],
            "delta_omega_original_opt": [float(np.nanmin(delta_opt_np)), float(np.nanmax(delta_opt_np))],
            "nu": [float(np.nanmin(nu_np)), float(np.nanmax(nu_np))],
        },
    }
    summary_path = run_dir / f"original_omega_heatmap_summary_{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), constrained_layout=True)
    fields = (
        (axes[0], omega_np, pi_mask_np, "omega_original under pi violations", "omega original"),
        (axes[1], delta_pi_np, pi_mask_np, "delta omega_original under pi", "Delta omega original"),
        (axes[2], omega_np, opt_mask_np, "omega_original optimal-label violations", "omega original"),
    )
    for ax, values, mask, title, colorbar_label in fields:
        cmap = "coolwarm"
        if cmap == "coolwarm":
            vmax = max(float(np.nanmax(np.abs(values))), 1e-8)
            cf = ax.contourf(x0_np, x1_np, values, levels=80, cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            cf = ax.contourf(x0_np, x1_np, values, levels=80, cmap=cmap)
        fig.colorbar(cf, ax=ax, label=colorbar_label)
        ax.contourf(x0_np, x1_np, mask.astype(float), levels=[0.5, 1.5], colors=["white"], alpha=0.95)
        if thresholds.get("c_analytic") is not None:
            safe_contour(ax, x0_np, x1_np, nu_np, [float(thresholds["c_analytic"])], colors="black", linewidths=1.0)
        if thresholds.get("c_nn_off") is not None:
            safe_contour(ax, x0_np, x1_np, nu_np, [float(thresholds["c_nn_off"])], colors="orange", linewidths=1.0)
        if thresholds.get("c_s") is not None:
            safe_contour(ax, x0_np, x1_np, nu_np, [float(thresholds["c_s"])], colors="red", linewidths=1.0)
        safe_contour(ax, x0_np, x1_np, nu_np, [c2], colors="magenta", linewidths=1.8)
        safe_contour(ax, x0_np, x1_np, omega_np, [w1], colors="cyan", linestyles=["--"], linewidths=1.8)
        safe_contour(ax, x0_np, x1_np, omega_np, [w2], colors="lime", linestyles=["--"], linewidths=1.8)
        ax.set_title(title)
        ax.grid(True, alpha=0.2)

    labels = system.state_labels or tuple(f"x{i}" for i in range(system.n_dim))
    for ax in axes:
        ax.set_xlabel(labels[axis_pair[0]])
        ax.set_ylabel(labels[axis_pair[1]])

    legend_handles = [
        Line2D([0], [0], color="white", lw=2.0, label="nu=c_analytic"),
        Line2D([0], [0], color="orange", lw=2.0, label="nu=C_NN_OFF"),
        Line2D([0], [0], color="red", lw=2.0, label="nu=C_S"),
        Line2D([0], [0], color="magenta", lw=1.8, label="nu=c2"),
        Line2D([0], [0], color="cyan", lw=1.8, linestyle="--", label="omega_original=w1"),
        Line2D([0], [0], color="lime", lw=1.8, linestyle="--", label="omega_original=w2"),
        Line2D([0], [0], color="white", lw=6, alpha=0.95, label="violation mask"),
    ]
    axes[2].legend(handles=legend_handles, loc="upper right", framealpha=0.9)
    fig.suptitle(
        "Original-Omega Heatmaps and Decrease-Violation Masks "
        f"({labels[axis_pair[0]]}, {labels[axis_pair[1]]})"
    )
    output_path = plot_dir / f"original_omega_heatmap_decrease_diagnostic_{suffix}.png"
    fig.savefig(output_path, dpi=180)
    if axis_pair == (0, 1):
        fig.savefig(plot_dir / "original_omega_heatmap_decrease_diagnostic.png", dpi=180)
    plt.close(fig)
    return output_path


def plot_state_space_landscapes(
    *,
    system: SystemSpec,
    models: dict[str, torch.nn.Module],
    run_dir: Path,
    device: torch.device,
    grid_points: int = 151,
    elev: float = 30.0,
    azim: float = -135.0,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Plot alpha, omega, and omega_cl as 3D surfaces for 2D state systems.

    Higher-dimensional systems need an explicit slice choice, so this function
    returns None for those models instead of guessing.
    """
    ensure_runtime_deps()
    if system.n_dim != 2:
        return None

    pi = models["pi"]
    omega_orig = models["omega"]
    omega_cl = models["omega_closed_loop"]
    p_matrix = pi.P_matrix.to(device=device, dtype=torch.float32)
    (x0_lo, x0_hi), (x1_lo, x1_hi) = landscape_bounds(system)

    x0_axis = torch.linspace(x0_lo, x0_hi, int(grid_points), device=device)
    x1_axis = torch.linspace(x1_lo, x1_hi, int(grid_points), device=device)
    x0_grid, x1_grid = torch.meshgrid(x0_axis, x1_axis, indexing="ij")
    points = torch.stack([x0_grid.reshape(-1), x1_grid.reshape(-1)], dim=1)

    with torch.no_grad():
        nu = nu_from_p(points, p_matrix)
        alpha = zubov_alpha_from_nu(
            nu,
            dt=system.dt,
            scale=system.training.zubov_alpha_scale,
        ).reshape(grid_points, grid_points)
        omega = omega_orig.query(points).reshape(grid_points, grid_points)
        omega_closed_loop = omega_cl.query(points).reshape(grid_points, grid_points)

    x0_np = x0_grid.detach().cpu().numpy()
    x1_np = x1_grid.detach().cpu().numpy()
    panels = (
        (alpha.detach().cpu().numpy(), "alpha(x)"),
        (omega.detach().cpu().numpy(), "omega original"),
        (omega_closed_loop.detach().cpu().numpy(), "omega closed-loop"),
    )

    fig = plt.figure(figsize=(15, 4.8), constrained_layout=True)
    for idx, (values, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.plot_surface(x0_np, x1_np, values, cmap="viridis", linewidth=0, antialiased=True, alpha=0.96)
        ax.set_title(title)
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_zlabel(title)
        ax.view_init(elev=float(elev), azim=float(azim))

    fig.suptitle(f"{system.name}: state-space threshold landscapes ({run_dir.name})", fontsize=12)
    if output_path is None:
        output_path = run_dir / "plots" / "threshold_state_space_landscapes.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _row_point(row: dict[str, str], n_dim: int) -> Optional[list[float]]:
    values = []
    for i in range(n_dim):
        raw = row.get(f"x{i}")
        if raw in (None, ""):
            return None
        values.append(float(raw))
    return values


def load_witness_points(run_dir: Path, n_dim: int, names: Optional[set[str]] = None) -> list[tuple[str, list[float]]]:
    witness_path = run_dir / "threshold_witness_points.csv"
    if not witness_path.is_file():
        return []
    points: list[tuple[str, list[float]]] = []
    with witness_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("name") or f"witness_{len(points)}"
            if names is not None and name not in names:
                continue
            point = _row_point(row, n_dim)
            if point is not None:
                points.append((name, point))
    return points


def load_text_counterexamples(cex_dir: Optional[Path], system_name: str, n_dim: int) -> list[tuple[str, list[float]]]:
    if cex_dir is None or not cex_dir.is_dir():
        return []
    points: list[tuple[str, list[float]]] = []
    pattern = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
    for path in sorted(cex_dir.glob(f"{system_name}_cex*.txt")):
        numbers = [float(match.group(0)) for match in pattern.finditer(path.read_text(encoding="utf-8", errors="ignore"))]
        if len(numbers) >= n_dim:
            points.append((path.stem, numbers[:n_dim]))
    return points


def unique_points(points: Sequence[tuple[str, Sequence[float]]]) -> list[tuple[str, list[float]]]:
    seen: set[tuple[float, ...]] = set()
    unique: list[tuple[str, list[float]]] = []
    for name, point in points:
        key = tuple(round(float(value), 10) for value in point)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, [float(value) for value in point]))
    return unique


def simulate_witness_trajectories(
    *,
    system: SystemSpec,
    models: dict[str, torch.nn.Module],
    starts: Sequence[tuple[str, Sequence[float]]],
    steps: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Simulate PiNet closed-loop trajectories from saved witness starts."""
    ensure_runtime_deps()
    if not starts:
        raise ValueError("No witness or counterexample starts were found.")

    pi = models["pi"]
    omega_orig = models["omega"]
    omega_cl = models["omega_closed_loop"]
    p_matrix = pi.P_matrix.to(device=device, dtype=torch.float32)

    state = torch.tensor([point for _, point in starts], device=device, dtype=torch.float32)
    n_traj = state.shape[0]
    states = torch.empty((n_traj, steps + 1, system.n_dim), device=device)
    omega_orig_values = torch.empty((n_traj, steps + 1), device=device)
    omega_cl_values = torch.empty((n_traj, steps + 1), device=device)
    alpha_values = torch.empty((n_traj, steps + 1), device=device)
    nu_values = torch.empty((n_traj, steps + 1), device=device)

    with torch.no_grad():
        for t in range(steps + 1):
            states[:, t, :] = state
            omega_orig_values[:, t] = omega_orig.query(state).reshape(-1)
            omega_cl_values[:, t] = omega_cl.query(state).reshape(-1)
            nu = nu_from_p(state, p_matrix)
            nu_values[:, t] = nu
            alpha_values[:, t] = zubov_alpha_from_nu(
                nu,
                dt=system.dt,
                scale=system.training.zubov_alpha_scale,
            )
            if t < steps:
                state = system.dynamics(state, pi(state))

    return {
        "states": states.detach().cpu().numpy(),
        "omega_original": omega_orig_values.detach().cpu().numpy(),
        "omega_cl": omega_cl_values.detach().cpu().numpy(),
        "alpha": alpha_values.detach().cpu().numpy(),
        "nu": nu_values.detach().cpu().numpy(),
    }


def plot_trajectory_values(
    values: dict[str, np.ndarray],
    starts: Sequence[tuple[str, Sequence[float]]],
    output_path: Path,
) -> None:
    ensure_runtime_deps()
    panels = (
        ("alpha", "alpha(x_t)"),
        ("omega_original", "omega original"),
        ("omega_cl", "omega closed-loop"),
        ("nu", "nu=x'Px"),
    )
    t = np.arange(values["alpha"].shape[1])
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(starts), 1)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (key, label) in zip(axes.flat, panels):
        for idx, (name, _point) in enumerate(starts):
            ax.plot(t, values[key][idx], linewidth=2.0, color=colors[idx % len(colors)], label=name)
        ax.set_xlabel("step")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_phase_trajectories(
    values: dict[str, np.ndarray],
    starts: Sequence[tuple[str, Sequence[float]]],
    output_path: Path,
) -> Optional[Path]:
    ensure_runtime_deps()
    states = values["states"]
    if states.shape[2] < 2:
        return None
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(starts), 1)))
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for idx, (name, _point) in enumerate(starts):
        xy = states[idx, :, :2]
        ax.plot(xy[:, 0], xy[:, 1], color=colors[idx % len(colors)], linewidth=2.0, label=name)
        ax.scatter(xy[0, 0], xy[0, 1], color=colors[idx % len(colors)], s=35)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def write_trajectory_csv(values: dict[str, np.ndarray], starts: Sequence[tuple[str, Sequence[float]]], output_path: Path) -> None:
    states = values["states"]
    state_columns = [f"x{i}" for i in range(states.shape[2])]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trajectory", "name", "step", *state_columns, "alpha", "omega_original", "omega_cl", "nu"])
        for traj_idx, (name, _point) in enumerate(starts):
            for step in range(states.shape[1]):
                writer.writerow(
                    [
                        traj_idx,
                        name,
                        step,
                        *states[traj_idx, step, :].tolist(),
                        values["alpha"][traj_idx, step],
                        values["omega_original"][traj_idx, step],
                        values["omega_cl"][traj_idx, step],
                        values["nu"][traj_idx, step],
                    ]
                )


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate threshold landscapes and closed-loop trajectory graphs "
            "from saved threshold-search witnesses."
        )
    )
    parser.add_argument("--system", default="2d_bench", help="System key from systems.py")
    parser.add_argument("--run-name", default=None, help="Run folder under Streamlined/pipeline_runs/<system>.")
    parser.add_argument("--run-dir", default=None, help="Explicit run directory containing artifacts/.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--grid-points", type=int, default=151, help="Grid resolution for 2D landscapes.")
    parser.add_argument(
        "--heatmap-grid-points",
        type=int,
        default=None,
        help="Override grid resolution for original-omega heatmaps.",
    )
    parser.add_argument(
        "--skip-original-heatmaps",
        action="store_true",
        help="Skip graph-script heatmaps that use original_omega.w1/w2.",
    )
    parser.add_argument("--elev", type=float, default=30.0, help="3D landscape camera elevation in degrees.")
    parser.add_argument("--azim", type=float, default=-135.0, help="3D landscape camera azimuth in degrees.")
    parser.add_argument("--steps", type=int, default=75, help="Closed-loop trajectory steps.")
    parser.add_argument("--cex-dir", default=str(STREAMLINED_ROOT / "counterexamples"))
    parser.add_argument(
        "--include-verifier-cex",
        action="store_true",
        help="Also simulate verifier counterexample text files from --cex-dir.",
    )
    parser.add_argument(
        "--include-all-witnesses",
        action="store_true",
        help="Use every row in threshold_witness_points.csv instead of only original-omega w1/w2 rows.",
    )
    args = parser.parse_args(argv)

    if args.steps < 0:
        raise ValueError("--steps must be nonnegative.")
    if args.grid_points < 5:
        raise ValueError("--grid-points must be at least 5.")
    if args.heatmap_grid_points is not None and args.heatmap_grid_points < 5:
        raise ValueError("--heatmap-grid-points must be at least 5.")

    ensure_runtime_deps()
    base_system = get_system(args.system)
    run_dir = resolve_run_dir(base_system, args)
    system = system_with_existing_run_domain(base_system, run_dir)
    device = pick_device(args.device)
    models = load_models(system, run_dir / "artifacts", device=device)

    plot_dir = run_dir / "plots"
    landscape_path = plot_state_space_landscapes(
        system=system,
        models=models,
        run_dir=run_dir,
        device=device,
        grid_points=args.grid_points,
        elev=args.elev,
        azim=args.azim,
    )
    heatmap_paths: list[str] = []
    if not args.skip_original_heatmaps:
        original_thresholds = load_original_omega_thresholds(run_dir, system)
        for axis_pair in normalize_plot_pairs(system):
            heatmap_path = plot_original_omega_heatmaps(
                system=system,
                models=models,
                run_dir=run_dir,
                device=device,
                axis_pair=axis_pair,
                thresholds=original_thresholds,
                grid_points=args.heatmap_grid_points,
            )
            if heatmap_path is not None:
                heatmap_paths.append(str(heatmap_path))

    witness_names = None if args.include_all_witnesses else {"w1_original", "w2_original"}
    starts = load_witness_points(run_dir, system.n_dim, names=witness_names)
    if args.include_verifier_cex:
        starts.extend(load_text_counterexamples(Path(args.cex_dir), system.name, system.n_dim))
    starts = unique_points(starts)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trajectory_paths: list[str] = []
    if starts:
        values = simulate_witness_trajectories(
            system=system,
            models=models,
            starts=starts,
            steps=args.steps,
            device=device,
        )
        value_plot = plot_dir / f"threshold_witness_trajectory_values_{timestamp}.png"
        phase_plot = plot_dir / f"threshold_witness_phase_{timestamp}.png"
        csv_path = run_dir / f"threshold_witness_trajectories_{timestamp}.csv"
        plot_trajectory_values(values, starts, value_plot)
        trajectory_paths.append(str(value_plot))
        phase_result = plot_phase_trajectories(values, starts, phase_plot)
        if phase_result is not None:
            trajectory_paths.append(str(phase_result))
        write_trajectory_csv(values, starts, csv_path)
        trajectory_paths.append(str(csv_path))

    summary = {
        "system": system.name,
        "run_dir": str(run_dir),
        "n_dim": int(system.n_dim),
        "landscape_camera": {"elev": float(args.elev), "azim": float(args.azim)},
        "landscape_plot": None if landscape_path is None else str(landscape_path),
        "landscape_skipped": system.n_dim != 2,
        "original_omega_heatmaps": heatmap_paths,
        "trajectory_starts": [{"name": name, "x": list(point)} for name, point in starts],
        "trajectory_outputs": trajectory_paths,
        "notes": [
            "3D alpha/omega landscapes are generated only for n_dim == 2.",
            "Original-omega heatmaps use original_omega.w1/w2 from verification_thresholds_rescaled.json.",
            "Default trajectory starts are w1_original and w2_original threshold witnesses.",
            "Verifier cex text files are included only when --include-verifier-cex is passed.",
        ],
    }
    summary_path = run_dir / f"threshold_graphs_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Loaded run: {run_dir}")
    if landscape_path is None:
        print(f"Skipped 3D landscapes because n_dim={system.n_dim}; configure a 2D slice for this system.")
    else:
        print(f"Saved landscapes: {landscape_path}")
    if heatmap_paths:
        print(f"Saved original-omega heatmaps: {', '.join(heatmap_paths)}")
    if starts:
        print(f"Saved trajectory outputs: {', '.join(trajectory_paths)}")
    else:
        print("No witness or counterexample starts found for trajectory simulation.")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
