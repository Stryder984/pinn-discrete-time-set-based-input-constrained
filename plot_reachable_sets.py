from __future__ import annotations

import argparse
import csv
import json
import os
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


def ensure_runtime_deps():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "plot_reachable_sets.py requires torch, numpy, and matplotlib. "
            "Run it from the same Python environment used for Streamlined training."
        ) from exc
    return plt, np, torch


def parse_plot_pairs(text: Optional[str], system) -> tuple[tuple[int, int], ...]:
    from training_pipeline import normalize_plot_pairs

    if text is None:
        return normalize_plot_pairs(system)
    pairs = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Bad plot pair {item!r}; use forms like '0,1;1,3'.")
        i, j = int(parts[0]), int(parts[1])
        if i == j or not (0 <= i < system.n_dim) or not (0 <= j < system.n_dim):
            raise ValueError(f"Plot pair {(i, j)} is invalid for n_dim={system.n_dim}.")
        pairs.append((i, j))
    if not pairs:
        raise ValueError("--plot-pairs did not contain any usable pairs.")
    return tuple(dict.fromkeys(pairs))


def parse_points_json(text: str, n_dim: int) -> list[list[float]]:
    payload = json.loads(text)
    if isinstance(payload, dict):
        payload = payload.get("points")
    if not isinstance(payload, list):
        raise ValueError("--points-json must be a JSON list or an object with a 'points' list.")
    points = []
    for idx, row in enumerate(payload):
        if not isinstance(row, list) or len(row) != n_dim:
            raise ValueError(f"Point {idx} must be a list of length {n_dim}.")
        points.append([float(value) for value in row])
    return points


def load_points_csv(path: Path, n_dim: int) -> list[list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    points = []
    for row_idx, row in enumerate(rows):
        point = []
        for dim in range(n_dim):
            key = f"x{dim}"
            if key not in row or row[key] in {None, ""}:
                raise ValueError(f"{path} row {row_idx} is missing {key}.")
            point.append(float(row[key]))
        points.append(point)
    return points


def default_points(system) -> list[list[float]]:
    points = [[0.0 for _ in range(system.n_dim)]]
    if system.verification is not None:
        lo = [float(a) for a, _ in system.verification.domain]
        hi = [float(b) for _, b in system.verification.domain]
    else:
        lo = [float(v) for v in system.x_lb]
        hi = [float(v) for v in system.x_hi]
    points.append([(a + b) / 4.0 for a, b in zip(lo, hi)])
    return points


def resolve_run_dir(system, args) -> Optional[Path]:
    if args.run_dir is not None:
        return Path(args.run_dir).expanduser().resolve()
    if args.run_name is not None:
        return RUNS_ROOT / system.name / args.run_name
    return None


def load_run_system(system, run_dir: Optional[Path]):
    if run_dir is None:
        return system
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    from training_pipeline import system_with_existing_run_domain

    return system_with_existing_run_domain(system, run_dir)


def embedding_mode_from_args(system, args, run_dir: Optional[Path]) -> str:
    return "zonotope_generators"


def reconstruct_from_embedding(system, z, mode: str, torch):
    normalized = str(mode).strip().lower()
    n = int(system.n_dim)
    m = int(system.u_dim)
    center = z[..., :n]
    if normalized == "zonotope_generators":
        expected = n + n * m
        if z.shape[-1] != expected:
            raise ValueError(
                f"zonotope_generators expected embedding width {expected}, got {z.shape[-1]}."
            )
        generators = z[..., n:].reshape(*z.shape[:-1], m, n)
        radius = generators.abs().sum(dim=-2)
        return {
            "kind": "zonotope",
            "center": center,
            "lb": center - radius,
            "ub": center + radius,
            "generators": generators,
        }
    raise ValueError(f"Unknown embedding mode {mode!r}.")


def projected_vertices(center, generators, pair, np):
    center_pair = center[list(pair)]
    generator_pair = generators[:, list(pair)]
    m = int(generator_pair.shape[0])
    if m == 0:
        return center_pair.reshape(1, 2)
    signs = np.array(
        [[1.0 if (mask >> j) & 1 else -1.0 for j in range(m)] for mask in range(2**m)],
        dtype=float,
    )
    return center_pair.reshape(1, 2) + signs @ generator_pair


def convex_hull(points, np):
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) <= 1:
        return np.array(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def interval_rectangle(lb, ub, pair, np):
    i, j = pair
    return np.array(
        [
            [lb[i], lb[j]],
            [ub[i], lb[j]],
            [ub[i], ub[j]],
            [lb[i], ub[j]],
        ],
        dtype=float,
    )


def axis_label(system, idx: int) -> str:
    if system.state_labels is not None and idx < len(system.state_labels):
        return f"x{idx} ({system.state_labels[idx]})"
    return f"x{idx}"


def plot_reachable_sets(system, points, pairs, mode: str, out_dir: Path, args) -> list[Path]:
    plt, np, torch = ensure_runtime_deps()
    from training_pipeline import build_successor_zonotope_embedding

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    x = torch.tensor(points, device=device, dtype=torch.float32)
    with torch.no_grad():
        z = build_successor_zonotope_embedding(system, x)
        reconstructed = reconstruct_from_embedding(system, z, mode, torch)

    x_np = x.detach().cpu().numpy()
    center_np = reconstructed["center"].detach().cpu().numpy()
    lb_np = reconstructed["lb"].detach().cpu().numpy()
    ub_np = reconstructed["ub"].detach().cpu().numpy()
    generators_np = reconstructed["generators"].detach().cpu().numpy()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = []
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(points), 1)))
    for pair in pairs:
        fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
        for point_idx, color in enumerate(colors):
            if reconstructed["kind"] == "interval":
                polygon = interval_rectangle(lb_np[point_idx], ub_np[point_idx], pair, np)
                edge_label = "interval hull" if point_idx == 0 else None
            else:
                vertices = projected_vertices(center_np[point_idx], generators_np[point_idx], pair, np)
                polygon = convex_hull(vertices, np)
                edge_label = "zonotope projection" if point_idx == 0 else None
            if polygon.shape[0] >= 3:
                closed = np.vstack([polygon, polygon[0]])
                ax.fill(closed[:, 0], closed[:, 1], color=color, alpha=0.18)
                ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.8, label=edge_label)
            elif polygon.shape[0] == 2:
                ax.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=3.0, label=edge_label)
            else:
                ax.scatter(polygon[:, 0], polygon[:, 1], color=color, s=60, label=edge_label)

            i, j = pair
            ax.scatter(
                [x_np[point_idx, i]],
                [x_np[point_idx, j]],
                marker="x",
                s=70,
                color=color,
                linewidths=2.0,
                label=f"x[{point_idx}]" if args.label_points else None,
            )
            ax.scatter(
                [center_np[point_idx, i]],
                [center_np[point_idx, j]],
                marker="o",
                s=34,
                facecolors="white",
                edgecolors=color,
                linewidths=1.6,
                label=f"center[{point_idx}]" if args.label_points else None,
            )

        ax.set_title(f"{system.name}: one-step reachable set, {mode}")
        ax.set_xlabel(axis_label(system, pair[0]))
        ax.set_ylabel(axis_label(system, pair[1]))
        ax.axhline(0.0, color="0.82", linewidth=0.8)
        ax.axvline(0.0, color="0.82", linewidth=0.8)
        ax.grid(True, color="0.90", linewidth=0.8)
        ax.set_aspect("equal", adjustable="datalim")
        handles, labels = ax.get_legend_handles_labels()
        if any(label for label in labels):
            ax.legend(loc="best", fontsize=8)
        filename = f"reachable_set_{system.name}_{mode}_x{pair[0]}_x{pair[1]}_{stamp}.png"
        path = out_dir / filename
        fig.savefig(path, dpi=int(args.dpi))
        plt.close(fig)
        paths.append(path)
    return paths


def write_summary(path: Path, system, points, pairs, mode: str, plots: Sequence[Path]) -> None:
    payload = {
        "system": system.name,
        "n_dim": system.n_dim,
        "u_dim": system.u_dim,
        "embedding_mode": mode,
        "points": points,
        "plot_pairs": [list(pair) for pair in pairs],
        "plots": [str(path) for path in plots],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one-step reachable sets represented by the fixed "
            "omega_orig zonotope embedding."
        )
    )
    parser.add_argument("--system", required=True, help="System key from systems.py.")
    parser.add_argument("--run-name", default=None, help="Run folder under Streamlinedexperimental/pipeline_runs/<system>.")
    parser.add_argument("--run-dir", default=None, help="Explicit run directory.")
    parser.add_argument(
        "--embedding-mode",
        default=None,
        choices=("zonotope_generators",),
        help="Compatibility option; omega_orig now always uses zonotope_generators.",
    )
    parser.add_argument("--plot-pairs", default=None, help="Semicolon-separated zero-based pairs, e.g. '0,1;1,3'.")
    parser.add_argument("--points-json", default=None, help="JSON list of states, e.g. '[[0,0,0,0],[.1,0,0,0]]'.")
    parser.add_argument("--points-csv", default=None, help="CSV with columns x0, x1, ...; threshold_witness_points.csv works.")
    parser.add_argument("--max-points", type=int, default=12, help="Maximum CSV/default points to plot.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to <run>/plots or Streamlinedexperimental/plots.")
    parser.add_argument("--device", default="cpu", help="Torch device for constructing embeddings; CPU is enough for plotting.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--label-points", action="store_true", help="Add legend labels for each point and reachable center.")
    args = parser.parse_args(argv)

    ensure_runtime_deps()
    from systems import get_system

    system = get_system(args.system)
    run_dir = resolve_run_dir(system, args)
    system = load_run_system(system, run_dir)
    mode = embedding_mode_from_args(system, args, run_dir)
    pairs = parse_plot_pairs(args.plot_pairs, system)

    if args.points_json is not None:
        points = parse_points_json(args.points_json, system.n_dim)
    elif args.points_csv is not None:
        points = load_points_csv(Path(args.points_csv).expanduser().resolve(), system.n_dim)
    elif run_dir is not None and (run_dir / "threshold_witness_points.csv").is_file():
        points = load_points_csv(run_dir / "threshold_witness_points.csv", system.n_dim)
    else:
        points = default_points(system)
    points = points[: max(1, int(args.max_points))]

    if args.out_dir is not None:
        out_dir = Path(args.out_dir).expanduser().resolve()
    elif run_dir is not None:
        out_dir = run_dir / "plots"
    else:
        out_dir = STREAMLINED_ROOT / "plots"

    plots = plot_reachable_sets(system, points, pairs, mode, out_dir, args)
    summary_path = out_dir / f"reachable_sets_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_summary(summary_path, system, points, pairs, mode, plots)

    print("Reachable-set plots saved:")
    for path in plots:
        print(f"  {path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
