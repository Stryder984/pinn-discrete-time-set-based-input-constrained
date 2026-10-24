"""Replay comparison archives into figures/tables, without simulating dynamics."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import torch

from models import load_omeganet


OBJECTIVE_NAMES = {"omega_orig": "omega_orig", "norm2": "Norm squared", "nu": "Quadratic nu"}
OBJECTIVE_MATH = {"omega_orig": r"$\omega_{\rm orig}(x)$", "norm2": r"$\|x\|^2$", "nu": r"$\nu(x)=x^\top P x$"}
SUCCESS, RED, YELLOW = "#17834b", "#c93636", "#e0b000"
OUTCOMES = ("reached_roa", "unconverged_in_domain", "no_in_domain_successor", "numerical_error")
ROA_NAMES = {"c1": "local ROA", "c2": "expanded ROA"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_path(root, item, filename_key="file"):
    path = root / item[filename_key]
    if item.get("sha256") and sha256(path) != item["sha256"]:
        raise ValueError(f"Archive hash mismatch: {path}")
    return path


def longest_true_streak(mask):
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def outcome_for_row(data, row, grid_count):
    """Translate raw stop codes without confusing rejected controls with exits."""
    reason = str(data["stop_reason"][row])
    if reason in ("reached_roa", "reached_c1"):
        # Schema 1 archives always stopped at c1; keep their histories readable.
        return "reached_roa"
    if reason == "horizon":
        # The maximum simulation length was exhausted. This does NOT prove instability.
        return "unconverged_in_domain"
    if reason == "nonfinite_state":
        return "numerical_error"
    if reason != "no_feasible_control":
        raise ValueError(f"Unknown stop reason: {reason}")
    attempt = int(data["attempt_lengths"][row]) - 1
    if attempt < 0:
        raise ValueError("Failed search lacks its diagnostic attempt.")
    # A nonfinite candidate might have been in-domain: do not claim all controls
    # leave the box when their successors/objectives could not be evaluated.
    numerical = (data["nonfinite_successor_count"][row, attempt]
                 + data["nonfinite_objective_count"][row, attempt])
    if numerical:
        return "numerical_error"
    if data["out_of_domain_count"][row, attempt] != grid_count:
        raise ValueError("Failed search is not explained by its rejection counts.")
    # No departing transition was taken. The terminal state is still in-domain.
    return "no_in_domain_successor"


def archive_roa_threshold(manifest, shared):
    """Resolve legacy c1 archives or validate the explicit schema-2 threshold."""
    if manifest["schema_version"] == 1:
        name, value = "c1", float(shared["c1"])
        saved_name, saved_value = "c1", float(manifest["c1"])
    else:
        name, value = str(shared["roa_threshold_name"]), float(shared["roa_threshold"])
        saved_name, saved_value = manifest["roa_threshold_name"], float(manifest["roa_threshold"])
    if (name not in ROA_NAMES or name != saved_name or value != saved_value
            or not np.isfinite(value) or value <= 0):
        raise ValueError("Archive ROA threshold is invalid or inconsistent with the manifest.")
    return name, value


def load_comparison(root, *, max_trajectories=60, seed=0, increase_tol=0.0):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in (1, 2) or manifest.get("status") != "complete":
        raise ValueError("Reporting requires a complete schema-version-1 or -2 comparison archive.")
    if not np.isfinite(increase_tol) or increase_tol < 0 or max_trajectories < 0:
        raise ValueError("Increase tolerance and plot limit must be nonnegative.")
    shared_path = checked_path(root, {"file": manifest["shared_data"], "sha256": manifest.get("shared_sha256")})
    with np.load(shared_path, allow_pickle=False) as stored:
        shared = {key: stored[key] for key in stored.files}
    name, value = archive_roa_threshold(manifest, shared)
    shared.update(roa_threshold_name=name, roa_threshold=value)
    ids = shared["point_ids"].astype(int).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate shared point IDs.")
    selected = ids if max_trajectories == 0 or len(ids) <= max_trajectories else sorted(
        np.random.default_rng(seed).choice(ids, max_trajectories, replace=False).tolist())
    selected_set = set(selected)
    metrics, histories = {}, {}
    for objective in manifest["objectives"]:
        if objective not in OBJECTIVE_NAMES or manifest["controllers"][objective]["status"] != "complete":
            raise ValueError(f"Invalid/incomplete objective: {objective}")
        metrics[objective], histories[objective] = {}, {}
        for shard in manifest["controllers"][objective]["shards"]:
            with np.load(checked_path(root, shard), allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            for row, pid in enumerate(data["point_ids"].astype(int)):
                if pid in metrics[objective]:
                    raise ValueError(f"Duplicate history for point {pid}")
                length = int(data["state_lengths"][row])
                transitions = int(data["transition_lengths"][row])
                if length != transitions + 1:
                    raise ValueError("State and transition lengths disagree.")
                outcome = outcome_for_row(data, row, len(shared["control_grid"]))
                # Use THIS objective's searched minimum minus its current value.
                # A failed search has no transition and must never become a yellow segment.
                delta = data["objective_delta"][row, :transitions]
                positive = np.isfinite(delta) & (delta > increase_tol)
                # Unknown candidate costs cannot establish that an increase was
                # unavoidable on the grid. Keep these cases out of yellow/counts.
                complete_search = ((data["nonfinite_successor_count"][row, :transitions] == 0)
                                   & (data["nonfinite_objective_count"][row, :transitions] == 0))
                increase = positive & complete_search
                metrics[objective][pid] = dict(
                    outcome=outcome, stop_time=float(data["stop_time"][row]),
                    already_inside=(outcome == "reached_roa" and int(data["stop_step"][row]) == 0),
                    increase_steps=int(increase.sum()), largest_increase=float(delta[increase].max()) if increase.any() else 0.0,
                    longest_increase_streak=longest_true_streak(increase),
                    nonfinite_increments=int((~np.isfinite(delta)).sum()),
                    unresolved_positive_steps=int((positive & ~complete_search).sum()),
                )
                if pid in selected_set:
                    histories[objective][pid] = dict(
                        states=data["states"][row, :length].copy(),
                        values=data[objective][row, :length].copy(),
                        controls=data["controls"][row, :transitions].copy(),
                        time=data["time"][:length].copy(),
                        control_time=data["attempt_time"][:transitions].copy(),
                        increase=increase.copy(), outcome=outcome,
                    )
        if set(metrics[objective]) != set(ids):
            raise ValueError("Controllers do not contain identical initial point IDs.")
    # Shared starts must have the same selected-ROA membership for every controller.
    first = metrics[manifest["objectives"][0]]
    for records in metrics.values():
        if any(records[pid]["already_inside"] != first[pid]["already_inside"] for pid in ids):
            raise ValueError("Initial selected-ROA membership differs between controllers.")
    return manifest, shared, metrics, histories, selected


def comparison_tables(shared, metrics):
    ids = shared["point_ids"].astype(int).tolist()
    inside_ids = {pid for pid, inside in zip(ids, shared["inside_mask"]) if inside}
    # "Inside/outside sublevel" always refers to the SINGLE saved sampling
    # classification, not to a different value function for each controller.
    groups = {"All sampled": ids, "Inside sublevel": [pid for pid in ids if pid in inside_ids],
              "Outside sublevel": [pid for pid in ids if pid not in inside_ids]}
    outcome_rows, pair_rows, increase_rows = [], [], []
    for group, group_ids in groups.items():
        for objective, records in metrics.items():
            # Denominator: every sampled point in this classification group that
            # STARTED OUTSIDE the selected ROA, including errors and failed searches.
            # Already-inside points are reported, but never inflate controller success.
            tested = [records[pid] for pid in group_ids if not records[pid]["already_inside"]]
            successful = [record for record in tested if record["outcome"] == "reached_roa"]
            counts = {outcome: sum(record["outcome"] == outcome for record in tested) for outcome in OUTCOMES}
            outcome_rows.append(dict(
                group=group, controller=objective, initially_inside_roa=len(group_ids) - len(tested),
                tested_outside_roa=len(tested), **counts,
                success_percent=100 * len(successful) / len(tested) if tested else None,
                # Median entry time is conditional on reaching the selected ROA.
                median_entry_time_s=float(np.median([r["stop_time"] for r in successful])) if successful else None,
            ))
            with_increases = [r for r in tested if r["increase_steps"]]
            # "Successful with increases" counts successful trajectories with
            # at least one highlighted step; "Any increases" counts trajectories
            # regardless of success. Neither field counts individual steps.
            # Nonfinite increments are counted separately, never as increases.
            increase_rows.append(dict(
                group=group, controller=objective, tested_outside_roa=len(tested),
                reached_roa=len(successful), successful_with_increases=sum(r["increase_steps"] > 0 for r in successful),
                trajectories_with_increases=len(with_increases),
                # These two extrema range over ALL tested trajectories, not only successes.
                largest_positive_increment=max((r["largest_increase"] for r in tested), default=None),
                longest_increase_streak_steps=max((r["longest_increase_streak"] for r in tested), default=None),
                nonfinite_objective_increments=sum(r["nonfinite_increments"] for r in tested),
            ))
        if "omega_orig" not in metrics:
            continue
        for alternative in (key for key in metrics if key != "omega_orig"):
            eligible = [pid for pid in group_ids if not metrics["omega_orig"][pid]["already_inside"]]
            both, omega_only, other_only, neither, differences = 0, 0, 0, 0, []
            for pid in eligible:
                omega, other = metrics["omega_orig"][pid], metrics[alternative][pid]
                a, b = omega["outcome"] == "reached_roa", other["outcome"] == "reached_roa"
                both += a and b
                omega_only += a and not b
                other_only += b and not a
                neither += not a and not b
                if a and b:
                    differences.append(omega["stop_time"] - other["stop_time"])
            # Paired denominator: identical IDs initially outside the selected ROA. "Neither"
            # includes horizon, missing contained successors, and numerical errors.
            # Negative time difference means omega reaches the ROA sooner, among BOTH-success pairs only.
            pair_rows.append(dict(group=group, alternative=alternative, paired_points=len(eligible), both_reach_roa=both,
                                  only_omega_reaches=omega_only, only_alternative_reaches=other_only, neither_reaches=neither,
                                  median_omega_minus_alternative_time_s=float(np.median(differences)) if differences else None))
    # Keep each CSV row self-describing when comparing runs with different ROAs.
    for rows in (outcome_rows, pair_rows, increase_rows):
        for row in rows:
            row.update(roa_threshold_name=str(shared["roa_threshold_name"]),
                       roa_threshold=float(shared["roa_threshold"]))
    return outcome_rows, pair_rows, increase_rows


OUTCOME_COLUMNS = [
    ("group", "Initial-point\ngroup"), ("controller", "Controller"),
    ("initially_inside_roa", "Already\nin {threshold}"), ("tested_outside_roa", "Tested\noutside {threshold}"),
    ("reached_roa", "Reached\n{threshold} ROA"), ("success_percent", "Success\n(%)"),
    ("median_entry_time_s", "Median entry\ntime (s)"),
    ("unconverged_in_domain", "Unconverged\nin-domain"),
    ("no_in_domain_successor", "No in-domain\nsuccessor"), ("numerical_error", "Numerical\nerror"),
]
PAIR_COLUMNS = [("group", "Initial-point\ngroup"), ("alternative", "Compared\nwith omega"), ("paired_points", "Paired\npoints"),
                ("both_reach_roa", "Both\nsucceed"), ("only_omega_reaches", "Only omega\nsucceeds"),
                ("only_alternative_reaches", "Only alternative\nsucceeds"), ("neither_reaches", "Neither\nsucceeds"),
                ("median_omega_minus_alternative_time_s", "Median time difference (s)\nomega minus alternative")]
INCREASE_COLUMNS = [("group", "Initial-point\ngroup"), ("controller", "Controller"), ("tested_outside_roa", "Tested\noutside {threshold}"),
                    ("reached_roa", "Reached\n{threshold} ROA"), ("successful_with_increases", "Successful with\nincreases"),
                    ("trajectories_with_increases", "Any\nincreases"), ("largest_positive_increment", "Largest positive\nincrement"),
                    ("longest_increase_streak_steps", "Longest increase\nstreak (steps)"),
                    ("nonfinite_objective_increments", "Nonfinite objective\nincrements")]


def display(value):
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def write_table_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key, _ in columns])
        writer.writeheader()
        writer.writerows(rows)


def render_tables(path, tables, classification, increase_tol, threshold_name, threshold_value):
    fig, axes = plt.subplots(3, 1, figsize=(20, 15), gridspec_kw={"height_ratios": [1.2, 1.0, 1.2]})
    titles = ("Controller outcomes", "Paired outcomes: omega_orig versus alternatives", "Transient objective increases")
    columns_list = (OUTCOME_COLUMNS, PAIR_COLUMNS, INCREASE_COLUMNS)
    for ax, title, rows, columns in zip(axes, titles, tables, columns_list):
        ax.axis("off")
        ax.set_title(title, fontsize=14, loc="left", pad=12)
        if not rows:
            ax.text(0, 0.5, "No applicable comparisons.", transform=ax.transAxes)
            continue
        cells = [[display(row[key]) for key, _ in columns] for row in rows]
        table = ax.table(cellText=cells, colLabels=[label.format(threshold=threshold_name) for _, label in columns], cellLoc="center", bbox=[0, 0.08, 1, 0.85])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#c6c6c6")
            if row == 0:
                cell.set_facecolor("#e8edf0")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("white" if row % 2 else "#f4f5f5")
    fig.suptitle(f"Grid-controller comparison | {classification} | stop: {threshold_name} = {threshold_value:.6g}", fontsize=16, y=0.99)
    fig.text(0.025, 0.035,
             f"All rates exclude points already in {threshold_name}; entry means first nu <= {threshold_name}, not simulated convergence to zero. "
             "Unconverged in-domain means the total step limit was reached.\n"
             "No in-domain successor: all grid successors were finite and outside the box; no departing step was taken. "
             "Numerical error: evaluation prevented a valid conclusion.\n"
             "Paired times use both-success points only; negative differences favor omega. Neither includes every non-success outcome. "
             "The inside/outside grouping is fixed for all controllers.\n"
             f"Increases use each controller's own feasible-grid minimum minus its current objective, strictly > {increase_tol:g}. "
             "Steps with nonfinite candidates are excluded. Tiny positives may reflect roundoff. Extrema include successes and non-successes.", fontsize=10)
    fig.subplots_adjust(left=0.025, right=0.99, top=0.95, bottom=0.12, hspace=0.22)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def load_original_for_contours(manifest, device):
    run_dir = Path(manifest["run_dir"])
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    filename = config["checkpoints"]["omega"]
    matches = [entry for entry in manifest["checkpoints"] if Path(entry["path"]).name == filename]
    if len(matches) != 1:
        raise ValueError("Cannot uniquely identify the original omega checkpoint in the archive.")
    entry = matches[0]
    path = Path(entry["path"])
    if sha256(path) != entry["sha256"]:
        raise ValueError("Original checkpoint changed since these trajectories were simulated.")
    return load_omeganet(path, n_dim=len(config["x_lb"]), device=device)


@torch.no_grad()
def contour_values(shared, model, device, resolution=151):
    domain = shared["domain"]
    if domain.shape != (2, 2):
        raise ValueError("True objective contours are currently supported only for 2D systems.")
    xs, ys = np.meshgrid(np.linspace(*domain[0], resolution), np.linspace(*domain[1], resolution))
    points = np.column_stack((xs.ravel(), ys.ravel())).astype(np.float32)
    values = []
    for start in range(0, len(points), 4096):
        x = torch.as_tensor(points[start:start + 4096], device=device)
        values.append(model.query(x).reshape(-1).cpu().numpy())
    p = shared["P"]
    return xs, ys, dict(omega_orig=np.concatenate(values).reshape(xs.shape),
                       norm2=(points * points).sum(-1).reshape(xs.shape),
                       nu=np.einsum("bi,ij,bj->b", points, p, points).reshape(xs.shape))


def increasing_segments(states, increase):
    """Only actual transitions can carry yellow; mask belongs to this objective."""
    if len(states) != len(increase) + 1:
        raise ValueError("Increase mask must align with state transitions.")
    return np.stack((states[:-1], states[1:]), axis=1)[increase]


def normalized_states(states, domain):
    """Display coordinates only: map each domain interval to [-1, 1]."""
    domain = np.asarray(domain)
    half_width = (domain[:, 1] - domain[:, 0]) / 2
    if not np.isfinite(domain).all() or np.any(half_width <= 0):
        raise ValueError("State-space normalization requires finite, positive domain widths.")
    return (np.asarray(states) - domain.mean(axis=1)) / half_width


def render_trajectories(path, manifest, shared, histories, selected, contours):
    objectives = list(histories)
    threshold_name = str(shared["roa_threshold_name"])
    fig, axes = plt.subplots(3, len(objectives), figsize=(5.3 * len(objectives), 12), squeeze=False)
    xs, ys, contour_data = contours
    # Transform contours, paths, markers and yellow segments together; objective
    # values and every simulation/report statistic retain physical coordinates.
    mesh = normalized_states(np.stack((xs, ys), axis=-1), shared["domain"])
    xs, ys = mesh[..., 0], mesh[..., 1]
    labels = manifest.get("state_labels", ["x1", "x2"])
    bounds = np.asarray(manifest.get("input_domain", []))
    for col, objective in enumerate(objectives):
        value_ax, state_ax, control_ax = axes[:, col]
        value_ax.set_title(f"Minimize {OBJECTIVE_MATH[objective]}", fontsize=13, pad=12)
        value_ax.set_ylabel(OBJECTIVE_MATH[objective])
        value_ax.set_xlabel("Time (s)")
        state_ax.set_facecolor("white")
        values = np.ma.masked_invalid(contour_data[objective])
        if values.count() and values.max() > values.min():
            levels = np.linspace(float(values.min()), float(values.max()), 7)[1:-1]
            contour = state_ax.contour(xs, ys, values, levels=levels, colors="#929292", linewidths=0.7)
            # Spread one label per level around the longest contour component;
            # steep/near-saturated omega contours otherwise stack labels together.
            center = np.zeros(2)
            positions = []
            for index, (level, components) in enumerate(zip(contour.levels, contour.allsegs)):
                components = [segment for segment in components if len(segment)]
                if not components:
                    continue
                segment = max(components, key=len)
                interior = ((segment > -0.88) & (segment < 0.88)).all(axis=1)
                segment = segment[interior]
                if not len(segment):
                    continue  # Do not place cut-off labels on short box-corner arcs.
                theta = np.arctan2(segment[:, 1] - center[1], segment[:, 0] - center[0])
                target = 2 * np.pi * index / len(levels) + 0.4
                position = segment[np.argmin(np.abs(np.angle(np.exp(1j * (theta - target)))))]
                positions.append(tuple(position))
            if positions:
                state_ax.clabel(contour, manual=positions, inline=True, fontsize=8, fmt="%.3g")
        nu = contour_data["nu"]
        if np.nanmin(nu) < float(shared["roa_threshold"]) < np.nanmax(nu):
            state_ax.contour(xs, ys, nu, levels=[float(shared["roa_threshold"])], colors="black", linestyles="--", linewidths=1.3)
        for pid in selected:
            trace = histories[objective][pid]
            color = SUCCESS if trace["outcome"] == "reached_roa" else RED
            states = normalized_states(trace["states"], shared["domain"])
            value_ax.plot(trace["time"], trace["values"], color=color, lw=1.0, alpha=0.75,
                          marker="." if len(states) == 1 else None)
            state_ax.plot(states[:, 0], states[:, 1], color=color, lw=1.1, alpha=0.8)
            state_ax.plot(states[0, 0], states[0, 1], marker=".", color=color, ms=4)
            state_ax.plot(states[-1, 0], states[-1, 1], marker="o" if color == SUCCESS else "x", color=color, ms=4, fillstyle="none")
            segments = increasing_segments(states, trace["increase"])
            if len(segments):
                state_ax.add_collection(LineCollection(segments, colors=YELLOW, linewidths=2.3, zorder=5))
            for j in range(trace["controls"].shape[1]):
                if len(trace["controls"]):
                    # The last chosen input acts until the terminal state time,
                    # but not beyond it. Extend the final held interval only.
                    control_ax.step(np.append(trace["control_time"], trace["time"][-1]),
                                    np.append(trace["controls"][:, j], trace["controls"][-1, j]),
                                    where="post", color=color, linestyle=("-", "--", ":", "-.")[j % 4], lw=1.0, alpha=0.7)
        if objective == "nu":
            value_ax.axhline(float(shared["roa_threshold"]), color="black", ls="--", lw=1)
            value_ax.text(0.98, float(shared["roa_threshold"]), f" {threshold_name}", transform=value_ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9)
        for lo, hi in bounds:
            control_ax.axhline(lo, color="#777777", lw=0.7, ls=":")
            control_ax.axhline(hi, color="#777777", lw=0.7, ls=":")
        state_ax.set(xlim=(-1, 1), ylim=(-1, 1),
                     xlabel=f"{labels[0]} (domain-normalized)", ylabel=f"{labels[1]} (domain-normalized)")
        state_ax.set_aspect("equal", adjustable="box")
        control_ax.set(xlabel="Time (s)", ylabel="Control input")
        for ax in (value_ax, control_ax):
            ax.grid(alpha=0.2)
            ax.set_xlim(0, max(float(manifest["horizon"]) * float(shared["dt"]), float(shared["dt"])))
        for ax in axes[:, col]:
            ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=SUCCESS, label=f"Reached {ROA_NAMES[threshold_name]} ({threshold_name})"),
               Line2D([], [], color=RED, label="Unconverged / numerical error"),
               Line2D([], [], color=YELLOW, lw=2.3, label="Feasible-grid minimum increases (own objective)"),
               Line2D([], [], color="black", ls="--", label=f"nu = {threshold_name}")]
    fig.suptitle(f"{manifest['system']}: matched initial conditions ({len(selected)} shown); {threshold_name} = {float(shared['roa_threshold']):.6g}", fontsize=16, y=0.985)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.951), ncol=2, frameon=False, fontsize=10)
    fig.text(0.04, 0.017, "State axes: (x - domain midpoint) / domain half-width. Contour labels retain physical objective values.\n"
             "Paths end at their actual stopping state. Yellow refers to the finite control grid, not every admissible input.", fontsize=9)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.07, top=0.86, hspace=0.38, wspace=0.30)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def generate_report(comparison_dir, *, output_dir=None, device="cuda", max_trajectories=60,
                    seed=0, increase_tol=0.0, contour_resolution=151):
    root = Path(comparison_dir).resolve()
    if contour_resolution < 3:
        raise ValueError("Contour resolution must be at least 3.")
    manifest, shared, metrics, histories, selected = load_comparison(
        root, max_trajectories=max_trajectories, seed=seed, increase_tol=increase_tol)
    output = Path(output_dir).resolve() if output_dir else root / ("report_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    tables = comparison_tables(shared, metrics)
    threshold_name, threshold_value = str(shared["roa_threshold_name"]), float(shared["roa_threshold"])
    for stem, rows, columns in zip(("outcomes", "paired_outcomes", "increases"), tables,
                                   (OUTCOME_COLUMNS, PAIR_COLUMNS, INCREASE_COLUMNS)):
        write_table_csv(output / f"{stem}.csv", rows,
                        [("roa_threshold_name", "ROA threshold"), ("roa_threshold", "ROA value")] + columns)
    classification = f"{manifest.get('classification_omega', 'saved omega')} < {display(manifest.get('classification_sublevel', 'saved threshold'))}"
    render_tables(output / "tables.png", tables, classification, increase_tol, threshold_name, threshold_value)
    notes = ["State-space coordinates are normalized separately to [-1, 1] using the saved domain midpoint and half-width. "
             "Contours and trajectories use the same transform; objectives, controls, saved states and statistics are unchanged."]
    unresolved = {key: sum(record["unresolved_positive_steps"] for record in records.values())
                  for key, records in metrics.items()}
    if any(unresolved.values()):
        notes.append(f"Positive increments with incomplete candidate evaluation were excluded from yellow/increase statistics: {unresolved}.")
    trajectory_file = None
    if shared["domain"].shape[0] == 2:
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no automatic CPU fallback.")
        model = load_original_for_contours(manifest, torch_device)
        contours = contour_values(shared, model, torch_device, contour_resolution)
        render_trajectories(output / "trajectories.png", manifest, shared, histories, selected, contours)
        trajectory_file = "trajectories.png"
    else:
        notes.append("Contour trajectory figures are restricted to 2D systems by design. No higher-dimensional projections are drawn. Tables include every point.")
    result = dict(comparison_manifest=str(root / "manifest.json"), report_dir=str(output), tables="tables.png",
                  roa_threshold_name=threshold_name, roa_threshold=threshold_value,
                  trajectories=trajectory_file, selected_point_ids=selected, selection_seed=seed,
                  increase_tolerance=increase_tol, contour_resolution=contour_resolution,
                  outcomes=tables[0], paired_outcomes=tables[1], increases=tables[2], notes=notes,
                  unresolved_positive_steps=unresolved,
                  reporting_source={"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))})
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    text = ["# Controller comparison", "", f"Classification: {classification}.",
            f"Stopping: first nu <= {threshold_name} = {threshold_value:.12g}, including step zero ({ROA_NAMES[threshold_name]}).",
            "Points initially inside this quadratic region are reported separately and excluded from success-rate denominators. Entry is not a simulation of convergence to zero or a new formal verification.",
            "All tables use every saved point; displayed trajectories use identical IDs across controllers.",
            "", "![Tables](tables.png)", ""]
    if trajectory_file:
        text.extend(["![Trajectories](trajectories.png)", ""])
    text.extend(["CSV tables: [outcomes](outcomes.csv), [paired outcomes](paired_outcomes.csv), [increases](increases.csv).", ""] + notes)
    (output / "README.md").write_text("\n".join(text), encoding="utf-8")
    print(f"Comparison report: {output / 'README.md'}")
    for row in tables[0]:
        if row["group"] == "All sampled":
            print(f"  {row['controller']}: {row['reached_roa']}/{row['tested_outside_roa']} reached {threshold_name}; "
                  f"{row['unconverged_in_domain']} unconverged in-domain; "
                  f"{row['no_in_domain_successor']} no in-domain successor; {row['numerical_error']} numerical error")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-trajectories", type=int, default=60, help="Same random IDs in every column; 0 plots all. Tables always use all points.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--increase-tol", type=float, default=0.0, help="Minimum positive searched-objective increment to highlight/count; not a proof tolerance.")
    parser.add_argument("--contour-resolution", type=int, default=151)
    args = parser.parse_args(argv)
    generate_report(args.comparison_dir, output_dir=args.output_dir, device=args.device,
                    max_trajectories=args.max_trajectories, seed=args.seed,
                    increase_tol=args.increase_tol, contour_resolution=args.contour_resolution)


if __name__ == "__main__":
    main()
