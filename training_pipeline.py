from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Windows/PyTorch can route small nn.Linear calls through cuBLASLt and fail
# with CUBLAS_STATUS_EXECUTION_FAILED after memory pressure or driver issues.
# This must be set before importing torch.
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")

import numpy as np
import torch

from models import OmegaNet, PiNet, hidden_sizes_from_state_dict, load_models, load_omeganet, load_pinet
from systems import DataDistributionCounts, SystemSpec, TrainingConfig, get_system, tensor_bounds


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(__file__).resolve().parent / "pipeline_runs"
ORIGINAL_OMEGA_EMBEDDING_MODE = "zonotope_generators"


@dataclass
class SublevelRecord:
    """One scalar nu-sublevel value and how it was obtained."""

    name: str
    value: float
    formula: str
    notes: Tuple[str, ...] = ()


@dataclass
class UnconstrainedLQRArtifacts:
    """First reviewed LQR substep: origin linearization plus DARE solution."""

    A: torch.Tensor
    B: torch.Tensor
    K: torch.Tensor
    P: torch.Tensor
    c_input: float
    c_input_by_dim: torch.Tensor
    sublevels: Tuple[SublevelRecord, ...]


@dataclass
class LQRBoxArtifacts:
    """Outputs of the initial local-controller calculation.

    This object is intentionally descriptive rather than final. We want the LQR
    setup to be reviewable before downstream training consumes it.
    """

    A: torch.Tensor
    B: torch.Tensor
    K: torch.Tensor
    P: torch.Tensor
    c_input: float
    c_input_by_dim: torch.Tensor
    c_box: float
    c_box_by_dim: torch.Tensor
    c_nonlinear_decrease: float
    c_nonlinear_decrease_grid: float
    c_nonlinear_decrease_analytic: float
    nonlinear_decrease_constants: Dict[str, object]
    c_analytic: float
    c_s: float
    c_sigmoid_center: float
    domain_box: Tuple[Tuple[float, float], ...]
    sublevels: Tuple[SublevelRecord, ...]
    notes: Tuple[str, ...] = ()


@dataclass
class OmegaLabelBatch:
    """Finite-horizon guidance labels for original omega.

    These labels are not the definition of omega. They are a smaller supervised
    anchor for the interval Zubov residual.
    """

    x: torch.Tensor
    z_x: torch.Tensor
    target_w: torch.Tensor


@dataclass
class OmegaResidualBatch:
    """Collocation data for the original interval Zubov equation."""

    x: torch.Tensor
    z_x: torch.Tensor
    z_next_interval: torch.Tensor
    xi: torch.Tensor
    successor_embedding_mode: str = ORIGINAL_OMEGA_EMBEDDING_MODE


@dataclass
class OmegaLossTerms:
    """Named pieces of the original omega objective."""

    data_mse: torch.Tensor
    interval_residual_mse: torch.Tensor
    boundary_condition_mse: torch.Tensor
    total: torch.Tensor


@dataclass
class ClosedLoopOmegaLossTerms:
    """Named pieces of the closed-loop omega objective."""

    zubov_mse: torch.Tensor
    decrease_penalty: torch.Tensor
    range_penalty: torch.Tensor
    boundary_condition_mse: torch.Tensor
    data_mse: torch.Tensor
    total: torch.Tensor


@dataclass
class PolicyLabelBatch:
    """Grid-search labels for the saturated NN controller branch."""

    x: torch.Tensor
    u_opt: torch.Tensor
    omega_opt_next: torch.Tensor
    control_weight: Optional[torch.Tensor] = None
    near_optimal_fraction: Optional[torch.Tensor] = None
    feasible_control_fraction: Optional[torch.Tensor] = None
    unmasked_fallback_mask: Optional[torch.Tensor] = None
    decisive_mask: Optional[torch.Tensor] = None
    neighbor_imputed_mask: Optional[torch.Tensor] = None
    midpoint_default_mask: Optional[torch.Tensor] = None


@dataclass
class HeatmapOptimalControlGrid:
    """Full optimal-control grid data used by value heatmap diagnostics."""

    grid_points: int
    axis_pair: Tuple[int, int]
    x0_grid: torch.Tensor
    x1_grid: torch.Tensor
    points: torch.Tensor
    labels: PolicyLabelBatch
    opt_next_points: torch.Tensor
    opt_next_in_domain: torch.Tensor
    omega_orig_values: torch.Tensor
    delta_orig_opt: torch.Tensor
    control_grid_points_per_dim: int
    control_grid_total_points: int


def normalize_plot_pairs(system: SystemSpec) -> Tuple[Tuple[int, int], ...]:
    """Validate and de-duplicate configured 2D diagnostic slice axes.

    `TrainingConfig.plot_pairs` uses zero-based state-axis indices. For each
    pair `(i, j)`, the heatmap varies state coordinates `i` and `j` over their
    configured bounds and fixes every other state coordinate at zero.
    """
    pairs = tuple(tuple(int(axis) for axis in pair) for pair in system.training.plot_pairs)
    if not pairs:
        raise ValueError(f"{system.name}: training.plot_pairs must contain at least one axis pair.")
    normalized: List[Tuple[int, int]] = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"{system.name}: each training.plot_pairs entry must contain exactly two axes.")
        i, j = pair
        if i == j:
            raise ValueError(f"{system.name}: plot pair {pair} repeats an axis.")
        if not (0 <= i < system.n_dim and 0 <= j < system.n_dim):
            raise ValueError(f"{system.name}: plot pair {pair} is outside n_dim={system.n_dim}.")
        if (i, j) not in normalized:
            normalized.append((i, j))
    return tuple(normalized)


def normalize_axis_pair(system: SystemSpec, axis_pair: Tuple[int, int]) -> Tuple[int, int]:
    """Validate one zero-based heatmap slice axis pair."""
    pair = tuple(int(axis) for axis in axis_pair)
    if len(pair) != 2:
        raise ValueError(f"{system.name}: axis_pair must contain exactly two axes.")
    i, j = pair
    if i == j:
        raise ValueError(f"{system.name}: axis_pair {pair} repeats an axis.")
    if not (0 <= i < system.n_dim and 0 <= j < system.n_dim):
        raise ValueError(f"{system.name}: axis_pair {pair} is outside n_dim={system.n_dim}.")
    return i, j


def heatmap_pair_suffix(axis_pair: Tuple[int, int]) -> str:
    """Filename suffix for a coordinate-slice heatmap pair."""
    return f"x{axis_pair[0]}_x{axis_pair[1]}"


@dataclass
class DomainBoundaryDiagnostic:
    """Finite-horizon boundary probe for deciding whether to expand a domain."""

    bounds: Tuple[Tuple[float, float], ...]
    horizon: int
    rollouts_per_state: int
    w_threshold: float
    n_points: int
    max_w: float
    min_w: float
    mean_w: float
    n_below_threshold: int
    should_expand: bool
    notes: Tuple[str, ...] = ()


@dataclass
class ReachabilityExpansionResult:
    """Domain-growth record from finite-horizon stepwise-min reachability."""

    initial_bounds: Tuple[Tuple[float, float], ...]
    final_bounds: Tuple[Tuple[float, float], ...]
    iterations: int
    converged: bool
    threshold: float
    history: Tuple[Dict[str, object], ...]
    notes: Tuple[str, ...] = ()


@dataclass
class SampledThresholdSearchResult:
    """Sampled c2/w1/w2 thresholds and their witness points."""

    n_samples: int
    domain: Tuple[Tuple[float, float], ...]
    eps: float
    scale: float
    c2_scale: float
    w1_scale: float
    w2_scale: float
    c1_raw: float
    c_s: float
    c2_cap: Optional[float]
    c2_raw: Optional[float]
    c2_verify: Optional[float]
    w1_raw: Optional[float]
    w1_verify: Optional[float]
    w2_raw: Optional[float]
    w2_verify: Optional[float]
    c2_witness: Optional[Dict[str, object]]
    w1_witness: Optional[Dict[str, object]]
    w2_witness: Optional[Dict[str, object]]
    w1_original_raw: Optional[float]
    w1_original_verify: Optional[float]
    w2_original_raw: Optional[float]
    w2_original_verify: Optional[float]
    w1_original_witness: Optional[Dict[str, object]]
    w2_original_witness: Optional[Dict[str, object]]
    shell_diagnostics: Tuple[Dict[str, object], ...]
    band_good_count: int
    band_bad_count: int
    band_bad_witness: Optional[Dict[str, object]]
    notes: Tuple[str, ...] = ()


@dataclass
class EarlyLqrC2SearchResult:
    """Early clamped-LQR c2 search used only to shape training samples."""

    n_samples: int
    domain: Tuple[Tuple[float, float], ...]
    eps: float
    c1_raw: float
    c2_raw: Optional[float]
    bad_count: int
    witness: Optional[Dict[str, object]]
    notes: Tuple[str, ...] = ()


@dataclass
class PreflightResult:
    ok: bool
    messages: Tuple[str, ...]


def _split_count(total: int, fractions: Tuple[float, ...]) -> Tuple[int, ...]:
    if any(frac < 0.0 for frac in fractions):
        raise ValueError(f"Sampling fractions must be nonnegative, got {fractions}.")
    frac_sum = sum(float(frac) for frac in fractions)
    if frac_sum <= 0.0:
        return tuple(0 for _ in fractions)
    raw = [float(total) * float(frac) / frac_sum for frac in fractions]
    counts = [int(math.floor(value)) for value in raw]
    remainder = int(total) - sum(counts)
    order = sorted(range(len(raw)), key=lambda idx: raw[idx] - counts[idx], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return tuple(counts)


def _stratified_quotas(total: int, sizes: Sequence[int]) -> Tuple[int, ...]:
    nonempty = [idx for idx, size in enumerate(sizes) if int(size) > 0]
    if total <= 0 or not nonempty:
        return tuple(0 for _ in sizes)
    if total < len(nonempty):
        order = sorted(nonempty, key=lambda idx: int(sizes[idx]), reverse=True)
        chosen = set(order[:total])
        return tuple(1 if idx in chosen else 0 for idx in range(len(sizes)))
    quotas = [0 for _ in sizes]
    for idx in nonempty:
        quotas[idx] = 1
    remaining = int(total) - len(nonempty)
    extras = _split_count(remaining, tuple(float(sizes[idx]) for idx in nonempty))
    for idx, extra in zip(nonempty, extras):
        quotas[idx] += int(extra)
    return tuple(quotas)


def source_groups_from_ids(source_ids: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    ids = source_ids.detach().cpu().to(dtype=torch.long)
    if ids.numel() == 0:
        return ()
    return tuple((ids == source).nonzero(as_tuple=False).squeeze(1) for source in sorted(ids.unique().tolist()))


class FixedStratifiedBatcher:
    """Cyclic, without-replacement minibatches from fixed source groups."""

    def __init__(self, groups: Sequence[torch.Tensor], batch_size: int):
        self.groups = [group.detach().cpu().to(dtype=torch.long) for group in groups if int(group.numel()) > 0]
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.quotas = _stratified_quotas(self.batch_size, [int(group.numel()) for group in self.groups])
        self.permutations = [torch.randperm(int(group.numel()), device=torch.device("cpu")) for group in self.groups]
        self.positions = [0 for _ in self.groups]

    @property
    def has_data(self) -> bool:
        return bool(self.groups)

    def batches_per_full_pass(self) -> int:
        if not self.groups:
            return 0
        required = []
        for group, quota in zip(self.groups, self.quotas):
            if quota <= 0:
                continue
            required.append(math.ceil(int(group.numel()) / int(quota)))
        return max(required) if required else 0

    def next_indices(self, *, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        parts = []
        for group_idx, quota in enumerate(self.quotas):
            if quota <= 0:
                continue
            group = self.groups[group_idx]
            remaining = int(quota)
            group_parts = []
            while remaining > 0:
                available = int(group.numel()) - int(self.positions[group_idx])
                if available <= 0:
                    self.permutations[group_idx] = torch.randperm(int(group.numel()), device=torch.device("cpu"))
                    self.positions[group_idx] = 0
                    available = int(group.numel())
                take = min(remaining, available)
                perm_slice = self.permutations[group_idx][self.positions[group_idx] : self.positions[group_idx] + take]
                group_parts.append(group[perm_slice])
                self.positions[group_idx] += take
                remaining -= take
            parts.append(torch.cat(group_parts, dim=0))
        if not parts:
            return torch.empty(0, dtype=torch.long, device=device)
        indices = torch.cat(parts, dim=0)
        indices = indices[torch.randperm(int(indices.numel()), device=torch.device("cpu"))]
        return indices.to(device=device)


def resolve_data_distribution_counts(system: SystemSpec) -> DataDistributionCounts:
    """Resolve the ratio-based data profile into explicit sampler counts."""
    profile = system.training.data_profile
    if profile.base_points_per_dim <= 0:
        raise ValueError("data_profile.base_points_per_dim must be positive.")
    if profile.base_exponent is not None and int(profile.base_exponent) < 0:
        raise ValueError("data_profile.base_exponent must be nonnegative.")
    if profile.base_reference_dim < 0:
        raise ValueError("data_profile.base_reference_dim must be nonnegative.")
    if profile.base_dimension_growth <= 0.0:
        raise ValueError("data_profile.base_dimension_growth must be positive.")
    if profile.global_scale < 0.0:
        raise ValueError("data_profile.global_scale must be nonnegative.")
    if profile.base_exponent is None:
        reference_dim = int(profile.base_reference_dim)
        extra_dim = max(0, int(system.n_dim) - reference_dim)
        base = int(
            round(
                float(profile.global_scale)
                * (int(profile.base_points_per_dim) ** reference_dim)
                * (float(profile.base_dimension_growth) ** extra_dim)
            )
        )
    else:
        base = int(
            round(
                float(profile.global_scale)
                * (int(profile.base_points_per_dim) ** int(profile.base_exponent))
            )
        )

    def total(ratio: float) -> int:
        if ratio < 0.0:
            raise ValueError(f"Sampling total ratios must be nonnegative, got {ratio}.")
        return int(round(base * float(ratio)))

    label_scale_base = float(profile.omega_label_dimension_scale_base)
    if label_scale_base <= 0.0:
        raise ValueError("data_profile.omega_label_dimension_scale_base must be positive.")
    omega_label_total = int(round(total(profile.omega_label_total_ratio) / (label_scale_base ** system.n_dim)))

    omega_label_ellipse, omega_label_uniform = _split_count(
        omega_label_total,
        (profile.omega_label_ellipse_fraction, profile.omega_label_uniform_fraction),
    )
    omega_collocation_ellipse, omega_collocation_uniform = _split_count(
        total(profile.omega_collocation_total_ratio),
        (profile.omega_collocation_ellipse_fraction, profile.omega_collocation_uniform_fraction),
    )
    policy_scale_base = float(profile.policy_dimension_scale_base)
    if policy_scale_base <= 0.0:
        raise ValueError("data_profile.policy_dimension_scale_base must be positive.")
    policy_total = int(round(total(profile.policy_total_ratio) / (policy_scale_base ** system.n_dim)))

    policy_ellipse, policy_uniform, policy_gd = _split_count(
        policy_total,
        (profile.policy_ellipse_fraction, profile.policy_uniform_fraction, profile.policy_gd_fraction),
    )
    cl_traj_scale_base = float(profile.closed_loop_trajectory_dimension_scale_base)
    if cl_traj_scale_base <= 0.0:
        raise ValueError("data_profile.closed_loop_trajectory_dimension_scale_base must be positive.")
    cl_trajectory_total = int(
        round(total(profile.closed_loop_trajectory_total_ratio) / (cl_traj_scale_base ** system.n_dim))
    )

    cl_traj_ellipse, cl_traj_uniform, cl_traj_gd = _split_count(
        cl_trajectory_total,
        (
            profile.closed_loop_trajectory_ellipse_fraction,
            profile.closed_loop_trajectory_uniform_fraction,
            profile.closed_loop_trajectory_gd_fraction,
        ),
    )
    cl_constraint_ellipse, cl_constraint_uniform, cl_constraint_gd = _split_count(
        total(profile.closed_loop_constraint_total_ratio),
        (
            profile.closed_loop_constraint_ellipse_fraction,
            profile.closed_loop_constraint_uniform_fraction,
            profile.closed_loop_constraint_gd_fraction,
        ),
    )
    return DataDistributionCounts(
        omega_trajectory_uniform_points=omega_label_uniform,
        omega_trajectory_ellipse_points=omega_label_ellipse,
        omega_collocation_uniform_points=omega_collocation_uniform,
        omega_collocation_ellipse_points=omega_collocation_ellipse,
        policy_ellipse_points=policy_ellipse,
        policy_uniform_points=policy_uniform,
        policy_gd_omega_sublevel_points=policy_gd,
        closed_loop_trajectory_ellipse_points=cl_traj_ellipse,
        closed_loop_trajectory_uniform_points=cl_traj_uniform,
        closed_loop_trajectory_gd_omega_sublevel_points=cl_traj_gd,
        closed_loop_constraint_ellipse_points=cl_constraint_ellipse,
        closed_loop_constraint_uniform_points=cl_constraint_uniform,
        closed_loop_constraint_gd_omega_sublevel_points=cl_constraint_gd,
    )


def pick_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best = 0
    best_free = -1
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free = free
                best = i
        except Exception:
            pass
    torch.cuda.set_device(best)
    return torch.device(f"cuda:{best}")


def check_machine_preflight(
    *,
    device: torch.device,
    min_free_ram_gb: float = 2.0,
    min_free_gpu_gb: float = 1.0,
) -> PreflightResult:
    """Best-effort resource sanity check before expensive stages."""
    messages = []
    ok = True

    try:
        import psutil  # type: ignore

        memory = psutil.virtual_memory()
        free_ram_gb = float(memory.available) / (1024**3)
        messages.append(f"available system RAM = {free_ram_gb:.2f} GB")
        if free_ram_gb < min_free_ram_gb:
            ok = False
            messages.append(
                f"available system RAM is below {min_free_ram_gb:.1f} GB; "
                "restart stale Python/Jupyter terminals or close unused apps before training."
            )

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
            shown = ", ".join(
                f"pid={pid} {name} age={age:.1f}h ram={rss:.2f}GB"
                for pid, name, age, rss in stale[:8]
            )
            messages.append(f"stale-looking Python/PowerShell processes: {shown}")
    except Exception as exc:
        messages.append(f"system RAM/process preflight skipped: {exc}")

    if device.type == "cuda":
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            free_gpu_gb = float(free_bytes) / (1024**3)
            total_gpu_gb = float(total_bytes) / (1024**3)
            messages.append(f"free GPU memory = {free_gpu_gb:.2f}/{total_gpu_gb:.2f} GB")
            if free_gpu_gb < min_free_gpu_gb:
                ok = False
                messages.append(
                    f"free GPU memory is below {min_free_gpu_gb:.1f} GB; "
                    "restart failed CUDA processes before training."
                )
        except Exception as exc:
            messages.append(f"GPU memory preflight skipped: {exc}")

    return PreflightResult(ok=ok, messages=tuple(messages))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def sample_uniform(system: SystemSpec, n: int, *, device: torch.device) -> torch.Tensor:
    """Uniform box sampler over the configured state bounds.

    This is only a default sampler. The training stages should accept alternate
    sampling strategies because we expect to experiment with domain-focused,
    boundary-focused, and low-discrepancy sampling later.
    """
    x = torch.empty(n, system.n_dim, device=device)
    for d, (lo, hi) in enumerate(zip(system.x_lb, system.x_hi)):
        x[:, d].uniform_(lo, hi)
    return x


def sample_uniform_bounds(
    bounds: Tuple[Tuple[float, float], ...],
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Uniform sampler over an explicit state hyperrectangle."""
    x = torch.empty(n, len(bounds), device=device, dtype=dtype)
    for d, (lo, hi) in enumerate(bounds):
        x[:, d].uniform_(float(lo), float(hi))
    return x


def sample_hyperrectangle_boundary(
    bounds: Tuple[Tuple[float, float], ...],
    points_per_face: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample points on every face of a hyperrectangle.

    Returns `2 * n_dim * points_per_face` points. This is a diagnostic sampler,
    not a proof-producing covering of the boundary.
    """
    if points_per_face <= 0:
        raise ValueError("points_per_face must be positive.")
    n_dim = len(bounds)
    lower = torch.tensor([lo for lo, _ in bounds], device=device, dtype=dtype)
    upper = torch.tensor([hi for _, hi in bounds], device=device, dtype=dtype)
    if torch.any(lower >= upper):
        raise ValueError("Each boundary lower bound must be below its upper bound.")

    faces = []
    for dim in range(n_dim):
        for side_value in (lower[dim], upper[dim]):
            x = lower + (upper - lower) * torch.rand(points_per_face, n_dim, device=device, dtype=dtype)
            x[:, dim] = side_value
            faces.append(x)
    return torch.cat(faces, dim=0)


def sample_hyperrectangle_boundary_total(
    bounds: Tuple[Tuple[float, float], ...],
    n_samples: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample a requested total number of points on random box faces."""
    if n_samples <= 0:
        return torch.empty(0, len(bounds), device=device, dtype=dtype)
    n_dim = len(bounds)
    lower = torch.tensor([lo for lo, _ in bounds], device=device, dtype=dtype)
    upper = torch.tensor([hi for _, hi in bounds], device=device, dtype=dtype)
    if torch.any(lower >= upper):
        raise ValueError("Each boundary lower bound must be below upper bound.")

    x = lower + (upper - lower) * torch.rand(n_samples, n_dim, device=device, dtype=dtype)
    face_dim = torch.randint(0, n_dim, (n_samples,), device=device)
    face_side = torch.randint(0, 2, (n_samples,), device=device, dtype=torch.bool)
    row = torch.arange(n_samples, device=device)
    x[row, face_dim] = torch.where(face_side, upper[face_dim], lower[face_dim])
    return x


def clamp_control(system: SystemSpec, u: torch.Tensor) -> torch.Tensor:
    lb = tensor_bounds(system.u_lb, device=u.device, dtype=u.dtype).view(1, -1)
    hi = tensor_bounds(system.u_hi, device=u.device, dtype=u.dtype).view(1, -1)
    return torch.clamp(u, lb, hi)


def clamp_state(system: SystemSpec, x: torch.Tensor) -> torch.Tensor:
    lb = tensor_bounds(system.x_lb, device=x.device, dtype=x.dtype).view(1, -1)
    hi = tensor_bounds(system.x_hi, device=x.device, dtype=x.dtype).view(1, -1)
    return torch.clamp(x, lb, hi)


def sample_uniform_controls(
    system: SystemSpec,
    shape_prefix: Tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Uniformly sample controls from the configured hyperinterval."""
    lower = tensor_bounds(system.u_lb, device=device, dtype=dtype).view(*((1,) * len(shape_prefix)), -1)
    upper = tensor_bounds(system.u_hi, device=device, dtype=dtype).view(*((1,) * len(shape_prefix)), -1)
    return lower + (upper - lower) * torch.rand(*shape_prefix, system.u_dim, device=device, dtype=dtype)


@torch.no_grad()
def original_omega_sublevel_mask(
    omega: OmegaNet,
    x: torch.Tensor,
    *,
    sublevel: float,
    batch_size: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Return the fixed-DOS mask for candidate controller-label states."""
    total = int(x.shape[0])
    if total == 0:
        return torch.empty(0, device=x.device, dtype=torch.bool), {
            "input_points": 0,
            "kept_points": 0,
            "removed_points": 0,
            "kept_fraction": None,
            "omega_range": None,
            "sublevel": float(sublevel),
        }
    if batch_size <= 0:
        raise ValueError("policy DOS filtration batch_size must be positive.")

    mask_parts = []
    kept_count = 0
    omega_min: Optional[float] = None
    omega_max: Optional[float] = None
    for start in range(0, total, int(batch_size)):
        xb = x[start : start + int(batch_size)]
        values = omega.query(xb).reshape(-1)
        batch_min = float(values.min().detach().cpu().item())
        batch_max = float(values.max().detach().cpu().item())
        omega_min = batch_min if omega_min is None else min(omega_min, batch_min)
        omega_max = batch_max if omega_max is None else max(omega_max, batch_max)
        mask = values <= float(sublevel)
        kept_count += int(mask.sum().detach().cpu().item())
        mask_parts.append(mask.detach())

    kept_mask = torch.cat(mask_parts, dim=0) if mask_parts else torch.empty(0, device=x.device, dtype=torch.bool)
    return kept_mask, {
        "input_points": total,
        "kept_points": kept_count,
        "removed_points": total - kept_count,
        "kept_fraction": float(kept_count / total),
        "omega_range": [float(omega_min), float(omega_max)],
        "sublevel": float(sublevel),
    }


@torch.no_grad()
def filter_states_by_original_omega_sublevel(
    omega: OmegaNet,
    x: torch.Tensor,
    *,
    sublevel: float,
    batch_size: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Keep controller-label states inside the proposed original-omega DOS."""
    kept_mask, stats = original_omega_sublevel_mask(
        omega,
        x,
        sublevel=sublevel,
        batch_size=batch_size,
    )
    return x[kept_mask].detach(), stats


def control_hyperrectangle_vertices(
    system: SystemSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return all vertices of the control hyperinterval.

    For `u_dim=m`, this returns `(2**m, m)`. This is appropriate for the
    current interval-reachable-set scaffold because we are assuming dynamics are
    affine in `u` over the constrained input hyperinterval.
    """
    lower = tensor_bounds(system.u_lb, device=device, dtype=dtype)
    upper = tensor_bounds(system.u_hi, device=device, dtype=dtype)
    vertices = []
    for mask in range(2 ** system.u_dim):
        coords = [upper[j] if (mask >> j) & 1 else lower[j] for j in range(system.u_dim)]
        vertices.append(torch.stack(coords))
    return torch.stack(vertices, dim=0)


def make_control_grid(
    system: SystemSpec,
    points_per_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a full tensor-product grid over the input hyperinterval.

    Returns shape `(points_per_dim ** u_dim, u_dim)`. This is deliberately
    exhaustive; callers should treat the size as a serious runtime/memory knob.
    """
    if points_per_dim <= 1:
        raise ValueError("points_per_dim must be at least 2 for a control grid.")
    axes = [
        torch.linspace(lo, hi, points_per_dim, device=device, dtype=dtype)
        for lo, hi in zip(system.u_lb, system.u_hi)
    ]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([axis.reshape(-1) for axis in mesh], dim=-1)


def reachable_interval_affine_control(
    system: SystemSpec,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-step reachable interval assuming dynamics are affine in control.

    We evaluate all vertices of the control hyperrectangle and take the
    coordinatewise min/max of the successor states. This gives the interval hull
    of the vertex successors. It relies on the explicit assumption that the
    dynamics are affine in `u`; do not use this for systems with nonlinear input
    dependence without replacing the reachable interval routine.
    """
    if x.shape[-1] != system.n_dim:
        raise ValueError("x last dimension must equal system.n_dim.")
    vertices = control_hyperrectangle_vertices(system, device=x.device, dtype=x.dtype)
    n_vertices = vertices.shape[0]
    x_expanded = x.unsqueeze(1).expand(-1, n_vertices, -1)
    u_expanded = vertices.unsqueeze(0).expand(x.shape[0], -1, -1)
    successors = system.dynamics(x_expanded, u_expanded)
    return successors.min(dim=1).values, successors.max(dim=1).values


def reachable_zonotope_affine_control(
    system: SystemSpec,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact one-step control zonotope for dynamics affine in control.

    With a fixed state x and box control set u = u_mid + diag(u_rad) xi,
    xi in [-1, 1]^m, affine-in-control dynamics produce:

        f(x, u) = center(x) + sum_j generator_j(x) xi_j.

    This representation preserves the fact that the same control can move
    several state coordinates together. The old interval hull drops that
    correlation after taking coordinatewise min/max.
    """
    if x.shape[-1] != system.n_dim:
        raise ValueError("x last dimension must equal system.n_dim.")
    u_lb = tensor_bounds(system.u_lb, device=x.device, dtype=x.dtype).view(1, system.u_dim)
    u_hi = tensor_bounds(system.u_hi, device=x.device, dtype=x.dtype).view(1, system.u_dim)
    u_mid = (u_lb + u_hi) / 2.0
    u_rad = (u_hi - u_lb) / 2.0
    center_u = u_mid.expand(x.shape[0], -1)
    center = system.dynamics(x, center_u)
    generators = []
    for control_dim in range(system.u_dim):
        delta = torch.zeros_like(center_u)
        delta[:, control_dim] = u_rad[:, control_dim]
        plus = system.dynamics(x, center_u + delta)
        minus = system.dynamics(x, center_u - delta)
        generators.append((plus - minus) / 2.0)
    if generators:
        generator_tensor = torch.stack(generators, dim=1)
    else:
        generator_tensor = torch.empty(x.shape[0], 0, system.n_dim, device=x.device, dtype=x.dtype)
    return center, generator_tensor


def original_omega_input_dim(system: SystemSpec) -> int:
    """Omega_orig uses `[center, vec(G)]`, with width `(m + 1) * n`."""
    return system.n_dim + system.n_dim * system.u_dim


def embed_singleton_zonotope(
    system: SystemSpec,
    x: torch.Tensor,
) -> torch.Tensor:
    """Embed a point state as a degenerate zonotope for omega_orig."""
    zeros = torch.zeros(*x.shape[:-1], system.n_dim * system.u_dim, device=x.device, dtype=x.dtype)
    return torch.cat([x, zeros], dim=-1)


def embed_pointwise_value_input(
    system: SystemSpec,
    x: torch.Tensor,
    *,
    input_dim: int,
) -> torch.Tensor:
    """Embed pointwise states for an OmegaNet that may use state or reachable-set inputs."""
    input_dim = int(input_dim)
    if input_dim == system.n_dim:
        return x
    expected = original_omega_input_dim(system)
    if input_dim != expected:
        raise ValueError(
            f"OmegaNet input_dim={input_dim} does not match omega_orig zonotope "
            f"input_dim={expected}."
        )
    return embed_singleton_zonotope(system, x)


def embed_zonotope_generators(center: torch.Tensor, generators: torch.Tensor) -> torch.Tensor:
    """Embed a zonotope as `[center, vec(G)]`.

    `generators` has shape `(..., u_dim, n_dim)`. The flattened order is
    generator-major: all state coordinates of control generator 0, then
    generator 1, and so on.
    """
    return torch.cat([center, generators.reshape(*center.shape[:-1], -1)], dim=-1)


def build_successor_zonotope_embedding(
    system: SystemSpec,
    x: torch.Tensor,
) -> torch.Tensor:
    """Build omega_orig successor input `[center(F({x})), vec(G_F)]`."""
    center, generators = reachable_zonotope_affine_control(system, x)
    return embed_zonotope_generators(center, generators)


@torch.no_grad()
def compute_optimal_next_omega_grid_labels(
    system: SystemSpec,
    omega: OmegaNet,
    x: torch.Tensor,
    control_grid: torch.Tensor,
    *,
    state_batch_size: int,
    control_chunk_size: int = 50_000,
    resolve_ambiguous_for_policy: bool = False,
    near_optimal_rel_tol: float = 1e-2,
    compute_control_confidence: bool = False,
    mask_out_of_domain_successors: bool = False,
    successor_domain: Optional[Tuple[Tuple[float, float], ...]] = None,
) -> PolicyLabelBatch:
    """Compute `argmin_u omega(f(x,u))` by exhaustive control-grid search.

    This is the label source for PiNet. For each state x_i, it finds:

        u_opt = argmin_{u in control_grid} omega.query(f(x_i, u))
        omega_opt_next = min_{u in control_grid} omega.query(f(x_i, u))

    The grid is exhaustive over the chosen discretization, not over the
    continuous input set. Increasing `points_per_dim` improves resolution but
    costs `points_per_dim ** u_dim`.
    """
    if x.shape[-1] != system.n_dim:
        raise ValueError("x last dimension must equal system.n_dim.")
    if control_grid.shape[-1] != system.u_dim:
        raise ValueError("control_grid last dimension must equal system.u_dim.")
    if state_batch_size <= 0:
        raise ValueError("state_batch_size must be positive.")
    if control_chunk_size <= 0:
        raise ValueError("control_chunk_size must be positive.")
    near_optimal_rel_tol = float(near_optimal_rel_tol)
    if near_optimal_rel_tol < 0.0:
        raise ValueError("near_optimal_rel_tol must be nonnegative.")
    if mask_out_of_domain_successors and successor_domain is None:
        successor_domain = tuple(zip(system.x_lb, system.x_hi))

    omega.eval()
    device = x.device
    dtype = x.dtype
    control_grid = control_grid.to(device=device, dtype=dtype)
    u_opt_chunks = []
    value_chunks = []
    control_weight_chunks = []
    near_fraction_chunks = []
    feasible_fraction_chunks = []
    fallback_mask_chunks = []
    midpoint_value_chunks = []
    midpoint = torch.tensor(
        [(lo + hi) / 2.0 for lo, hi in zip(system.u_lb, system.u_hi)],
        device=device,
        dtype=dtype,
    )
    u_lb = torch.tensor(system.u_lb, device=device, dtype=dtype)
    u_hi = torch.tensor(system.u_hi, device=device, dtype=dtype)
    need_landscape = bool(compute_control_confidence)

    for start in range(0, x.shape[0], state_batch_size):
        xb = x[start : start + state_batch_size]
        bsz = xb.shape[0]
        best_values = torch.full((bsz,), torch.inf, device=device, dtype=dtype)
        best_controls = torch.empty(bsz, system.u_dim, device=device, dtype=dtype)
        worst_values = torch.full((bsz,), -torch.inf, device=device, dtype=dtype)
        feasible_counts = torch.zeros((bsz,), device=device, dtype=dtype)
        fallback_best_values = torch.full((bsz,), torch.inf, device=device, dtype=dtype)
        fallback_best_controls = torch.empty(bsz, system.u_dim, device=device, dtype=dtype)
        stored_value_chunks = []
        stored_control_starts = []

        for u_start in range(0, control_grid.shape[0], control_chunk_size):
            ug = control_grid[u_start : u_start + control_chunk_size]
            k = ug.shape[0]
            x_expanded = xb.unsqueeze(1).expand(-1, k, -1).reshape(bsz * k, system.n_dim)
            u_expanded = ug.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * k, system.u_dim)
            next_x = system.dynamics(x_expanded, u_expanded)
            values = omega.query(next_x).reshape(bsz, k)
            raw_values = values
            if mask_out_of_domain_successors:
                assert successor_domain is not None
                feasible = points_in_bounds(next_x, successor_domain).reshape(bsz, k)
                feasible_counts = feasible_counts + feasible.to(dtype=dtype).sum(dim=1)
                values = torch.where(feasible, values, torch.full_like(values, torch.inf))
                raw_chunk_values, raw_chunk_indices = raw_values.min(dim=1)
                raw_improved = raw_chunk_values < fallback_best_values
                if raw_improved.any():
                    fallback_best_values[raw_improved] = raw_chunk_values[raw_improved]
                    fallback_best_controls[raw_improved] = ug[raw_chunk_indices[raw_improved]]
            chunk_values, chunk_indices = values.min(dim=1)
            improved = chunk_values < best_values
            if improved.any():
                best_values[improved] = chunk_values[improved]
                best_controls[improved] = ug[chunk_indices[improved]]
            if need_landscape:
                finite_values = torch.where(torch.isfinite(values), values, torch.full_like(values, -torch.inf))
                worst_values = torch.maximum(worst_values, finite_values.max(dim=1).values)
            if compute_control_confidence:
                stored_value_chunks.append(values.detach())
                stored_control_starts.append(int(u_start))

        fallback_mask = torch.zeros((bsz,), device=device, dtype=torch.bool)
        if mask_out_of_domain_successors:
            fallback_mask = ~torch.isfinite(best_values)
            if fallback_mask.any():
                best_values[fallback_mask] = fallback_best_values[fallback_mask]
                best_controls[fallback_mask] = fallback_best_controls[fallback_mask]
            feasible_fraction = feasible_counts / float(control_grid.shape[0])
            feasible_fraction_chunks.append(feasible_fraction.detach())
            fallback_mask_chunks.append(fallback_mask.detach())

        selected_controls = best_controls
        selected_values = best_values
        value_range = (worst_values - best_values).clamp_min(1e-12) if need_landscape else None

        if compute_control_confidence:
            assert value_range is not None
            near_limit = best_values + near_optimal_rel_tol * value_range
            near_count = torch.zeros((bsz,), device=device, dtype=dtype)
            for values, u_start in zip(stored_value_chunks, stored_control_starts):
                near_count = near_count + (values <= near_limit.unsqueeze(1)).to(dtype=dtype).sum(dim=1)
            near_fraction = near_count / float(control_grid.shape[0])
            control_weight = (1.0 - near_fraction).clamp(0.0, 1.0)
            near_fraction_chunks.append(near_fraction.detach())
            control_weight_chunks.append(control_weight.detach())

        u_opt_chunks.append(selected_controls.detach())
        value_chunks.append(selected_values.detach())
        if resolve_ambiguous_for_policy:
            midpoint_controls = midpoint.expand(bsz, -1)
            midpoint_next = system.dynamics(xb, midpoint_controls)
            midpoint_value_chunks.append(omega.query(midpoint_next).detach())

    u_opt = torch.cat(u_opt_chunks, dim=0)
    omega_opt_next = torch.cat(value_chunks, dim=0)
    control_weight = torch.cat(control_weight_chunks, dim=0) if control_weight_chunks else None
    near_optimal_fraction = torch.cat(near_fraction_chunks, dim=0) if near_fraction_chunks else None
    feasible_control_fraction = torch.cat(feasible_fraction_chunks, dim=0) if feasible_fraction_chunks else None
    unmasked_fallback_mask = torch.cat(fallback_mask_chunks, dim=0) if fallback_mask_chunks else None
    if not resolve_ambiguous_for_policy:
        return PolicyLabelBatch(
            x=x,
            u_opt=u_opt,
            omega_opt_next=omega_opt_next,
            control_weight=control_weight,
            near_optimal_fraction=near_optimal_fraction,
            feasible_control_fraction=feasible_control_fraction,
            unmasked_fallback_mask=unmasked_fallback_mask,
        )

    midpoint_values = torch.cat(midpoint_value_chunks, dim=0)
    # Omega is sigmoid-bounded. Smaller gains than this are not informative
    # enough to justify an arbitrary grid-corner label.
    decisive_mask = (midpoint_values - omega_opt_next) > 1e-4
    neighbor_imputed_mask = torch.zeros_like(decisive_mask)
    midpoint_default_mask = torch.zeros_like(decisive_mask)
    resolved_controls = u_opt.clone()

    ambiguous_mask = ~decisive_mask
    if ambiguous_mask.any():
        if decisive_mask.any():
            from scipy.spatial import cKDTree

            x_lb = torch.tensor(system.x_lb, device=device, dtype=dtype)
            x_hi = torch.tensor(system.x_hi, device=device, dtype=dtype)
            normalized_x = ((x - x_lb) / (x_hi - x_lb)).detach().cpu().numpy()
            decisive_indices = decisive_mask.nonzero(as_tuple=False).squeeze(1)
            ambiguous_indices = ambiguous_mask.nonzero(as_tuple=False).squeeze(1)
            tree = cKDTree(normalized_x[decisive_indices.detach().cpu().numpy()])
            _, nearest = tree.query(normalized_x[ambiguous_indices.detach().cpu().numpy()], k=1)
            nearest_indices = decisive_indices[
                torch.as_tensor(nearest, device=device, dtype=torch.long)
            ]
            resolved_controls[ambiguous_indices] = u_opt[nearest_indices]
            neighbor_imputed_mask[ambiguous_indices] = True
        else:
            resolved_controls[ambiguous_mask] = midpoint
            midpoint_default_mask[ambiguous_mask] = True

        resolved_next = system.dynamics(x, resolved_controls)
        omega_opt_next = omega.query(resolved_next).detach()

    return PolicyLabelBatch(
        x=x,
        u_opt=resolved_controls,
        omega_opt_next=omega_opt_next,
        control_weight=control_weight,
        near_optimal_fraction=near_optimal_fraction,
        feasible_control_fraction=feasible_control_fraction,
        unmasked_fallback_mask=unmasked_fallback_mask,
        decisive_mask=decisive_mask,
        neighbor_imputed_mask=neighbor_imputed_mask,
        midpoint_default_mask=midpoint_default_mask,
    )


def normalize_policy_label_selection_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    aliases = {
        "raw": "argmin",
        "grid_argmin": "argmin",
        "min": "argmin",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized != "argmin":
        raise ValueError(f"policy label_selection_mode must be 'argmin', got {mode!r}.")
    return normalized


def nu_from_p(x: torch.Tensor, p_matrix: torch.Tensor) -> torch.Tensor:
    """Quadratic function nu(x) = x^T P x.

    Zubov stages use alpha(x) = scale * dt * nu(x), via zubov_alpha_from_nu.
    """
    p_matrix = p_matrix.to(device=x.device, dtype=x.dtype)
    return (x @ p_matrix * x).sum(dim=-1)


def zubov_alpha_from_nu(nu: torch.Tensor, *, dt: float, scale: float = 1.0) -> torch.Tensor:
    """Zubov stage cost exponent alpha(x) = scale * dt * nu(x)."""
    return float(scale) * float(dt) * nu


def zubov_alpha_from_p(
    x: torch.Tensor,
    p_matrix: torch.Tensor,
    *,
    dt: float,
    scale: float = 1.0,
) -> torch.Tensor:
    return zubov_alpha_from_nu(nu_from_p(x, p_matrix), dt=dt, scale=scale)


def xi_stage_cost_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """xi(x) = 1 - exp(-alpha(x))."""
    return 1.0 - torch.exp(-alpha)


def xi_stage_cost(
    x: torch.Tensor,
    p_matrix: torch.Tensor,
    *,
    dt: float,
    scale: float = 1.0,
) -> torch.Tensor:
    alpha = zubov_alpha_from_p(x, p_matrix, dt=dt, scale=scale)
    return xi_stage_cost_from_alpha(alpha)


def embed_singleton_state(x: torch.Tensor) -> torch.Tensor:
    """Notebook singleton embedding: [x, 0]."""
    return torch.cat([x, torch.zeros_like(x)], dim=-1)


def embed_interval_bounds(lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    """Notebook interval embedding: [(lb + ub) / 2, ub - lb].

    Preserve this convention for original omega training. The second half is the
    full interval width, matching `2D Bench verifiable`, not half-width radius.
    """
    return torch.cat([(lb + ub) / 2.0, ub - lb], dim=-1)


def zubov_residual_from_values(
    w_x: torch.Tensor,
    w_next: torch.Tensor,
    xi: torch.Tensor,
) -> torch.Tensor:
    """Notebook residual: w(x) - w(next) - xi(x) * (1 - w(next))."""
    return w_x - w_next - xi * (1.0 - w_next)


def interval_zubov_residual(
    vnet,
    z_x: torch.Tensor,
    z_next: torch.Tensor,
    xi: torch.Tensor,
):
    """Original omega residual copied from notebook Block 5.

    `z_x` and `z_next` must both live in the selected OmegaNet embedding space.
    For the historical mode this is `[x, 0]` and `[(lb_F + ub_F)/2, ub_F-lb_F]`.
    For detailed zonotopes this is `[x, 0_{n*m}]` and `[center, vec(G)]`.
    """
    w_x = vnet(z_x)
    w_next = vnet(z_next)
    residual = zubov_residual_from_values(w_x, w_next, xi)
    return residual, w_x, w_next, xi


def closed_loop_zubov_residual(
    vnet,
    x: torch.Tensor,
    closed_loop_step,
    p_matrix: torch.Tensor,
    *,
    dt: float,
    scale: float = 1.0,
):
    """Closed-loop residual copied from notebook Block 8."""
    w_x = vnet.query(x)
    w_next = vnet.query(closed_loop_step(x))
    xi = xi_stage_cost(x, p_matrix, dt=dt, scale=scale).clamp(0.0, 1.0)
    residual = zubov_residual_from_values(w_x, w_next, xi)
    return residual, w_x, w_next, xi


def lqr_control(system: SystemSpec, x: torch.Tensor, k_matrix: torch.Tensor) -> torch.Tensor:
    """Saturated LQR control, u = clamp(-Kx).

    Shape convention:
      x: (..., n_dim)
      K: (u_dim, n_dim)
      output: (..., u_dim)
    """
    k_matrix = k_matrix.to(device=x.device, dtype=x.dtype)
    u = -(x @ k_matrix.T)
    return clamp_control(system, u)


@torch.no_grad()
def linearize_at_origin(
    system: SystemSpec,
    *,
    eps: float = 1e-6,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finite-difference linearization of x_next=f(x,u) at the origin.

    Returns discrete-time matrices A and B for the already-discretized dynamics:

        x_{k+1} ~= A x_k + B u_k

    This assumes only that `system.dynamics` accepts batched tensors whose final
    dimensions are `(n_dim,)` and `(u_dim,)`.
    """
    device = device or torch.device("cpu")
    dtype = torch.float64
    n_dim = system.n_dim
    u_dim = system.u_dim
    x0 = torch.zeros(1, n_dim, device=device, dtype=dtype)
    u0 = torch.zeros(1, u_dim, device=device, dtype=dtype)

    a_matrix = torch.empty(n_dim, n_dim, device=device, dtype=dtype)
    b_matrix = torch.empty(n_dim, u_dim, device=device, dtype=dtype)

    for j in range(n_dim):
        dx = torch.zeros_like(x0)
        dx[0, j] = eps
        f_plus = system.dynamics(x0 + dx, u0)
        f_minus = system.dynamics(x0 - dx, u0)
        a_matrix[:, j] = ((f_plus - f_minus) / (2.0 * eps)).reshape(-1)

    for j in range(u_dim):
        du = torch.zeros_like(u0)
        du[0, j] = eps
        f_plus = system.dynamics(x0, u0 + du)
        f_minus = system.dynamics(x0, u0 - du)
        b_matrix[:, j] = ((f_plus - f_minus) / (2.0 * eps)).reshape(-1)

    return a_matrix.to(dtype=torch.float32), b_matrix.to(dtype=torch.float32)


def solve_unconstrained_discrete_lqr(
    a_matrix: torch.Tensor,
    b_matrix: torch.Tensor,
    *,
    q_scale: float,
    r_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve the unconstrained discrete-time LQR problem using DARE.

    The DARE is solved for the origin-linearized dynamics. The returned feedback
    is `u = -Kx`, with `K` shaped `(u_dim, n_dim)`.
    """
    from scipy.linalg import solve_discrete_are

    a_np = a_matrix.detach().cpu().double().numpy()
    b_np = b_matrix.detach().cpu().double().numpy()
    n_dim = a_np.shape[0]
    u_dim = b_np.shape[1]
    q_np = float(q_scale) * np.eye(n_dim)
    r_np = float(r_scale) * np.eye(u_dim)

    p_np = solve_discrete_are(a_np, b_np, q_np, r_np)
    k_np = np.linalg.solve(r_np + b_np.T @ p_np @ b_np, b_np.T @ p_np @ a_np)
    return (
        torch.tensor(k_np, dtype=torch.float32),
        torch.tensor(p_np, dtype=torch.float32),
    )


def closed_loop_linear_matrix(
    a_matrix: torch.Tensor,
    b_matrix: torch.Tensor,
    k_matrix: torch.Tensor,
) -> torch.Tensor:
    """Closed-loop origin-linearized dynamics matrix for `u=-Kx`."""
    b_matrix = b_matrix.to(device=a_matrix.device, dtype=a_matrix.dtype)
    k_matrix = k_matrix.to(device=a_matrix.device, dtype=a_matrix.dtype)
    return a_matrix - b_matrix @ k_matrix


def closed_loop_decrease_matrix(
    a_closed_loop: torch.Tensor,
    p_matrix: torch.Tensor,
) -> torch.Tensor:
    """Return Q such that A_cl^T P A_cl - P = -Q."""
    p_matrix = p_matrix.to(device=a_closed_loop.device, dtype=a_closed_loop.dtype)
    return p_matrix - a_closed_loop.T @ p_matrix @ a_closed_loop


def lqr_input_constraint_sublevel(
    system: SystemSpec,
    k_matrix: torch.Tensor,
    p_matrix: torch.Tensor,
    *,
    zero_tol: float = 1e-12,
) -> Tuple[float, torch.Tensor]:
    """Largest nu-sublevel respecting hyperinterval input limits under LQR.

    For `nu(x)=x^T P x` and LQR row `k_i`, the maximum squared input component
    over `nu(x)<=c` is:

        max (k_i x)^2 = c * k_i P^{-1} k_i^T

    Therefore each input dimension gives:

        c_i = u_i,max^2 / (k_i P^{-1} k_i^T)

    where `u_i,max = min(-u_lb[i], u_hi[i])` for a zero-containing input
    interval. The returned scalar is `min_i c_i`.
    """
    k_matrix = torch.as_tensor(k_matrix, dtype=torch.float64)
    p_matrix = torch.as_tensor(p_matrix, dtype=torch.float64)
    p_inv = torch.linalg.inv(p_matrix)
    u_lb = torch.tensor(system.u_lb, dtype=torch.float64)
    u_hi = torch.tensor(system.u_hi, dtype=torch.float64)

    if torch.any(u_lb >= 0.0) or torch.any(u_hi <= 0.0):
        raise ValueError(
            "LQR input sublevel calculation assumes each input interval contains zero."
        )

    u_abs = torch.minimum(-u_lb, u_hi)
    c_by_dim = torch.empty(system.u_dim, dtype=torch.float64)
    for i in range(system.u_dim):
        k_i = k_matrix[i : i + 1, :]
        denom = (k_i @ p_inv @ k_i.T).squeeze()
        if denom <= zero_tol:
            c_by_dim[i] = torch.inf
        else:
            c_by_dim[i] = (u_abs[i] ** 2) / denom

    c_input = float(torch.min(c_by_dim).item())
    return c_input, c_by_dim.to(dtype=torch.float32)


def box_containment_sublevel(
    box_radii: torch.Tensor,
    p_matrix: torch.Tensor,
) -> Tuple[float, torch.Tensor]:
    """Largest `nu=x^T P x` sublevel guaranteed inside a state box.

    If `B=[-R_B, R_B]`, then the ellipsoid `{x: x^T P x <= c}` is inside the
    box when, for every coordinate i:

        c <= R_B,i^2 / (P^{-1})_ii

    This is the containment condition shown in the screenshot. It is separate
    from input constraints and separate from nonlinear-decrease constraints.
    """
    radii = torch.as_tensor(box_radii, dtype=torch.float64)
    p_inv = torch.linalg.inv(torch.as_tensor(p_matrix, dtype=torch.float64))
    diag = torch.diagonal(p_inv)
    if torch.any(radii <= 0.0):
        raise ValueError("Box radii must be positive.")
    if torch.any(diag <= 0.0):
        raise ValueError("P inverse diagonal must be positive for a positive definite P.")
    c_by_dim = radii.pow(2) / diag
    return float(torch.min(c_by_dim).item()), c_by_dim.to(dtype=torch.float32)


def domain_containment_sublevel(
    bounds: Sequence[Tuple[float, float]],
    p_matrix: torch.Tensor,
) -> Tuple[float, torch.Tensor]:
    """Sublevel sufficient for `{x: x^T P x <= c}` to stay in a domain box."""
    normalized = tuple((float(lo), float(hi)) for lo, hi in bounds)
    for lo, hi in normalized:
        if not (lo < 0.0 < hi):
            raise ValueError(
                "Domain containment requires each state interval to strictly contain the origin."
            )
    radii = torch.tensor(
        [min(abs(lo), abs(hi)) for lo, hi in normalized],
        dtype=torch.float32,
    )
    return box_containment_sublevel(radii, p_matrix)


def cap_lqr_artifacts_to_state_domain(
    artifacts: LQRBoxArtifacts,
    bounds: Sequence[Tuple[float, float]],
    *,
    c1_scale: float,
) -> None:
    """Finalize the analytic and NN-off levels for the active state domain."""
    domain_cap, domain_cap_by_dim = domain_containment_sublevel(bounds, artifacts.P)
    old_c_analytic = float(artifacts.c_analytic)
    old_c_s = float(artifacts.c_s)
    artifacts.c_box = float(domain_cap)
    artifacts.c_box_by_dim = domain_cap_by_dim
    artifacts.sublevels = artifacts.sublevels + (
        SublevelRecord(
            name="active_domain_box_containment",
            value=float(domain_cap),
            formula="min_i min(|x_lb_i|, |x_hi_i|)^2 / (P^{-1})_ii",
            notes=(
                "Ensures the nu sublevel is contained in the active state domain.",
                f"active_domain={tuple((float(lo), float(hi)) for lo, hi in bounds)}",
                f"c_analytic: {old_c_analytic} -> {artifacts.c_analytic}",
                f"C_S: {old_c_s} -> pending refresh",
            ),
        ),
    )
    refresh_lqr_switch_levels(artifacts, bounds, c1_scale=c1_scale)
    artifacts.notes = artifacts.notes + (
        "The analytic verification level is capped by active state-domain containment.",
        "C_S is min(active-domain containment, LQR input constraint).",
        "The sigmoid blend center is refreshed from the capped C_S after those caps.",
    )


def replace_sublevel_record(
    records: Sequence[SublevelRecord],
    name: str,
    replacement: SublevelRecord,
) -> Tuple[SublevelRecord, ...]:
    replaced = False
    updated = []
    for record in records:
        if record.name == name:
            updated.append(replacement)
            replaced = True
        else:
            updated.append(record)
    if not replaced:
        updated.append(replacement)
    return tuple(updated)


def refresh_lqr_switch_levels(
    artifacts: LQRBoxArtifacts,
    bounds: Sequence[Tuple[float, float]],
    *,
    c1_scale: float,
) -> None:
    """Refresh C_S, c_analytic, and the sampling cap for the active domain."""
    domain_cap, domain_cap_by_dim = domain_containment_sublevel(bounds, artifacts.P)
    local_box_cap, _ = domain_containment_sublevel(artifacts.domain_box, artifacts.P)
    c_s = min(
        float(artifacts.c_input),
        float(domain_cap),
    )
    c_analytic = min(
        float(artifacts.c_input),
        float(local_box_cap),
        float(domain_cap),
        float(artifacts.c_nonlinear_decrease_analytic),
    )
    if not (0.0 <= float(c1_scale) <= 1.5):
        raise ValueError(f"c1_scale must lie in [0, 1.5], got {c1_scale}.")
    if not c_s > 0.0:
        raise ValueError(
            "The sampling cap construction requires a positive capped level C_S: "
            f"C_S={c_s}."
        )
    if c_analytic > c_s:
        raise ValueError(
            "Cannot place the sample cap between c1 and C_S: "
            f"c1={c_analytic}, C_S={c_s}."
        )
    c_sigmoid_center = float(c_analytic) + float(c1_scale) * (
        float(c_s) - float(c_analytic)
    )
    artifacts.c_box = float(domain_cap)
    artifacts.c_box_by_dim = domain_cap_by_dim
    artifacts.c_s = float(c_s)
    artifacts.c_analytic = float(c_analytic)
    artifacts.c_sigmoid_center = float(c_sigmoid_center)
    artifacts.sublevels = replace_sublevel_record(
        artifacts.sublevels,
        "controller_lqr_off_level_C_S",
        SublevelRecord(
            name="controller_lqr_off_level_C_S",
            value=float(c_s),
            formula="min(c_input, active_domain_box_containment)",
            notes=(
                "This capped level is the outer reference for band sampling.",
                "This level is capped by active-domain containment, not the small Hessian box.",
                "No pre-training nonlinear LQR grid search is used for C_S.",
            ),
        ),
    )
    artifacts.sublevels = replace_sublevel_record(
        artifacts.sublevels,
        "analytic_verification_c1",
        SublevelRecord(
            name="analytic_verification_c1",
            value=float(c_analytic),
            formula="min(c_input, active_domain_box_containment, c_nonlinear_decrease_analytic)",
            notes=(
                "This is the lower eta-analytic level used by verification.",
                "The Hessian-bound nonlinear term still comes from the small local box.",
                "The small local-box containment level still caps this analytic floor.",
                "The nonlinear decrease grid is not used to cap C_S.",
            ),
        ),
    )
    artifacts.sublevels = replace_sublevel_record(
        artifacts.sublevels,
        "controller_sample_cap",
        SublevelRecord(
            name="controller_sample_cap",
            value=float(c_sigmoid_center),
            formula="c1 + c1_scale * (C_S - c1)",
            notes=(
                "The c1-to-C_S band sampler uses this as its scaled sample cap.",
                "Values above 1.0 place the sample cap slightly beyond capped C_S.",
                f"c1_scale={float(c1_scale)}",
                f"c1={c_analytic}",
                f"C_S={c_s}",
            ),
        ),
    )


def refresh_nonlinear_lqr_decrease_after_expansion(
    artifacts: LQRBoxArtifacts,
    system: SystemSpec,
    *,
    device: Optional[torch.device],
) -> None:
    """Run the true-dynamics LQR decrease grid over the active expanded domain."""
    local_box_cap, _ = domain_containment_sublevel(artifacts.domain_box, artifacts.P)
    certified_floor = min(
        float(artifacts.c_input),
        float(local_box_cap),
        float(artifacts.c_nonlinear_decrease_analytic),
    )
    bounds = tuple((float(lo), float(hi)) for lo, hi in zip(system.x_lb, system.x_hi))
    threshold_points_per_dim, _ = threshold_grid_shape(bounds, system.training.threshold_search_samples)
    effective_points_per_dim = max(
        int(system.training.lqr_decrease_grid_points_per_dim),
        int(threshold_points_per_dim),
    )
    c_nonlinear_grid, nonlinear_grid_constants = nonlinear_decrease_sublevel_from_grid_search(
        system,
        artifacts.K,
        artifacts.P,
        bounds=bounds,
        points_per_dim=effective_points_per_dim,
        batch_size=system.training.lqr_decrease_grid_batch_size,
        epsilon=system.local_decrease.epsilon,
        certified_floor=certified_floor,
        device=device,
    )
    constants = dict(artifacts.nonlinear_decrease_constants)
    constants["grid_search"] = nonlinear_grid_constants
    constants["grid_search_refreshed_after_expansion"] = True
    artifacts.c_nonlinear_decrease = float(c_nonlinear_grid)
    artifacts.c_nonlinear_decrease_grid = float(c_nonlinear_grid)
    artifacts.nonlinear_decrease_constants = constants
    artifacts.sublevels = replace_sublevel_record(
        artifacts.sublevels,
        "nonlinear_lqr_decrease_grid",
        SublevelRecord(
            name="nonlinear_lqr_decrease_grid",
            value=float(c_nonlinear_grid),
            formula="min grid nu where nu(f(x,-Kx))-nu(x) >= 0 on active domain",
            notes=(
                "Uses true nonlinear dynamics and the linear controller u=-Kx.",
                "This grid search is run after reachability expansion.",
                f"active_domain={bounds}",
                f"points_per_dim={system.training.lqr_decrease_grid_points_per_dim}",
                f"effective_points_per_dim={effective_points_per_dim}",
                f"threshold_search_samples={system.training.threshold_search_samples}",
                f"epsilon={system.local_decrease.epsilon}",
                f"certified_floor={certified_floor}",
            ),
        ),
    )


def nonlinear_decrease_sublevel_from_remainder_bound(
    a_closed_loop: torch.Tensor,
    p_matrix: torch.Tensor,
    q_matrix: torch.Tensor,
    eta_b: torch.Tensor,
    *,
    epsilon: float,
    zero_tol: float = 1e-12,
) -> Tuple[float, Dict[str, float]]:
    """Serry-Liu-style nonlinear-decrease sublevel from a remainder bound.

    Setup from the screenshot:

        f(x) = A x + h(x)
        nu(x) = x^T P x
        A^T P A - P = -Q
        ||h(x)|| <= (||x||^2 / 2) * eta_B, for x in the chosen box B

    With d = lambda_min(Q) - epsilon, require:

        alpha * c + beta * sqrt(c) <= d

    where:

        alpha = ||P|| * ||eta_B||^2 / (4 * lambda_min(P))
        beta  = ||P^{1/2} eta_B|| * ||P^{1/2} A P^{-1/2}||

    Then the sufficient sublevel is:

        c <= ((-beta + sqrt(beta^2 + 4 alpha d)) / (2 alpha))^2

    If alpha is numerically zero, this reduces to `(d / beta)^2`; if both alpha
    and beta are zero, the nonlinear remainder bound contributes no restriction.

    This helper does not compute eta_B. That bound must be supplied by a
    reviewed system-specific Hessian/interval-arithmetic calculation.
    Never treat a toy eta_B used in a smoke test as a certificate.
    """
    a = torch.as_tensor(a_closed_loop, dtype=torch.float64).detach().cpu()
    p = torch.as_tensor(p_matrix, dtype=torch.float64).detach().cpu()
    q = torch.as_tensor(q_matrix, dtype=torch.float64).detach().cpu()
    eta = torch.as_tensor(eta_b, dtype=torch.float64).detach().cpu().reshape(-1)

    lambda_min_p = float(torch.linalg.eigvalsh(p).min().item())
    lambda_min_q = float(torch.linalg.eigvalsh(q).min().item())
    d_value = lambda_min_q - float(epsilon)
    if lambda_min_p <= zero_tol:
        raise ValueError("P must be positive definite.")
    if d_value <= 0.0:
        raise ValueError("epsilon must be smaller than lambda_min(Q).")

    p_norm = float(torch.linalg.matrix_norm(p, ord=2).item())
    eta_norm = float(torch.linalg.vector_norm(eta, ord=2).item())
    p_sqrt = _symmetric_matrix_power(p, 0.5)
    p_inv_sqrt = _symmetric_matrix_power(p, -0.5)

    alpha_value = p_norm * eta_norm**2 / (4.0 * lambda_min_p)
    beta_left = torch.linalg.vector_norm(p_sqrt @ eta, ord=2)
    beta_right = torch.linalg.matrix_norm(p_sqrt @ a @ p_inv_sqrt, ord=2)
    beta_value = float((beta_left * beta_right).item())

    if alpha_value <= zero_tol:
        if beta_value <= zero_tol:
            c_value = float("inf")
        else:
            c_value = (d_value / beta_value) ** 2
    else:
        numerator = -beta_value + float(np.sqrt(beta_value**2 + 4.0 * alpha_value * d_value))
        c_value = (numerator / (2.0 * alpha_value)) ** 2

    constants = {
        "alpha": alpha_value,
        "beta": beta_value,
        "d": d_value,
        "lambda_min_p": lambda_min_p,
        "lambda_min_q": lambda_min_q,
        "p_norm": p_norm,
        "eta_norm": eta_norm,
    }
    return c_value, constants


@torch.no_grad()
def nonlinear_decrease_sublevel_from_grid_search(
    system: SystemSpec,
    k_matrix: torch.Tensor,
    p_matrix: torch.Tensor,
    *,
    box_radii: Optional[torch.Tensor] = None,
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    points_per_dim: int,
    batch_size: int,
    epsilon: float,
    certified_floor: float,
    device: Optional[torch.device] = None,
) -> Tuple[float, Dict[str, float]]:
    """Grid-searched true-dynamics LQR decrease level inside supplied bounds.

    This is not a proof. It searches the rectangular grid in B for the smallest
    sampled value of `nu=x^T P x` where the true nonlinear closed-loop map
    `f(x,-Kx)` fails the strict decrease test above the analytic certificate:

        nu(f(x,-Kx)) - nu(x) >= 0.

    The floor is the Hessian-bound analytic switching level. Below that level
    we already have the conservative certificate and do not need the sampled
    search to rediscover the origin equality.

    The returned value is intended for the practical switching threshold only;
    the analytic Hessian-based value remains recorded separately for later
    verification.
    """
    if points_per_dim < 2:
        raise ValueError("lqr_decrease_grid_points_per_dim must be at least 2.")
    if batch_size <= 0:
        raise ValueError("lqr_decrease_grid_batch_size must be positive.")

    device = device or torch.device("cpu")
    dtype = torch.float32
    k = k_matrix.to(device=device, dtype=dtype)
    p = p_matrix.to(device=device, dtype=dtype)
    if bounds is None:
        if box_radii is None:
            raise ValueError("Provide either box_radii or bounds for nonlinear decrease grid search.")
        radii = torch.as_tensor(box_radii, device=device, dtype=dtype)
        axes = [
            torch.linspace(
                -float(radius),
                float(radius),
                int(points_per_dim),
                device=device,
                dtype=dtype,
            )
            for radius in radii
        ]
        search_bounds = tuple((-float(radius), float(radius)) for radius in radii.detach().cpu())
    else:
        search_bounds = tuple((float(lo), float(hi)) for lo, hi in bounds)
        if len(search_bounds) != system.n_dim:
            raise ValueError("Nonlinear decrease grid bounds must match system.n_dim.")
        axes = [
            torch.linspace(float(lo), float(hi), int(points_per_dim), device=device, dtype=dtype)
            for lo, hi in search_bounds
        ]
    total_points = int(points_per_dim) ** int(system.n_dim)

    best_level = float("inf")
    best_delta = float("nan")
    bad_count = 0
    checked_count = 0

    index_batch = torch.empty(batch_size, system.n_dim, device=device, dtype=torch.long)
    point_batch = torch.empty(batch_size, system.n_dim, device=device, dtype=dtype)
    strides = [int(points_per_dim) ** dim for dim in range(system.n_dim)]

    for start in range(0, total_points, batch_size):
        count = min(batch_size, total_points - start)
        flat = torch.arange(start, start + count, device=device, dtype=torch.long)
        for dim, stride in enumerate(strides):
            index_batch[:count, dim] = (flat // stride) % int(points_per_dim)
            point_batch[:count, dim] = axes[dim][index_batch[:count, dim]]

        x = point_batch[:count]
        u = -(x @ k.T)
        x_next = system.dynamics(x, u)
        nu = nu_from_p(x, p)
        nu_next = nu_from_p(x_next, p)
        delta = nu_next - nu
        bad = (nu > float(certified_floor)) & (delta >= 0.0)
        checked_count += count
        bad_count += int(bad.sum().item())
        if bad.any():
            candidate_values = nu.masked_fill(~bad, float("inf"))
            idx = int(torch.argmin(candidate_values).item())
            candidate_level = float(nu[idx].item())
            if candidate_level < best_level:
                best_level = candidate_level
                best_delta = float(delta[idx].item())

    constants = {
        "points_per_dim": float(points_per_dim),
        "total_grid_points": float(total_points),
        "checked_points": float(checked_count),
        "bad_points": float(bad_count),
        "epsilon_recorded_not_used_as_margin": float(epsilon),
        "certified_floor": float(certified_floor),
        "failure_condition": "nu > certified_floor and nu(f(x,-Kx))-nu(x) >= 0",
        "witness_delta_nu": best_delta,
        "bounds": tuple((float(lo), float(hi)) for lo, hi in search_bounds),
    }
    return best_level, constants


def _symmetric_matrix_power(matrix: torch.Tensor, exponent: float) -> torch.Tensor:
    """Symmetric positive-definite matrix power via eigen-decomposition."""
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    if torch.any(eigvals <= 0.0):
        raise ValueError("Matrix must be symmetric positive definite.")
    return eigvecs @ torch.diag(eigvals.pow(exponent)) @ eigvecs.T


def sample_near_nu_boundary(
    p_matrix: torch.Tensor,
    level: float,
    n_samples: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    radial_noise_std: float = 0.0,
) -> torch.Tensor:
    """Sample near the ellipsoid boundary `{x: x^T P x = level}`.

    The sampler draws random Euclidean directions `z/||z||`, maps them through
    `P^{-1/2}`, and scales by `sqrt(level)`. Optional radial multiplicative
    noise expands outward from the boundary so samples are never generated
    inside the requested ellipsoid.
    """
    if level <= 0.0:
        raise ValueError("Boundary level must be positive.")
    if n_samples <= 0:
        return torch.empty(0, p_matrix.shape[0], device=device, dtype=dtype)

    p_cpu = torch.as_tensor(p_matrix, dtype=torch.float64).detach().cpu()
    p_inv_sqrt_cpu = _symmetric_matrix_power(p_cpu, -0.5)
    p_inv_sqrt = p_inv_sqrt_cpu.to(device=device, dtype=dtype)
    n_dim = p_inv_sqrt.shape[0]
    directions = torch.randn(n_samples, n_dim, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    radii = torch.full((n_samples, 1), float(level) ** 0.5, device=device, dtype=dtype)
    if radial_noise_std > 0.0:
        multipliers = 1.0 + float(radial_noise_std) * torch.randn(
            n_samples,
            1,
            device=device,
            dtype=dtype,
        ).abs()
        radii = radii * multipliers
    return directions @ p_inv_sqrt.T * radii


def sample_nu_band(
    p_matrix: torch.Tensor,
    c1: float,
    c2: float,
    n_samples: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    radial_noise_std: float = 0.0,
    c_scale: float = 1.0,
    batch_size: int = 100_000,
) -> torch.Tensor:
    """Sample directions on P-ellipsoids with levels from c1 to beyond c2.

    Levels are drawn uniformly in the scaled band:

        rho = U(0, c_scale)
        level = c1 + rho * (c2 - c1)

    so c1 remains the inner anchor and c_scale controls how far past the early
    LQR c2 search the band may extend.
    """
    if c1 <= 0.0:
        raise ValueError("Band lower level c1 must be positive.")
    if c2 <= c1:
        raise ValueError(f"Band upper level c2 must exceed c1, got c1={c1}, c2={c2}.")
    if c_scale <= 0.0:
        raise ValueError("Band c_scale must be positive.")
    if batch_size <= 0:
        raise ValueError("Band sampler batch_size must be positive.")
    if n_samples <= 0:
        return torch.empty(0, p_matrix.shape[0], device=device, dtype=dtype)

    p_cpu = torch.as_tensor(p_matrix, dtype=torch.float64).detach().cpu()
    p_inv_sqrt_cpu = _symmetric_matrix_power(p_cpu, -0.5)
    p_inv_sqrt = p_inv_sqrt_cpu.to(device=device, dtype=dtype)
    n_dim = p_inv_sqrt.shape[0]
    del radial_noise_std
    max_elements = 100_000_000
    requested_elements = int(n_samples) * int(n_dim)
    if requested_elements > max_elements:
        raise ValueError(
            "Requested LQR band sample is too large to materialize safely: "
            f"n_samples={n_samples}, n_dim={n_dim}, elements={requested_elements}. "
            "Reduce the data profile total for this dataset before sampling."
        )

    chunks = []
    for start in range(0, int(n_samples), int(batch_size)):
        count = min(int(batch_size), int(n_samples) - start)
        directions = torch.randn(count, n_dim, device=device, dtype=dtype)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        rho = torch.rand(count, 1, device=device, dtype=dtype) * float(c_scale)
        levels = float(c1) + rho * (float(c2) - float(c1))
        radii = levels.clamp(min=0.0).sqrt()
        chunks.append(directions @ p_inv_sqrt.T * radii)
    return torch.cat(chunks, dim=0)


def compute_unconstrained_lqr_artifacts(
    system: SystemSpec,
    *,
    device: Optional[torch.device] = None,
    linearization_eps: float = 1e-6,
) -> UnconstrainedLQRArtifacts:
    """Compute origin-linearized unconstrained LQR and input-limit sublevel."""
    a_matrix, b_matrix = linearize_at_origin(system, eps=linearization_eps, device=device)
    k_matrix, p_matrix = solve_unconstrained_discrete_lqr(
        a_matrix,
        b_matrix,
        q_scale=system.training.q_scale,
        r_scale=system.training.r_scale,
    )
    c_input, c_input_by_dim = lqr_input_constraint_sublevel(system, k_matrix, p_matrix)
    sublevels = (
        SublevelRecord(
            name="lqr_input_constraint",
            value=c_input,
            formula="min_i u_i,max^2 / (k_i P^{-1} k_i^T)",
            notes=(
                "Depends on q_scale/r_scale through DARE P and K.",
                "Assumes input constraints form a zero-containing hyperinterval.",
            ),
        ),
    )
    return UnconstrainedLQRArtifacts(
        A=a_matrix,
        B=b_matrix,
        K=k_matrix,
        P=p_matrix,
        c_input=c_input,
        c_input_by_dim=c_input_by_dim,
        sublevels=sublevels,
    )


def compute_lqr_box_artifacts(
    system: SystemSpec,
    *,
    device: Optional[torch.device] = None,
    linearization_eps: float = 1e-6,
) -> LQRBoxArtifacts:
    """Compute the reviewed analytic and outer LQR sublevels.

    The outer level is finalized after reachability expansion:

        C_S = min(c_input, active_domain_box_containment)

    It also records the tighter diagnostic/certified display value:

        c_analytic = min(c_input, local_box_containment,
                         c_nonlinear_decrease_analytic)

    Each term is recorded separately:
      - `c_input`: LQR inputs stay inside the constrained input hyperinterval.
      - `c_box`: initially records containment in the supplied rectangle B;
        the pipeline later replaces it with active-domain containment and caps
        the analytic verification level against that active domain.
      - `c_nonlinear_decrease_grid`: retained as an optional diagnostic only.
      - `c_nonlinear_decrease_analytic`: Hessian-bound sufficient level kept
        for later verification and the inner NN-off construction.

    If `system.local_decrease` is missing or invalid, system validation should
    fail before this point. Do not train or export a PiNet switch threshold from
    only the unconstrained LQR/input calculation.
    """
    system.validate()
    lqr = compute_unconstrained_lqr_artifacts(
        system,
        device=device,
        linearization_eps=linearization_eps,
    )
    a_closed_loop = closed_loop_linear_matrix(lqr.A, lqr.B, lqr.K)
    q_decrease = closed_loop_decrease_matrix(a_closed_loop, lqr.P)
    c_box, c_box_by_dim = box_containment_sublevel(
        torch.tensor(system.local_decrease.box_radii, dtype=torch.float32),
        lqr.P,
    )
    c_nonlinear, nonlinear_constants = nonlinear_decrease_sublevel_from_remainder_bound(
        a_closed_loop,
        lqr.P,
        q_decrease,
        torch.tensor(system.local_decrease.eta_bound, dtype=torch.float32),
        epsilon=system.local_decrease.epsilon,
    )
    c_analytic_certified = min(lqr.c_input, c_box, c_nonlinear)
    c_nonlinear_grid = float("inf")
    nonlinear_grid_constants = {
        "deferred_until_after_reachability_expansion": True,
        "certified_floor": float(c_analytic_certified),
        "reason": "C_S should use the active expanded domain, while eta_B remains local.",
    }
    nonlinear_constants = {
        "analytic": nonlinear_constants,
        "grid_search": nonlinear_grid_constants,
    }
    c_s = min(lqr.c_input, c_box)
    c_analytic = c_analytic_certified
    c_sigmoid_center = float(c_analytic) + float(system.training.c1_scale) * (
        float(c_s) - float(c_analytic)
    )
    domain_box = tuple((-float(r), float(r)) for r in system.local_decrease.box_radii)

    sublevels = lqr.sublevels + (
        SublevelRecord(
            name="lqr_box_containment",
            value=c_box,
            formula="min_i R_B,i^2 / (P^{-1})_ii",
            notes=(
                "Ensures {x: nu(x)<=c} is contained in supplied local rectangle B.",
                "The pipeline later applies an active-domain containment cap.",
                f"B radii source: {system.local_decrease.source}",
            ),
        ),
        SublevelRecord(
            name="nonlinear_lqr_decrease_grid",
            value=c_nonlinear_grid,
            formula="unused for C_S",
            notes=(
                "Uses true nonlinear dynamics and the linear controller u=-Kx.",
                "This grid search is not run during pre-training setup.",
                "C_S does not depend on this diagnostic.",
            ),
        ),
        SublevelRecord(
            name="nonlinear_lqr_decrease_analytic",
            value=c_nonlinear,
            formula="Serry-Liu remainder bound alpha*c + beta*sqrt(c) <= d",
            notes=(
                "This eta-governed bound remains part of the analytic verification level.",
                "Requires supplied eta_B; no eta_B means no certified analytic threshold.",
                f"eta_B={tuple(float(v) for v in system.local_decrease.eta_bound)}",
                f"epsilon={system.local_decrease.epsilon}",
            ),
        ),
        SublevelRecord(
            name="analytic_certified_search_floor",
            value=c_analytic_certified,
            formula="min(c_input, c_box, c_nonlinear_decrease_analytic)",
            notes=(
                "Kept as the analytic lower verification floor.",
                "The outer C_S uses only input and active-domain containment.",
            ),
        ),
        SublevelRecord(
            name="controller_lqr_off_level_C_S",
            value=c_s,
            formula="min(c_input, local_box_containment)",
            notes=(
                "This capped level is the initial outer reference for band sampling.",
                "This initial value is temporary and is refreshed after reachability expansion.",
                "The small Hessian box does not cap the final C_S.",
            ),
        ),
        SublevelRecord(
            name="analytic_verification_c1",
            value=c_analytic,
            formula="min(c_input, local_box_containment, c_nonlinear_decrease_analytic)",
            notes=(
                "This is the lower eta-analytic level used by verification.",
                "This remains tied to the small local box used by the Hessian bound.",
            ),
        ),
    )
    return LQRBoxArtifacts(
        A=lqr.A,
        B=lqr.B,
        K=lqr.K,
        P=lqr.P,
        c_input=lqr.c_input,
        c_input_by_dim=lqr.c_input_by_dim,
        c_box=c_box,
        c_box_by_dim=c_box_by_dim,
        c_nonlinear_decrease=c_nonlinear_grid,
        c_nonlinear_decrease_grid=c_nonlinear_grid,
        c_nonlinear_decrease_analytic=c_nonlinear,
        nonlinear_decrease_constants=nonlinear_constants,
        c_analytic=c_analytic,
        c_s=c_s,
        c_sigmoid_center=c_sigmoid_center,
        domain_box=domain_box,
        sublevels=sublevels,
        notes=(
            "C_S is finalized after reachability expansion as min(active-domain containment, input constraint).",
            "Local-box containment remains recorded for the analytic Hessian-bound floor.",
            "The analytic Hessian-based decrease level remains the lower verification threshold.",
            "Training and verification use the saturated NN branch and clamped LQR branch separately.",
        ),
    )


@torch.no_grad()
def compute_stepwise_min_zubov_values(
    system: SystemSpec,
    x_batch: torch.Tensor,
    p_matrix: torch.Tensor,
    *,
    horizon: int,
    ntraj: int,
    control_sampler: Optional[Callable[[Tuple[int, ...], torch.device, torch.dtype], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return finite-horizon stepwise-min reachability values `(V, W)`.

    This follows the intended notebook logic using sampled controls. It is a
    finite-horizon approximation to the stepwise minimum over the reachable set;
    it is not a proof that the true reachable-set minimum has been found.

    Intended tensor shapes:
      x_batch: (N, n_dim)
      x_cur:   (N, ntraj, n_dim)
      u:       (N, ntraj, u_dim)
      nu_step: (N, ntraj)
      V:       (N,)

    Intended accumulation:
      For k = 0..horizon:
        alpha_step = zubov_alpha_scale * dt * nu(x_cur)
        V += min over trajectory axis of alpha_step
        sample random controls and step all trajectories if k < horizon

    Important: the minimum is stepwise. It is not the minimum total cost over a
    single complete sampled trajectory.
    """
    device = x_batch.device
    dtype = x_batch.dtype
    n_batch = x_batch.shape[0]
    if x_batch.shape[-1] != system.n_dim:
        raise ValueError("x_batch last dimension must equal system.n_dim.")
    if horizon < 0:
        raise ValueError("horizon must be nonnegative.")
    if ntraj <= 0:
        raise ValueError("ntraj must be positive.")

    sampler = control_sampler
    x_cur = x_batch.unsqueeze(1).expand(n_batch, ntraj, system.n_dim).clone()
    accumulated = torch.zeros(n_batch, device=device, dtype=dtype)

    for k in range(horizon + 1):
        nu_step = nu_from_p(x_cur, p_matrix)
        alpha_step = zubov_alpha_from_nu(
            nu_step,
            dt=system.dt,
            scale=system.training.zubov_alpha_scale,
        )
        step_min = alpha_step.min(dim=1).values
        accumulated += step_min
        if k < horizon:
            if sampler is None:
                u = sample_uniform_controls(
                    system,
                    (n_batch, ntraj),
                    device=device,
                    dtype=dtype,
                )
            else:
                u = sampler((n_batch, ntraj), device, dtype)
            if u.shape != (n_batch, ntraj, system.u_dim):
                raise ValueError(
                    "control_sampler must return shape "
                    f"{(n_batch, ntraj, system.u_dim)}, got {tuple(u.shape)}"
                )
            x_cur = system.dynamics(x_cur, u)

    target_w = 1.0 - torch.exp(-accumulated)
    return accumulated, target_w.clamp(0.0, 1.0)


@torch.no_grad()
def compute_stepwise_min_zubov_targets(
    system: SystemSpec,
    x_batch: torch.Tensor,
    p_matrix: torch.Tensor,
    *,
    horizon: int,
    ntraj: int,
    control_sampler: Optional[Callable[[Tuple[int, ...], torch.device, torch.dtype], torch.Tensor]] = None,
) -> OmegaLabelBatch:
    """Finite-horizon guidance labels for original omega.

    Each source state contributes one degenerate-zonotope embedding for the
    full stepwise-min finite horizon.
    """
    device = x_batch.device
    dtype = x_batch.dtype
    n_batch = x_batch.shape[0]
    if x_batch.shape[-1] != system.n_dim:
        raise ValueError("x_batch last dimension must equal system.n_dim.")
    if horizon < 0:
        raise ValueError("horizon must be nonnegative.")
    if ntraj <= 0:
        raise ValueError("ntraj must be positive.")

    x_cur = x_batch.unsqueeze(1).expand(n_batch, ntraj, system.n_dim).clone()
    accumulated = torch.zeros(n_batch, device=device, dtype=dtype)

    for k in range(horizon + 1):
        nu_step = nu_from_p(x_cur, p_matrix)
        alpha_step = zubov_alpha_from_nu(
            nu_step,
            dt=system.dt,
            scale=system.training.zubov_alpha_scale,
        )
        step_min = alpha_step.min(dim=1).values
        accumulated += step_min
        if k < horizon:
            if control_sampler is None:
                u = sample_uniform_controls(
                    system,
                    (n_batch, ntraj),
                    device=device,
                    dtype=dtype,
                )
            else:
                u = control_sampler((n_batch, ntraj), device, dtype)
            if u.shape != (n_batch, ntraj, system.u_dim):
                raise ValueError(
                    "control_sampler must return shape "
                    f"{(n_batch, ntraj, system.u_dim)}, got {tuple(u.shape)}"
                )
            x_cur = system.dynamics(x_cur, u)

    target_w = (1.0 - torch.exp(-accumulated)).clamp(0.0, 1.0)
    return OmegaLabelBatch(
        x=x_batch.detach(),
        z_x=embed_singleton_zonotope(system, x_batch).detach(),
        target_w=target_w.detach(),
    )


def build_original_omega_residual_batch(
    system: SystemSpec,
    x_collocation: torch.Tensor,
    p_matrix: torch.Tensor,
) -> OmegaResidualBatch:
    """Build the Bellman-type residual batch for original omega.

    Omega_orig uses the fixed zonotope input:

        z_x    = [x, 0]
        z_next = [center(F({x})), vec(G_F)]
        xi = 1 - exp(-zubov_alpha_scale * dt * nu(x))
    """
    z_next = build_successor_zonotope_embedding(system, x_collocation)
    z_x = embed_singleton_zonotope(system, x_collocation)
    xi = xi_stage_cost(
        x_collocation,
        p_matrix,
        dt=system.dt,
        scale=system.training.zubov_alpha_scale,
    ).clamp(0.0, 1.0)
    return OmegaResidualBatch(
        x=x_collocation,
        z_x=z_x,
        z_next_interval=z_next,
        xi=xi,
        successor_embedding_mode=ORIGINAL_OMEGA_EMBEDDING_MODE,
    )


def original_omega_loss_terms(
    vnet,
    residual_batch: OmegaResidualBatch,
    label_batch: OmegaLabelBatch,
    *,
    lambda_data: float,
    lambda_residual: float,
    lambda_boundary: float,
    x_origin: Optional[torch.Tensor] = None,
) -> OmegaLossTerms:
    """Compute the original omega objective terms from notebook Block 5.

    This helper assumes the caller has already built the interval residual
    batch correctly. It deliberately does not know how samples were generated.
    """
    data_mse = ((vnet(label_batch.z_x) - label_batch.target_w) ** 2).mean()
    if residual_batch.x.shape[0] > 0:
        residual, _, _, _ = interval_zubov_residual(
            vnet,
            residual_batch.z_x,
            residual_batch.z_next_interval,
            residual_batch.xi,
        )
        interval_residual_mse = (residual ** 2).mean()
    else:
        interval_residual_mse = data_mse * 0.0
    if x_origin is None:
        boundary_condition_mse = torch.zeros((), device=data_mse.device, dtype=data_mse.dtype)
    else:
        boundary_condition_mse = (vnet.query(x_origin) ** 2).mean()
    total = (
        float(lambda_data) * data_mse
        + float(lambda_residual) * interval_residual_mse
        + float(lambda_boundary) * boundary_condition_mse
    )
    return OmegaLossTerms(
        data_mse=data_mse,
        interval_residual_mse=interval_residual_mse,
        boundary_condition_mse=boundary_condition_mse,
        total=total,
    )


def closed_loop_omega_loss_terms(
    vnet,
    omega_orig,
    constraint_x_batch: torch.Tensor,
    trajectory_label_batch: OmegaLabelBatch,
    closed_loop_step,
    p_matrix: torch.Tensor,
    *,
    system: SystemSpec,
    lambda_zubov: float,
    lambda_data: float,
    lambda_decrease: float,
    lambda_range: float,
    lambda_boundary: float,
    dos_gate: float,
    kappa: float,
    anchor_filter: str = "none",
    x_origin: Optional[torch.Tensor] = None,
) -> ClosedLoopOmegaLossTerms:
    """Compute the closed-loop omega losses.

    The trajectory and constraint datasets are deliberately disjoint:

      1. finite closed-loop trajectory label MSE is evaluated only on
         `trajectory_label_batch`;
      2. the closed-loop Zubov residual is evaluated only on
         `constraint_x_batch`;
      3. the multiplicative decrease hinge is evaluated only on constraint
         points inside the original-omega proposed DOS.

    Empty batches produce zero for their associated terms so individual data
    sources can be disabled. The training stage separately checks that each
    active loss family has a non-empty combined dataset.
    """
    reference = next(vnet.parameters())
    zero = reference.sum() * 0.0

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.numel() != mask.numel():
            raise ValueError("Masked loss values and mask must have the same number of elements.")
        weights = mask.to(device=values.device, dtype=values.dtype)
        denom = weights.sum().clamp(min=1.0)
        return (values.reshape(-1) * weights).sum() / denom

    if constraint_x_batch.shape[0] > 0:
        zubov_residual, w_x, w_next, _ = closed_loop_zubov_residual(
            vnet,
            constraint_x_batch,
            closed_loop_step,
            p_matrix,
            dt=system.dt,
            scale=system.training.zubov_alpha_scale,
        )
        with torch.no_grad():
            omega_orig_x = omega_orig.query(constraint_x_batch).reshape(-1)
            proposed_dos_mask = omega_orig_x <= float(dos_gate)
        decrease_hinge = torch.relu(w_next - (1.0 - float(kappa)) * w_x)
        zubov_mse = zubov_residual.pow(2).mean()
        decrease_penalty = masked_mean(decrease_hinge, proposed_dos_mask)
        range_penalty = masked_mean((1.0 - w_x).pow(2), ~proposed_dos_mask)
    else:
        zubov_mse = zero
        decrease_penalty = zero
        range_penalty = zero

    if trajectory_label_batch.x.shape[0] > 0:
        data_error = (vnet(trajectory_label_batch.z_x) - trajectory_label_batch.target_w).pow(2)
        normalized_anchor_filter = str(anchor_filter).strip().lower()
        if normalized_anchor_filter in {"none", ""}:
            data_mse = data_error.mean()
        elif normalized_anchor_filter in {"omega_orig_nonviolating", "nonviolating", "decrease"}:
            with torch.no_grad():
                anchor_x = trajectory_label_batch.x
                omega_orig_x = omega_orig.query(anchor_x).reshape(-1)
                omega_orig_next = omega_orig.query(closed_loop_step(anchor_x)).reshape(-1)
                anchor_mask = (omega_orig_next - omega_orig_x) < -float(system.verification.eps)
            data_mse = masked_mean(data_error.reshape(-1), anchor_mask)
        else:
            raise ValueError(
                "closed_loop_anchor_filter must be 'none' or "
                f"'omega_orig_nonviolating', got {anchor_filter!r}."
            )
    else:
        data_mse = zero
    if x_origin is None:
        boundary_condition_mse = zero
    else:
        boundary_condition_mse = (vnet.query(x_origin) ** 2).mean()
    total = (
        float(lambda_zubov) * zubov_mse
        + float(lambda_decrease) * decrease_penalty
        + float(lambda_data) * data_mse
        + float(lambda_range) * range_penalty
        + float(lambda_boundary) * boundary_condition_mse
    )
    return ClosedLoopOmegaLossTerms(
        zubov_mse=zubov_mse,
        decrease_penalty=decrease_penalty,
        range_penalty=range_penalty,
        boundary_condition_mse=boundary_condition_mse,
        data_mse=data_mse,
        total=total,
    )


def policy_control_for_mode(pi: PiNet, x: torch.Tensor, mode: str) -> torch.Tensor:
    """Return the controller branch used by closed-loop value training."""
    normalized = str(mode).strip().lower()
    if normalized in {"nn", "neural", "neural_net", "saturated_nn", "tanh_nn"}:
        return pi.nn_control(x)
    if normalized in {"lqr", "linear"}:
        return pi.lqr_control(x)
    raise ValueError(
        "closed_loop_controller_mode must be 'nn' or 'lqr'; blended/sigmoid "
        "controllers are not part of the discontinuous-switch proof path, "
        f"got {mode!r}."
    )


@torch.no_grad()
def evaluate_closed_loop_decrease_report(
    system: SystemSpec,
    pi: PiNet,
    omega_orig: OmegaNet,
    omega_cl: OmegaNet,
    p_matrix: torch.Tensor,
    states: torch.Tensor,
    source_ids: torch.Tensor,
    *,
    controller_mode: str,
    batch_size: int,
    dos_gate: float,
    eps: float,
    domain: Optional[Tuple[Tuple[float, float], ...]] = None,
) -> Dict[str, object]:
    """Held-out sign diagnostic for both value functions under f(x, pi_mode(x))."""
    if batch_size <= 0:
        raise ValueError("closed-loop validation batch_size must be positive.")
    if states.shape[0] != source_ids.shape[0]:
        raise ValueError("validation states and source ids must have matching length.")

    pi.eval()
    omega_orig.eval()
    omega_cl.eval()
    p_matrix = p_matrix.to(device=states.device, dtype=states.dtype)
    domain = domain or tuple(zip(system.x_lb, system.x_hi))

    pieces: Dict[str, List[torch.Tensor]] = {
        "source": [],
        "nu": [],
        "omega_orig": [],
        "omega_cl": [],
        "delta_orig": [],
        "delta_cl": [],
        "in_domain": [],
    }
    for start in range(0, states.shape[0], int(batch_size)):
        x = states[start : start + int(batch_size)]
        u = policy_control_for_mode(pi, x, controller_mode)
        x_next = system.dynamics(x, u)
        pieces["source"].append(source_ids[start : start + int(batch_size)].detach().cpu())
        pieces["nu"].append(nu_from_p(x, p_matrix).detach().cpu())
        w_orig = omega_orig.query(x).reshape(-1)
        w_orig_next = omega_orig.query(x_next).reshape(-1)
        w_cl = omega_cl.query(x).reshape(-1)
        w_cl_next = omega_cl.query(x_next).reshape(-1)
        pieces["omega_orig"].append(w_orig.detach().cpu())
        pieces["omega_cl"].append(w_cl.detach().cpu())
        pieces["delta_orig"].append((w_orig_next - w_orig).detach().cpu())
        pieces["delta_cl"].append((w_cl_next - w_cl).detach().cpu())
        pieces["in_domain"].append(points_in_bounds(x_next, domain).detach().cpu())

    arrays = {key: torch.cat(value, dim=0) for key, value in pieces.items()}
    source_names = {
        0: "c1_to_lqr_c2_band",
        1: "uniform_domain",
        2: "omega_orig_rejection_dos",
    }

    def summarize(mask: torch.Tensor) -> Dict[str, object]:
        mask = mask.to(dtype=torch.bool)
        n = int(mask.sum().item())
        if n == 0:
            return {"count": 0}
        delta_orig = arrays["delta_orig"][mask]
        delta_cl = arrays["delta_cl"][mask]
        in_domain = arrays["in_domain"][mask]
        orig_violation = (delta_orig > -float(eps)) | (~in_domain)
        cl_violation = (delta_cl > -float(eps)) | (~in_domain)
        cl_positive = delta_cl[cl_violation]
        orig_positive = delta_orig[orig_violation]
        omega_cl = arrays["omega_cl"][mask]
        omega_cl_bad = omega_cl[cl_violation]
        omega_cl_good = omega_cl[~cl_violation]
        omega_orig = arrays["omega_orig"][mask]
        proposed_dos = omega_orig <= float(dos_gate)

        def maybe_float(value: torch.Tensor) -> Optional[float]:
            return None if value.numel() == 0 else float(value.item())

        def maybe_quantile(values: torch.Tensor, q: float) -> Optional[float]:
            return None if values.numel() == 0 else float(torch.quantile(values, q).item())

        return {
            "count": n,
            "inside_dos_gate_count": int(proposed_dos.sum().item()),
            "inside_dos_gate_fraction": float(proposed_dos.float().mean().item()),
            "next_out_of_domain_count": int((~in_domain).sum().item()),
            "omega_orig_violation_count": int(orig_violation.sum().item()),
            "omega_orig_violation_fraction": float(orig_violation.float().mean().item()),
            "omega_orig_max_delta": float(delta_orig.max().item()),
            "omega_orig_p95_delta": maybe_quantile(delta_orig, 0.95),
            "omega_orig_mean_positive_delta": maybe_float(orig_positive.mean()) if orig_positive.numel() else None,
            "omega_cl_violation_count": int(cl_violation.sum().item()),
            "omega_cl_violation_fraction": float(cl_violation.float().mean().item()),
            "omega_cl_max_delta": float(delta_cl.max().item()),
            "omega_cl_p95_delta": maybe_quantile(delta_cl, 0.95),
            "omega_cl_mean_positive_delta": maybe_float(cl_positive.mean()) if cl_positive.numel() else None,
            "omega_cl_min_value_among_violations": maybe_float(omega_cl_bad.min()) if omega_cl_bad.numel() else None,
            "omega_cl_max_value_among_decreasing": maybe_float(omega_cl_good.max()) if omega_cl_good.numel() else None,
        }

    per_source = {}
    for source_id in sorted(int(v) for v in arrays["source"].unique().tolist()):
        per_source[source_names.get(source_id, f"source_{source_id}")] = summarize(arrays["source"] == source_id)

    return {
        "controller_mode": str(controller_mode),
        "eps": float(eps),
        "dos_gate": float(dos_gate),
        "domain": [[float(lo), float(hi)] for lo, hi in domain],
        "overall": summarize(torch.ones(states.shape[0], dtype=torch.bool)),
        "by_source": per_source,
    }


@torch.no_grad()
def evaluate_policy_decrease_report(
    system: SystemSpec,
    pi: PiNet,
    omega_orig: OmegaNet,
    p_matrix: torch.Tensor,
    states: torch.Tensor,
    source_ids: torch.Tensor,
    *,
    batch_size: int,
    kappa: float,
    eps: float,
    c_s: float,
    domain: Optional[Tuple[Tuple[float, float], ...]] = None,
) -> Dict[str, object]:
    """Policy-side sign diagnostic for omega_orig under the learned NN branch."""
    if batch_size <= 0:
        raise ValueError("policy decrease report batch_size must be positive.")
    if states.shape[0] != source_ids.shape[0]:
        raise ValueError("policy decrease report states and source ids must match.")

    pi.eval()
    omega_orig.eval()
    p_matrix = p_matrix.to(device=states.device, dtype=states.dtype)
    domain = domain or tuple(zip(system.x_lb, system.x_hi))

    pieces: Dict[str, List[torch.Tensor]] = {
        "source": [],
        "nu": [],
        "omega_orig": [],
        "delta_orig": [],
        "hinge": [],
        "in_domain": [],
    }
    for start in range(0, states.shape[0], int(batch_size)):
        x = states[start : start + int(batch_size)]
        u = pi.nn_control(x)
        x_next = system.dynamics(x, u)
        omega_x = omega_orig.query(x).reshape(-1)
        omega_next = omega_orig.query(x_next).reshape(-1)
        delta = omega_next - omega_x
        hinge = torch.relu(omega_next - (1.0 - float(kappa)) * omega_x).pow(2)
        pieces["source"].append(source_ids[start : start + int(batch_size)].detach().cpu())
        pieces["nu"].append(nu_from_p(x, p_matrix).detach().cpu())
        pieces["omega_orig"].append(omega_x.detach().cpu())
        pieces["delta_orig"].append(delta.detach().cpu())
        pieces["hinge"].append(hinge.detach().cpu())
        pieces["in_domain"].append(points_in_bounds(x_next, domain).detach().cpu())

    arrays = {key: torch.cat(value, dim=0) for key, value in pieces.items()}
    source_names = {
        0: "c1_to_lqr_c2_band",
        1: "uniform_domain",
        2: "omega_orig_rejection_dos",
    }

    def maybe_quantile(values: torch.Tensor, q: float) -> Optional[float]:
        return None if values.numel() == 0 else float(torch.quantile(values, q).item())

    def summarize(mask: torch.Tensor) -> Dict[str, object]:
        mask = mask.to(dtype=torch.bool)
        n = int(mask.sum().item())
        if n == 0:
            return {"count": 0}
        delta = arrays["delta_orig"][mask]
        hinge = arrays["hinge"][mask]
        in_domain = arrays["in_domain"][mask]
        violation = (delta > -float(eps)) | (~in_domain)
        positive = delta[violation]
        return {
            "count": n,
            "next_out_of_domain_count": int((~in_domain).sum().item()),
            "omega_orig_violation_count": int(violation.sum().item()),
            "omega_orig_violation_fraction": float(violation.float().mean().item()),
            "omega_orig_max_delta": float(delta.max().item()),
            "omega_orig_p95_delta": maybe_quantile(delta, 0.95),
            "omega_orig_mean_delta": float(delta.mean().item()),
            "omega_orig_mean_positive_delta": (
                None if positive.numel() == 0 else float(positive.mean().item())
            ),
            "multiplicative_hinge_mean": float(hinge.mean().item()),
            "multiplicative_hinge_p95": maybe_quantile(hinge, 0.95),
        }

    omega = arrays["omega_orig"]
    omega_edges = [0.0, 0.25, 0.5, 0.75, 0.9, float("inf")]
    omega_bins = {}
    for lo, hi in zip(omega_edges[:-1], omega_edges[1:]):
        label = f"{lo:g}_to_{hi:g}" if math.isfinite(hi) else f"{lo:g}_plus"
        omega_bins[label] = summarize((omega >= lo) & (omega < hi))

    nu = arrays["nu"]
    c_s_value = max(float(c_s), 1e-12)
    nu_scaled = nu / c_s_value
    nu_edges = [0.0, 0.25, 0.5, 0.75, 1.0, float("inf")]
    nu_bins = {}
    for lo, hi in zip(nu_edges[:-1], nu_edges[1:]):
        label = f"{lo:g}_to_{hi:g}_C_S" if math.isfinite(hi) else f"{lo:g}_C_S_plus"
        nu_bins[label] = summarize((nu_scaled >= lo) & (nu_scaled < hi))

    per_source = {}
    for source_id in sorted(int(v) for v in arrays["source"].unique().tolist()):
        per_source[source_names.get(source_id, f"source_{source_id}")] = summarize(
            arrays["source"] == source_id
        )

    return {
        "controller_mode": "nn",
        "eps": float(eps),
        "kappa": float(kappa),
        "c_s": float(c_s),
        "domain": [[float(lo), float(hi)] for lo, hi in domain],
        "overall": summarize(torch.ones(states.shape[0], dtype=torch.bool)),
        "by_source": per_source,
        "by_omega_orig": omega_bins,
        "by_nu_over_C_S": nu_bins,
    }


@torch.no_grad()
def compute_closed_loop_zubov_targets(
    system: SystemSpec,
    pi: PiNet,
    x_batch: torch.Tensor,
    p_matrix: torch.Tensor,
    *,
    horizon: int,
    controller_mode: str,
    value_input_dim: Optional[int] = None,
) -> OmegaLabelBatch:
    """Finite-horizon W labels along the learned closed-loop dynamics.

    For closed-loop omega there is no minimization over controls at each step:
    the controller has already been chosen. We accumulate

        V_N(x) = sum_{k=0}^N alpha(x_k),  x_{k+1}=f(x_k, u_mode(x)),
        W_N(x) = 1 - exp(-V_N(x)).

    This data term weakly guides the closed-loop value. The closed-loop Zubov,
    range, and decrease conditions are optimized on a separate constraint set.
    """
    pi.eval()
    p_matrix = p_matrix.to(device=x_batch.device, dtype=x_batch.dtype)
    x_cur = x_batch
    accumulated = torch.zeros(x_batch.shape[0], device=x_batch.device, dtype=x_batch.dtype)
    for _ in range(int(horizon) + 1):
        accumulated = accumulated + zubov_alpha_from_p(
            x_cur,
            p_matrix,
            dt=system.dt,
            scale=system.training.zubov_alpha_scale,
        )
        x_cur = system.dynamics(x_cur, policy_control_for_mode(pi, x_cur, controller_mode))
    target_w = 1.0 - torch.exp(-accumulated)
    input_dim = int(value_input_dim) if value_input_dim is not None else system.n_dim
    return OmegaLabelBatch(
        x=x_batch.detach(),
        z_x=embed_pointwise_value_input(
            system,
            x_batch,
            input_dim=input_dim,
        ).detach(),
        target_w=target_w.detach(),
    )


def policy_next_omega_mse(
    system: SystemSpec,
    pi: PiNet,
    omega: OmegaNet,
    x_batch: torch.Tensor,
    omega_opt_next: torch.Tensor,
    *,
    use_stitched_controller: bool = False,
) -> torch.Tensor:
    """MSE between omega(f(x, controller(x))) and grid-search optimal next omega."""
    u = pi(x_batch) if use_stitched_controller else pi.nn_control(x_batch)
    next_x = system.dynamics(x_batch, u)
    omega_next = omega.query(next_x)
    return ((omega_next - omega_opt_next) ** 2).mean()


def policy_omega_decrease_hinge(
    system: SystemSpec,
    pi: PiNet,
    omega: OmegaNet,
    x_batch: torch.Tensor,
    *,
    kappa: float,
    use_stitched_controller: bool = False,
) -> torch.Tensor:
    """Multiplicative one-step decrease hinge for omega(f(x, pi(x)))."""
    u = pi(x_batch) if use_stitched_controller else pi.nn_control(x_batch)
    next_x = system.dynamics(x_batch, u)
    omega_x = omega.query(x_batch).reshape(-1)
    omega_next = omega.query(next_x).reshape(-1)
    hinge = torch.relu(omega_next - (1.0 - float(kappa)) * omega_x)
    return hinge.mean()


def policy_control_imitation_mse(
    system: SystemSpec,
    pi: PiNet,
    x_batch: torch.Tensor,
    u_opt: torch.Tensor,
    *,
    use_stitched_controller: bool = False,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalized MSE between the selected controller branch and grid-search control labels."""
    u = pi(x_batch) if use_stitched_controller else pi.nn_control(x_batch)
    u_lb = torch.tensor(system.u_lb, device=x_batch.device, dtype=x_batch.dtype)
    u_hi = torch.tensor(system.u_hi, device=x_batch.device, dtype=x_batch.dtype)
    scale = (u_hi - u_lb).clamp_min(1e-12)
    per_state = (((u - u_opt) / scale) ** 2).mean(dim=-1)
    if sample_weight is None:
        return per_state.mean()
    weights = sample_weight.to(device=x_batch.device, dtype=x_batch.dtype).reshape(-1)
    return (per_state * weights).sum() / weights.sum().clamp_min(1e-12)


def require_trainable_parameters(module: torch.nn.Module, *, name: str) -> None:
    n_total = 0
    n_trainable = 0
    for param in module.parameters():
        n_total += int(param.numel())
        if param.requires_grad:
            n_trainable += int(param.numel())
    if n_total <= 0:
        raise ValueError(f"{name} has no parameters.")
    if n_trainable <= 0:
        raise RuntimeError(f"{name} has no trainable parameters; check no_grad state handling.")


@torch.no_grad()
def evaluate_boundary_stepwise_min_reachability(
    system: SystemSpec,
    p_matrix: torch.Tensor,
    *,
    bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
    points_per_face: Optional[int] = None,
    horizon: Optional[int] = None,
    ntraj: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute stepwise-min reachability values on a hyperrectangle boundary."""
    cfg = system.training
    device = device or torch.device("cpu")
    bounds = bounds or tuple(zip(system.x_lb, system.x_hi))
    points_per_face = int(points_per_face or cfg.boundary_diagnostic_points_per_face)
    horizon = int(horizon if horizon is not None else cfg.boundary_diagnostic_horizon)
    ntraj = int(ntraj if ntraj is not None else cfg.boundary_diagnostic_rollouts_per_state)
    batch_size = int(batch_size or cfg.boundary_diagnostic_batch_size)

    boundary = sample_hyperrectangle_boundary(
        bounds,
        points_per_face,
        device=device,
        dtype=torch.float32,
    )
    p_matrix = p_matrix.to(device=device, dtype=torch.float32)
    v_chunks = []
    w_chunks = []
    for start in range(0, boundary.shape[0], batch_size):
        xb = boundary[start : start + batch_size]
        v_values, w_values = compute_stepwise_min_zubov_values(
            system,
            xb,
            p_matrix,
            horizon=horizon,
            ntraj=ntraj,
        )
        v_chunks.append(v_values.detach().cpu())
        w_chunks.append(w_values.detach().cpu())

    return boundary.detach().cpu(), torch.cat(v_chunks, dim=0), torch.cat(w_chunks, dim=0)


@torch.no_grad()
def diagnose_domain_boundary_stepwise_min(
    system: SystemSpec,
    p_matrix: torch.Tensor,
    *,
    bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
    points_per_face: Optional[int] = None,
    horizon: Optional[int] = None,
    ntraj: Optional[int] = None,
    batch_size: Optional[int] = None,
    w_threshold: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> Tuple[DomainBoundaryDiagnostic, torch.Tensor, torch.Tensor]:
    """Probe whether the configured rectangle appears large enough.

    The diagnostic samples points on the rectangle boundary and computes the
    finite-horizon, stepwise-min value approximation. Boundary points with
    `W < w_threshold` are evidence that the rectangle may be too small and
    should be expanded before training omega on it.

    This is intentionally a diagnostic. It uses sampled controls, so it can
    miss better controls or miss problematic boundary regions.
    """
    cfg = system.training
    bounds = bounds or tuple(zip(system.x_lb, system.x_hi))
    horizon = int(horizon if horizon is not None else cfg.boundary_diagnostic_horizon)
    ntraj = int(ntraj if ntraj is not None else cfg.boundary_diagnostic_rollouts_per_state)
    w_threshold = float(w_threshold if w_threshold is not None else cfg.boundary_diagnostic_w_threshold)
    boundary, _, w_values = evaluate_boundary_stepwise_min_reachability(
        system,
        p_matrix,
        bounds=bounds,
        points_per_face=points_per_face,
        horizon=horizon,
        ntraj=ntraj,
        batch_size=batch_size,
        device=device,
    )
    n_below = int((w_values < w_threshold).sum().item())
    diagnostic = DomainBoundaryDiagnostic(
        bounds=tuple((float(lo), float(hi)) for lo, hi in bounds),
        horizon=horizon,
        rollouts_per_state=ntraj,
        w_threshold=w_threshold,
        n_points=int(boundary.shape[0]),
        max_w=float(w_values.max().item()),
        min_w=float(w_values.min().item()),
        mean_w=float(w_values.mean().item()),
        n_below_threshold=n_below,
        should_expand=n_below > 0,
        notes=(
            "Uses sampled controls and stepwise minima; this is not a proof.",
            "If any boundary W is below threshold, expand the training/search rectangle.",
        ),
    )
    return diagnostic, boundary.detach().cpu(), w_values


def _expand_symmetric_bounds(
    bounds: Tuple[Tuple[float, float], ...],
    *,
    factor: float,
    max_bounds: Tuple[Tuple[float, float], ...],
) -> Tuple[Tuple[float, float], ...]:
    if factor <= 1.0:
        raise ValueError("reachability_expansion_factor must be greater than 1.")
    expanded = []
    for (lo, hi), (max_lo, max_hi) in zip(bounds, max_bounds):
        center = 0.5 * (float(lo) + float(hi))
        radius = 0.5 * (float(hi) - float(lo)) * factor
        new_lo = max(float(max_lo), center - radius)
        new_hi = min(float(max_hi), center + radius)
        if new_lo >= new_hi:
            raise ValueError("Expanded bounds collapsed; check expansion factor and max radius scale.")
        expanded.append((new_lo, new_hi))
    return tuple(expanded)


def _expand_bounds_on_failed_faces(
    bounds: Tuple[Tuple[float, float], ...],
    *,
    factor: float,
    max_bounds: Tuple[Tuple[float, float], ...],
    expand_lower: Tuple[bool, ...],
    expand_upper: Tuple[bool, ...],
) -> Tuple[Tuple[float, float], ...]:
    if factor <= 1.0:
        raise ValueError("reachability_expansion_factor must be greater than 1.")
    expanded = []
    for dim, ((lo, hi), (max_lo, max_hi)) in enumerate(zip(bounds, max_bounds)):
        lo = float(lo)
        hi = float(hi)
        expand_axis = bool(expand_lower[dim] or expand_upper[dim])
        radius_delta = 0.5 * (hi - lo) * (factor - 1.0)
        new_lo = max(float(max_lo), lo - radius_delta) if expand_axis else lo
        new_hi = min(float(max_hi), hi + radius_delta) if expand_axis else hi
        if new_lo >= new_hi:
            raise ValueError("Expanded bounds collapsed; check expansion factor and max radius scale.")
        expanded.append((new_lo, new_hi))
    return tuple(expanded)


def _system_with_state_domain(
    system: SystemSpec,
    bounds: Tuple[Tuple[float, float], ...],
) -> SystemSpec:
    """Return a SystemSpec whose system and verification domains match bounds."""
    normalized = tuple((float(lo), float(hi)) for lo, hi in bounds)
    verification = (
        replace(system.verification, domain=normalized)
        if system.verification is not None
        else None
    )
    return replace(
        system,
        x_lb=tuple(lo for lo, _ in normalized),
        x_hi=tuple(hi for _, hi in normalized),
        verification=verification,
    )


@torch.no_grad()
def expand_domain_by_stepwise_min_reachability(
    system: SystemSpec,
    p_matrix: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[ReachabilityExpansionResult, Tuple[Tuple[float, float], ...], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand the state box until boundary stepwise-min W saturates near one.

    The expansion criterion is the same finite-horizon set-based approximation
    used for original omega targets:

        V(x) = sum_k min_{sampled reachable states at step k} alpha(x_k)
        W(x) = 1 - exp(-V(x))

    The helper records V and W on every evaluated boundary, and returns the
    final boundary samples so original omega can keep those points in its
    finite-horizon guidance set.
    """
    cfg = system.training
    device = device or torch.device("cpu")
    bounds = tuple((float(lo), float(hi)) for lo, hi in zip(system.x_lb, system.x_hi))
    initial_bounds = bounds
    threshold = float(cfg.boundary_diagnostic_w_threshold)
    max_iters = int(cfg.reachability_expansion_max_iters)
    if max_iters < 0:
        raise ValueError("reachability_expansion_max_iters must be nonnegative.")

    max_bounds = []
    max_scale = float(cfg.reachability_expansion_max_radius_scale)
    if max_scale < 1.0:
        raise ValueError("reachability_expansion_max_radius_scale must be at least 1.")
    for lo, hi in initial_bounds:
        center = 0.5 * (lo + hi)
        radius = 0.5 * (hi - lo) * max_scale
        max_bounds.append((center - radius, center + radius))
    max_bounds_tuple = tuple(max_bounds)

    history = []
    final_boundary = torch.empty(0, system.n_dim)
    final_v = torch.empty(0)
    final_w = torch.empty(0)
    converged = False

    for iteration in range(max_iters + 1):
        boundary, v_values, w_values = evaluate_boundary_stepwise_min_reachability(
            system,
            p_matrix,
            bounds=bounds,
            points_per_face=cfg.boundary_diagnostic_points_per_face,
            horizon=cfg.boundary_diagnostic_horizon,
            ntraj=cfg.boundary_diagnostic_rollouts_per_state,
            batch_size=cfg.boundary_diagnostic_batch_size,
            device=device,
        )
        below = w_values < threshold
        points_per_face = int(cfg.boundary_diagnostic_points_per_face)
        face_failures = []
        expand_lower = []
        expand_upper = []
        for dim in range(system.n_dim):
            lo_start = (2 * dim) * points_per_face
            lo_end = lo_start + points_per_face
            hi_start = lo_end
            hi_end = hi_start + points_per_face
            lower_count = int(below[lo_start:lo_end].sum().item())
            upper_count = int(below[hi_start:hi_end].sum().item())
            expand_lower.append(lower_count > 0)
            expand_upper.append(upper_count > 0)
            expand_axis = (lower_count > 0) or (upper_count > 0)
            face_failures.append(
                {
                    "dim": int(dim),
                    "lower_below_threshold": lower_count,
                    "upper_below_threshold": upper_count,
                    "expand_lower": expand_axis,
                    "expand_upper": expand_axis,
                    "expand_axis": expand_axis,
                }
            )
        history.append(
            {
                "iteration": int(iteration),
                "bounds": tuple((float(lo), float(hi)) for lo, hi in bounds),
                "n_points": int(boundary.shape[0]),
                "threshold": threshold,
                "n_below_threshold": int(below.sum().item()),
                "face_failures": tuple(face_failures),
                "V_min": float(v_values.min().item()),
                "V_mean": float(v_values.mean().item()),
                "V_max": float(v_values.max().item()),
                "W_min": float(w_values.min().item()),
                "W_mean": float(w_values.mean().item()),
                "W_max": float(w_values.max().item()),
            }
        )
        final_boundary = boundary
        final_v = v_values
        final_w = w_values
        if not bool(below.any()):
            converged = True
            break
        if iteration >= max_iters:
            break
        new_bounds = _expand_bounds_on_failed_faces(
            bounds,
            factor=float(cfg.reachability_expansion_factor),
            max_bounds=max_bounds_tuple,
            expand_lower=tuple(expand_lower),
            expand_upper=tuple(expand_upper),
        )
        if new_bounds == bounds:
            break
        bounds = new_bounds

    result = ReachabilityExpansionResult(
        initial_bounds=initial_bounds,
        final_bounds=tuple((float(lo), float(hi)) for lo, hi in bounds),
        iterations=len(history),
        converged=converged,
        threshold=threshold,
        history=tuple(history),
        notes=(
            "Expansion uses finite-horizon stepwise-min reachability values before original omega training.",
            "If either face on an axis fails, that whole coordinate axis is expanded symmetrically.",
            "Boundary data records V=sum_k min_j alpha(x_k^j) and W=1-exp(-V).",
            "This is a sampled controllability estimate, not a formal proof.",
        ),
    )
    return result, result.final_bounds, final_boundary, final_v, final_w


class PipelineRun:
    def __init__(self, system: SystemSpec, run_root: Path = RUNS_ROOT, run_name: Optional[str] = None):
        timestamp = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.system = system
        self.run_name = timestamp
        self.run_dir = run_root / system.name / timestamp
        self.artifact_dir = self.run_dir / "artifacts"
        self.plot_dir = self.run_dir / "plots"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def write_config(self) -> None:
        payload = {
            "system": self.system.name,
            "n_dim": self.system.n_dim,
            "u_dim": self.system.u_dim,
            "x_lb": self.system.x_lb,
            "x_hi": self.system.x_hi,
            "u_lb": self.system.u_lb,
            "u_hi": self.system.u_hi,
            "checkpoints": asdict(self.system.checkpoints),
            "training": asdict(self.system.training),
            "verification": asdict(self.system.verification) if self.system.verification else None,
        }
        (self.run_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_sublevels(self, records: Iterable[SublevelRecord], filename: str = "sublevels.json") -> None:
        payload = [asdict(record) for record in records]
        (self.run_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_json(self, filename: str, payload) -> None:
        (self.run_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_csv(self, filename: str, rows: Iterable[Dict[str, object]]) -> None:
        rows = list(rows)
        if not rows:
            (self.run_dir / filename).write_text("", encoding="utf-8")
            return
        columns = list(rows[0].keys())
        lines = [",".join(columns)]
        for row in rows:
            lines.append(",".join(str(row.get(column, "")) for column in columns))
        (self.run_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def system_with_existing_run_domain(system: SystemSpec, run_dir: Path) -> SystemSpec:
    """Reload an existing run's expanded domain for standalone rerun stages."""
    expansion_path = run_dir / "reachability_domain_expansion.json"
    if expansion_path.exists():
        payload = json.loads(expansion_path.read_text(encoding="utf-8"))
        final_bounds = payload.get("final_bounds")
        if final_bounds:
            return _system_with_state_domain(
                system,
                tuple((float(lo), float(hi)) for lo, hi in final_bounds),
            )

    config_path = run_dir / "config.json"
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        x_lb = payload.get("x_lb")
        x_hi = payload.get("x_hi")
        if x_lb and x_hi:
            return _system_with_state_domain(
                system,
                tuple((float(lo), float(hi)) for lo, hi in zip(x_lb, x_hi)),
            )
    return system


def save_loss_history_artifacts(
    run: Optional[PipelineRun],
    artifact_dir: Path,
    *,
    stem: str,
    title: str,
    history: Iterable[Dict[str, float]],
) -> None:
    """Persist scalar loss histories as JSON, CSV, and a semilog PNG."""
    history = list(history)
    if not history:
        return

    if run is not None:
        run.write_json(f"{stem}_loss_history.json", history)
        run.write_csv(f"{stem}_loss_history.csv", history)
        plot_dir = run.plot_dir
    else:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / f"{stem}_loss_history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        columns = list(history[0].keys())
        lines = [",".join(columns)]
        for row in history:
            lines.append(",".join(str(row.get(column, "")) for column in columns))
        (artifact_dir / f"{stem}_loss_history.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        plot_dir = artifact_dir / "plots"

    plot_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in history[0]:
        if key == "epoch":
            continue
        values = [max(float(row[key]), 1e-16) for row in history]
        ax.semilogy(epochs, values, lw=2, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{stem}_losses.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def save_finite_horizon_target_plot(
    run: Optional[PipelineRun],
    artifact_dir: Path,
    *,
    stem: str,
    title: str,
    labels: OmegaLabelBatch,
    p_matrix: torch.Tensor,
    c_s: float,
    max_points: int = 20_000,
    max_candidate_points: int = 1_000_000,
) -> None:
    """Save a diagnostic plot of finite-horizon W targets used for MSE guidance."""
    n = labels.x.shape[0]
    if n == 0:
        return
    candidate_count = min(int(max_candidate_points), n)
    count = min(int(max_points), candidate_count)
    if count <= 0:
        return
    device = labels.x.device
    if count == 1:
        indices = torch.zeros(1, device=device, dtype=torch.long)
    else:
        # Avoid materializing or floating-point indexing over huge label sets.
        # These indices evenly span at most max_candidate_points labels.
        indices = torch.arange(count, device=device, dtype=torch.long)
        indices = indices * int(candidate_count - 1) // int(count - 1)
    x_plot = labels.x[indices].detach().cpu()
    w_plot = labels.target_w[indices].detach().cpu()
    p_cpu = p_matrix.detach().cpu().to(dtype=torch.float32)
    nu_plot = nu_from_p(x_plot, p_cpu).detach().cpu()

    summary = {
        "count_plotted": int(count),
        "count_total": int(n),
        "count_candidate_cap": int(candidate_count),
        "target_w_min": float(w_plot.min().item()),
        "target_w_max": float(w_plot.max().item()),
        "target_w_mean": float(w_plot.mean().item()),
        "nu_min": float(nu_plot.min().item()),
        "nu_max": float(nu_plot.max().item()),
        "C_S": float(c_s),
    }
    if run is not None:
        run.write_json(f"{stem}_finite_horizon_targets_summary.json", summary)
        plot_dir = run.plot_dir
    else:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / f"{stem}_finite_horizon_targets_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        plot_dir = artifact_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if x_plot.shape[1] > 2:
        off_slice_norm = x_plot[:, 2:].norm(dim=1)
        keep_count = min(count, max(1, count // 3))
        keep = torch.argsort(off_slice_norm)[:keep_count]
        x_plot = x_plot[keep]
        w_plot = w_plot[keep]
        summary["slice_mode"] = "x0_x1_with_remaining_coordinates_nearest_zero"
        summary["slice_points_plotted"] = int(keep_count)
        summary["max_off_slice_norm_plotted"] = float(off_slice_norm[keep].max().item())
        if run is not None:
            run.write_json(f"{stem}_finite_horizon_targets_summary.json", summary)
        else:
            (artifact_dir / f"{stem}_finite_horizon_targets_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        x_plot[:, 0].numpy(),
        x_plot[:, 1].numpy(),
        c=w_plot.numpy(),
        s=5,
        cmap="viridis",
        alpha=0.85,
        linewidths=0,
    )
    fig.colorbar(sc, ax=ax, label="finite-horizon W target")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{stem}_finite_horizon_targets.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def save_policy_control_comparison_plot(
    run: Optional[PipelineRun],
    artifact_dir: Path,
    *,
    pi: PiNet,
    labels: PolicyLabelBatch,
    max_points: int,
    title: str = "Saturated NN Controller vs Grid-Search Control",
) -> None:
    """Compare learned saturated-NN controller outputs against grid-search labels."""
    n = labels.x.shape[0]
    if n == 0:
        return
    count = min(int(max_points), n)
    if count <= 0:
        return
    if count == 1:
        indices = torch.zeros(1, device=labels.x.device, dtype=torch.long)
    else:
        indices = torch.arange(count, device=labels.x.device, dtype=torch.long)
        indices = indices * int(n - 1) // int(count - 1)
    x_plot = labels.x[indices]
    u_opt = labels.u_opt[indices].detach().cpu()
    u_pi = pi.nn_control(x_plot).detach().cpu()

    if run is not None:
        plot_dir = run.plot_dir
        rows = []
        for row_idx, (uo, up) in enumerate(zip(u_opt.tolist(), u_pi.tolist())):
            row = {"index": row_idx}
            for j, value in enumerate(uo):
                row[f"u{j}_opt"] = float(value)
            for j, value in enumerate(up):
                row[f"u{j}_nn"] = float(value)
            rows.append(row)
        run.write_csv("policy_control_comparison.csv", rows)
    else:
        plot_dir = artifact_dir / "plots"

    plot_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u_dim = u_opt.shape[1]
    fig, axes = plt.subplots(1, u_dim, figsize=(5 * u_dim, 4.5), squeeze=False)
    for j in range(u_dim):
        ax = axes[0, j]
        opt_j = u_opt[:, j].numpy()
        pi_j = u_pi[:, j].numpy()
        lo = float(min(opt_j.min(), pi_j.min()))
        hi = float(max(opt_j.max(), pi_j.max()))
        ax.scatter(opt_j, pi_j, s=8, alpha=0.45)
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.2, ls="--")
        ax.set_xlabel(f"grid-search u{j}")
        ax.set_ylabel(f"PiNet saturated-NN u{j}")
        ax.set_title(f"u{j}")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(plot_dir / "policy_control_comparison.png", dpi=180)
    plt.close(fig)


def sample_uniform_bounds(
    bounds: Tuple[Tuple[float, float], ...],
    n: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Uniform sampler over an explicit hyperrectangle."""
    x = torch.empty(n, len(bounds), device=device)
    for d, (lo, hi) in enumerate(bounds):
        x[:, d].uniform_(float(lo), float(hi))
    return x


def points_in_bounds(x: torch.Tensor, bounds: Tuple[Tuple[float, float], ...]) -> torch.Tensor:
    """Return a boolean mask for membership in a closed hyperrectangle."""
    mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    for d, (lo, hi) in enumerate(bounds):
        mask = mask & (x[:, d] >= float(lo)) & (x[:, d] <= float(hi))
    return mask


@torch.no_grad()
def sample_omega_rejection_points(
    omega: OmegaNet,
    bounds: Tuple[Tuple[float, float], ...],
    count: int,
    *,
    device: torch.device,
    sublevel: float,
    batch_size: int,
    max_attempts: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Uniform rejection sampler for points inside an omega sublevel."""
    if count <= 0:
        return torch.empty(0, len(bounds), device=device), {
            "mode": "uniform_rejection_omega_sublevel",
            "requested_points": int(count),
            "returned_points": 0,
            "candidate_points": 0,
            "accepted_points": 0,
            "sublevel": float(sublevel),
            "acceptance_rate": None,
            "max_attempts": int(max_attempts),
        }
    if batch_size <= 0:
        raise ValueError("Rejection sampler batch_size must be positive.")
    if max_attempts <= 0:
        raise ValueError("Rejection sampler max_attempts must be positive.")

    omega.eval()
    accepted: List[torch.Tensor] = []
    accepted_points = 0
    candidate_points = 0
    attempts = 0
    while accepted_points < count and attempts < max_attempts:
        attempts += 1
        remaining = int(count) - int(accepted_points)
        draw_count = max(int(batch_size), remaining)
        candidates = sample_uniform_bounds(bounds, draw_count, device=device).to(dtype=torch.float32)
        values = omega.query(candidates)
        mask = values <= float(sublevel)
        candidate_points += int(candidates.shape[0])
        if not mask.any():
            continue
        kept = candidates[mask][:remaining].detach()
        accepted.append(kept)
        accepted_points += int(kept.shape[0])

    if accepted:
        samples = torch.cat(accepted, dim=0)[:count]
    else:
        samples = torch.empty(0, len(bounds), device=device, dtype=torch.float32)
    stats = {
        "mode": "uniform_rejection_omega_sublevel",
        "requested_points": int(count),
        "returned_points": int(samples.shape[0]),
        "candidate_points": int(candidate_points),
        "accepted_points": int(accepted_points),
        "sublevel": float(sublevel),
        "acceptance_rate": (
            float(accepted_points / candidate_points) if candidate_points > 0 else None
        ),
        "attempts": int(attempts),
        "max_attempts": int(max_attempts),
        "bounds": [[float(lo), float(hi)] for lo, hi in bounds],
    }
    return samples, stats


def threshold_witness_payload(
    x: torch.Tensor,
    fx: torch.Tensor,
    *,
    nu: float,
    nu_next: Optional[float],
    omega_cl: float,
    omega_cl_next: Optional[float],
    omega_original: Optional[float] = None,
    omega_original_next: Optional[float] = None,
    fx_in_domain: Optional[bool],
    reason: str,
) -> Dict[str, object]:
    """Small JSON-friendly record for the sampled point attaining a threshold."""
    return {
        "x": [float(v) for v in x.detach().cpu().tolist()],
        "f_pi_x": [float(v) for v in fx.detach().cpu().tolist()],
        "nu": float(nu),
        "nu_next": None if nu_next is None else float(nu_next),
        "delta_nu": None if nu_next is None else float(nu_next - nu),
        "omega_cl": float(omega_cl),
        "omega_cl_next": None if omega_cl_next is None else float(omega_cl_next),
        "delta_omega_cl": None if omega_cl_next is None else float(omega_cl_next - omega_cl),
        "omega_original": None if omega_original is None else float(omega_original),
        "omega_original_next": None if omega_original_next is None else float(omega_original_next),
        "delta_omega_original": (
            None
            if omega_original is None or omega_original_next is None
            else float(omega_original_next - omega_original)
        ),
        "f_pi_x_in_domain": None if fx_in_domain is None else bool(fx_in_domain),
        "reason": reason,
    }


def threshold_grid_shape(
    bounds: Tuple[Tuple[float, float], ...],
    target_points: int,
) -> Tuple[int, int]:
    """Return `(points_per_dim, total_points)` for a full threshold grid."""
    n_dim = len(bounds)
    if n_dim <= 0:
        raise ValueError("Threshold grid requires at least one state dimension.")
    if target_points <= 0:
        raise ValueError("threshold_search_samples must be positive.")
    points_per_dim = max(2, int(math.ceil(float(target_points) ** (1.0 / float(n_dim)))))
    total_points = int(points_per_dim) ** n_dim
    return points_per_dim, total_points


def iter_threshold_grid_batches(
    bounds: Tuple[Tuple[float, float], ...],
    *,
    points_per_dim: int,
    total_points: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    """Yield deterministic tensor-product grid batches over the threshold domain."""
    if batch_size <= 0:
        raise ValueError("threshold_search_batch_size must be positive.")
    lower = torch.tensor([lo for lo, _ in bounds], device=device, dtype=dtype)
    upper = torch.tensor([hi for _, hi in bounds], device=device, dtype=dtype)
    axes = [
        torch.linspace(lower[d], upper[d], int(points_per_dim), device=device, dtype=dtype)
        for d in range(len(bounds))
    ]
    strides = [int(points_per_dim) ** dim for dim in range(len(bounds))]
    index_batch = torch.empty(batch_size, len(bounds), device=device, dtype=torch.long)
    point_batch = torch.empty(batch_size, len(bounds), device=device, dtype=dtype)

    for start in range(0, int(total_points), int(batch_size)):
        count = min(int(batch_size), int(total_points) - start)
        flat = torch.arange(start, start + count, device=device, dtype=torch.long)
        for dim, stride in enumerate(strides):
            index_batch[:count, dim] = (flat // stride) % int(points_per_dim)
            point_batch[:count, dim] = axes[dim][index_batch[:count, dim]]
        yield point_batch[:count]


@torch.no_grad()
def early_lqr_c2_search(
    system: SystemSpec,
    artifacts: LQRBoxArtifacts,
    *,
    n_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> EarlyLqrC2SearchResult:
    """Grid-search a clamped-LQR c2 prior for training-sample placement."""
    cfg = system.training
    if system.verification is not None:
        domain = tuple((float(lo), float(hi)) for lo, hi in system.verification.domain)
        eps = float(system.verification.eps)
    else:
        domain = tuple((float(lo), float(hi)) for lo, hi in zip(system.x_lb, system.x_hi))
        eps = float(system.local_decrease.epsilon)
    requested_points = int(n_samples or cfg.threshold_search_samples)
    batch_size = int(batch_size or cfg.threshold_search_batch_size)
    points_per_dim, total_points = threshold_grid_shape(domain, requested_points)
    device = device or torch.device("cpu")
    p_matrix = artifacts.P.to(device=device, dtype=torch.float32)
    k_matrix = artifacts.K.to(device=device, dtype=torch.float32)
    c1_raw = float(artifacts.c_analytic)

    c2_raw = None
    witness = None
    bad_count = 0
    for x in iter_threshold_grid_batches(
        domain,
        points_per_dim=points_per_dim,
        total_points=total_points,
        batch_size=batch_size,
        device=device,
    ):
        u = lqr_control(system, x, k_matrix)
        fx = system.dynamics(x, u)
        nu = nu_from_p(x, p_matrix)
        nu_next = nu_from_p(fx, p_matrix)
        delta = nu_next - nu
        fx_in_domain = points_in_bounds(fx, domain)
        bad = (nu >= c1_raw) & ((delta > -eps) | (~fx_in_domain))
        bad_count += int(bad.sum().item())
        if not bad.any():
            continue
        candidate_values = nu.masked_fill(~bad, float("inf"))
        idx = int(torch.argmin(candidate_values).item())
        value = float(nu[idx].item())
        if c2_raw is None or value < c2_raw:
            c2_raw = value
            witness = {
                "x": [float(v) for v in x[idx].detach().cpu().tolist()],
                "f_pi_x": [float(v) for v in fx[idx].detach().cpu().tolist()],
                "nu": value,
                "nu_next": float(nu_next[idx].item()),
                "delta_nu": float(delta[idx].item()),
                "f_pi_x_in_domain": bool(fx_in_domain[idx].item()),
                "reason": (
                    "clamped_lqr_nu_decrease_violation"
                    if bool((delta[idx] > -eps).item())
                    else "clamped_lqr_next_state_outside_domain"
                ),
                "controller_branch": "clamped_lqr",
            }

    notes: Tuple[str, ...]
    if c2_raw is None:
        notes = (
            "No early clamped-LQR c2 violation was found on the training grid.",
            "Band sampling should fall back to c1 boundary sampling.",
        )
    else:
        notes = (
            "This c2 search is used only as a training-sample prior.",
            "Final verifier thresholds are still produced by search_thresholds.",
        )
    return EarlyLqrC2SearchResult(
        n_samples=total_points,
        domain=domain,
        eps=eps,
        c1_raw=c1_raw,
        c2_raw=c2_raw,
        bad_count=bad_count,
        witness=witness,
        notes=notes,
    )


@torch.no_grad()
def sampled_threshold_search(
    system: SystemSpec,
    pi: PiNet,
    omega_cl: OmegaNet,
    p_matrix: torch.Tensor,
    *,
    omega_original: Optional[OmegaNet] = None,
    c1_raw: float,
    c_s: float,
    c2_cap: Optional[float] = None,
    n_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    scale: Optional[float] = None,
    c2_scale: Optional[float] = None,
    w1_scale: Optional[float] = None,
    w2_scale: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> SampledThresholdSearchResult:
    """Approximate c2, w1, and w2 from grid-searched violation conditions.

    The search follows the discontinuous-switch proof structure:
      1. find raw grid c2 from clamped-LQR nu-decrease/domain violations;
      2. compute c2_verify by shrinking the interval from c1 to c2_raw;
      3. find raw grid w1 from the bridge failure nu(x)>c2_verify;
      4. find raw grid w2 from saturated-NN omega_cl-decrease/domain violations above w1_raw;
      5. compute w1_verify and shrink the interval from w1_verify to w2_raw;
      6. count grid bad points in [w1_verify, w2_verify].

    When `omega_original` is supplied, the same grid containment/decrease
    logic is also applied to the original omega model. Those values are recorded
    as diagnostics and witness starts; the verifier-facing w1/w2 remain the
    closed-loop omega thresholds.
    """
    if system.verification is None:
        raise ValueError(f"{system.name}: verification domain and eps are required for threshold search.")
    vcfg = system.verification
    cfg = system.training
    device = device or torch.device("cpu")
    requested_points = int(n_samples or cfg.threshold_search_samples)
    batch_size = int(batch_size or cfg.threshold_search_batch_size)
    scale = float(scale if scale is not None else cfg.threshold_search_scale)
    c2_scale = float(
        c2_scale
        if c2_scale is not None
        else (
            cfg.threshold_search_c2_scale
            if cfg.threshold_search_c2_scale is not None
            else scale
        )
    )
    w1_scale = float(
        w1_scale
        if w1_scale is not None
        else (
            cfg.threshold_search_w1_scale
            if cfg.threshold_search_w1_scale is not None
            else scale
        )
    )
    w2_scale = float(
        w2_scale
        if w2_scale is not None
        else (
            cfg.threshold_search_w2_scale
            if cfg.threshold_search_w2_scale is not None
            else scale
        )
    )
    for name, value in (
        ("threshold search scale", scale),
        ("threshold_search_c2_scale", c2_scale),
        ("threshold_search_w1_scale", w1_scale),
        ("threshold_search_w2_scale", w2_scale),
    ):
        if not (0.0 < value <= 1.0):
            raise ValueError(f"{name} must lie in (0, 1].")

    domain = tuple((float(lo), float(hi)) for lo, hi in vcfg.domain)
    points_per_dim, n_grid_points = threshold_grid_shape(domain, requested_points)
    eps = float(vcfg.eps)
    c2_cap_value = float(c2_cap) if c2_cap is not None else None
    p_matrix = p_matrix.to(device=device, dtype=torch.float32)
    pi.eval()
    omega_cl.eval()
    if omega_original is not None:
        omega_original.eval()

    c2_raw = None
    c2_witness = None

    def grid_batches():
        return iter_threshold_grid_batches(
            domain,
            points_per_dim=points_per_dim,
            total_points=n_grid_points,
            batch_size=batch_size,
            device=device,
        )

    def evaluate_threshold_batch(x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        u_lqr = pi.lqr_control(x)
        fx_lqr = system.dynamics(x, u_lqr)
        u_nn = pi.nn_control(x)
        fx_nn = system.dynamics(x, u_nn)
        nu = nu_from_p(x, p_matrix)
        nu_lqr_next = nu_from_p(fx_lqr, p_matrix)
        nu_nn_next = nu_from_p(fx_nn, p_matrix)
        omega = omega_cl.query(x)
        omega_lqr_next = omega_cl.query(fx_lqr)
        omega_nn_next = omega_cl.query(fx_nn)
        omega_orig = omega_original.query(x) if omega_original is not None else None
        omega_orig_lqr_next = omega_original.query(fx_lqr) if omega_original is not None else None
        omega_orig_nn_next = omega_original.query(fx_nn) if omega_original is not None else None
        delta_nu_lqr = nu_lqr_next - nu
        delta_omega_nn = omega_nn_next - omega
        delta_omega_orig = (
            omega_orig_nn_next - omega_orig
            if omega_orig is not None and omega_orig_nn_next is not None
            else None
        )
        fx_lqr_in_domain = points_in_bounds(fx_lqr, domain)
        fx_nn_in_domain = points_in_bounds(fx_nn, domain)
        return {
            "x": x,
            "fx_lqr": fx_lqr,
            "fx_nn": fx_nn,
            "nu": nu,
            "nu_lqr_next": nu_lqr_next,
            "nu_nn_next": nu_nn_next,
            "omega": omega,
            "omega_lqr_next": omega_lqr_next,
            "omega_nn_next": omega_nn_next,
            "delta_nu_lqr": delta_nu_lqr,
            "delta_omega_nn": delta_omega_nn,
            "omega_original": omega_orig,
            "omega_original_lqr_next": omega_orig_lqr_next,
            "omega_original_nn_next": omega_orig_nn_next,
            "delta_omega_original": delta_omega_orig,
            "fx_lqr_in_domain": fx_lqr_in_domain,
            "fx_nn_in_domain": fx_nn_in_domain,
        }

    for x in grid_batches():
        chunk = evaluate_threshold_batch(x)
        c2_bad = (chunk["nu"] >= float(c1_raw)) & (
            (chunk["delta_nu_lqr"] > -eps) | (~chunk["fx_lqr_in_domain"])
        )
        if c2_bad.any():
            candidate_values = chunk["nu"].masked_fill(~c2_bad, float("inf"))
            idx = int(torch.argmin(candidate_values).item())
            value = float(chunk["nu"][idx].item())
            if c2_raw is None or value < c2_raw:
                c2_raw = value
                c2_witness = threshold_witness_payload(
                    x[idx],
                    chunk["fx_lqr"][idx],
                    nu=float(chunk["nu"][idx].item()),
                    nu_next=float(chunk["nu_lqr_next"][idx].item()),
                    omega_cl=float(chunk["omega"][idx].item()),
                    omega_cl_next=float(chunk["omega_lqr_next"][idx].item()),
                    omega_original=(
                        None
                        if chunk["omega_original_lqr_next"] is None
                        else float(chunk["omega_original_lqr_next"][idx].item())
                    ),
                    omega_original_next=(
                        None
                        if chunk["omega_original_lqr_next"] is None
                        else float(chunk["omega_original_lqr_next"][idx].item())
                    ),
                    fx_in_domain=bool(chunk["fx_lqr_in_domain"][idx].item()),
                    reason=(
                        "clamped_lqr_nu_decrease_violation"
                        if bool((chunk["delta_nu_lqr"][idx] > -eps).item())
                        else "clamped_lqr_next_state_outside_domain"
                    ),
                )
                c2_witness["controller_branch"] = "clamped_lqr"

    c2_notes: List[str] = []
    if c2_raw is None and c2_cap_value is not None:
        c2_raw = c2_cap_value
        c2_notes.append(
            "No grid c2 violation was found; c2_raw was set to the active-domain containment cap."
        )
    elif c2_raw is not None and c2_cap_value is not None and c2_raw > c2_cap_value:
        uncapped_c2_raw = c2_raw
        c2_raw = c2_cap_value
        c2_notes.append(
            f"Grid c2_raw={uncapped_c2_raw} was capped to active-domain containment {c2_cap_value}."
        )

    if c2_raw is None:
        return SampledThresholdSearchResult(
            n_samples=n_grid_points,
            domain=domain,
            eps=eps,
            scale=scale,
            c2_scale=c2_scale,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            c1_raw=float(c1_raw),
            c_s=float(c_s),
            c2_cap=c2_cap_value,
            c2_raw=None,
            c2_verify=None,
            w1_raw=None,
            w1_verify=None,
            w2_raw=None,
            w2_verify=None,
            c2_witness=None,
            w1_witness=None,
            w2_witness=None,
            w1_original_raw=None,
            w1_original_verify=None,
            w2_original_raw=None,
            w2_original_verify=None,
            w1_original_witness=None,
            w2_original_witness=None,
            shell_diagnostics=(),
            band_good_count=0,
            band_bad_count=0,
            band_bad_witness=None,
            notes=(
                "No grid c2 violation was found.",
                "No w1/w2 search was attempted because c2_verify is undefined.",
                f"threshold_grid_points_per_dim={points_per_dim}",
                f"threshold_grid_requested_points={requested_points}",
            ),
        )

    c2_unboxed = float(c1_raw) + c2_scale * (c2_raw - float(c1_raw))
    c2_verify = c2_unboxed
    w1_raw = None
    w1_witness = None
    w1_original_raw = None
    w1_original_witness = None
    for x in grid_batches():
        chunk = evaluate_threshold_batch(x)
        outside_rescaled_c2 = chunk["nu"] > float(c2_verify)
        if outside_rescaled_c2.any():
            candidate_values = chunk["omega"].masked_fill(~outside_rescaled_c2, float("inf"))
            idx = int(torch.argmin(candidate_values).item())
            value = float(chunk["omega"][idx].item())
            if w1_raw is None or value < w1_raw:
                w1_raw = value
                w1_witness = threshold_witness_payload(
                    chunk["x"][idx],
                    chunk["fx_lqr"][idx],
                    nu=float(chunk["nu"][idx].item()),
                    nu_next=float(chunk["nu_lqr_next"][idx].item()),
                    omega_cl=float(chunk["omega"][idx].item()),
                    omega_cl_next=float(chunk["omega_lqr_next"][idx].item()),
                    omega_original=(
                        None
                        if chunk["omega_original"] is None
                        else float(chunk["omega_original"][idx].item())
                    ),
                    omega_original_next=(
                        None
                        if chunk["omega_original"] is None
                        else float(chunk["omega_original"][idx].item())
                    ),
                    fx_in_domain=bool(chunk["fx_lqr_in_domain"][idx].item()),
                    reason="omega_cl_sublevel_outside_c2",
                )
        if chunk["omega_original"] is not None and outside_rescaled_c2.any():
            candidate_values = chunk["omega_original"].masked_fill(~outside_rescaled_c2, float("inf"))
            idx = int(torch.argmin(candidate_values).item())
            value = float(chunk["omega_original"][idx].item())
            if w1_original_raw is None or value < w1_original_raw:
                w1_original_raw = value
                w1_original_witness = threshold_witness_payload(
                    chunk["x"][idx],
                    chunk["fx_lqr"][idx],
                    nu=float(chunk["nu"][idx].item()),
                    nu_next=float(chunk["nu_lqr_next"][idx].item()),
                    omega_cl=float(chunk["omega"][idx].item()),
                    omega_cl_next=float(chunk["omega_lqr_next"][idx].item()),
                    omega_original=float(chunk["omega_original"][idx].item()),
                    omega_original_next=float(chunk["omega_original_lqr_next"][idx].item()),
                    fx_in_domain=bool(chunk["fx_lqr_in_domain"][idx].item()),
                    reason="omega_original_sublevel_outside_c2",
                )

    w1_verify = w1_scale * w1_raw if w1_raw is not None else None
    w1_original_verify = w1_scale * w1_original_raw if w1_original_raw is not None else None
    w2_raw = None
    w2_witness = None
    w2_original_raw = None
    w2_original_witness = None
    if w1_verify is not None:
        for x in grid_batches():
            chunk = evaluate_threshold_batch(x)
            w2_bad = (chunk["omega"] >= float(w1_verify)) & (
                (chunk["delta_omega_nn"] > -eps) | (~chunk["fx_nn_in_domain"])
            )
            if w2_bad.any():
                candidate_values = chunk["omega"].masked_fill(~w2_bad, float("inf"))
                idx = int(torch.argmin(candidate_values).item())
                value = float(chunk["omega"][idx].item())
                if w2_raw is None or value < w2_raw:
                    w2_raw = value
                    w2_witness = threshold_witness_payload(
                        chunk["x"][idx],
                        chunk["fx_nn"][idx],
                        nu=float(chunk["nu"][idx].item()),
                        nu_next=float(chunk["nu_nn_next"][idx].item()),
                        omega_cl=float(chunk["omega"][idx].item()),
                        omega_cl_next=float(chunk["omega_nn_next"][idx].item()),
                        omega_original=(
                            None
                            if chunk["omega_original"] is None
                            else float(chunk["omega_original"][idx].item())
                        ),
                        omega_original_next=(
                            None
                            if chunk["omega_original_nn_next"] is None
                            else float(chunk["omega_original_nn_next"][idx].item())
                        ),
                        fx_in_domain=bool(chunk["fx_nn_in_domain"][idx].item()),
                        reason=(
                            "saturated_nn_omega_cl_decrease_violation"
                            if bool((chunk["delta_omega_nn"][idx] > -eps).item())
                            else "saturated_nn_next_state_outside_domain"
                        ),
                    )
                    w2_witness["controller_branch"] = "saturated_nn"
    if w1_original_verify is not None:
        for x in grid_batches():
            chunk = evaluate_threshold_batch(x)
            if chunk["omega_original"] is None or chunk["delta_omega_original"] is None:
                continue
            w2_original_bad = (chunk["omega_original"] >= float(w1_original_verify)) & (
                (chunk["delta_omega_original"] > -eps) | (~chunk["fx_nn_in_domain"])
            )
            if w2_original_bad.any():
                candidate_values = chunk["omega_original"].masked_fill(~w2_original_bad, float("inf"))
                idx = int(torch.argmin(candidate_values).item())
                value = float(chunk["omega_original"][idx].item())
                if w2_original_raw is None or value < w2_original_raw:
                    w2_original_raw = value
                    w2_original_witness = threshold_witness_payload(
                        chunk["x"][idx],
                        chunk["fx_nn"][idx],
                        nu=float(chunk["nu"][idx].item()),
                        nu_next=float(chunk["nu_nn_next"][idx].item()),
                        omega_cl=float(chunk["omega"][idx].item()),
                        omega_cl_next=float(chunk["omega_nn_next"][idx].item()),
                        omega_original=float(chunk["omega_original"][idx].item()),
                        omega_original_next=float(chunk["omega_original_nn_next"][idx].item()),
                        fx_in_domain=bool(chunk["fx_nn_in_domain"][idx].item()),
                        reason=(
                            "saturated_nn_omega_original_decrease_violation"
                            if bool((chunk["delta_omega_original"][idx] > -eps).item())
                            else "saturated_nn_omega_original_next_state_outside_domain"
                        ),
                    )
                    w2_original_witness["controller_branch"] = "saturated_nn"

    # Rescale the upper omega violation threshold inside [w1_verify, w2_raw].
    # This keeps the verified band ordered even when omega values are negative
    # or when the sampled w2 violation is very close to w1.
    w2_verify = (
        float(w1_verify) + w2_scale * (w2_raw - float(w1_verify))
        if w1_verify is not None and w2_raw is not None
        else None
    )
    w2_original_verify = (
        float(w1_original_verify) + w2_scale * (w2_original_raw - float(w1_original_verify))
        if w1_original_verify is not None and w2_original_raw is not None
        else None
    )
    shell_specs = (
        ("inside_c2", 0.0, 1.0),
        ("outside_1x_2x", 1.0, 2.0),
        ("outside_2x_5x", 2.0, 5.0),
        ("outside_5x_10x", 5.0, 10.0),
        ("outside_10x_25x", 10.0, 25.0),
        ("outside_25x_50x", 25.0, 50.0),
        ("outside_50x_100x", 50.0, 100.0),
        ("outside_100x_inf", 100.0, float("inf")),
    )
    shell_records = [
        {
            "label": label,
            "nu_over_c2_lower": float(lower),
            "nu_over_c2_upper": None if math.isinf(upper) else float(upper),
            "count": 0,
            "min_omega_cl": None,
            "min_omega_cl_witness": None,
            "min_omega_original": None,
            "min_omega_original_witness": None,
        }
        for label, lower, upper in shell_specs
    ]
    for x in grid_batches():
        chunk = evaluate_threshold_batch(x)
        nu_ratio = chunk["nu"] / float(c2_verify)
        for record, (_, lower, upper) in zip(shell_records, shell_specs):
            in_shell = nu_ratio >= float(lower)
            if math.isfinite(upper):
                in_shell = in_shell & (nu_ratio < float(upper))
            if not in_shell.any():
                continue
            record["count"] = int(record["count"]) + int(in_shell.sum().item())
            candidate_values = chunk["omega"].masked_fill(~in_shell, float("inf"))
            idx = int(torch.argmin(candidate_values).item())
            value = float(chunk["omega"][idx].item())
            if record["min_omega_cl"] is None or value < float(record["min_omega_cl"]):
                witness = threshold_witness_payload(
                    chunk["x"][idx],
                    chunk["fx_lqr"][idx],
                    nu=float(chunk["nu"][idx].item()),
                    nu_next=float(chunk["nu_lqr_next"][idx].item()),
                    omega_cl=float(chunk["omega"][idx].item()),
                    omega_cl_next=float(chunk["omega_lqr_next"][idx].item()),
                    omega_original=(
                        None
                        if chunk["omega_original"] is None
                        else float(chunk["omega_original"][idx].item())
                    ),
                    omega_original_next=(
                        None
                        if chunk["omega_original_lqr_next"] is None
                        else float(chunk["omega_original_lqr_next"][idx].item())
                    ),
                    fx_in_domain=bool(chunk["fx_lqr_in_domain"][idx].item()),
                    reason=f"shell_min_omega_cl_{record['label']}",
                )
                witness["nu_over_c2_verify"] = float(nu_ratio[idx].item())
                witness["controller_branch"] = "clamped_lqr_shell_context"
                record["min_omega_cl"] = value
                record["min_omega_cl_witness"] = witness
            if chunk["omega_original"] is not None:
                candidate_values = chunk["omega_original"].masked_fill(~in_shell, float("inf"))
                idx = int(torch.argmin(candidate_values).item())
                value = float(chunk["omega_original"][idx].item())
                if (
                    record["min_omega_original"] is None
                    or value < float(record["min_omega_original"])
                ):
                    witness = threshold_witness_payload(
                        chunk["x"][idx],
                        chunk["fx_lqr"][idx],
                        nu=float(chunk["nu"][idx].item()),
                        nu_next=float(chunk["nu_lqr_next"][idx].item()),
                        omega_cl=float(chunk["omega"][idx].item()),
                        omega_cl_next=float(chunk["omega_lqr_next"][idx].item()),
                        omega_original=float(chunk["omega_original"][idx].item()),
                        omega_original_next=float(chunk["omega_original_lqr_next"][idx].item()),
                        fx_in_domain=bool(chunk["fx_lqr_in_domain"][idx].item()),
                        reason=f"shell_min_omega_original_{record['label']}",
                    )
                    witness["nu_over_c2_verify"] = float(nu_ratio[idx].item())
                    witness["controller_branch"] = "clamped_lqr_shell_context"
                    record["min_omega_original"] = value
                    record["min_omega_original_witness"] = witness

    band_good_count = 0
    band_bad_count = 0
    band_bad_witness = None
    if w1_verify is not None and w2_verify is not None:
        for x in grid_batches():
            chunk = evaluate_threshold_batch(x)
            in_band = (chunk["omega"] >= float(w1_verify)) & (chunk["omega"] <= float(w2_verify))
            good = in_band & (chunk["delta_omega_nn"] <= -eps) & chunk["fx_nn_in_domain"]
            bad = in_band & ~((chunk["delta_omega_nn"] <= -eps) & chunk["fx_nn_in_domain"])
            band_good_count += int(good.sum().item())
            band_bad_count += int(bad.sum().item())
            if band_bad_witness is None and bad.any():
                candidate_values = chunk["omega"].masked_fill(~bad, float("inf"))
                idx = int(torch.argmin(candidate_values).item())
                band_bad_witness = threshold_witness_payload(
                    chunk["x"][idx],
                    chunk["fx_nn"][idx],
                    nu=float(chunk["nu"][idx].item()),
                    nu_next=float(chunk["nu_nn_next"][idx].item()),
                    omega_cl=float(chunk["omega"][idx].item()),
                    omega_cl_next=float(chunk["omega_nn_next"][idx].item()),
                    omega_original=(
                        None
                        if chunk["omega_original"] is None
                        else float(chunk["omega_original"][idx].item())
                    ),
                    omega_original_next=(
                        None
                        if chunk["omega_original_nn_next"] is None
                        else float(chunk["omega_original_nn_next"][idx].item())
                    ),
                    fx_in_domain=bool(chunk["fx_nn_in_domain"][idx].item()),
                    reason=(
                        "band_saturated_nn_omega_cl_decrease_violation"
                        if bool((chunk["delta_omega_nn"][idx] > -eps).item())
                        else "band_saturated_nn_next_state_outside_domain"
                    ),
                )
                band_bad_witness["controller_branch"] = "saturated_nn"

    return SampledThresholdSearchResult(
        n_samples=n_grid_points,
        domain=domain,
        eps=eps,
        scale=scale,
        c2_scale=c2_scale,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        c1_raw=float(c1_raw),
        c_s=float(c_s),
        c2_cap=c2_cap_value,
        c2_raw=c2_raw,
        c2_verify=c2_verify,
        w1_raw=w1_raw,
        w1_verify=w1_verify,
        w2_raw=w2_raw,
        w2_verify=w2_verify,
        c2_witness=c2_witness,
        w1_witness=w1_witness,
        w2_witness=w2_witness,
        w1_original_raw=w1_original_raw,
        w1_original_verify=w1_original_verify,
        w2_original_raw=w2_original_raw,
        w2_original_verify=w2_original_verify,
        w1_original_witness=w1_original_witness,
        w2_original_witness=w2_original_witness,
        shell_diagnostics=tuple(shell_records),
        band_good_count=band_good_count,
        band_bad_count=band_bad_count,
        band_bad_witness=band_bad_witness,
        notes=(
            "All thresholds are deterministic grid approximations to infima.",
            f"threshold_grid_points_per_dim={points_per_dim}",
            f"threshold_grid_requested_points={requested_points}",
            *c2_notes,
            "c2_raw is the first clamped-LQR nu decrease/domain violation; no c2 cap is used unless explicitly supplied to the helper.",
            "c2_verify = c1 + c2_scale*(c2_raw-c1).",
            "w1_raw is the minimum omega_cl(x) outside nu(x)<=c2_verify.",
            "w1_verify is rescaled from w1_raw toward zero.",
            "w2_raw is the first saturated-NN omega_cl decrease/domain violation above w1_verify.",
            "w2_verify is rescaled inside the interval [w1_verify, w2_raw].",
            "c2, w1, and w2 each use independent scale knobs.",
            "Next-state domain membership is included in both clamped-LQR and saturated-NN bad-point searches.",
            "The band check is grid-based; formal verification is still required.",
        ),
    )


def resolve_plot_thresholds(system: SystemSpec, thresholds: Optional[Dict[str, float]]) -> Tuple[float, float, float, float]:
    if system.verification is None:
        raise ValueError(f"{system.name}: verification config is required for threshold diagnostics.")
    vcfg = system.verification
    values = {
        "c1": thresholds.get("c1") if thresholds else vcfg.c1,
        "c2": thresholds.get("c2") if thresholds else vcfg.c2,
        "w1": thresholds.get("w1") if thresholds else vcfg.w1,
        "w2": thresholds.get("w2") if thresholds else vcfg.w2,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(
            f"{system.name}: threshold diagnostics need {missing}. "
            "Run search_thresholds first so verification_thresholds_rescaled.json exists."
        )
    return float(values["c1"]), float(values["c2"]), float(values["w1"]), float(values["w2"])


@torch.no_grad()
def save_threshold_diagnostic_plot_2d(
    run: Optional[PipelineRun],
    artifact_dir: Path,
    *,
    system: SystemSpec,
    pi: PiNet,
    omega_cl: OmegaNet,
    p_matrix: torch.Tensor,
    thresholds: Optional[Dict[str, float]] = None,
) -> None:
    """Plot the manually supplied verifier thresholds for a 2D system.

    Current assumption:
      - `c2` is the clamped-LQR nu threshold from the violation search.
      - `w1` bridges omega_cl below w1 into nu below c2.
      - `w2` is the saturated-NN omega_cl threshold from the same search.
    """
    if system.verification is None:
        raise ValueError(f"{system.name}: verification thresholds c2/w1/w2 are required.")
    if system.n_dim != 2:
        raise NotImplementedError(
            "threshold diagnostic plotting is currently 2D-only; add an explicit "
            "slice specification before using it for higher-dimensional systems."
        )

    vcfg = system.verification
    c1, c2, w1, w2 = resolve_plot_thresholds(system, thresholds)
    plot_dir = run.plot_dir if run is not None else artifact_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "c2": c2,
        "w1": w1,
        "w2": w2,
        "c1": c1,
        "source": "sampled_threshold_search" if thresholds else "manual_or_previous_violation_search",
        "notes": [
            "c2 is interpreted as a nu=x^T P x threshold.",
            "w1 and w2 are interpreted as omega_cl thresholds.",
            "Replace these manual values with the violation-search producer later.",
        ],
    }
    if run is not None:
        run.write_json("manual_thresholds.json", payload)
    else:
        (artifact_dir / "manual_thresholds.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid_n = int(system.training.diagnostic_plot_grid_points)
    (x0_lo, x0_hi), (x1_lo, x1_hi) = tuple(zip(system.x_lb, system.x_hi))
    x0 = torch.linspace(float(x0_lo), float(x0_hi), grid_n, device=p_matrix.device)
    x1 = torch.linspace(float(x1_lo), float(x1_hi), grid_n, device=p_matrix.device)
    x0_grid, x1_grid = torch.meshgrid(x0, x1, indexing="xy")
    points = torch.stack([x0_grid.reshape(-1), x1_grid.reshape(-1)], dim=1)

    pi.eval()
    omega_cl.eval()
    p_matrix = p_matrix.to(device=points.device, dtype=points.dtype)
    u_lqr = pi.lqr_control(points)
    next_lqr_points = system.dynamics(points, u_lqr)
    u_nn = pi.nn_control(points)
    next_nn_points = system.dynamics(points, u_nn)
    lqr_next_in_domain = points_in_bounds(next_lqr_points, tuple(tuple(v) for v in vcfg.domain)).reshape(grid_n, grid_n)
    nn_next_in_domain = points_in_bounds(next_nn_points, tuple(tuple(v) for v in vcfg.domain)).reshape(grid_n, grid_n)
    nu_tensor = nu_from_p(points, p_matrix).reshape(grid_n, grid_n)
    nu_lqr_next = nu_from_p(next_lqr_points, p_matrix).reshape(grid_n, grid_n)
    delta_nu_lqr = (nu_lqr_next - nu_tensor).detach().cpu().numpy()
    nu_values = nu_tensor.detach().cpu().numpy()
    omega_values = omega_cl.query(points).reshape(grid_n, grid_n)
    omega_next = omega_cl.query(next_nn_points).reshape(grid_n, grid_n)
    delta_values = (omega_next - omega_values).detach().cpu().numpy()
    omega_values_np = omega_values.detach().cpu().numpy()
    omega_next_np = omega_next.detach().cpu().numpy()
    u0 = u_nn[:, 0].reshape(grid_n, grid_n).detach().cpu().numpy()
    x0_np = x0_grid.detach().cpu().numpy()
    x1_np = x1_grid.detach().cpu().numpy()
    lqr_next_in_domain_np = lqr_next_in_domain.detach().cpu().numpy()
    nn_next_in_domain_np = nn_next_in_domain.detach().cpu().numpy()

    eps = float(vcfg.eps)
    c2_mask = (nu_values >= c1) & (nu_values <= c2) & (
        (delta_nu_lqr > -eps) | (~lqr_next_in_domain_np)
    )
    w1_mask = (omega_values_np <= w1) & (nu_values > c2)
    w2_mask = (omega_values_np >= w1) & ((delta_values > -eps) | (~nn_next_in_domain_np))
    band_mask = (omega_values_np >= w1) & (omega_values_np <= w2)
    band_bad_mask = band_mask & ((delta_values > -eps) | (~nn_next_in_domain_np))
    out_of_domain_mask = ~nn_next_in_domain_np

    def summarize_mask(name: str, mask: np.ndarray) -> Dict[str, object]:
        idx = np.argwhere(mask)
        summary: Dict[str, object] = {
            "count": int(idx.shape[0]),
            "fraction": float(idx.shape[0] / mask.size),
        }
        if idx.shape[0] == 0:
            return summary
        omega_at_mask = omega_values_np[mask]
        nu_at_mask = nu_values[mask]
        delta_at_mask = delta_values[mask]
        best_flat = int(np.nanargmin(omega_at_mask if name != "c2_violation" else nu_at_mask))
        row, col = idx[best_flat]
        summary.update(
            {
                "representative_x": [float(x0_np[row, col]), float(x1_np[row, col])],
                "representative_nu": float(nu_values[row, col]),
                "representative_omega_cl": float(omega_values_np[row, col]),
                "representative_delta_omega_cl": float(delta_values[row, col]),
                "representative_next_in_domain": bool(nn_next_in_domain_np[row, col]),
                "min_nu": float(np.nanmin(nu_at_mask)),
                "min_omega_cl": float(np.nanmin(omega_at_mask)),
                "max_delta_omega_cl": float(np.nanmax(delta_at_mask)),
            }
        )
        return summary

    mask_payload = {
        "thresholds": {
            "c2": c2,
            "w1": w1,
            "w2": w2,
            "eps": eps,
        },
        "grid_points": grid_n,
        "masks": {
            "c2_violation": summarize_mask("c2_violation", c2_mask),
            "w1_containment_violation": summarize_mask("w1_containment_violation", w1_mask),
            "w2_decrease_violation": summarize_mask("w2_decrease_violation", w2_mask),
            "band": summarize_mask("band", band_mask),
            "band_bad": summarize_mask("band_bad", band_bad_mask),
            "next_state_outside_domain": summarize_mask("next_state_outside_domain", out_of_domain_mask),
        },
        "field_ranges": {
            "nu": [float(np.nanmin(nu_values)), float(np.nanmax(nu_values))],
            "omega_cl": [float(np.nanmin(omega_values_np)), float(np.nanmax(omega_values_np))],
            "delta_omega_cl": [float(np.nanmin(delta_values)), float(np.nanmax(delta_values))],
            "delta_nu_lqr": [
                float(np.nanmin(delta_nu_lqr)),
                float(np.nanmax(delta_nu_lqr)),
            ],
            "u0": [float(np.nanmin(u0)), float(np.nanmax(u0))],
        },
    }

    mask_rows = []
    for name, mask in (
        ("c2_violation", c2_mask),
        ("w1_containment_violation", w1_mask),
        ("w2_decrease_violation", w2_mask),
        ("band_bad", band_bad_mask),
        ("next_state_outside_domain", out_of_domain_mask),
    ):
        idx = np.argwhere(mask)
        if idx.shape[0] == 0:
            continue
        scores = nu_values[mask] if name == "c2_violation" else omega_values_np[mask]
        order = np.argsort(scores)[: min(25, idx.shape[0])]
        for local_idx in order:
            row, col = idx[int(local_idx)]
            mask_rows.append(
                {
                    "mask": name,
                    "x0": float(x0_np[row, col]),
                    "x1": float(x1_np[row, col]),
                    "nu": float(nu_values[row, col]),
                    "omega_cl": float(omega_values_np[row, col]),
                    "omega_cl_next": float(omega_next_np[row, col]),
                    "delta_omega_cl": float(delta_values[row, col]),
                    "next_state_in_domain": bool(
                        lqr_next_in_domain_np[row, col]
                        if name == "c2_violation"
                        else nn_next_in_domain_np[row, col]
                    ),
                }
            )

    if run is not None:
        run.write_json("threshold_plot_mask_summary.json", mask_payload)
        run.write_csv("threshold_plot_mask_points.csv", mask_rows)
    else:
        (artifact_dir / "threshold_plot_mask_summary.json").write_text(
            json.dumps(mask_payload, indent=2),
            encoding="utf-8",
        )
        if mask_rows:
            import csv

            with (artifact_dir / "threshold_plot_mask_points.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(mask_rows[0]))
                writer.writeheader()
                writer.writerows(mask_rows)

    def safe_contour(ax, z_values: np.ndarray, levels: Sequence[float], **kwargs) -> None:
        finite = z_values[np.isfinite(z_values)]
        if finite.size == 0:
            return
        z_min = float(np.min(finite))
        z_max = float(np.max(finite))
        usable_raw = [float(level) for level in levels if z_min <= float(level) <= z_max]
        usable = sorted(set(usable_raw))
        if not usable:
            return
        try:
            ax.contour(x0_np, x1_np, z_values, levels=usable, **kwargs)
        except Exception:
            return

    def draw_value_contours(ax, z_values: np.ndarray) -> None:
        finite = z_values[np.isfinite(z_values)]
        if finite.size == 0:
            return
        z_min = float(np.min(finite))
        z_max = float(np.max(finite))
        if not z_max > z_min:
            return
        levels = np.linspace(z_min, z_max, 13)[1:-1]
        ax.contour(
            x0_np,
            x1_np,
            z_values,
            levels=levels,
            colors="white",
            linewidths=1.15,
            alpha=0.8,
        )
        contours = ax.contour(
            x0_np,
            x1_np,
            z_values,
            levels=levels,
            colors="black",
            linewidths=0.45,
            alpha=0.9,
        )
        try:
            ax.clabel(contours, inline=True, fontsize=6, fmt="%.2g")
        except Exception:
            pass

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    cf0 = axes[0].contourf(x0_np, x1_np, omega_values_np, levels=50, cmap="plasma")
    fig.colorbar(cf0, ax=axes[0], label="omega_cl")
    draw_value_contours(axes[0], omega_values_np)
    safe_contour(axes[0], nu_values, [c2], colors="magenta", linewidths=2)
    safe_contour(
        axes[0],
        omega_values_np,
        [w1, w2],
        colors="cyan",
        linestyles=["--", "--"],
        linewidths=1.8,
    )
    axes[0].scatter(x0_np[w1_mask], x1_np[w1_mask], s=3, c="white", alpha=0.2, linewidths=0)
    axes[0].set_title("omega_cl with thresholds")

    vmax = float(np.nanmax(np.abs(delta_values)))
    vmax = max(vmax, 1e-8)
    cf1 = axes[1].contourf(x0_np, x1_np, delta_values, levels=50, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    fig.colorbar(cf1, ax=axes[1], label="Delta omega_cl")
    safe_contour(axes[1], delta_values, [0.0], colors="black", linewidths=1.4)
    draw_value_contours(axes[1], omega_values_np)
    safe_contour(axes[1], nu_values, [c2], colors="magenta", linewidths=2)
    safe_contour(
        axes[1],
        omega_values_np,
        [w1, w2],
        colors="cyan",
        linestyles=["--", "--"],
        linewidths=1.8,
    )
    axes[1].scatter(x0_np[c2_mask], x1_np[c2_mask], s=3, c="yellow", alpha=0.3, linewidths=0)
    axes[1].scatter(x0_np[w2_mask], x1_np[w2_mask], s=3, c="black", alpha=0.25, linewidths=0)
    axes[1].set_title("closed-loop decrease")

    cf2 = axes[2].contourf(x0_np, x1_np, u0, levels=50, cmap="viridis")
    fig.colorbar(cf2, ax=axes[2], label="u0")
    safe_contour(axes[2], nu_values, [c2], colors="magenta", linewidths=2)
    safe_contour(
        axes[2],
        omega_values_np,
        [w1, w2],
        colors="cyan",
        linestyles=["--", "--"],
        linewidths=1.8,
    )
    axes[2].scatter(x0_np[out_of_domain_mask], x1_np[out_of_domain_mask], s=3, c="red", alpha=0.25, linewidths=0)
    axes[2].set_title("NN branch control")

    labels = system.state_labels or tuple(f"x{i}" for i in range(system.n_dim))
    for ax in axes:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.grid(True, alpha=0.2)

    fig.suptitle("Manual Threshold Diagnostic")
    fig.savefig(plot_dir / "threshold_diagnostic.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def prepare_value_heatmap_optimal_control_grid_2d(
    run: Optional[PipelineRun],
    artifact_dir: Path,
    *,
    system: SystemSpec,
    omega_orig: OmegaNet,
    p_matrix: torch.Tensor,
    k_matrix: Optional[torch.Tensor] = None,
    axis_pair: Tuple[int, int] = (0, 1),
) -> HeatmapOptimalControlGrid:
    """Compute and persist the full optimal-control grid used by heatmaps.

    This mirrors the diagnostic heatmap slice exactly, but writes the complete
    per-state argmin data before plotting so unusual optimal-control violations
    can be inspected without reverse-engineering the image.
    """
    if system.verification is None:
        raise ValueError(f"{system.name}: verification thresholds c2/w1/w2 are required.")
    if system.n_dim < 2:
        raise NotImplementedError("value heatmap diagnostics require at least two state dimensions.")

    axis_pair = normalize_axis_pair(system, axis_pair)
    vcfg = system.verification
    grid_n = int(system.training.diagnostic_plot_grid_points)
    x_axis, y_axis = axis_pair
    x0_lo, x0_hi = system.x_lb[x_axis], system.x_hi[x_axis]
    x1_lo, x1_hi = system.x_lb[y_axis], system.x_hi[y_axis]
    x0 = torch.linspace(float(x0_lo), float(x0_hi), grid_n, device=p_matrix.device)
    x1 = torch.linspace(float(x1_lo), float(x1_hi), grid_n, device=p_matrix.device)
    x0_grid, x1_grid = torch.meshgrid(x0, x1, indexing="xy")
    points = torch.zeros(
        x0_grid.numel(),
        system.n_dim,
        device=p_matrix.device,
        dtype=torch.float32,
    )
    points[:, x_axis] = x0_grid.reshape(-1)
    points[:, y_axis] = x1_grid.reshape(-1)

    omega_orig.eval()
    control_grid = make_control_grid(
        system,
        int(system.training.control_grid_points),
        device=points.device,
        dtype=points.dtype,
    )
    labels = compute_optimal_next_omega_grid_labels(
        system,
        omega_orig,
        points,
        control_grid,
        state_batch_size=max(1, int(system.training.control_grid_batch_size)),
        mask_out_of_domain_successors=bool(system.training.policy_label_mask_out_of_domain_successors),
        successor_domain=tuple(tuple(v) for v in vcfg.domain),
    )
    opt_next_points = system.dynamics(points, labels.u_opt)
    opt_next_in_domain = points_in_bounds(opt_next_points, tuple(tuple(v) for v in vcfg.domain))
    omega_orig_values = omega_orig.query(points).reshape(-1)
    delta_orig_opt = labels.omega_opt_next.reshape(-1) - omega_orig_values

    grid = HeatmapOptimalControlGrid(
        grid_points=grid_n,
        axis_pair=axis_pair,
        x0_grid=x0_grid,
        x1_grid=x1_grid,
        points=points.detach(),
        labels=labels,
        opt_next_points=opt_next_points.detach(),
        opt_next_in_domain=opt_next_in_domain.detach(),
        omega_orig_values=omega_orig_values.detach(),
        delta_orig_opt=delta_orig_opt.detach(),
        control_grid_points_per_dim=int(system.training.control_grid_points),
        control_grid_total_points=int(control_grid.shape[0]),
    )

    eps = float(vcfg.eps)
    delta_np = grid.delta_orig_opt.detach().cpu().numpy()
    in_domain_np = grid.opt_next_in_domain.detach().cpu().numpy()
    violation_mask = (delta_np > -eps) | (~in_domain_np)
    summary = {
        "grid_points": grid.grid_points,
        "total_state_points": int(points.shape[0]),
        "slice": {
            "axes": [int(x_axis), int(y_axis)],
            "fixed_coordinates": {
                str(i): 0.0 for i in range(system.n_dim) if i not in axis_pair
            },
        },
        "control_grid_points_per_dim": grid.control_grid_points_per_dim,
        "control_grid_total_points": grid.control_grid_total_points,
        "optimal_original_omega_violation_mask": {
            "count": int(violation_mask.sum()),
            "fraction": float(violation_mask.mean()),
            "condition": (
                "min_u_grid omega_orig(f(x,u))-omega_orig(x) > -eps "
                "or f(x,u_opt_grid) outside X_v"
            ),
        },
        "field_ranges": {
            "omega_orig": [
                float(torch.min(grid.omega_orig_values).detach().cpu()),
                float(torch.max(grid.omega_orig_values).detach().cpu()),
            ],
            "omega_orig_opt_next": [
                float(torch.min(grid.labels.omega_opt_next).detach().cpu()),
                float(torch.max(grid.labels.omega_opt_next).detach().cpu()),
            ],
            "delta_omega_orig_opt": [
                float(torch.min(grid.delta_orig_opt).detach().cpu()),
                float(torch.max(grid.delta_orig_opt).detach().cpu()),
            ],
        },
    }

    points_np = grid.points.detach().cpu().numpy()
    u_np = grid.labels.u_opt.detach().cpu().numpy()
    next_np = grid.opt_next_points.detach().cpu().numpy()
    omega_np = grid.omega_orig_values.detach().cpu().numpy()
    omega_next_np = grid.labels.omega_opt_next.detach().cpu().numpy()
    rows = []
    for idx in range(points_np.shape[0]):
        row = {"grid_index": idx}
        row.update({f"x{dim}": float(points_np[idx, dim]) for dim in range(system.n_dim)})
        row.update({f"u_opt{dim}": float(u_np[idx, dim]) for dim in range(system.u_dim)})
        row.update({f"f_opt_x{dim}": float(next_np[idx, dim]) for dim in range(system.n_dim)})
        row["omega_orig"] = float(omega_np[idx])
        row["omega_orig_opt_next"] = float(omega_next_np[idx])
        row["delta_omega_orig_opt"] = float(delta_np[idx])
        row["f_opt_x_in_domain"] = bool(in_domain_np[idx])
        row["optimal_violation"] = bool(violation_mask[idx])
        rows.append(row)

    suffix = heatmap_pair_suffix(axis_pair)
    summary_name = f"value_heatmap_optimal_controls_summary_{suffix}.json"
    csv_name = f"value_heatmap_optimal_controls_{suffix}.csv"
    if run is not None:
        run.write_json(summary_name, summary)
        run.write_csv(csv_name, rows)
        if axis_pair == (0, 1):
            run.write_json("value_heatmap_optimal_controls_summary.json", summary)
            run.write_csv("value_heatmap_optimal_controls.csv", rows)
    else:
        (artifact_dir / summary_name).write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        if rows:
            columns = list(rows[0].keys())
            lines = [",".join(columns)]
            for row in rows:
                lines.append(",".join(str(row.get(column, "")) for column in columns))
            (artifact_dir / csv_name).write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )
        else:
            (artifact_dir / csv_name).write_text("", encoding="utf-8")
        if axis_pair == (0, 1):
            (artifact_dir / "value_heatmap_optimal_controls_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            if rows:
                columns = list(rows[0].keys())
                lines = [",".join(columns)]
                for row in rows:
                    lines.append(",".join(str(row.get(column, "")) for column in columns))
                (artifact_dir / "value_heatmap_optimal_controls.csv").write_text(
                    "\n".join(lines) + "\n",
                    encoding="utf-8",
                )
            else:
                (artifact_dir / "value_heatmap_optimal_controls.csv").write_text("", encoding="utf-8")

    return grid


@torch.no_grad()
def save_value_heatmap_decrease_diagnostic_2d(
    run: Optional[PipelineRun],
    artifact_dir: Path,
    *,
    system: SystemSpec,
    pi: PiNet,
    omega_orig: OmegaNet,
    omega_cl: OmegaNet,
    p_matrix: torch.Tensor,
    c_analytic: float,
    c_sigmoid_center: Optional[float] = None,
    c_s: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
    optimal_grid: Optional[HeatmapOptimalControlGrid] = None,
    k_matrix: Optional[torch.Tensor] = None,
    axis_pair: Tuple[int, int] = (0, 1),
) -> None:
    """Side-by-side omega heatmaps with saturated-NN decrease violations.

    The mask is computed from the closed-loop value under the learned saturated NN controller:

        omega_cl(f(x, pi_nn(x))) - omega_cl(x) > -eps

    or next state leaves X_v. The mask is overlaid on both heatmaps so the same
    problematic region can be compared against original omega and omega_cl.
    """
    if system.verification is None:
        raise ValueError(f"{system.name}: verification thresholds c2/w1/w2 are required.")
    if system.n_dim < 2:
        raise NotImplementedError("value heatmap diagnostics require at least two state dimensions.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    vcfg = system.verification
    _, c2, w1, w2 = resolve_plot_thresholds(system, thresholds)
    eps = float(vcfg.eps)
    plot_dir = run.plot_dir if run is not None else artifact_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    axis_pair = normalize_axis_pair(system, axis_pair)

    if optimal_grid is None:
        optimal_grid = prepare_value_heatmap_optimal_control_grid_2d(
            run,
            artifact_dir,
            system=system,
            omega_orig=omega_orig,
            p_matrix=p_matrix,
            k_matrix=k_matrix,
            axis_pair=axis_pair,
        )
    elif normalize_axis_pair(system, optimal_grid.axis_pair) != axis_pair:
        raise ValueError(
            f"{system.name}: optimal_grid axis_pair {optimal_grid.axis_pair} "
            f"does not match requested axis_pair {axis_pair}."
        )

    grid_n = int(optimal_grid.grid_points)
    x0_grid = optimal_grid.x0_grid
    x1_grid = optimal_grid.x1_grid
    points = optimal_grid.points.to(device=p_matrix.device, dtype=torch.float32)

    pi.eval()
    omega_orig.eval()
    omega_cl.eval()
    p_matrix = p_matrix.to(device=points.device, dtype=points.dtype)
    u_nn = pi.nn_control(points)
    next_points = system.dynamics(points, u_nn)
    next_in_domain = points_in_bounds(next_points, tuple(tuple(v) for v in vcfg.domain)).reshape(grid_n, grid_n)
    nu_values = nu_from_p(points, p_matrix).reshape(grid_n, grid_n).detach().cpu().numpy()
    omega_orig_np = optimal_grid.omega_orig_values.reshape(grid_n, grid_n).detach().cpu().numpy()
    omega_orig_next = omega_orig.query(next_points).reshape(grid_n, grid_n)
    omega_cl_values = omega_cl.query(points).reshape(grid_n, grid_n)
    omega_cl_next = omega_cl.query(next_points).reshape(grid_n, grid_n)
    omega_cl_np = omega_cl_values.detach().cpu().numpy()
    delta_orig_pi_np = (omega_orig_next - optimal_grid.omega_orig_values.reshape(grid_n, grid_n)).detach().cpu().numpy()
    delta_cl_np = (omega_cl_next - omega_cl_values).detach().cpu().numpy()
    omega_orig_opt_next_np = optimal_grid.labels.omega_opt_next.reshape(grid_n, grid_n).detach().cpu().numpy()
    delta_orig_opt_np = omega_orig_opt_next_np - omega_orig_np
    next_in_domain_np = next_in_domain.detach().cpu().numpy()
    opt_next_in_domain_np = optimal_grid.opt_next_in_domain.reshape(grid_n, grid_n).detach().cpu().numpy()
    orig_pi_violation_mask = (delta_orig_pi_np > -eps) | (~next_in_domain_np)
    cl_pi_violation_mask = (delta_cl_np > -eps) | (~next_in_domain_np)
    opt_violation_mask = (delta_orig_opt_np > -eps) | (~opt_next_in_domain_np)
    omega_cl_dos_mask = omega_cl_np < w2
    omega_cl_outside_dos_mask = ~omega_cl_dos_mask
    omega_orig_dos_mask = omega_orig_np < w2
    omega_orig_outside_dos_mask = ~omega_orig_dos_mask
    nu_c2_mask = nu_values < c2
    x0_np = x0_grid.detach().cpu().numpy()
    x1_np = x1_grid.detach().cpu().numpy()

    def masked_count(mask: np.ndarray, region: np.ndarray) -> Dict[str, object]:
        total = int(region.sum())
        count = int((mask & region).sum())
        return {
            "region_points": total,
            "count": count,
            "fraction": float(count / total) if total else None,
        }

    mask_payload = {
        "thresholds": {
            "c_analytic": float(c_analytic),
            "sample_cap": float(c_sigmoid_center) if c_sigmoid_center is not None else None,
            "C_S": float(c_s) if c_s is not None else None,
            "c2": c2,
            "w1": w1,
            "w2": w2,
            "eps": eps,
        },
        "grid_points": grid_n,
        "slice": {
            "axes": [int(axis_pair[0]), int(axis_pair[1])],
            "fixed_coordinates": {
                str(i): 0.0 for i in range(system.n_dim) if i not in axis_pair
            },
        },
        "omega_orig_under_pi_nn_violation_mask": {
            "count": int(orig_pi_violation_mask.sum()),
            "fraction": float(orig_pi_violation_mask.mean()),
            "condition": "omega_orig(f(x,pi_nn(x)))-omega_orig(x) > -eps or f(x,pi_nn(x)) outside X_v",
            "breakdown": {
                "inside_omega_cl_lt_w2": masked_count(orig_pi_violation_mask, omega_cl_dos_mask),
                "outside_omega_cl_lt_w2": masked_count(orig_pi_violation_mask, omega_cl_outside_dos_mask),
                "inside_omega_orig_lt_w2": masked_count(orig_pi_violation_mask, omega_orig_dos_mask),
                "outside_omega_orig_lt_w2": masked_count(orig_pi_violation_mask, omega_orig_outside_dos_mask),
                "inside_nu_lt_c2": masked_count(orig_pi_violation_mask, nu_c2_mask),
                "outside_nu_lt_c2": masked_count(orig_pi_violation_mask, ~nu_c2_mask),
            },
        },
        "omega_cl_under_pi_nn_violation_mask": {
            "count": int(cl_pi_violation_mask.sum()),
            "fraction": float(cl_pi_violation_mask.mean()),
            "condition": "omega_cl(f(x,pi_nn(x)))-omega_cl(x) > -eps or f(x,pi_nn(x)) outside X_v",
            "breakdown": {
                "inside_omega_cl_lt_w2": masked_count(cl_pi_violation_mask, omega_cl_dos_mask),
                "outside_omega_cl_lt_w2": masked_count(cl_pi_violation_mask, omega_cl_outside_dos_mask),
                "inside_omega_orig_lt_w2": masked_count(cl_pi_violation_mask, omega_orig_dos_mask),
                "outside_omega_orig_lt_w2": masked_count(cl_pi_violation_mask, omega_orig_outside_dos_mask),
                "inside_nu_lt_c2": masked_count(cl_pi_violation_mask, nu_c2_mask),
                "outside_nu_lt_c2": masked_count(cl_pi_violation_mask, ~nu_c2_mask),
            },
        },
        # Backward-compatible alias for older scripts that read this key.
        "violation_mask": {
            "count": int(cl_pi_violation_mask.sum()),
            "fraction": float(cl_pi_violation_mask.mean()),
            "condition": "omega_cl(f(x,pi_nn(x)))-omega_cl(x) > -eps or f(x,pi_nn(x)) outside X_v",
        },
        "optimal_original_omega_violation_mask": {
            "count": int(opt_violation_mask.sum()),
            "fraction": float(opt_violation_mask.mean()),
            "condition": (
                "over the full diagnostic slice: min_u_grid omega_orig(f(x,u))-omega_orig(x) > -eps "
                "or f(x,u_opt_grid) outside X_v"
            ),
            "control_grid_points_per_dim": int(optimal_grid.control_grid_points_per_dim),
            "control_grid_total_points": int(optimal_grid.control_grid_total_points),
            "breakdown": {
                "inside_omega_cl_lt_w2": masked_count(opt_violation_mask, omega_cl_dos_mask),
                "outside_omega_cl_lt_w2": masked_count(opt_violation_mask, omega_cl_outside_dos_mask),
                "inside_omega_orig_lt_w2": masked_count(opt_violation_mask, omega_orig_dos_mask),
                "outside_omega_orig_lt_w2": masked_count(opt_violation_mask, omega_orig_outside_dos_mask),
                "inside_nu_lt_c2": masked_count(opt_violation_mask, nu_c2_mask),
                "outside_nu_lt_c2": masked_count(opt_violation_mask, ~nu_c2_mask),
                "delta_only": int(((delta_orig_opt_np > -eps) & opt_next_in_domain_np).sum()),
                "out_of_domain_only": int(((delta_orig_opt_np <= -eps) & (~opt_next_in_domain_np)).sum()),
                "delta_and_out_of_domain": int(((delta_orig_opt_np > -eps) & (~opt_next_in_domain_np)).sum()),
            },
        },
        "field_ranges": {
            "nu": [float(np.nanmin(nu_values)), float(np.nanmax(nu_values))],
            "omega_orig": [float(np.nanmin(omega_orig_np)), float(np.nanmax(omega_orig_np))],
            "omega_cl": [float(np.nanmin(omega_cl_np)), float(np.nanmax(omega_cl_np))],
            "delta_omega_orig_pi_nn": [float(np.nanmin(delta_orig_pi_np)), float(np.nanmax(delta_orig_pi_np))],
            "delta_omega_cl": [float(np.nanmin(delta_cl_np)), float(np.nanmax(delta_cl_np))],
            "delta_omega_orig_opt": [
                float(np.nanmin(delta_orig_opt_np)),
                float(np.nanmax(delta_orig_opt_np)),
            ],
        },
    }
    suffix = heatmap_pair_suffix(axis_pair)
    mask_summary_name = f"value_heatmap_decrease_mask_summary_{suffix}.json"
    if run is not None:
        run.write_json(mask_summary_name, mask_payload)
        if axis_pair == (0, 1):
            run.write_json("value_heatmap_decrease_mask_summary.json", mask_payload)
    else:
        (artifact_dir / mask_summary_name).write_text(
            json.dumps(mask_payload, indent=2),
            encoding="utf-8",
        )
        if axis_pair == (0, 1):
            (artifact_dir / "value_heatmap_decrease_mask_summary.json").write_text(
                json.dumps(mask_payload, indent=2),
                encoding="utf-8",
            )

    def safe_contour(ax, z_values: np.ndarray, levels: Sequence[float], **kwargs) -> None:
        finite = z_values[np.isfinite(z_values)]
        if finite.size == 0:
            return
        z_min = float(np.min(finite))
        z_max = float(np.max(finite))
        usable = sorted(set(float(level) for level in levels if z_min <= float(level) <= z_max))
        if not usable:
            return
        try:
            ax.contour(x0_np, x1_np, z_values, levels=usable, **kwargs)
        except Exception:
            return

    def draw_value_contours(ax, z_values: np.ndarray) -> None:
        finite = z_values[np.isfinite(z_values)]
        if finite.size == 0:
            return
        z_min = float(np.min(finite))
        z_max = float(np.max(finite))
        if not z_max > z_min:
            return
        levels = np.linspace(z_min, z_max, 15)[1:-1]
        ax.contour(
            x0_np,
            x1_np,
            z_values,
            levels=levels,
            colors="white",
            linewidths=1.2,
            alpha=0.85,
        )
        contours = ax.contour(
            x0_np,
            x1_np,
            z_values,
            levels=levels,
            colors="black",
            linewidths=0.5,
            alpha=0.95,
        )
        try:
            ax.clabel(contours, inline=True, fontsize=6, fmt="%.2g")
        except Exception:
            pass

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), constrained_layout=True)
    fields = (
        (axes[0], omega_orig_np, orig_pi_violation_mask, "white", "omega_orig under saturated-NN violations"),
        (axes[1], omega_cl_np, cl_pi_violation_mask, "white", "omega_cl under saturated-NN violations"),
        (axes[2], omega_orig_np, opt_violation_mask, "white", "omega_orig under optimal-label violations"),
    )
    for ax, values, mask, mask_color, title in fields:
        cf = ax.contourf(x0_np, x1_np, values, levels=80, cmap="plasma", alpha=0.88)
        fig.colorbar(cf, ax=ax, label=title)
        ax.contourf(
            x0_np,
            x1_np,
            mask.astype(float),
            levels=[0.5, 1.5],
            colors=[mask_color],
            alpha=0.50,
        )
        draw_value_contours(ax, values)
        safe_contour(ax, nu_values, [float(c_analytic)], colors="white", linewidths=2.0)
        if c_sigmoid_center is not None:
            safe_contour(ax, nu_values, [float(c_sigmoid_center)], colors="orange", linewidths=2.0)
        if c_s is not None:
            safe_contour(ax, nu_values, [float(c_s)], colors="red", linewidths=2.0)
        safe_contour(ax, nu_values, [c2], colors="magenta", linewidths=1.8)
        safe_contour(ax, values, [w1], colors="cyan", linestyles=["--"], linewidths=1.8)
        safe_contour(ax, values, [w2], colors="lime", linestyles=["--"], linewidths=1.8)
        ax.set_title(title)
        ax.grid(True, alpha=0.2)

    labels = system.state_labels or tuple(f"x{i}" for i in range(system.n_dim))
    for ax in axes:
        ax.set_xlabel(labels[axis_pair[0]])
        ax.set_ylabel(labels[axis_pair[1]])

    legend_handles = [
        Line2D([0], [0], color="white", lw=2.0, label="nu=c_analytic"),
        Line2D([0], [0], color="magenta", lw=1.8, label="nu=c2"),
        Line2D([0], [0], color="cyan", lw=1.8, linestyle="--", label="omega=w1"),
        Line2D([0], [0], color="lime", lw=1.8, linestyle="--", label="omega=w2"),
        Line2D([0], [0], color="black", lw=0.8, label="value level curves"),
        Line2D([0], [0], color="white", lw=6, alpha=0.5, label="saturated-NN violation"),
        Line2D([0], [0], color="white", lw=6, alpha=0.5, label="optimal-label violation"),
    ]
    if c_s is not None:
        legend_handles.insert(1, Line2D([0], [0], color="red", lw=2.0, label="nu=C_s cap"))
    if c_sigmoid_center is not None:
        legend_handles.insert(1, Line2D([0], [0], color="orange", lw=2.0, label="nu=sample cap"))
    axes[2].legend(handles=legend_handles, loc="upper right", framealpha=0.9)
    fig.suptitle(
        "Value Heatmaps and Decrease-Violation Masks "
        f"({labels[axis_pair[0]]}, {labels[axis_pair[1]]})"
    )
    fig.savefig(plot_dir / f"value_heatmap_decrease_diagnostic_{suffix}.png", dpi=180)
    if axis_pair == (0, 1):
        fig.savefig(plot_dir / "value_heatmap_decrease_diagnostic.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def save_pinet_control_slice_diagnostic_2d(
    run: Optional[PipelineRun],
    artifact_dir: Path,
    *,
    system: SystemSpec,
    pi: PiNet,
    p_matrix: torch.Tensor,
    c_analytic: float,
    c_sigmoid_center: Optional[float] = None,
    c_s: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
    optimal_grid: Optional[HeatmapOptimalControlGrid] = None,
    axis_pair: Tuple[int, int] = (0, 1),
) -> None:
    """Plot the saturated-NN PiNet branch beside grid-search optimal controls."""
    if system.verification is None:
        raise ValueError(f"{system.name}: verification domain is required for PiNet control diagnostics.")
    if system.n_dim < 2:
        raise NotImplementedError("PiNet control diagnostics require at least two state dimensions.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    vcfg = system.verification
    if thresholds is not None or system.verification.c2 is not None:
        _, c2, _, _ = resolve_plot_thresholds(system, thresholds)
    else:
        c2 = float(c_s) if c_s is not None else float(c_analytic)
    plot_dir = run.plot_dir if run is not None else artifact_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    axis_pair = normalize_axis_pair(system, axis_pair)
    suffix = heatmap_pair_suffix(axis_pair)

    if optimal_grid is None:
        raise ValueError(
            f"{system.name}: PiNet control diagnostics need optimal_grid when plotting "
            "side-by-side optimal controls."
        )
    if normalize_axis_pair(system, optimal_grid.axis_pair) != axis_pair:
        raise ValueError(
            f"{system.name}: optimal_grid axis_pair {optimal_grid.axis_pair} "
            f"does not match requested axis_pair {axis_pair}."
        )
    grid_n = int(optimal_grid.grid_points)
    x0_grid = optimal_grid.x0_grid
    x1_grid = optimal_grid.x1_grid
    points = optimal_grid.points.to(device=p_matrix.device, dtype=torch.float32)
    u_opt = optimal_grid.labels.u_opt.to(device=p_matrix.device, dtype=torch.float32)

    pi.eval()
    p_matrix = p_matrix.to(device=points.device, dtype=points.dtype)
    controls = pi.nn_control(points).reshape(grid_n, grid_n, system.u_dim)
    u_opt_values = u_opt.reshape(grid_n, grid_n, system.u_dim)
    nu_values = nu_from_p(points, p_matrix).reshape(grid_n, grid_n)
    omega_policy_dos_level = float(system.training.policy_label_dos_sublevel)
    omega_orig_values = optimal_grid.omega_orig_values.to(device=points.device, dtype=points.dtype).reshape(grid_n, grid_n)

    x0_np = x0_grid.detach().cpu().numpy()
    x1_np = x1_grid.detach().cpu().numpy()
    controls_np = controls.detach().cpu().numpy()
    u_opt_np = u_opt_values.detach().cpu().numpy()
    nu_np = nu_values.detach().cpu().numpy()
    omega_orig_np = omega_orig_values.detach().cpu().numpy()

    summary = {
        "grid_points": grid_n,
        "slice": {
            "axes": [int(axis_pair[0]), int(axis_pair[1])],
            "fixed_coordinates": {
                str(i): 0.0 for i in range(system.n_dim) if i not in axis_pair
            },
        },
        "thresholds": {
            "c_analytic": float(c_analytic),
            "C_S": float(c_s) if c_s is not None else None,
            "c2": float(c2),
            "policy_label_omega_o_dos": omega_policy_dos_level,
        },
        "control_bounds": {
            f"u{j}": [float(system.u_lb[j]), float(system.u_hi[j])]
            for j in range(system.u_dim)
        },
        "pinet_saturated_nn": {},
        "grid_search_optimal": {},
        "pinet_minus_optimal": {},
    }
    for j in range(system.u_dim):
        values = controls_np[:, :, j]
        opt_values = u_opt_np[:, :, j]
        diff_values = values - opt_values
        summary["pinet_saturated_nn"][f"u{j}"] = {
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }
        summary["grid_search_optimal"][f"u{j}"] = {
            "min": float(np.nanmin(opt_values)),
            "max": float(np.nanmax(opt_values)),
            "mean": float(np.nanmean(opt_values)),
            "std": float(np.nanstd(opt_values)),
        }
        summary["pinet_minus_optimal"][f"u{j}"] = {
            "min": float(np.nanmin(diff_values)),
            "max": float(np.nanmax(diff_values)),
            "mean": float(np.nanmean(diff_values)),
            "std": float(np.nanstd(diff_values)),
            "rmse": float(np.sqrt(np.nanmean(diff_values * diff_values))),
        }

    summary_name = f"pinet_control_slice_summary_{suffix}.json"
    if run is not None:
        run.write_json(summary_name, summary)
        if axis_pair == (0, 1):
            run.write_json("pinet_control_slice_summary.json", summary)
    else:
        (artifact_dir / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if axis_pair == (0, 1):
            (artifact_dir / "pinet_control_slice_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )

    def safe_contour(ax, z_values: np.ndarray, levels: Sequence[float], **kwargs) -> None:
        finite = z_values[np.isfinite(z_values)]
        if finite.size == 0:
            return
        z_min = float(np.min(finite))
        z_max = float(np.max(finite))
        usable = sorted(set(float(level) for level in levels if z_min <= float(level) <= z_max))
        if not usable:
            return
        try:
            ax.contour(x0_np, x1_np, z_values, levels=usable, **kwargs)
        except Exception:
            return

    labels = system.state_labels or tuple(f"x{i}" for i in range(system.n_dim))
    fig, axes = plt.subplots(
        2,
        system.u_dim,
        figsize=(5.8 * system.u_dim, 9.4),
        squeeze=False,
        constrained_layout=True,
    )
    for j in range(system.u_dim):
        pi_values = controls_np[:, :, j]
        opt_values = u_opt_np[:, :, j]
        finite = np.concatenate(
            [
                values[np.isfinite(values)]
                for values in (pi_values, opt_values)
                if values[np.isfinite(values)].size > 0
            ]
        )
        if finite.size:
            vmax = float(np.max(np.abs(finite)))
        else:
            vmax = 1.0
        if vmax <= 0.0:
            vmax = 1.0
        for row, (row_label, values) in enumerate(
            (("PiNet saturated NN", pi_values), ("grid-search optimal", opt_values))
        ):
            ax = axes[row, j]
            cf = ax.contourf(
                x0_np,
                x1_np,
                values,
                levels=80,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                alpha=0.9,
            )
            fig.colorbar(cf, ax=ax, label=f"{row_label} u{j}")
            safe_contour(ax, nu_np, [float(c_analytic)], colors="white", linewidths=2.0)
            if c_s is not None:
                safe_contour(ax, nu_np, [float(c_s)], colors="red", linewidths=2.0)
            safe_contour(ax, nu_np, [float(c2)], colors="magenta", linewidths=1.8)
            safe_contour(
                ax,
                omega_orig_np,
                [omega_policy_dos_level],
                colors="yellow",
                linestyles=["--"],
                linewidths=1.8,
            )
            ax.set_xlabel(labels[axis_pair[0]])
            ax.set_ylabel(labels[axis_pair[1]])
            ax.set_title(f"{row_label} u{j}")
            ax.grid(True, alpha=0.2)

    legend_handles = [
        Line2D([0], [0], color="white", lw=2.0, label="nu=c_analytic"),
        Line2D([0], [0], color="magenta", lw=1.8, label="nu=c2"),
        Line2D([0], [0], color="yellow", lw=1.8, linestyle="--", label="omega_o=policy DOS"),
    ]
    if c_s is not None:
        legend_handles.insert(1, Line2D([0], [0], color="red", lw=2.0, label="nu=C_S"))
    axes[0, -1].legend(handles=legend_handles, loc="upper right", framealpha=0.9)
    fig.suptitle(f"PiNet Saturated NN vs Optimal Control Slice ({labels[axis_pair[0]]}, {labels[axis_pair[1]]})")
    fig.savefig(plot_dir / f"pinet_control_slice_{suffix}.png", dpi=180)
    if axis_pair == (0, 1):
        fig.savefig(plot_dir / "pinet_control_slice.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def save_closed_loop_difference_field_diagnostic_2d(
    run: Optional[PipelineRun],
    artifact_dir: Path,
    *,
    system: SystemSpec,
    pi: PiNet,
    omega_orig: OmegaNet,
    omega_cl: OmegaNet,
    p_matrix: torch.Tensor,
    thresholds: Optional[Dict[str, float]] = None,
    axis_pair: Tuple[int, int] = (0, 1),
) -> None:
    """Plot one-step value differences and the closed-loop displacement field."""
    if system.verification is None:
        raise ValueError(f"{system.name}: verification thresholds c2/w1/w2 are required.")
    if system.n_dim < 2:
        raise NotImplementedError("closed-loop field diagnostics require at least two state dimensions.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D

    vcfg = system.verification
    _, c2, _, w2 = resolve_plot_thresholds(system, thresholds)
    plot_dir = run.plot_dir if run is not None else artifact_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    axis_pair = normalize_axis_pair(system, axis_pair)
    grid_n = int(system.training.diagnostic_plot_grid_points)
    bounds = tuple(tuple(v) for v in vcfg.domain)
    device = p_matrix.device
    dtype = torch.float32
    x0 = torch.linspace(float(bounds[axis_pair[0]][0]), float(bounds[axis_pair[0]][1]), grid_n, device=device)
    x1 = torch.linspace(float(bounds[axis_pair[1]][0]), float(bounds[axis_pair[1]][1]), grid_n, device=device)
    x0_grid, x1_grid = torch.meshgrid(x0, x1, indexing="ij")
    points = torch.zeros(grid_n * grid_n, system.n_dim, device=device, dtype=dtype)
    points[:, axis_pair[0]] = x0_grid.reshape(-1)
    points[:, axis_pair[1]] = x1_grid.reshape(-1)

    pi.eval()
    omega_orig.eval()
    omega_cl.eval()
    p_matrix = p_matrix.to(device=device, dtype=dtype)
    u = pi.nn_control(points)
    next_points = system.dynamics(points, u)
    next_in_domain = points_in_bounds(next_points, bounds).reshape(grid_n, grid_n)
    omega_orig_values = omega_orig.query(points).reshape(grid_n, grid_n)
    omega_orig_next = omega_orig.query(next_points).reshape(grid_n, grid_n)
    omega_cl_values = omega_cl.query(points).reshape(grid_n, grid_n)
    omega_cl_next = omega_cl.query(next_points).reshape(grid_n, grid_n)
    delta_orig = omega_orig_next - omega_orig_values
    delta_cl = omega_cl_next - omega_cl_values
    displacement = (next_points - points).reshape(grid_n, grid_n, system.n_dim)
    nu_values = nu_from_p(points, p_matrix).reshape(grid_n, grid_n)

    x0_np = x0_grid.detach().cpu().numpy()
    x1_np = x1_grid.detach().cpu().numpy()
    delta_orig_np = delta_orig.detach().cpu().numpy()
    delta_cl_np = delta_cl.detach().cpu().numpy()
    omega_cl_np = omega_cl_values.detach().cpu().numpy()
    nu_np = nu_values.detach().cpu().numpy()
    dx0_np = displacement[:, :, axis_pair[0]].detach().cpu().numpy()
    dx1_np = displacement[:, :, axis_pair[1]].detach().cpu().numpy()
    next_in_domain_np = next_in_domain.detach().cpu().numpy()
    eps = float(vcfg.eps)
    orig_violation = (delta_orig_np > -eps) | (~next_in_domain_np)
    cl_violation = (delta_cl_np > -eps) | (~next_in_domain_np)

    def delta_norm(*fields: np.ndarray) -> TwoSlopeNorm:
        finite_parts = [field[np.isfinite(field)] for field in fields]
        finite = np.concatenate([part for part in finite_parts if part.size > 0])
        if finite.size == 0:
            return TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
        vmax = float(np.max(np.abs(finite)))
        if vmax <= 0.0:
            vmax = 1.0
        return TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    def safe_contour(ax, z_values: np.ndarray, levels: Sequence[float], **kwargs) -> None:
        finite = z_values[np.isfinite(z_values)]
        if finite.size == 0:
            return
        z_min = float(np.min(finite))
        z_max = float(np.max(finite))
        usable = sorted(set(float(level) for level in levels if z_min <= float(level) <= z_max))
        if not usable:
            return
        try:
            ax.contour(x0_np, x1_np, z_values, levels=usable, **kwargs)
        except Exception:
            return

    labels = system.state_labels or tuple(f"x{i}" for i in range(system.n_dim))
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), constrained_layout=True)
    norm = delta_norm(delta_orig_np, delta_cl_np)
    fields = (
        (axes[0], delta_orig_np, orig_violation, "Delta omega_orig under pi_nn"),
        (axes[1], delta_cl_np, cl_violation, "Delta omega_cl under pi_nn"),
    )
    for ax, field, mask, title in fields:
        cf = ax.contourf(x0_np, x1_np, field, levels=80, cmap="coolwarm", norm=norm)
        fig.colorbar(cf, ax=ax, label=title)
        ax.contourf(
            x0_np,
            x1_np,
            mask.astype(float),
            levels=[0.5, 1.5],
            colors=["black"],
            alpha=0.22,
        )
        safe_contour(ax, field, [0.0], colors="black", linewidths=1.4)
        safe_contour(ax, nu_np, [c2], colors="magenta", linewidths=1.8)
        safe_contour(ax, omega_cl_np, [w2], colors="lime", linestyles=["--"], linewidths=1.8)
        ax.set_title(title)
        ax.grid(True, alpha=0.2)

    skip = max(1, grid_n // 24)
    speed = np.sqrt(dx0_np ** 2 + dx1_np ** 2)
    cf = axes[2].contourf(x0_np, x1_np, speed, levels=80, cmap="viridis", alpha=0.88)
    fig.colorbar(cf, ax=axes[2], label="||f_cl(x)-x||")
    arrow_dx = dx0_np[::skip, ::skip]
    arrow_dy = dx1_np[::skip, ::skip]
    arrow_norm = np.sqrt(arrow_dx ** 2 + arrow_dy ** 2)
    arrow_dx = np.divide(arrow_dx, arrow_norm, out=np.zeros_like(arrow_dx), where=arrow_norm > 0.0)
    arrow_dy = np.divide(arrow_dy, arrow_norm, out=np.zeros_like(arrow_dy), where=arrow_norm > 0.0)
    axes[2].quiver(
        x0_np[::skip, ::skip],
        x1_np[::skip, ::skip],
        arrow_dx,
        arrow_dy,
        color="white",
        alpha=0.9,
        width=0.004,
        scale=34,
        scale_units="width",
        pivot="mid",
    )
    safe_contour(axes[2], nu_np, [c2], colors="magenta", linewidths=1.8)
    safe_contour(axes[2], omega_cl_np, [w2], colors="lime", linestyles=["--"], linewidths=1.8)
    axes[2].set_title("closed-loop direction field")
    axes[2].grid(True, alpha=0.2)

    for ax in axes:
        ax.set_xlabel(labels[axis_pair[0]])
        ax.set_ylabel(labels[axis_pair[1]])
    axes[2].legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.4, label="Delta=0"),
            Line2D([0], [0], color="magenta", lw=1.8, label="nu=c2"),
            Line2D([0], [0], color="lime", lw=1.8, linestyle="--", label="omega_cl=w2"),
            Line2D([0], [0], color="black", lw=6, alpha=0.22, label="decrease violation"),
        ],
        loc="upper right",
        framealpha=0.9,
    )
    fig.suptitle(f"Closed-Loop Difference Fields ({labels[axis_pair[0]]}, {labels[axis_pair[1]]})")
    suffix = heatmap_pair_suffix(axis_pair)
    fig.savefig(plot_dir / f"closed_loop_difference_field_{suffix}.png", dpi=180)
    if axis_pair == (0, 1):
        fig.savefig(plot_dir / "closed_loop_difference_field.png", dpi=180)
    plt.close(fig)

    payload = {
        "grid_points": grid_n,
        "slice": {
            "axes": [int(axis_pair[0]), int(axis_pair[1])],
            "fixed_coordinates": {
                str(i): 0.0 for i in range(system.n_dim) if i not in axis_pair
            },
        },
        "thresholds": {"c2": float(c2), "w2": float(w2), "eps": eps},
        "delta_omega_orig": {
            "min": float(np.nanmin(delta_orig_np)),
            "max": float(np.nanmax(delta_orig_np)),
            "violation_count": int(orig_violation.sum()),
            "violation_fraction": float(orig_violation.mean()),
        },
        "delta_omega_cl": {
            "min": float(np.nanmin(delta_cl_np)),
            "max": float(np.nanmax(delta_cl_np)),
            "violation_count": int(cl_violation.sum()),
            "violation_fraction": float(cl_violation.mean()),
        },
    }
    summary_name = f"closed_loop_difference_field_summary_{suffix}.json"
    if run is not None:
        run.write_json(summary_name, payload)
        if axis_pair == (0, 1):
            run.write_json("closed_loop_difference_field_summary.json", payload)
    else:
        (artifact_dir / summary_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if axis_pair == (0, 1):
            (artifact_dir / "closed_loop_difference_field_summary.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )


@torch.no_grad()
def save_policy_label_gap_diagnostic_2d(
    run: Optional[PipelineRun],
    artifact_dir: Path,
    *,
    system: SystemSpec,
    omega_orig: OmegaNet,
    p_matrix: torch.Tensor,
    c_analytic: float,
    c_s: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
    axis_pair: Tuple[int, int] = (0, 1),
) -> None:
    """Plot how decisive the grid-search policy labels are on one 2D slice."""
    if system.verification is None:
        raise ValueError(f"{system.name}: verification domain is required for policy label gap diagnostics.")
    if system.n_dim < 2:
        raise NotImplementedError("policy label gap diagnostics require at least two state dimensions.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if thresholds is not None or system.verification.c2 is not None:
        _, c2, _, _ = resolve_plot_thresholds(system, thresholds)
    else:
        c2 = float(c_s) if c_s is not None else float(c_analytic)
    axis_pair = normalize_axis_pair(system, axis_pair)
    suffix = heatmap_pair_suffix(axis_pair)
    grid_n = int(system.training.diagnostic_plot_grid_points)
    x_axis, y_axis = axis_pair
    x0 = torch.linspace(float(system.x_lb[x_axis]), float(system.x_hi[x_axis]), grid_n, device=p_matrix.device)
    x1 = torch.linspace(float(system.x_lb[y_axis]), float(system.x_hi[y_axis]), grid_n, device=p_matrix.device)
    x0_grid, x1_grid = torch.meshgrid(x0, x1, indexing="xy")
    points = torch.zeros(x0_grid.numel(), system.n_dim, device=p_matrix.device, dtype=torch.float32)
    points[:, x_axis] = x0_grid.reshape(-1)
    points[:, y_axis] = x1_grid.reshape(-1)

    control_grid = make_control_grid(
        system,
        int(system.training.control_grid_points),
        device=points.device,
        dtype=points.dtype,
    )
    state_batch_size = max(1, int(system.training.control_grid_batch_size))
    control_chunk_size = 50_000
    near_tol = float(system.training.policy_label_gap_near_optimal_rel_tol)
    eps = 1e-12

    omega_orig.eval()
    best_chunks = []
    second_chunks = []
    worst_chunks = []
    near_fraction_chunks = []
    for start in range(0, points.shape[0], state_batch_size):
        xb = points[start : start + state_batch_size]
        bsz = xb.shape[0]
        best = torch.full((bsz,), torch.inf, device=points.device, dtype=points.dtype)
        second = torch.full((bsz,), torch.inf, device=points.device, dtype=points.dtype)
        worst = torch.full((bsz,), -torch.inf, device=points.device, dtype=points.dtype)
        stored_values = []
        for u_start in range(0, control_grid.shape[0], control_chunk_size):
            ug = control_grid[u_start : u_start + control_chunk_size]
            k = ug.shape[0]
            x_expanded = xb.unsqueeze(1).expand(-1, k, -1).reshape(bsz * k, system.n_dim)
            u_expanded = ug.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * k, system.u_dim)
            values = omega_orig.query(system.dynamics(x_expanded, u_expanded)).reshape(bsz, k)
            stored_values.append(values.detach())
            chunk_top = torch.topk(values, k=min(2, k), dim=1, largest=False).values
            if chunk_top.shape[1] == 1:
                chunk_second = torch.full_like(chunk_top[:, 0], torch.inf)
            else:
                chunk_second = chunk_top[:, 1]
            candidates = torch.stack((best, second, chunk_top[:, 0], chunk_second), dim=1)
            top = torch.topk(candidates, k=2, dim=1, largest=False).values
            best = top[:, 0]
            second = top[:, 1]
            worst = torch.maximum(worst, values.max(dim=1).values)

        all_values = torch.cat(stored_values, dim=1)
        value_range = (worst - best).clamp_min(eps)
        near_limit = best + near_tol * value_range
        near_fraction = (all_values <= near_limit.unsqueeze(1)).to(dtype=points.dtype).mean(dim=1)

        best_chunks.append(best.detach().cpu())
        second_chunks.append(second.detach().cpu())
        worst_chunks.append(worst.detach().cpu())
        near_fraction_chunks.append(near_fraction.detach().cpu())

    best_values = torch.cat(best_chunks, dim=0)
    second_values = torch.cat(second_chunks, dim=0)
    worst_values = torch.cat(worst_chunks, dim=0)
    near_fraction = torch.cat(near_fraction_chunks, dim=0)
    abs_gap = second_values - best_values
    value_range = (worst_values - best_values).clamp_min(eps)
    relative_gap = abs_gap / value_range
    log_abs_gap = torch.log10(abs_gap.clamp_min(eps))

    p_matrix = p_matrix.to(device=points.device, dtype=points.dtype)
    nu_values = nu_from_p(points, p_matrix).reshape(grid_n, grid_n).detach().cpu().numpy()
    omega_policy_dos_level = float(system.training.policy_label_dos_sublevel)
    omega_orig_values = omega_orig.query(points).reshape(grid_n, grid_n).detach().cpu().numpy()
    x0_np = x0_grid.detach().cpu().numpy()
    x1_np = x1_grid.detach().cpu().numpy()
    fields = {
        "relative winner gap": relative_gap.reshape(grid_n, grid_n).numpy(),
        "near-optimal fraction": near_fraction.reshape(grid_n, grid_n).numpy(),
        "log10 absolute gap": log_abs_gap.reshape(grid_n, grid_n).numpy(),
    }

    def stats(values: torch.Tensor) -> Dict[str, float]:
        return {
            "min": float(values.min().item()),
            "p05": float(torch.quantile(values, 0.05).item()),
            "median": float(torch.quantile(values, 0.50).item()),
            "p95": float(torch.quantile(values, 0.95).item()),
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
        }

    summary = {
        "grid_points": grid_n,
        "total_state_points": int(points.shape[0]),
        "slice": {
            "axes": [int(x_axis), int(y_axis)],
            "fixed_coordinates": {str(i): 0.0 for i in range(system.n_dim) if i not in axis_pair},
        },
        "control_grid_points_per_dim": int(system.training.control_grid_points),
        "control_grid_total_points": int(control_grid.shape[0]),
        "near_optimal_relative_tolerance": near_tol,
        "definition": {
            "J": "J_x(u)=omega_orig(f(x,u)) over the discrete control grid.",
            "absolute_gap": "second_smallest(J_x)-smallest(J_x).",
            "relative_gap": "absolute_gap/(max(J_x)-min(J_x)+1e-12).",
            "near_optimal_fraction": "fraction of grid controls with J_x(u)<=min(J_x)+tol*(max(J_x)-min(J_x)+1e-12).",
        },
        "thresholds": {
            "c_analytic": float(c_analytic),
            "C_S": float(c_s) if c_s is not None else None,
            "c2": float(c2),
            "policy_label_omega_o_dos": omega_policy_dos_level,
        },
        "absolute_gap": stats(abs_gap),
        "relative_gap": stats(relative_gap),
        "near_optimal_fraction": stats(near_fraction),
        "value_range": stats(value_range),
    }

    plot_dir = run.plot_dir if run is not None else artifact_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary_name = f"policy_label_gap_summary_{suffix}.json"
    if run is not None:
        run.write_json(summary_name, summary)
        if axis_pair == (0, 1):
            run.write_json("policy_label_gap_summary.json", summary)
    else:
        (artifact_dir / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if axis_pair == (0, 1):
            (artifact_dir / "policy_label_gap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    labels = system.state_labels or tuple(f"x{i}" for i in range(system.n_dim))
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 4.8), constrained_layout=True)
    cmaps = ("viridis", "magma", "viridis")
    for ax, (title, values), cmap in zip(axes, fields.items(), cmaps):
        cf = ax.contourf(x0_np, x1_np, values, levels=80, cmap=cmap)
        fig.colorbar(cf, ax=ax, label=title)
        try:
            ax.contour(x0_np, x1_np, nu_values, levels=[float(c_analytic)], colors="white", linewidths=1.6)
            if c_s is not None:
                ax.contour(x0_np, x1_np, nu_values, levels=[float(c_s)], colors="red", linewidths=1.6)
            ax.contour(x0_np, x1_np, nu_values, levels=[float(c2)], colors="magenta", linewidths=1.4)
            ax.contour(
                x0_np,
                x1_np,
                omega_orig_values,
                levels=[omega_policy_dos_level],
                colors="yellow",
                linestyles=["--"],
                linewidths=1.4,
            )
        except Exception:
            pass
        ax.set_xlabel(labels[x_axis])
        ax.set_ylabel(labels[y_axis])
        ax.set_title(title)
        ax.grid(True, alpha=0.2)

    legend_handles = [
        Line2D([0], [0], color="white", lw=1.6, label="nu=c_analytic"),
        Line2D([0], [0], color="magenta", lw=1.4, label="nu=c2"),
        Line2D([0], [0], color="yellow", lw=1.4, linestyle="--", label="omega_o=policy DOS"),
    ]
    if c_s is not None:
        legend_handles.insert(1, Line2D([0], [0], color="red", lw=1.6, label="nu=C_S"))
    axes[-1].legend(handles=legend_handles, loc="upper right", framealpha=0.9)
    fig.suptitle(f"Policy Label Gap Diagnostic ({labels[x_axis]}, {labels[y_axis]})")
    fig.savefig(plot_dir / f"policy_label_gap_diagnostic_{suffix}.png", dpi=180)
    if axis_pair == (0, 1):
        fig.savefig(plot_dir / "policy_label_gap_diagnostic.png", dpi=180)
    plt.close(fig)


class TrainingPipeline:
    """Stage-based skeleton for replacing one-notebook-per-system workflows."""

    RUNNABLE_TRAINING_STAGES = (
        "setup_lqr_box",
        "train_omega",
        "train_policy",
        "train_closed_loop_omega",
        "search_thresholds",
        "plot_threshold_diagnostics",
    )

    STAGES = (
        "plan",
        "smoke",
        "setup_lqr_box",
        "diagnose_training_domain",
        "train_omega",
        "train_policy",
        "train_closed_loop_omega",
        "search_thresholds",
        "plot_threshold_diagnostics",
        "plot_policy_label_gap_diagnostics",
        "export",
        "all",
    )

    def __init__(
        self,
        system: SystemSpec,
        *,
        device: Optional[torch.device] = None,
        run: Optional[PipelineRun] = None,
        skip_preflight: bool = False,
    ):
        self.system = system
        self.device = device or pick_device()
        self.run_context = run
        self.lqr_box_artifacts: Optional[LQRBoxArtifacts] = None
        self.early_lqr_c2_result: Optional[EarlyLqrC2SearchResult] = None
        self.skip_preflight = bool(skip_preflight)
        set_seed(system.training.seed)

    def artifact_dir(self) -> Path:
        if self.run_context is not None:
            return self.run_context.artifact_dir
        return RUNS_ROOT / self.system.name / "manual_artifacts"

    def checkpoint_path(self, checkpoint_name: str, description: str) -> Path:
        path = self.artifact_dir() / checkpoint_name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {description} checkpoint for system {self.system.name!r}.\n"
                f"Expected: {path}\n"
                "Run the prerequisite training stage with the same --run-name, "
                "or supply the run name that already contains this artifact."
            )
        return path

    def plan(self) -> None:
        cfg = self.system.training
        counts = resolve_data_distribution_counts(self.system)
        print(f"System: {self.system.name}")
        print(f"Dimensions: n={self.system.n_dim}, m={self.system.u_dim}")
        print(f"State bounds: {self.system.x_lb} to {self.system.x_hi}")
        print(f"Control bounds: {self.system.u_lb} to {self.system.u_hi}")
        print("Configured stages:")
        print("  1. setup_lqr_box: compute A, B, K, P and the largest LQR-certified box")
        print("  2. diagnose_training_domain: optional boundary W probe")
        print("  3. train_omega: reachability expansion, then interval Zubov residual plus finite-horizon guidance")
        print("  4. train_policy: start from the LQR box and grid-search control labels")
        print("  5. train_closed_loop_omega: closed-loop Zubov fine-tuning")
        print("  6. search_thresholds: grid c2/w1/w2 violation search")
        print("  7. plot_threshold_diagnostics: plot searched or supplied c2/w1/w2 thresholds")
        print("  8. plot_policy_label_gap_diagnostics: optional expensive policy-label ambiguity plots")
        print("  9. export: save PyTorch, TorchScript, ONNX artifacts")
        print(
            "  all: setup_lqr_box -> train_omega -> train_policy -> "
            "train_closed_loop_omega -> search_thresholds -> plot_threshold_diagnostics"
        )
        print("Review points before long runs:")
        print(f"  Zubov alpha: alpha(x)={cfg.zubov_alpha_scale}*dt*nu(x), dt={self.system.dt}")
        print(
            "  omega labels: "
            f"trajectory_uniform={counts.omega_trajectory_uniform_points}, "
            f"trajectory_c1_to_lqr_c2_band={counts.omega_trajectory_ellipse_points}, "
            f"collocation_uniform={counts.omega_collocation_uniform_points}, "
            f"collocation_c1_to_lqr_c2_band={counts.omega_collocation_ellipse_points}, "
            f"Ntraj={cfg.omega_rollouts_per_state}, "
            f"label_batch={cfg.omega_batch_size}, "
            "train_loss=full dataset"
        )
        print(f"  training/search rectangle: {self.system.x_lb} to {self.system.x_hi}")
        print(f"  domain expansion strategy: {cfg.domain_expansion_strategy}")
        print(
            "  policy labels: "
            f"c1_to_lqr_c2_band={counts.policy_ellipse_points}, "
            f"uniform={counts.policy_uniform_points}, "
            f"rejection_from_gd_ratio={counts.policy_gd_omega_sublevel_points}, "
            f"omega_dos<={cfg.policy_label_dos_sublevel}, grid={cfg.control_grid_points}"
        )
        print(
            "  boundary diagnostic: "
            f"points_per_face={cfg.boundary_diagnostic_points_per_face}, "
            f"Ntraj={cfg.boundary_diagnostic_rollouts_per_state}, "
            f"horizon={cfg.boundary_diagnostic_horizon}, "
            f"W_threshold={cfg.boundary_diagnostic_w_threshold}"
        )
        print(
            "  CL trajectory labels: "
            f"c1_to_lqr_c2_band={counts.closed_loop_trajectory_ellipse_points}, "
            f"uniform={counts.closed_loop_trajectory_uniform_points}, "
            f"rejection_from_gd_ratio={counts.closed_loop_trajectory_gd_omega_sublevel_points}, "
            f"label_batch={cfg.closed_loop_label_batch_size}"
        )
        print(
            "  CL constraint points: "
            f"c1_to_lqr_c2_band={counts.closed_loop_constraint_ellipse_points}, "
            f"uniform={counts.closed_loop_constraint_uniform_points}, "
            f"rejection_from_gd_ratio={counts.closed_loop_constraint_gd_omega_sublevel_points}, "
            f"DOS gate={cfg.closed_loop_loss_dos_gate}, "
            f"train_batch={cfg.closed_loop_train_batch_size}, "
            f"kappa={cfg.kappa}"
        )
        print(f"  plot pairs: {normalize_plot_pairs(self.system)}")

    def smoke(self) -> None:
        checkpoint_dir = self.artifact_dir()
        models = load_models(self.system, checkpoint_dir, device=self.device)
        x = torch.tensor([[0.0, 0.0], [0.2, -0.1], [1.0, 0.5]], device=self.device)
        with torch.no_grad():
            u = models["pi"](x)
            x_next = self.system.dynamics(x, u)
            omega = models["omega"].query(x)
            omega_cl = models["omega_closed_loop"].query(x)
        print(f"Loaded checkpoints from: {checkpoint_dir}")
        print(f"x = {x.detach().cpu()}")
        print(f"pi(x) = {u.detach().cpu()}")
        print(f"f_sys(x, pi(x)) = {x_next.detach().cpu()}")
        print(f"omega(x) = {omega.detach().cpu()}")
        print(f"omega_closed_loop(x) = {omega_cl.detach().cpu()}")


# [CODEX INSTRUCTIONS HERE:] This V_traj needs to be read carefully, the details are important.
## Use exactly Ntraj total trajectories per initial state, with random sampled
## controls at each step. For each k in range(Ns + 1), compute nu=x^T P x for
## all Ntraj states and add the minimum step value to V. This intentionally may
## choose different trajectories at different time steps; it is a pointwise
## minimum over sampled reachable states, not a minimum over complete trajectory
## costs. Use alpha = zubov_alpha_scale * dt * nu in this accumulation.


# def compute_V_traj(x_batch, Ns=OMEGA_Ns, Ntraj=OMEGA_Ntraj):
#     """W(x) = 1 - exp(-V(x)) where V accumulates min_j alpha(x_k^j)."""
#     x_batch = x_batch.to(device, non_blocking=True)
#     N       = x_batch.shape[0]
#     x_cur   = x_batch.unsqueeze(1).expand(N, Ntraj, N_DIM).clone()
#     V       = torch.zeros(N, device=device, dtype=x_batch.dtype)
#     u_lb    = _u_lb.to(device=device, dtype=x_batch.dtype).view(1, 1, U_DIM)
#     u_span  = (_u_ub - _u_lb).to(device=device, dtype=x_batch.dtype).view(1, 1, U_DIM)

#     for k in range(Ns + 1):
        
#         nu_step = nu_from_p(x_cur, P_torch)
#         alpha_step = zubov_alpha_from_nu(
#             nu_step,
#             dt=system.dt,
#             scale=system.training.zubov_alpha_scale,
#         )
#         V += alpha_step.min(dim=1).values
#         if k < Ns:
#             u = u_lb + u_span * torch.rand(N, Ntraj, U_DIM, device=device, dtype=x_batch.dtype)
#             x_cur = f_sys(x_cur, u)

#     return V, 1.0 - torch.exp(-V)

##
##
    def setup_lqr_box(self) -> None:
        # Working notebook reference: Block 3.
        #
        # This stage should produce a reviewed `LQRBoxArtifacts` instance.
        # Keep the logic deliberately local:
        #   - origin linearization,
        #   - DARE solve,
        #   - input-limit sublevel from `compute_unconstrained_lqr_artifacts`,
        #   - saturated LQR controller,
        #   - quadratic nu=x^T P x,
        #   - largest LQR-certified box/sublevel inside configured bounds.
        #
        # Do not train omega or choose expansion samples here. This stage is the
        # small sturdy platform everything else stands on.
        artifacts = compute_lqr_box_artifacts(self.system, device=self.device)
        cap_lqr_artifacts_to_state_domain(
            artifacts,
            tuple(zip(self.system.x_lb, self.system.x_hi)),
            c1_scale=self.system.training.c1_scale,
        )
        self.lqr_box_artifacts = artifacts
        self.early_lqr_c2_result = None
        if self.run_context is not None:
            self.run_context.write_sublevels(artifacts.sublevels)

        print("LQR setup complete.")
        print(f"  K                    = {artifacts.K.detach().cpu().numpy()}")
        print(f"  P                    = {artifacts.P.detach().cpu().numpy()}")
        print(f"  c_input              = {artifacts.c_input:.9g}")
        print(f"  c_box                = {artifacts.c_box:.9g}")
        print(f"  c_nonlinear_decrease = {artifacts.c_nonlinear_decrease:.9g}")
        print(f"  c_nonlinear_analytic = {artifacts.c_nonlinear_decrease_analytic:.9g}")
        print(f"  c_analytic           = {artifacts.c_analytic:.9g}")
        print(f"  sample cap           = {artifacts.c_sigmoid_center:.9g}")
        print(f"  C_S                  = {artifacts.c_s:.9g}")
        print(f"  local box B          = {artifacts.domain_box}")
        print(f"  eta_B                = {self.system.local_decrease.eta_bound}")

    def get_early_lqr_c2_search(self) -> EarlyLqrC2SearchResult:
        """Return the cached clamped-LQR c2 search used for local sampling."""
        if self.early_lqr_c2_result is not None:
            return self.early_lqr_c2_result
        if self.lqr_box_artifacts is None:
            self.setup_lqr_box()
        assert self.lqr_box_artifacts is not None
        cfg = self.system.training
        result = early_lqr_c2_search(
            self.system,
            self.lqr_box_artifacts,
            n_samples=cfg.threshold_search_samples,
            batch_size=cfg.threshold_search_batch_size,
            device=self.device,
        )
        self.early_lqr_c2_result = result
        if self.run_context is not None:
            self.run_context.write_json("early_lqr_c2_sampling_search.json", asdict(result))
        else:
            artifact_dir = self.artifact_dir()
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "early_lqr_c2_sampling_search.json").write_text(
                json.dumps(asdict(result), indent=2),
                encoding="utf-8",
            )
        print("Early clamped-LQR c2 sampling search complete.")
        print(f"  c1        = {result.c1_raw:.9g}")
        print(f"  c2_raw    = {result.c2_raw}")
        print(f"  bad count = {result.bad_count}")
        return result

    def sample_lqr_c2_band(
        self,
        p_matrix: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """Sample the local c1-to-C_S band for training datasets."""
        if self.lqr_box_artifacts is None:
            self.setup_lqr_box()
        assert self.lqr_box_artifacts is not None
        cfg = self.system.training
        c1 = float(self.lqr_box_artifacts.c_analytic)
        c_s = float(self.lqr_box_artifacts.c_s)
        if not c_s > c1:
            return sample_near_nu_boundary(
                p_matrix,
                c1,
                n_samples,
                device=self.device,
                dtype=torch.float32,
                radial_noise_std=cfg.omega_boundary_noise_std,
            )
        return sample_nu_band(
            p_matrix,
            c1,
            c_s,
            n_samples,
            device=self.device,
            dtype=torch.float32,
            radial_noise_std=cfg.omega_boundary_noise_std,
            c_scale=float(cfg.c1_scale),
        )

    def diagnose_training_domain(self) -> None:
        # This stage asks whether the configured training/search rectangle is
        # likely large enough to contain the practical DOS. It probes boundary
        # points using the finite-horizon stepwise-min value approximation:
        #
        #   W(x) = 1 - exp(-sum_k min_{sampled reachable states} alpha(x_k))
        #
        # If boundary points do not saturate near W=1, the rectangle may cut
        # through the DOS and should be expanded before expensive training.
        if self.lqr_box_artifacts is None:
            self.setup_lqr_box()
        assert self.lqr_box_artifacts is not None

        diagnostic, boundary, w_values = diagnose_domain_boundary_stepwise_min(
            self.system,
            self.lqr_box_artifacts.P,
            device=self.device,
        )
        print("Boundary domain diagnostic complete.")
        print(f"  sampled boundary points = {diagnostic.n_points}")
        print(f"  W min / mean / max      = {diagnostic.min_w:.6g} / {diagnostic.mean_w:.6g} / {diagnostic.max_w:.6g}")
        print(f"  W threshold             = {diagnostic.w_threshold:.6g}")
        print(f"  points below threshold  = {diagnostic.n_below_threshold}")
        print(f"  should_expand           = {diagnostic.should_expand}")

        if self.run_context is not None:
            diag_path = self.run_context.run_dir / "domain_boundary_diagnostic.json"
            diag_path.write_text(json.dumps(asdict(diagnostic), indent=2), encoding="utf-8")
            raw_path = self.run_context.run_dir / "domain_boundary_values.csv"
            rows = ["index," + ",".join(f"x{i+1}" for i in range(self.system.n_dim)) + ",W"]
            for idx, (x_row, w_value) in enumerate(zip(boundary.tolist(), w_values.tolist())):
                rows.append(
                    ",".join([str(idx), *(f"{float(v):.9g}" for v in x_row), f"{float(w_value):.9g}"])
                )
            raw_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def train_omega(self) -> None:
        # Working notebook reference: Block 5.
        #
        # Original omega training has two data streams:
        #
        #   A. Interval Zubov residual collocation, the main structural loss.
        #      z_x    = embed_singleton_state(x)
        #      lb, ub = reachable_interval_affine_control(...)
        #      z_next = embed_interval_bounds(lb, ub)
        #      xi     = 1 - exp(-zubov_alpha_scale * dt * nu(x))
        #      L_pi   = mean((omega(z_x) - omega(z_next)
        #                    - xi * (1 - omega(z_next)))^2)
        #
        #   B. Smaller finite-horizon guidance set.
        #      W_target is generated by random rollouts and the stepwise minimum
        #      alpha accumulation. It is used in an MSE term:
        #      L_d = mean((omega(z_data) - W_target)^2)
        #
        # The interval residual must not be replaced by singleton successor
        # samples. The finite-horizon data are guidance, not the defining
        # equation.
        if self.lqr_box_artifacts is None:
            self.setup_lqr_box()
        assert self.lqr_box_artifacts is not None

        cfg = self.system.training
        artifact_dir = (
            self.run_context.artifact_dir
            if self.run_context is not None
            else RUNS_ROOT / self.system.name / "manual_artifacts"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if cfg.omega_batch_size <= 0:
            raise ValueError("omega_batch_size must be positive.")
        omega_label_batch_size = int(cfg.omega_label_batch_size or cfg.omega_batch_size)
        if omega_label_batch_size <= 0:
            raise ValueError("omega_label_batch_size must be positive.")

        p_matrix = self.lqr_box_artifacts.P.to(device=self.device, dtype=torch.float32)
        k_matrix = self.lqr_box_artifacts.K.to(device=self.device, dtype=torch.float32)
        expansion_result = None
        boundary_points_cpu = torch.empty(0, self.system.n_dim)
        boundary_v_cpu = torch.empty(0)
        boundary_w_cpu = torch.empty(0)
        if cfg.domain_expansion_strategy == "reachability_stepwise_min":
            print("Expanding training domain by finite-horizon stepwise-min reachability...")
            (
                expansion_result,
                expanded_bounds,
                boundary_points_cpu,
                boundary_v_cpu,
                boundary_w_cpu,
            ) = expand_domain_by_stepwise_min_reachability(
                self.system,
                p_matrix,
                device=self.device,
            )
            self.system = _system_with_state_domain(self.system, expanded_bounds)
            if self.run_context is not None:
                self.run_context.system = self.system
                self.run_context.write_config()
            print(f"  initial bounds          = {expansion_result.initial_bounds}")
            print(f"  final bounds            = {expansion_result.final_bounds}")
            print(f"  iterations              = {expansion_result.iterations}")
            print(f"  converged               = {expansion_result.converged}")
            last = expansion_result.history[-1]
            print(
                "  final boundary W        = "
                f"{last['W_min']:.6g} / {last['W_mean']:.6g} / {last['W_max']:.6g}"
            )
            if self.run_context is not None:
                self.run_context.write_json(
                    "reachability_domain_expansion.json",
                    asdict(expansion_result),
                )
                raw_path = self.run_context.run_dir / "reachability_domain_expansion_boundary.csv"
            else:
                raw_path = artifact_dir / "reachability_domain_expansion_boundary.csv"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.with_name("reachability_domain_expansion.json").write_text(
                    json.dumps(asdict(expansion_result), indent=2),
                    encoding="utf-8",
                )
            rows = ["index," + ",".join(f"x{i+1}" for i in range(self.system.n_dim)) + ",V,W"]
            for idx, (x_row, v_value, w_value) in enumerate(
                zip(boundary_points_cpu.tolist(), boundary_v_cpu.tolist(), boundary_w_cpu.tolist())
            ):
                rows.append(
                    ",".join(
                        [
                            str(idx),
                            *(f"{float(v):.9g}" for v in x_row),
                            f"{float(v_value):.9g}",
                            f"{float(w_value):.9g}",
                        ]
                    )
                )
            raw_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        elif cfg.domain_expansion_strategy != "defer":
            print(
                "Skipping automatic domain expansion for unsupported "
                f"domain_expansion_strategy={cfg.domain_expansion_strategy!r}."
            )

        print("Refreshing LQR containment levels on the active domain...")
        cap_lqr_artifacts_to_state_domain(
            self.lqr_box_artifacts,
            tuple(zip(self.system.x_lb, self.system.x_hi)),
            c1_scale=self.system.training.c1_scale,
        )
        if self.run_context is not None:
            self.run_context.write_sublevels(self.lqr_box_artifacts.sublevels)
        print(f"  active-domain c_box      = {self.lqr_box_artifacts.c_box:.9g}")
        print(f"  refreshed C_S            = {self.lqr_box_artifacts.c_s:.9g}")
        print(f"  refreshed sample cap = {self.lqr_box_artifacts.c_sigmoid_center:.9g}")

        counts = resolve_data_distribution_counts(self.system)
        omega_trajectory_ellipse = self.sample_lqr_c2_band(
            p_matrix,
            counts.omega_trajectory_ellipse_points,
        )
        omega_collocation_ellipse = self.sample_lqr_c2_band(
            p_matrix,
            counts.omega_collocation_ellipse_points,
        )
        reachability_boundary_label = torch.empty(0, self.system.n_dim, device=self.device)
        reachability_boundary_residual = torch.empty(0, self.system.n_dim, device=self.device)

        print("Preparing original omega finite-horizon guidance labels...")
        print(f"  trajectory uniform     = {counts.omega_trajectory_uniform_points}")
        print(f"  trajectory ellipse     = {omega_trajectory_ellipse.shape[0]}")
        print(f"  reachability boundary  = {reachability_boundary_label.shape[0]}")
        print(f"  label batch size       = {omega_label_batch_size}")
        print(f"  training loss batch    = {cfg.omega_batch_size}")
        x_data_uniform = sample_uniform(
            self.system,
            counts.omega_trajectory_uniform_points,
            device=self.device,
        )
        x_data = torch.cat([omega_trajectory_ellipse, reachability_boundary_label, x_data_uniform], dim=0)
        label_source_ids = torch.cat(
            [
                torch.full((omega_trajectory_ellipse.shape[0],), 0, dtype=torch.long),
                torch.full((reachability_boundary_label.shape[0],), 1, dtype=torch.long),
                torch.full((x_data_uniform.shape[0],), 2, dtype=torch.long),
            ],
            dim=0,
        )

        label_batches = []
        for start in range(0, x_data.shape[0], omega_label_batch_size):
            xb = x_data[start : start + omega_label_batch_size]
            label_batches.append(
                compute_stepwise_min_zubov_targets(
                    self.system,
                    xb,
                    p_matrix,
                    horizon=cfg.omega_rollout_horizon,
                    ntraj=cfg.omega_rollouts_per_state,
                )
            )

        label_batch = OmegaLabelBatch(
            x=torch.cat([batch.x.detach().cpu() for batch in label_batches], dim=0),
            z_x=torch.cat([batch.z_x.detach().cpu() for batch in label_batches], dim=0),
            target_w=torch.cat([batch.target_w.detach().cpu() for batch in label_batches], dim=0),
        )
        save_finite_horizon_target_plot(
            self.run_context,
            artifact_dir,
            stem="omega_original_reachable_min",
            title="Original Omega Reachable-Min Finite-Horizon Targets",
            labels=label_batch,
            p_matrix=p_matrix,
            c_s=self.lqr_box_artifacts.c_s,
        )

        print("Preparing original omega interval residual collocation set...")
        print(f"  collocation uniform    = {counts.omega_collocation_uniform_points}")
        print(f"  collocation ellipse    = {omega_collocation_ellipse.shape[0]}")
        print(f"  reachability boundary  = {reachability_boundary_residual.shape[0]}")
        print(f"  omega embedding        = {ORIGINAL_OMEGA_EMBEDDING_MODE}")
        x_res_uniform = sample_uniform(
            self.system,
            counts.omega_collocation_uniform_points,
            device=self.device,
        )
        x_residual = torch.cat([omega_collocation_ellipse, reachability_boundary_residual, x_res_uniform], dim=0)
        residual_source_ids = torch.cat(
            [
                torch.full((omega_collocation_ellipse.shape[0],), 0, dtype=torch.long),
                torch.full((reachability_boundary_residual.shape[0],), 1, dtype=torch.long),
                torch.full((x_res_uniform.shape[0],), 2, dtype=torch.long),
            ],
            dim=0,
        )
        residual_batches = []
        for start in range(0, x_residual.shape[0], cfg.omega_batch_size):
            xb = x_residual[start : start + cfg.omega_batch_size]
            residual_batches.append(
                build_original_omega_residual_batch(
                    self.system,
                    xb,
                    p_matrix,
                )
            )
        if residual_batches:
            residual_batch = OmegaResidualBatch(
                x=torch.cat([batch.x.detach().cpu() for batch in residual_batches], dim=0),
                z_x=torch.cat([batch.z_x.detach().cpu() for batch in residual_batches], dim=0),
                z_next_interval=torch.cat(
                    [batch.z_next_interval.detach().cpu() for batch in residual_batches],
                    dim=0,
                ),
                xi=torch.cat([batch.xi.detach().cpu() for batch in residual_batches], dim=0),
                successor_embedding_mode=ORIGINAL_OMEGA_EMBEDDING_MODE,
            )
        else:
            residual_batch = OmegaResidualBatch(
                x=torch.empty(0, self.system.n_dim),
                z_x=torch.empty(0, original_omega_input_dim(self.system)),
                z_next_interval=torch.empty(0, original_omega_input_dim(self.system)),
                xi=torch.empty(0),
                successor_embedding_mode=ORIGINAL_OMEGA_EMBEDDING_MODE,
            )

        omega_input_dim = original_omega_input_dim(self.system)
        omega = OmegaNet(
            n_dim=self.system.n_dim,
            hidden_sizes=cfg.omega_hidden_sizes,
            input_dim=omega_input_dim,
            embedding_mode=ORIGINAL_OMEGA_EMBEDDING_MODE,
        ).to(self.device)
        optimizer = torch.optim.Adam(omega.parameters(), lr=cfg.omega_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, cfg.omega_epochs // 2),
            gamma=0.5,
        )
        x_origin = torch.zeros(1, self.system.n_dim, device=self.device, dtype=torch.float32)

        print(f"Training original omega for {cfg.omega_epochs} epochs...")
        n_label = label_batch.x.shape[0]
        n_residual = residual_batch.x.shape[0]
        if n_label == 0:
            raise ValueError("Original omega training requires a non-empty label dataset.")
        if n_residual == 0 and cfg.omega_lambda_bellman != 0.0:
            raise ValueError(
                "Original omega interval residual is active but its collocation dataset is empty."
            )
        full_batch_limit = 250_000
        use_full_batch = (n_label + n_residual) <= full_batch_limit
        if use_full_batch:
            print(
                "  original omega batches = full dataset "
                f"({n_label} labels + {n_residual} residual points)"
            )
            train_label_batch = OmegaLabelBatch(
                x=label_batch.x.to(self.device),
                z_x=label_batch.z_x.to(self.device),
                target_w=label_batch.target_w.to(self.device),
            )
            train_residual_batch = OmegaResidualBatch(
                x=residual_batch.x.to(self.device),
                z_x=residual_batch.z_x.to(self.device),
                z_next_interval=residual_batch.z_next_interval.to(self.device),
                xi=residual_batch.xi.to(self.device),
                successor_embedding_mode=residual_batch.successor_embedding_mode,
            )
        else:
            print(
                "  original omega batches = fixed stratified minibatches "
                f"({n_label} labels + {n_residual} residual points)"
            )
            label_batcher = FixedStratifiedBatcher(source_groups_from_ids(label_source_ids), cfg.omega_batch_size)
            residual_batcher = FixedStratifiedBatcher(source_groups_from_ids(residual_source_ids), cfg.omega_batch_size)
        omega_batches_per_epoch = 1
        if not use_full_batch and bool(cfg.omega_full_pass_epochs):
            batch_counts = [label_batcher.batches_per_full_pass()]
            if n_residual > 0:
                batch_counts.append(residual_batcher.batches_per_full_pass())
            omega_batches_per_epoch = max(1, *batch_counts)
            print(
                "  original omega full-pass mode = "
                f"{omega_batches_per_epoch} minibatches/epoch at batch_size={cfg.omega_batch_size}"
            )
        loss_history = []
        for epoch in range(cfg.omega_epochs):
            omega.train()
            epoch_data_loss = 0.0
            epoch_residual_loss = 0.0
            epoch_boundary_loss = 0.0
            epoch_total_loss = 0.0
            epoch_weight = 0
            epoch_label_weight = 0
            epoch_residual_weight = 0
            for _ in range(omega_batches_per_epoch):
                if use_full_batch:
                    label_mini_batch = train_label_batch
                    residual_mini_batch = train_residual_batch
                else:
                    label_indices = label_batcher.next_indices()
                    residual_indices = (
                        residual_batcher.next_indices()
                        if n_residual > 0
                        else torch.empty(0, dtype=torch.long)
                    )
                    label_mini_batch = OmegaLabelBatch(
                        x=label_batch.x[label_indices].to(self.device),
                        z_x=label_batch.z_x[label_indices].to(self.device),
                        target_w=label_batch.target_w[label_indices].to(self.device),
                    )
                    residual_mini_batch = OmegaResidualBatch(
                        x=residual_batch.x[residual_indices].to(self.device),
                        z_x=residual_batch.z_x[residual_indices].to(self.device),
                        z_next_interval=residual_batch.z_next_interval[residual_indices].to(self.device),
                        xi=residual_batch.xi[residual_indices].to(self.device),
                        successor_embedding_mode=residual_batch.successor_embedding_mode,
                    )
                optimizer.zero_grad()
                losses = original_omega_loss_terms(
                    omega,
                    residual_mini_batch,
                    label_mini_batch,
                    lambda_data=cfg.omega_lambda_data,
                    lambda_residual=cfg.omega_lambda_bellman,
                    lambda_boundary=cfg.omega_lambda_boundary,
                    x_origin=x_origin,
                )
                losses.total.backward()
                optimizer.step()
                label_count = int(label_mini_batch.x.shape[0])
                residual_count = int(residual_mini_batch.x.shape[0])
                batch_count = label_count + residual_count
                epoch_weight += batch_count
                epoch_label_weight += label_count
                epoch_residual_weight += residual_count
                epoch_data_loss += float(losses.data_mse.detach().cpu().item()) * label_count
                epoch_residual_loss += float(losses.interval_residual_mse.detach().cpu().item()) * residual_count
                epoch_boundary_loss += float(losses.boundary_condition_mse.detach().cpu().item()) * batch_count
                epoch_total_loss += float(losses.total.detach().cpu().item()) * batch_count
            scheduler.step()
            denom = max(1, epoch_weight)
            data_loss_scalar = epoch_data_loss / max(1, epoch_label_weight)
            residual_loss_scalar = epoch_residual_loss / max(1, epoch_residual_weight)
            boundary_loss_scalar = epoch_boundary_loss / denom
            total_loss_scalar = epoch_total_loss / denom
            loss_history.append(
                {
                    "epoch": int(epoch),
                    "data_mse": data_loss_scalar,
                    "interval_residual_mse": residual_loss_scalar,
                    "boundary_condition_mse": boundary_loss_scalar,
                    "total": total_loss_scalar,
                    "minibatches": int(omega_batches_per_epoch),
                    "examples_seen": int(epoch_weight),
                }
            )

            if epoch % 500 == 0 or epoch == cfg.omega_epochs - 1:
                print(
                    f"  epoch {epoch:5d} | "
                    f"L_d={data_loss_scalar:.6g} | "
                    f"L_interval={residual_loss_scalar:.6g} | "
                    f"L_bc={boundary_loss_scalar:.6g} | "
                    f"total={total_loss_scalar:.6g}"
                )

        omega.eval()
        checkpoint = {
            "model_state_dict": omega.state_dict(),
            "n_dim": int(self.system.n_dim),
            "input_dim": int(omega.input_dim),
            "embedding_mode": str(omega.embedding_mode),
            "reachable_set_embedding_mode": ORIGINAL_OMEGA_EMBEDDING_MODE,
            "output_activation": "sigmoid",
            "hidden_sizes": list(cfg.omega_hidden_sizes),
            "x_lb": [float(v) for v in self.system.x_lb],
            "x_hi": [float(v) for v in self.system.x_hi],
            "u_lb": [float(v) for v in self.system.u_lb],
            "u_hi": [float(v) for v in self.system.u_hi],
            "dt": float(self.system.dt),
            "P": self.lqr_box_artifacts.P.detach().cpu().numpy().astype(float).tolist(),
            "K": self.lqr_box_artifacts.K.detach().cpu().numpy().astype(float).tolist(),
            "c_analytic": float(self.lqr_box_artifacts.c_analytic),
            "sample_cap": float(self.lqr_box_artifacts.c_sigmoid_center),
            "C_S": float(self.lqr_box_artifacts.c_s),
            "zubov_alpha_scale": float(cfg.zubov_alpha_scale),
            "omega_loss": {
                "lambda_data": float(cfg.omega_lambda_data),
                "lambda_interval_residual": float(cfg.omega_lambda_bellman),
                "lambda_boundary": float(cfg.omega_lambda_boundary),
            },
            "sampling": {
                "band": {
                    "mode": "c1_to_C_S_band",
                    "c1": float(self.lqr_box_artifacts.c_analytic),
                    "C_S": float(self.lqr_box_artifacts.c_s),
                    "band_c_scale": float(cfg.c1_scale),
                },
                "band_c_scale": float(cfg.c1_scale),
                "omega_trajectory_uniform_points": int(counts.omega_trajectory_uniform_points),
                "omega_trajectory_ellipse_points": int(omega_trajectory_ellipse.shape[0]),
                "omega_collocation_uniform_points": int(counts.omega_collocation_uniform_points),
                "omega_collocation_ellipse_points": int(omega_collocation_ellipse.shape[0]),
                "finite_horizon_uniform_points": int(counts.omega_trajectory_uniform_points),
                "interval_residual_uniform_points": int(counts.omega_collocation_uniform_points),
                "c1_to_C_S_band_points": int(
                    omega_trajectory_ellipse.shape[0] + omega_collocation_ellipse.shape[0]
                ),
                "finite_horizon_c1_to_C_S_band_points": int(omega_trajectory_ellipse.shape[0]),
                "interval_residual_c1_to_C_S_band_points": int(omega_collocation_ellipse.shape[0]),
                "distribution_schema": (
                    "omega_o has independent trajectory-uniform, trajectory-ellipse, "
                    "collocation-uniform, and collocation-ellipse counts"
                ),
                "reachability_boundary_points": int(
                    reachability_boundary_label.shape[0] + reachability_boundary_residual.shape[0]
                ),
                "finite_horizon_reachability_boundary_points": int(reachability_boundary_label.shape[0]),
                "interval_residual_reachability_boundary_points": int(
                    reachability_boundary_residual.shape[0]
                ),
                "domain_expansion_strategy": cfg.domain_expansion_strategy,
                "reachability_domain_expansion": (
                    asdict(expansion_result) if expansion_result is not None else None
                ),
                "boundary_noise_std": float(cfg.omega_boundary_noise_std),
                "rollout_horizon": int(cfg.omega_rollout_horizon),
                "rollouts_per_state": int(cfg.omega_rollouts_per_state),
                "label_generation_batch_size": int(omega_label_batch_size),
                "training_loss_batch": int(cfg.omega_batch_size),
                "full_pass_epochs": bool(cfg.omega_full_pass_epochs),
                "minibatches_per_epoch": int(omega_batches_per_epoch),
                "reachable_set_embedding_mode": ORIGINAL_OMEGA_EMBEDDING_MODE,
                "reachable_set_embedding_input_dim": int(omega.input_dim),
                "successor_set_finite_horizon_targets_enabled": False,
                "reachable_interval": ORIGINAL_OMEGA_EMBEDDING_MODE,
            },
        }
        save_path = artifact_dir / self.system.checkpoints.omega
        torch.save(checkpoint, save_path)
        save_loss_history_artifacts(
            self.run_context,
            artifact_dir,
            stem="omega",
            title="Original Omega Training Losses",
            history=loss_history,
        )
        print(f"Original omega checkpoint saved to {save_path}")

    def train_policy(self) -> None:
        # Working notebook reference: policy/Grid-search Block 6-ish section.
        #
        # High-level intent:
        #   1. Sample candidate states from the initial LQR box/domain chosen by
        #      `setup_lqr_box`. Sampling strategy is intentionally not fixed.
        #   2. For each state, search over admissible controls and choose labels
        #      that minimize original omega at the successor state.
        #   3. Train the zero-at-origin saturated NN branch. The clamped LQR
        #      branch is used separately by threshold search and verification.
        #
        # The control search is likely system- and dimension-sensitive. Do not
        # assume one-dimensional control grids beyond the 2D benchmark.
        if self.lqr_box_artifacts is None:
            self.setup_lqr_box()
        assert self.lqr_box_artifacts is not None

        cfg = self.system.training
        counts = resolve_data_distribution_counts(self.system)
        artifact_dir = self.artifact_dir()
        artifact_dir.mkdir(parents=True, exist_ok=True)

        omega_path = self.checkpoint_path(self.system.checkpoints.omega, "original omega")
        omega = load_omeganet(omega_path, n_dim=self.system.n_dim, device=self.device)
        omega.eval()
        for param in omega.parameters():
            param.requires_grad_(False)

        control_grid = make_control_grid(
            self.system,
            cfg.control_grid_points,
            device=self.device,
            dtype=torch.float32,
        )

        if cfg.policy_label_cache_path:
            cache_path = Path(cfg.policy_label_cache_path).expanduser()
            if not cache_path.is_absolute():
                cache_path = Path.cwd() / cache_path
            payload = torch.load(cache_path, map_location=self.device)
            metadata = dict(payload.get("metadata", {}))
            if int(metadata.get("n_dim", -1)) != int(self.system.n_dim):
                raise ValueError(f"Policy label cache n_dim mismatch: {cache_path}")
            if int(metadata.get("u_dim", -1)) != int(self.system.u_dim):
                raise ValueError(f"Policy label cache u_dim mismatch: {cache_path}")
            if int(metadata.get("control_grid_points", -1)) != int(cfg.control_grid_points):
                raise ValueError(f"Policy label cache control_grid_points mismatch: {cache_path}")
            cached_mask_domain = bool(metadata.get("policy_label_mask_out_of_domain_successors", False))
            if cached_mask_domain != bool(cfg.policy_label_mask_out_of_domain_successors):
                raise ValueError(
                    "Policy label cache successor-domain-mask mismatch: "
                    f"cache has {cached_mask_domain}, config wants "
                    f"{bool(cfg.policy_label_mask_out_of_domain_successors)}. Regenerate labels."
                )
            cached_selection_mode = normalize_policy_label_selection_mode(
                metadata.get("policy_label_selection_mode", "argmin")
            )
            if cached_selection_mode != "argmin":
                raise ValueError(
                    "Policy label cache selection-mode mismatch: "
                    f"cache has {cached_selection_mode!r}, config wants 'argmin'. "
                    "Regenerate labels or point policy_label_cache_path at a matching cache."
                )
            if bool(cfg.policy_label_use_confidence_weight) and (
                "control_weight" not in payload or "near_optimal_fraction" not in payload
            ):
                raise ValueError(
                    "Policy label cache is configured for confidence-weighted control loss but does not contain "
                    "control_weight/near_optimal_fraction. Regenerate the cache."
                )
            print(f"Loading cached PiNet grid-search labels from {cache_path}")
            labels = PolicyLabelBatch(
                x=payload["x"].to(device=self.device, dtype=torch.float32),
                u_opt=payload["u_opt"].to(device=self.device, dtype=torch.float32),
                omega_opt_next=payload["omega_opt_next"].to(device=self.device, dtype=torch.float32),
                control_weight=payload.get("control_weight", None),
                near_optimal_fraction=payload.get("near_optimal_fraction", None),
                feasible_control_fraction=payload.get("feasible_control_fraction", None),
                unmasked_fallback_mask=payload.get("unmasked_fallback_mask", None),
                decisive_mask=payload.get(
                    "decisive_mask",
                    torch.ones_like(payload["omega_opt_next"], dtype=torch.bool),
                ).to(device=self.device, dtype=torch.bool),
                neighbor_imputed_mask=payload.get(
                    "neighbor_imputed_mask",
                    torch.zeros_like(payload["omega_opt_next"], dtype=torch.bool),
                ).to(device=self.device, dtype=torch.bool),
                midpoint_default_mask=payload.get(
                    "midpoint_default_mask",
                    torch.zeros_like(payload["omega_opt_next"], dtype=torch.bool),
                ).to(device=self.device, dtype=torch.bool),
            )
            if labels.control_weight is not None:
                labels.control_weight = labels.control_weight.to(device=self.device, dtype=torch.float32)
            if labels.near_optimal_fraction is not None:
                labels.near_optimal_fraction = labels.near_optimal_fraction.to(device=self.device, dtype=torch.float32)
            if labels.feasible_control_fraction is not None:
                labels.feasible_control_fraction = labels.feasible_control_fraction.to(device=self.device, dtype=torch.float32)
            if labels.unmasked_fallback_mask is not None:
                labels.unmasked_fallback_mask = labels.unmasked_fallback_mask.to(device=self.device, dtype=torch.bool)
            if labels.control_weight is None:
                labels.control_weight = torch.ones(labels.x.shape[0], device=self.device, dtype=torch.float32)
            if labels.near_optimal_fraction is None:
                labels.near_optimal_fraction = torch.zeros(labels.x.shape[0], device=self.device, dtype=torch.float32)
            if labels.feasible_control_fraction is None:
                labels.feasible_control_fraction = torch.ones(labels.x.shape[0], device=self.device, dtype=torch.float32)
            if labels.unmasked_fallback_mask is None:
                labels.unmasked_fallback_mask = torch.zeros(labels.x.shape[0], device=self.device, dtype=torch.bool)
            if not bool(cfg.policy_label_use_confidence_weight):
                labels.control_weight = torch.ones(labels.x.shape[0], device=self.device, dtype=torch.float32)
            policy_source_ids = payload["source_ids"].detach().cpu().to(dtype=torch.long)
            label_filter_summary = dict(payload.get("label_filter_summary", {}))
            label_filter_summary["loaded_policy_label_cache"] = str(cache_path)
            print(f"  cached policy states    = {labels.x.shape[0]}")
            print(f"  control grid points      = {metadata.get('control_grid_total_points')}")
            print(f"  grid points per dim      = {metadata.get('control_grid_points')}")
        else:
            rejection_batch_size = max(1, int(max(cfg.control_grid_batch_size, 50_000)))
            policy_dos_target_points = int(counts.policy_gd_omega_sublevel_points)
            policy_dos_points, policy_dos_stats = sample_omega_rejection_points(
                omega,
                tuple(zip(self.system.x_lb, self.system.x_hi)),
                policy_dos_target_points,
                device=self.device,
                sublevel=cfg.policy_label_dos_sublevel,
                batch_size=rejection_batch_size,
                max_attempts=cfg.optimized_sampler_max_attempts,
            )
            if cfg.optimized_sampler_strict_sublevel and policy_dos_points.shape[0] == 0:
                raise ValueError(
                    f"{self.system.name}: strict policy sublevel sampler returned no "
                    f"omega_orig(x)<={cfg.policy_label_dos_sublevel} points."
                )
            policy_ellipse_boundary = self.sample_lqr_c2_band(
                self.lqr_box_artifacts.P.to(device=self.device, dtype=torch.float32),
                counts.policy_ellipse_points,
            )
            policy_ellipse_boundary_stats = {
                "mode": "c1_to_C_S_band",
                "requested_points": int(counts.policy_ellipse_points),
                "returned_points": int(policy_ellipse_boundary.shape[0]),
                "c1": float(self.lqr_box_artifacts.c_analytic),
                "C_S": float(self.lqr_box_artifacts.c_s),
                "band_c_scale": float(cfg.c1_scale),
            }
            policy_uniform = sample_uniform_bounds(
                tuple(zip(self.system.x_lb, self.system.x_hi)),
                counts.policy_uniform_points,
                device=self.device,
            ).to(dtype=torch.float32)
            policy_uniform_stats = {
                "mode": "uniform_hyperrectangle",
                "requested_points": int(counts.policy_uniform_points),
                "returned_points": int(policy_uniform.shape[0]),
                "bounds": [
                    [float(lo), float(hi)]
                    for lo, hi in zip(self.system.x_lb, self.system.x_hi)
                ],
            }
            x_candidates = torch.cat([policy_ellipse_boundary, policy_uniform, policy_dos_points], dim=0)
            candidate_source_ids = torch.cat(
                [
                    torch.full((policy_ellipse_boundary.shape[0],), 0, device=self.device, dtype=torch.long),
                    torch.full((policy_uniform.shape[0],), 1, device=self.device, dtype=torch.long),
                    torch.full((policy_dos_points.shape[0],), 2, device=self.device, dtype=torch.long),
                ],
                dim=0,
            )
            x_train = x_candidates.detach()
            policy_source_ids = candidate_source_ids.detach().cpu()
            label_filter_summary = {
                "condition": (
                    "policy states are sampled from c1-to-early-LQR-c2 band "
                    "candidates, uniform domain candidates, and uniform-rejection "
                    "original-omega sublevel candidates. No post-hoc DOS filter is "
                    "applied to policy band or uniform points."
                ),
                "sublevel": float(cfg.policy_label_dos_sublevel),
                "batch_size": int(rejection_batch_size),
                "omega_filtration": {
                    "enabled": False,
                    "removed_points": 0,
                    "kept_fraction": 1.0,
                },
                "rejection_sublevel": policy_dos_stats,
                "policy_ellipse_boundary": policy_ellipse_boundary_stats,
                "policy_uniform": policy_uniform_stats,
                "state_box_boundary_points": 0,
                "policy_ellipse_boundary_points": int(policy_ellipse_boundary.shape[0]),
                "policy_uniform_points": int(policy_uniform.shape[0]),
                "policy_rejection_omega_sublevel_points": int(policy_dos_points.shape[0]),
                "policy_rejection_points_requested": int(policy_dos_target_points),
                "policy_ellipse_points_requested": int(counts.policy_ellipse_points),
                "policy_uniform_points_requested": int(counts.policy_uniform_points),
                "policy_gd_points_reallocated_to_rejection": int(
                    counts.policy_gd_omega_sublevel_points
                ),
                "input_points": int(
                    policy_ellipse_boundary.shape[0] + policy_uniform.shape[0] + policy_dos_points.shape[0]
                ),
                "kept_points": int(x_train.shape[0]),
                "removed_points": 0,
            }
            label_filter_summary["kept_fraction"] = 1.0
            if x_train.shape[0] == 0:
                raise ValueError(
                    f"{self.system.name}: policy_label_dos_sublevel={cfg.policy_label_dos_sublevel} "
                    "left no Omega_orig-DOS controller-label states."
                )
            if self.run_context is not None:
                self.run_context.write_json("policy_label_dos_filtration.json", label_filter_summary)
            else:
                (artifact_dir / "policy_label_dos_filtration.json").write_text(
                    json.dumps(label_filter_summary, indent=2),
                    encoding="utf-8",
                )

            print("Preparing PiNet grid-search labels...")
            print(f"  band points             = {policy_ellipse_boundary.shape[0]}")
            print(f"  uniform points          = {policy_uniform.shape[0]}")
            print(f"  rejection DOS points    = {policy_dos_points.shape[0]}")
            print(f"  policy states           = {x_train.shape[0]} / {x_candidates.shape[0]} (no post-filter)")
            print(
                "  requested rejection DOS = "
                f"{policy_dos_target_points}"
            )
            print(f"  optimized states        = {x_train.shape[0]}")
            print(f"  DOS target              = omega_orig(x) <= {cfg.policy_label_dos_sublevel}")
            print(
                "  rejection acceptance    = "
                f"{policy_dos_stats.get('acceptance_rate')}"
            )
            print(f"  control grid points      = {control_grid.shape[0]}")
            print(f"  grid points per dim      = {cfg.control_grid_points}")
            print("  label selection mode     = argmin")
            print(f"  confidence weights       = {cfg.policy_label_use_confidence_weight}")
            print(f"  mask OOD successors      = {cfg.policy_label_mask_out_of_domain_successors}")
            print(f"  near-optimal rel tol     = {cfg.policy_label_near_optimal_rel_tol}")
            labels = compute_optimal_next_omega_grid_labels(
                self.system,
                omega,
                x_train,
                control_grid,
                state_batch_size=cfg.control_grid_batch_size,
                resolve_ambiguous_for_policy=False,
                near_optimal_rel_tol=cfg.policy_label_near_optimal_rel_tol,
                compute_control_confidence=bool(cfg.policy_label_use_confidence_weight),
                mask_out_of_domain_successors=bool(cfg.policy_label_mask_out_of_domain_successors),
                successor_domain=tuple(tuple(v) for v in self.system.verification.domain),
            )
            if labels.control_weight is None:
                labels.control_weight = torch.ones(labels.x.shape[0], device=self.device, dtype=torch.float32)
            if labels.near_optimal_fraction is None:
                labels.near_optimal_fraction = torch.zeros(labels.x.shape[0], device=self.device, dtype=torch.float32)
            if labels.feasible_control_fraction is None:
                labels.feasible_control_fraction = torch.ones(labels.x.shape[0], device=self.device, dtype=torch.float32)
            if labels.unmasked_fallback_mask is None:
                labels.unmasked_fallback_mask = torch.zeros(labels.x.shape[0], device=self.device, dtype=torch.bool)
            if not bool(cfg.policy_label_use_confidence_weight):
                labels.control_weight = torch.ones(labels.x.shape[0], device=self.device, dtype=torch.float32)
            if labels.decisive_mask is None:
                labels.decisive_mask = torch.ones(labels.x.shape[0], device=self.device, dtype=torch.bool)
            if labels.neighbor_imputed_mask is None:
                labels.neighbor_imputed_mask = torch.zeros(labels.x.shape[0], device=self.device, dtype=torch.bool)
            if labels.midpoint_default_mask is None:
                labels.midpoint_default_mask = torch.zeros(labels.x.shape[0], device=self.device, dtype=torch.bool)
            fallback_mask = labels.unmasked_fallback_mask
            label_filter_summary["control_label_resolution"] = {
                "mode": "argmin",
                "successor_domain_mask": bool(cfg.policy_label_mask_out_of_domain_successors),
                "successor_domain": [
                    [float(lo), float(hi)]
                    for lo, hi in self.system.verification.domain
                ],
                "label_target": "exact_grid_argmin",
                "weighted_term": (
                    "control_imitation_mse_only"
                    if cfg.policy_label_use_confidence_weight
                    else None
                ),
                "control_weight": (
                    "1 - near_optimal_fraction"
                    if cfg.policy_label_use_confidence_weight
                    else "1"
                ),
                "minimum_midpoint_improvement": None,
                "near_optimal_relative_tolerance": (
                    float(cfg.policy_label_near_optimal_rel_tol)
                    if cfg.policy_label_use_confidence_weight
                    else None
                ),
                "feasible_control_fraction_min": float(labels.feasible_control_fraction.min().detach().cpu().item()),
                "feasible_control_fraction_mean": float(labels.feasible_control_fraction.mean().detach().cpu().item()),
                "feasible_control_fraction_p05": float(torch.quantile(labels.feasible_control_fraction, 0.05).detach().cpu().item()),
                "unmasked_fallback_points": int(fallback_mask.sum().item()),
                "control_weight_min": float(labels.control_weight.min().detach().cpu().item()),
                "control_weight_mean": float(labels.control_weight.mean().detach().cpu().item()),
                "control_weight_p05": float(torch.quantile(labels.control_weight, 0.05).detach().cpu().item()),
                "near_optimal_fraction_mean": float(labels.near_optimal_fraction.mean().detach().cpu().item()),
                "near_optimal_fraction_p95": float(torch.quantile(labels.near_optimal_fraction, 0.95).detach().cpu().item()),
                "decisive_points": int(labels.decisive_mask.sum().item()),
                "neighbor_imputed_points": int(labels.neighbor_imputed_mask.sum().item()),
                "midpoint_default_points": int(labels.midpoint_default_mask.sum().item()),
            }
            cache_payload = {
                "x": labels.x.detach().cpu(),
                "u_opt": labels.u_opt.detach().cpu(),
                "omega_opt_next": labels.omega_opt_next.detach().cpu(),
                "control_weight": labels.control_weight.detach().cpu(),
                "near_optimal_fraction": labels.near_optimal_fraction.detach().cpu(),
                "feasible_control_fraction": labels.feasible_control_fraction.detach().cpu(),
                "unmasked_fallback_mask": labels.unmasked_fallback_mask.detach().cpu(),
                "decisive_mask": labels.decisive_mask.detach().cpu(),
                "neighbor_imputed_mask": labels.neighbor_imputed_mask.detach().cpu(),
                "midpoint_default_mask": labels.midpoint_default_mask.detach().cpu(),
                "source_ids": policy_source_ids.detach().cpu(),
                "label_filter_summary": label_filter_summary,
                "metadata": {
                    "system": self.system.name,
                    "n_dim": int(self.system.n_dim),
                    "u_dim": int(self.system.u_dim),
                    "control_grid_points": int(cfg.control_grid_points),
                    "control_grid_total_points": int(control_grid.shape[0]),
                    "policy_label_selection_mode": "argmin",
                    "policy_label_use_confidence_weight": bool(cfg.policy_label_use_confidence_weight),
                    "policy_label_mask_out_of_domain_successors": bool(cfg.policy_label_mask_out_of_domain_successors),
                    "policy_label_near_optimal_rel_tol": float(cfg.policy_label_near_optimal_rel_tol),
                    "policy_label_dos_sublevel": float(cfg.policy_label_dos_sublevel),
                    "omega_checkpoint": str(omega_path),
                },
            }
            label_cache_path = artifact_dir / "policy_grid_labels.pt"
            torch.save(cache_payload, label_cache_path)
            label_filter_summary["saved_policy_label_cache"] = str(label_cache_path)
        x_train = labels.x
        if self.run_context is not None:
            self.run_context.write_json("policy_label_dos_filtration.json", label_filter_summary)
        else:
            (artifact_dir / "policy_label_dos_filtration.json").write_text(
                json.dumps(label_filter_summary, indent=2),
                encoding="utf-8",
            )

        pi = PiNet(
            n_dim=self.system.n_dim,
            u_dim=self.system.u_dim,
            hidden_sizes=cfg.pi_hidden_sizes,
            activation=cfg.pi_activation,
            p_matrix=self.lqr_box_artifacts.P.detach().cpu(),
            k_matrix=self.lqr_box_artifacts.K.detach().cpu(),
            u_lb=self.system.u_lb,
            u_hi=self.system.u_hi,
            c_s=self.lqr_box_artifacts.c_s,
            gate_type="branch_only",
        ).to(self.device)
        optimizer = torch.optim.Adam(pi.parameters(), lr=cfg.pi_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cfg.pi_epochs),
            eta_min=cfg.pi_lr_min,
        )

        print(f"Training saturated-NN PiNet branch for {cfg.pi_epochs} epochs...")
        n_train = labels.x.shape[0]
        if cfg.pi_train_max_points is not None and int(cfg.pi_train_max_points) < n_train:
            n_selected = max(1, int(cfg.pi_train_max_points))
            selection_batcher = FixedStratifiedBatcher(
                source_groups_from_ids(policy_source_ids),
                n_selected,
            )
            selected_indices_cpu = selection_batcher.next_indices(device=torch.device("cpu"))
            labels.x = labels.x[selected_indices_cpu.to(device=self.device)]
            labels.u_opt = labels.u_opt[selected_indices_cpu.to(device=self.device)]
            labels.omega_opt_next = labels.omega_opt_next[selected_indices_cpu.to(device=self.device)]
            if labels.control_weight is not None:
                labels.control_weight = labels.control_weight[selected_indices_cpu.to(device=self.device)]
            if labels.near_optimal_fraction is not None:
                labels.near_optimal_fraction = labels.near_optimal_fraction[selected_indices_cpu.to(device=self.device)]
            if labels.feasible_control_fraction is not None:
                labels.feasible_control_fraction = labels.feasible_control_fraction[selected_indices_cpu.to(device=self.device)]
            if labels.unmasked_fallback_mask is not None:
                labels.unmasked_fallback_mask = labels.unmasked_fallback_mask[selected_indices_cpu.to(device=self.device)]
            if labels.decisive_mask is not None:
                labels.decisive_mask = labels.decisive_mask[selected_indices_cpu.to(device=self.device)]
            if labels.neighbor_imputed_mask is not None:
                labels.neighbor_imputed_mask = labels.neighbor_imputed_mask[selected_indices_cpu.to(device=self.device)]
            if labels.midpoint_default_mask is not None:
                labels.midpoint_default_mask = labels.midpoint_default_mask[selected_indices_cpu.to(device=self.device)]
            policy_source_ids = policy_source_ids[selected_indices_cpu]
            n_train = labels.x.shape[0]
            print(f"  policy train subset    = {n_train} / {label_filter_summary['kept_points']}")
        policy_batcher = FixedStratifiedBatcher(source_groups_from_ids(policy_source_ids), cfg.pi_batch_size)
        policy_eval_batcher = FixedStratifiedBatcher(source_groups_from_ids(policy_source_ids), cfg.pi_eval_points)
        policy_batches_per_epoch = (
            max(1, policy_batcher.batches_per_full_pass())
            if bool(cfg.pi_full_pass_epochs)
            else 1
        )
        if bool(cfg.pi_full_pass_epochs):
            print(
                "  policy full-pass mode  = "
                f"{policy_batches_per_epoch} minibatches/epoch at batch_size={cfg.pi_batch_size}"
            )
        loss_history = []
        for epoch in range(cfg.pi_epochs):
            pi.train()
            epoch_value_loss = 0.0
            epoch_decrease_loss = 0.0
            epoch_control_loss = 0.0
            epoch_total_loss = 0.0
            epoch_weight = 0
            for _ in range(policy_batches_per_epoch):
                indices = policy_batcher.next_indices(device=self.device)
                xb = labels.x[indices]
                target = labels.omega_opt_next[indices]
                u_target = labels.u_opt[indices]
                control_weight = (
                    labels.control_weight[indices]
                    if bool(cfg.policy_label_use_confidence_weight) and labels.control_weight is not None
                    else None
                )
                optimizer.zero_grad()
                value_loss = policy_next_omega_mse(
                    self.system,
                    pi,
                    omega,
                    xb,
                    target,
                    use_stitched_controller=False,
                )
                decrease_loss = policy_omega_decrease_hinge(
                    self.system,
                    pi,
                    omega,
                    xb,
                    kappa=cfg.kappa,
                    use_stitched_controller=False,
                )
                control_loss = policy_control_imitation_mse(
                    self.system,
                    pi,
                    xb,
                    u_target,
                    use_stitched_controller=False,
                    sample_weight=control_weight,
                )
                loss = (
                    value_loss
                    + float(cfg.pi_lambda_decrease) * decrease_loss
                    + float(cfg.pi_lambda_control) * control_loss
                )
                loss.backward()
                if cfg.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(pi.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                batch_count = int(indices.numel())
                epoch_weight += batch_count
                epoch_value_loss += float(value_loss.detach().cpu().item()) * batch_count
                epoch_decrease_loss += float(decrease_loss.detach().cpu().item()) * batch_count
                epoch_control_loss += float(control_loss.detach().cpu().item()) * batch_count
                epoch_total_loss += float(loss.detach().cpu().item()) * batch_count
            scheduler.step()
            denom = max(1, epoch_weight)
            value_loss_scalar = epoch_value_loss / denom
            decrease_loss_scalar = epoch_decrease_loss / denom
            control_loss_scalar = epoch_control_loss / denom
            total_loss_scalar = epoch_total_loss / denom
            loss_history.append(
                {
                    "epoch": int(epoch),
                    "policy_next_omega_mse": value_loss_scalar,
                    "policy_decrease_hinge": decrease_loss_scalar,
                    "policy_control_mse": control_loss_scalar,
                    "total": total_loss_scalar,
                }
            )

            if epoch % cfg.pi_log_every == 0 or epoch == cfg.pi_epochs - 1:
                with torch.no_grad():
                    eval_indices = policy_eval_batcher.next_indices(device=self.device)
                    eval_loss = policy_next_omega_mse(
                        self.system,
                        pi,
                        omega,
                        labels.x[eval_indices],
                        labels.omega_opt_next[eval_indices],
                        use_stitched_controller=False,
                    )
                    eval_decrease_loss = policy_omega_decrease_hinge(
                        self.system,
                        pi,
                        omega,
                        labels.x[eval_indices],
                        kappa=cfg.kappa,
                        use_stitched_controller=False,
                    )
                    eval_control_loss = policy_control_imitation_mse(
                        self.system,
                        pi,
                        labels.x[eval_indices],
                        labels.u_opt[eval_indices],
                        use_stitched_controller=False,
                        sample_weight=(
                            labels.control_weight[eval_indices]
                            if bool(cfg.policy_label_use_confidence_weight) and labels.control_weight is not None
                            else None
                        ),
                    )
                print(
                    f"  epoch {epoch:5d} | "
                    f"value_loss={value_loss_scalar:.6g} | "
                    f"decrease={decrease_loss_scalar:.6g} | "
                    f"control_loss={control_loss_scalar:.6g} | "
                    f"total={total_loss_scalar:.6g} | "
                    f"eval_value={eval_loss.item():.6g} | "
                    f"eval_decrease={eval_decrease_loss.item():.6g} | "
                    f"eval_control={eval_control_loss.item():.6g}"
                )

        pi.eval()
        policy_report = None
        report_points = min(int(cfg.policy_decrease_report_points), int(n_train))
        if report_points > 0:
            report_batcher = FixedStratifiedBatcher(
                source_groups_from_ids(policy_source_ids),
                report_points,
            )
            report_indices = report_batcher.next_indices(device=self.device)
            report_source_ids = policy_source_ids[report_indices.detach().cpu()].to(device=self.device)
            policy_report = evaluate_policy_decrease_report(
                self.system,
                pi,
                omega,
                self.lqr_box_artifacts.P.to(device=self.device, dtype=torch.float32),
                labels.x[report_indices],
                report_source_ids,
                batch_size=cfg.policy_decrease_report_batch_size,
                kappa=cfg.kappa,
                eps=self.system.verification.eps if self.system.verification else 0.0,
                c_s=self.lqr_box_artifacts.c_s,
                domain=tuple(zip(self.system.x_lb, self.system.x_hi)),
            )
            if self.run_context is not None:
                self.run_context.write_json("policy_decrease_report.json", policy_report)
            else:
                (artifact_dir / "policy_decrease_report.json").write_text(
                    json.dumps(policy_report, indent=2),
                    encoding="utf-8",
                )
            overall = policy_report["overall"]
            print("Policy omega_orig decrease report:")
            print(
                "  violations = "
                f"{overall['omega_orig_violation_count']} / {overall['count']} "
                f"({100.0 * overall['omega_orig_violation_fraction']:.3f}%), "
                f"max_delta={overall['omega_orig_max_delta']:.6g}, "
                f"p95_delta={overall['omega_orig_p95_delta']:.6g}"
            )
        checkpoint = {
            "model_state_dict": pi.state_dict(),
            "n_dim": int(self.system.n_dim),
            "u_dim": int(self.system.u_dim),
            "hidden_sizes": list(cfg.pi_hidden_sizes),
            "P": self.lqr_box_artifacts.P.detach().cpu().numpy().astype(float).tolist(),
            "K": self.lqr_box_artifacts.K.detach().cpu().numpy().astype(float).tolist(),
            "u_lb": [float(v) for v in self.system.u_lb],
            "u_hi": [float(v) for v in self.system.u_hi],
            "activation": str(cfg.pi_activation),
            "gate_type": "branch_only",
            "c_analytic": float(self.lqr_box_artifacts.c_analytic),
            "C_S": float(self.lqr_box_artifacts.c_s),
            "c_sigmoid_center": None,
            "sample_cap": float(self.lqr_box_artifacts.c_sigmoid_center),
            "c1_scale": float(cfg.c1_scale),
            "sharpness": None,
            "policy_loss": (
                "Train the zero-at-origin saturated NN branch: "
                "MSE(omega(f(x, pi_nn(x))), min_u_grid omega(f(x,u))) "
                "+ pi_lambda_decrease * ReLU(omega(f(x,pi_nn(x))) - (1-kappa)*omega(x)) "
                "+ pi_lambda_control * weighted_MSE((pi_nn(x)-u_opt_grid)/(u_hi-u_lb))"
            ),
            "policy_label_selection_mode": "argmin",
            "policy_label_near_optimal_rel_tol": float(cfg.policy_label_near_optimal_rel_tol),
            "control_imitation_weighting": (
                "control_weight=1-near_optimal_fraction over the full control grid"
                if bool(cfg.policy_label_use_confidence_weight)
                else "uniform control_weight=1"
            ),
            "training_controller": "saturated_nn_branch",
            "saved_controller": (
                "Use PiNet.nn_control(x) for the learned branch and "
                "PiNet.lqr_control(x) for the clamped linear branch."
            ),
            "forward_controller_note": "For branch_only checkpoints, PiNet.forward(x) returns PiNet.nn_control(x).",
            "pi_lambda_control": float(cfg.pi_lambda_control),
            "pi_lambda_decrease": float(cfg.pi_lambda_decrease),
            "policy_decrease_report": policy_report,
            "label_source": {
                "omega_checkpoint": str(omega_path),
                "states": int(x_train.shape[0]),
                "raw_states": int(label_filter_summary["input_points"]),
                "state_box_boundary_points": 0,
                "raw_state_box_boundary_points": 0,
                "policy_ellipse_boundary_points": int(label_filter_summary.get("policy_ellipse_boundary_points", 0)),
                "policy_c1_to_lqr_c2_band_points": int(label_filter_summary.get("policy_ellipse_boundary_points", 0)),
                "raw_policy_ellipse_boundary_points": int(label_filter_summary.get("policy_ellipse_points_requested", 0)),
                "policy_uniform_points": int(label_filter_summary.get("policy_uniform_points", 0)),
                "raw_policy_uniform_points": int(label_filter_summary.get("policy_uniform_points_requested", 0)),
                "rejection_sublevel_points": int(label_filter_summary.get("policy_rejection_omega_sublevel_points", 0)),
                "raw_gd_points_reallocated_to_rejection": int(label_filter_summary.get("policy_gd_points_reallocated_to_rejection", 0)),
                "optimized_domain_boundary_omega_one_points": 0,
                "uniform_candidate_points": int(label_filter_summary.get("policy_uniform_points", 0)),
                "raw_uniform_candidate_points": int(label_filter_summary.get("policy_uniform_points_requested", 0)),
                "dos_filtration": label_filter_summary,
                "control_grid_points_per_dim": int(cfg.control_grid_points),
                "control_grid_total_points": int(control_grid.shape[0]),
            },
        }
        save_path = artifact_dir / self.system.checkpoints.pi
        torch.save(checkpoint, save_path)
        save_loss_history_artifacts(
            self.run_context,
            artifact_dir,
            stem="policy",
            title="PiNet Training Loss",
            history=loss_history,
        )
        save_policy_control_comparison_plot(
            self.run_context,
            artifact_dir,
            pi=pi,
            labels=labels,
            max_points=cfg.pi_plot_points,
        )
        print(f"PiNet checkpoint saved to {save_path}")

    def train_closed_loop_omega(self) -> None:
        # Working notebook reference: Block 8.
        #
        # Closed-loop omega can either copy original omega's embedded architecture
        # or train a fresh state-input value under the selected controller:
        #
        #   f_cl(x) = f_sys(x, pi_branch(x))
        #   zubov residual = omega_cl(x) - omega_cl(f_cl(x))
        #                    - xi(x) * (1 - omega_cl(f_cl(x)))
        #
        # The optimized objective uses two independently sampled datasets:
        #   - trajectory points receive only finite-horizon label MSE;
        #   - constraint points receive the Zubov residual, range guard, and
        #     DOS-masked multiplicative decrease hinge
        #     relu(w_next - (1-kappa) * w_x).
        #
        # This stage is closed-loop and uses singleton successor states; that is
        # a different object from original omega's interval residual.
        if self.lqr_box_artifacts is None:
            self.setup_lqr_box()
        assert self.lqr_box_artifacts is not None

        cfg = self.system.training
        closed_loop_controller_mode = str(cfg.closed_loop_controller_mode).strip().lower()
        if closed_loop_controller_mode in {"nn", "neural", "neural_net", "saturated_nn", "tanh_nn"}:
            closed_loop_controller_mode = "nn"
        if closed_loop_controller_mode not in {"nn", "lqr"}:
            raise ValueError(
                "closed_loop_controller_mode must be 'nn' or 'lqr'; blended/sigmoid "
                "controllers are not part of the discontinuous-switch proof path, "
                f"got {cfg.closed_loop_controller_mode!r}."
            )
        counts = resolve_data_distribution_counts(self.system)
        artifact_dir = self.artifact_dir()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if cfg.closed_loop_label_batch_size <= 0:
            raise ValueError("closed_loop_label_batch_size must be positive.")
        if cfg.closed_loop_train_batch_size <= 0:
            raise ValueError("closed_loop_train_batch_size must be positive.")

        pi_path = self.checkpoint_path(self.system.checkpoints.pi, "policy")
        omega_path = self.checkpoint_path(self.system.checkpoints.omega, "original omega")

        pi = load_pinet(pi_path, device=self.device)
        if pi.n_dim != self.system.n_dim or pi.u_dim != self.system.u_dim:
            raise ValueError(
                f"Policy checkpoint is incompatible with system {self.system.name!r}: "
                f"checkpoint has n_dim={pi.n_dim}, u_dim={pi.u_dim}, "
                f"but system requires n_dim={self.system.n_dim}, u_dim={self.system.u_dim}. "
                f"Checkpoint path: {pi_path}"
            )
        pi.eval()
        for param in pi.parameters():
            param.requires_grad_(False)

        original_omega = load_omeganet(
            omega_path,
            n_dim=self.system.n_dim,
            device=self.device,
        )
        original_omega.eval()
        for param in original_omega.parameters():
            param.requires_grad_(False)
        initialize_from_original_omega = bool(cfg.closed_loop_initialize_from_original_omega)
        if initialize_from_original_omega:
            closed_loop_hidden_sizes = tuple(hidden_sizes_from_state_dict(original_omega.state_dict()))
            omega_cl = OmegaNet(
                n_dim=self.system.n_dim,
                hidden_sizes=closed_loop_hidden_sizes,
                input_dim=original_omega.input_dim,
                embedding_mode=original_omega.embedding_mode,
            ).to(self.device)
            omega_cl.load_state_dict(original_omega.state_dict())
            omega_initialization = "copy_original_omega"
            initialized_from_original_omega = str(omega_path)
        else:
            closed_loop_hidden_sizes = tuple(cfg.closed_loop_hidden_sizes)
            omega_cl = OmegaNet(
                n_dim=self.system.n_dim,
                hidden_sizes=closed_loop_hidden_sizes,
                input_dim=self.system.n_dim,
                embedding_mode="state",
            ).to(self.device)
            omega_initialization = "fresh_state_input"
            initialized_from_original_omega = None
        require_trainable_parameters(omega_cl, name="closed-loop omega")

        p_matrix = self.lqr_box_artifacts.P.to(device=self.device, dtype=torch.float32)
        state_bounds = tuple(zip(self.system.x_lb, self.system.x_hi))

        def sample_closed_loop_states(
            *,
            ellipse_points: int,
            uniform_points: int,
            optimized_points: int,
        ) -> Tuple[torch.Tensor, Dict[str, object], torch.Tensor]:
            ellipse = self.sample_lqr_c2_band(
                p_matrix,
                ellipse_points,
            )
            rejection_points = int(optimized_points)
            rejected, rejection_stats = sample_omega_rejection_points(
                original_omega,
                state_bounds,
                rejection_points,
                device=self.device,
                sublevel=cfg.closed_loop_dos_sublevel,
                batch_size=max(1, int(cfg.closed_loop_label_batch_size)),
                max_attempts=cfg.optimized_sampler_max_attempts,
            )
            uniform = sample_uniform_bounds(
                state_bounds,
                uniform_points,
                device=self.device,
            ).to(dtype=torch.float32)
            if cfg.closed_loop_optimized_sampler_strict_sublevel and rejected.shape[0] == 0:
                raise ValueError(
                    f"{self.system.name}: strict closed-loop rejection sampler returned no "
                    f"omega_orig(x)<={cfg.closed_loop_dos_sublevel} points."
                )
            states = torch.cat([ellipse, uniform, rejected], dim=0)
            source_ids = torch.cat(
                [
                    torch.full((ellipse.shape[0],), 0, dtype=torch.long),
                    torch.full((uniform.shape[0],), 1, dtype=torch.long),
                    torch.full((rejected.shape[0],), 2, dtype=torch.long),
                ],
                dim=0,
            )
            return states, {
                "ellipse_points": int(ellipse.shape[0]),
                "uniform_points": int(uniform.shape[0]),
                "optimized_points": 0,
                "rejection_points": int(rejected.shape[0]),
                "requested_rejection_points": int(rejection_points),
                "gd_points_reallocated_to_rejection": int(optimized_points),
                "band": {
                    "mode": "c1_to_C_S_band",
                    "requested_points": int(ellipse_points),
                    "returned_points": int(ellipse.shape[0]),
                    "c1": float(self.lqr_box_artifacts.c_analytic),
                    "C_S": float(self.lqr_box_artifacts.c_s),
                    "band_c_scale": float(cfg.c1_scale),
                },
                "returned_points": int(states.shape[0]),
                "uniform_bounds": [
                    [float(lo), float(hi)]
                    for lo, hi in zip(self.system.x_lb, self.system.x_hi)
                ],
                "rejection": rejection_stats,
            }, source_ids

        trajectory_states, trajectory_sampling_stats, trajectory_source_ids = sample_closed_loop_states(
            ellipse_points=counts.closed_loop_trajectory_ellipse_points,
            uniform_points=counts.closed_loop_trajectory_uniform_points,
            optimized_points=counts.closed_loop_trajectory_gd_omega_sublevel_points,
        )
        constraint_states, constraint_sampling_stats, constraint_source_ids = sample_closed_loop_states(
            ellipse_points=counts.closed_loop_constraint_ellipse_points,
            uniform_points=counts.closed_loop_constraint_uniform_points,
            optimized_points=counts.closed_loop_constraint_gd_omega_sublevel_points,
        )

        print(f"  pi checkpoint          = {pi_path}")
        print(f"  omega_cl init          = {omega_initialization}")
        print(f"  initialized from omega = {initialized_from_original_omega}")
        print(f"  omega_cl input dim     = {omega_cl.input_dim}")
        print(f"  omega_cl embedding     = {omega_cl.embedding_mode}")
        print(f"  trajectory points      = {trajectory_states.shape[0]}")
        print(f"  constraint points      = {constraint_states.shape[0]}")
        print(f"  omega_cl controller    = {closed_loop_controller_mode}")
        print(f"  optimized sublevel     = {cfg.closed_loop_dos_sublevel}")
        print(f"  decrease DOS gate      = {cfg.closed_loop_loss_dos_gate}")
        print(f"  horizon                = {cfg.closed_loop_horizon}")
        print(f"  label batch size       = {cfg.closed_loop_label_batch_size}")
        print(f"  train batch size       = {cfg.closed_loop_train_batch_size}")

        closed_loop_data_target = str(cfg.closed_loop_data_target).strip().lower()
        if closed_loop_data_target not in {"trajectory", "finite_horizon", "omega_orig"}:
            raise ValueError(
                "closed_loop_data_target must be 'trajectory' or 'omega_orig', "
                f"got {cfg.closed_loop_data_target!r}."
            )

        def build_closed_loop_label_batch(x_train: torch.Tensor) -> OmegaLabelBatch:
            if closed_loop_data_target == "omega_orig":
                print(
                    "Preparing omega_orig pointwise targets for the trajectory dataset only..."
                )
            else:
                print(
                    "Preparing finite-horizon labels for the trajectory dataset only..."
                )
            if x_train.shape[0] == 0:
                return OmegaLabelBatch(
                    x=torch.empty(0, self.system.n_dim),
                    z_x=torch.empty(0, omega_cl.input_dim),
                    target_w=torch.empty(0),
                )
            label_batches = []
            for start in range(0, x_train.shape[0], cfg.closed_loop_label_batch_size):
                xb = x_train[start : start + cfg.closed_loop_label_batch_size]
                if closed_loop_data_target == "omega_orig":
                    xb_device = xb.to(self.device)
                    target_w = original_omega.query(xb_device).reshape(-1)
                    labels = OmegaLabelBatch(
                        x=xb_device.detach().cpu(),
                        z_x=embed_pointwise_value_input(
                            self.system,
                            xb_device,
                            input_dim=omega_cl.input_dim,
                        ).detach().cpu(),
                        target_w=target_w.detach().cpu(),
                    )
                else:
                    labels = compute_closed_loop_zubov_targets(
                        self.system,
                        pi,
                        xb,
                        p_matrix,
                        horizon=cfg.closed_loop_horizon,
                        controller_mode=closed_loop_controller_mode,
                        value_input_dim=omega_cl.input_dim,
                    )
                label_batches.append(
                    OmegaLabelBatch(
                        x=labels.x.detach().cpu(),
                        z_x=labels.z_x.detach().cpu(),
                        target_w=labels.target_w.detach().cpu(),
                    )
                )
            return OmegaLabelBatch(
                x=torch.cat([batch.x for batch in label_batches], dim=0),
                z_x=torch.cat([batch.z_x for batch in label_batches], dim=0),
                target_w=torch.cat([batch.target_w for batch in label_batches], dim=0),
            )

        trajectory_label_batch = build_closed_loop_label_batch(trajectory_states)
        if cfg.closed_loop_lambda_data != 0.0 and trajectory_label_batch.x.shape[0] == 0:
            raise ValueError(
                "Closed-loop trajectory MSE is active but its combined dataset is empty."
            )
        if constraint_states.shape[0] == 0:
            raise ValueError(
                "Closed-loop Zubov residual is active but its combined constraint dataset is empty."
            )
        require_trainable_parameters(omega_cl, name="closed-loop omega after label construction")
        save_finite_horizon_target_plot(
            self.run_context,
            artifact_dir,
            stem="omega_closed_loop_trajectory",
            title="Closed-Loop Omega Trajectory Finite-Horizon Targets",
            labels=trajectory_label_batch,
            p_matrix=p_matrix,
            c_s=self.lqr_box_artifacts.c_s,
        )

        def closed_loop_step(x: torch.Tensor) -> torch.Tensor:
            return self.system.dynamics(
                x,
                policy_control_for_mode(pi, x, closed_loop_controller_mode),
            )

        optimizer = torch.optim.Adam(omega_cl.parameters(), lr=cfg.closed_loop_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, cfg.closed_loop_epochs // 2),
            gamma=0.5,
        )

        print(f"Training closed-loop omega for {cfg.closed_loop_epochs} epochs...")
        x_origin = torch.zeros(1, self.system.n_dim, device=self.device, dtype=torch.float32)
        n_trajectory = trajectory_label_batch.x.shape[0]
        constraint_states_cpu = constraint_states.detach().cpu()
        n_constraint = constraint_states_cpu.shape[0]
        trajectory_batcher = FixedStratifiedBatcher(source_groups_from_ids(trajectory_source_ids), cfg.closed_loop_train_batch_size)
        constraint_batcher = FixedStratifiedBatcher(source_groups_from_ids(constraint_source_ids), cfg.closed_loop_train_batch_size)
        closed_loop_batches_per_epoch = 1
        if bool(cfg.closed_loop_full_pass_epochs):
            batch_counts = [constraint_batcher.batches_per_full_pass()]
            if n_trajectory > 0:
                batch_counts.append(trajectory_batcher.batches_per_full_pass())
            closed_loop_batches_per_epoch = max(1, *batch_counts)
            print(
                "  closed-loop omega full-pass mode = "
                f"{closed_loop_batches_per_epoch} minibatches/epoch "
                f"at batch_size={cfg.closed_loop_train_batch_size}"
            )
        loss_history = []
        for epoch in range(cfg.closed_loop_epochs):
            omega_cl.train()
            epoch_zubov_loss = 0.0
            epoch_data_loss = 0.0
            epoch_decrease_loss = 0.0
            epoch_range_loss = 0.0
            epoch_boundary_loss = 0.0
            epoch_total_loss = 0.0
            epoch_weight = 0
            epoch_trajectory_weight = 0
            epoch_constraint_weight = 0
            epoch_grad_max = 0.0
            for _ in range(closed_loop_batches_per_epoch):
                if n_trajectory > 0:
                    trajectory_indices = trajectory_batcher.next_indices()
                    trajectory_batch = OmegaLabelBatch(
                        x=trajectory_label_batch.x[trajectory_indices].to(self.device),
                        z_x=trajectory_label_batch.z_x[trajectory_indices].to(self.device),
                        target_w=trajectory_label_batch.target_w[trajectory_indices].to(self.device),
                    )
                else:
                    trajectory_batch = OmegaLabelBatch(
                        x=torch.empty(0, self.system.n_dim, device=self.device),
                        z_x=torch.empty(0, omega_cl.input_dim, device=self.device),
                        target_w=torch.empty(0, device=self.device),
                    )
                constraint_indices = constraint_batcher.next_indices()
                constraint_batch = constraint_states_cpu[constraint_indices].to(self.device)
                optimizer.zero_grad()
                losses = closed_loop_omega_loss_terms(
                    omega_cl,
                    original_omega,
                    constraint_batch,
                    trajectory_batch,
                    closed_loop_step,
                    p_matrix,
                    system=self.system,
                    lambda_zubov=cfg.closed_loop_lambda_zubov,
                    lambda_data=cfg.closed_loop_lambda_data,
                    lambda_decrease=cfg.closed_loop_lambda_decrease,
                    lambda_range=cfg.closed_loop_lambda_range,
                    lambda_boundary=cfg.closed_loop_lambda_boundary,
                    dos_gate=cfg.closed_loop_loss_dos_gate,
                    kappa=cfg.kappa,
                    anchor_filter=cfg.closed_loop_anchor_filter,
                    x_origin=x_origin,
                )
                losses.total.backward()
                grad_max = max(
                    (
                        float(param.grad.detach().abs().max().cpu().item())
                        for param in omega_cl.parameters()
                        if param.grad is not None
                    ),
                    default=0.0,
                )
                epoch_grad_max = max(epoch_grad_max, grad_max)
                if cfg.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(omega_cl.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                trajectory_count = int(trajectory_batch.x.shape[0])
                constraint_count = int(constraint_batch.shape[0])
                batch_count = trajectory_count + constraint_count
                epoch_weight += batch_count
                epoch_trajectory_weight += trajectory_count
                epoch_constraint_weight += constraint_count
                epoch_zubov_loss += float(losses.zubov_mse.detach().cpu().item()) * constraint_count
                epoch_data_loss += float(losses.data_mse.detach().cpu().item()) * trajectory_count
                epoch_decrease_loss += float(losses.decrease_penalty.detach().cpu().item()) * constraint_count
                epoch_range_loss += float(losses.range_penalty.detach().cpu().item()) * constraint_count
                epoch_boundary_loss += float(losses.boundary_condition_mse.detach().cpu().item()) * batch_count
                epoch_total_loss += float(losses.total.detach().cpu().item()) * batch_count
            scheduler.step()
            denom = max(1, epoch_weight)
            zubov_loss_scalar = epoch_zubov_loss / max(1, epoch_constraint_weight)
            data_loss_scalar = epoch_data_loss / max(1, epoch_trajectory_weight)
            decrease_loss_scalar = epoch_decrease_loss / max(1, epoch_constraint_weight)
            range_loss_scalar = epoch_range_loss / max(1, epoch_constraint_weight)
            boundary_loss_scalar = epoch_boundary_loss / denom
            total_loss_scalar = epoch_total_loss / denom
            loss_history.append(
                {
                    "epoch": int(epoch),
                    "zubov_mse": zubov_loss_scalar,
                    "finite_trajectory_mse": data_loss_scalar,
                    "decrease_violation": decrease_loss_scalar,
                    "range_penalty": range_loss_scalar,
                    "boundary_condition_mse": boundary_loss_scalar,
                    "grad_max": epoch_grad_max,
                    "total": total_loss_scalar,
                    "minibatches": int(closed_loop_batches_per_epoch),
                    "examples_seen": int(epoch_weight),
                }
            )

            if epoch % cfg.closed_loop_log_every == 0 or epoch == cfg.closed_loop_epochs - 1:
                print(
                    f"  epoch {epoch:5d} | "
                    f"L_zubov={zubov_loss_scalar:.6g} | "
                    f"L_data={data_loss_scalar:.6g} | "
                    f"L_decrease={decrease_loss_scalar:.6g} | "
                    f"L_range={range_loss_scalar:.6g} | "
                    f"L_bc={boundary_loss_scalar:.6g} | "
                    f"grad_max={epoch_grad_max:.3g} | "
                    f"total={total_loss_scalar:.6g}"
                )

        omega_cl.eval()
        checkpoint = {
            "model_state_dict": omega_cl.state_dict(),
            "n_dim": int(self.system.n_dim),
            "input_dim": int(omega_cl.input_dim),
            "embedding_mode": str(omega_cl.embedding_mode),
            "reachable_set_embedding_mode": str(omega_cl.embedding_mode),
            "hidden_sizes": list(closed_loop_hidden_sizes),
            "configured_closed_loop_hidden_sizes": list(cfg.closed_loop_hidden_sizes),
            "output_activation": "sigmoid",
            "x_lb": [float(v) for v in self.system.x_lb],
            "x_hi": [float(v) for v in self.system.x_hi],
            "u_lb": [float(v) for v in self.system.u_lb],
            "u_hi": [float(v) for v in self.system.u_hi],
            "dt": float(self.system.dt),
            "P": self.lqr_box_artifacts.P.detach().cpu().numpy().astype(float).tolist(),
            "K": self.lqr_box_artifacts.K.detach().cpu().numpy().astype(float).tolist(),
            "c_analytic": float(self.lqr_box_artifacts.c_analytic),
            "sample_cap": float(self.lqr_box_artifacts.c_sigmoid_center),
            "C_S": float(self.lqr_box_artifacts.c_s),
            "zubov_alpha_scale": float(cfg.zubov_alpha_scale),
            "closed_loop_loss": {
                "residual": "closed-loop Zubov residual MSE on constraint points only",
                "data_mse": "closed-loop data-target MSE on trajectory points only",
                "data_target": str(closed_loop_data_target),
                "anchor_filter": str(cfg.closed_loop_anchor_filter),
                "finite_trajectory_mse": "legacy name for closed-loop data-target MSE on trajectory points only",
                "decrease_violation": "multiplicative decrease hinge relu(w_next - (1-kappa)*w_x) on constraint points satisfying omega_orig(x)<=dos_gate",
                "range_penalty": "outside original-omega DOS gate, penalize (1 - omega_cl(x))^2",
                "boundary_condition_mse": "origin anchor omega_cl(0)^2",
                "lambda_zubov": float(cfg.closed_loop_lambda_zubov),
                "lambda_data": float(cfg.closed_loop_lambda_data),
                "lambda_decrease": float(cfg.closed_loop_lambda_decrease),
                "lambda_range": float(cfg.closed_loop_lambda_range),
                "lambda_boundary": float(cfg.closed_loop_lambda_boundary),
                "configured_lambda_range": float(cfg.closed_loop_lambda_range),
                "dos_gate": float(cfg.closed_loop_loss_dos_gate),
                "optimized_sampling_sublevel": float(cfg.closed_loop_dos_sublevel),
                "datasets_are_disjoint": True,
                "kappa": float(cfg.kappa),
                "controller_mode": closed_loop_controller_mode,
                "omega_initialization": omega_initialization,
                "closed_loop_initialize_from_original_omega": initialize_from_original_omega,
            },
            "sampling": {
                "closed_loop_dos_sublevel": float(cfg.closed_loop_dos_sublevel),
                "closed_loop_loss_dos_gate": float(cfg.closed_loop_loss_dos_gate),
                "trajectory": {
                    "losses": ["finite_trajectory_mse"],
                    "requested_ellipse_points": int(
                        counts.closed_loop_trajectory_ellipse_points
                    ),
                    "requested_uniform_points": int(
                        counts.closed_loop_trajectory_uniform_points
                    ),
                    "gd_points_reallocated_to_rejection": int(
                        counts.closed_loop_trajectory_gd_omega_sublevel_points
                    ),
                    "requested_rejection_points": int(
                        counts.closed_loop_trajectory_ellipse_points
                        + counts.closed_loop_trajectory_gd_omega_sublevel_points
                    ),
                    **trajectory_sampling_stats,
                },
                "constraint": {
                    "losses": ["zubov_residual", "range", "lyapunov_decrease"],
                    "requested_ellipse_points": int(
                        counts.closed_loop_constraint_ellipse_points
                    ),
                    "requested_uniform_points": int(
                        counts.closed_loop_constraint_uniform_points
                    ),
                    "gd_points_reallocated_to_rejection": int(
                        counts.closed_loop_constraint_gd_omega_sublevel_points
                    ),
                    "requested_rejection_points": int(
                        counts.closed_loop_constraint_ellipse_points
                        + counts.closed_loop_constraint_gd_omega_sublevel_points
                    ),
                    **constraint_sampling_stats,
                },
                "sampler": "independent_c1_to_lqr_c2_band_plus_uniform_plus_uniform_rejection_omega_orig_sublevel_sets",
                "closed_loop_horizon": int(cfg.closed_loop_horizon),
                "closed_loop_label_batch_size": int(cfg.closed_loop_label_batch_size),
                "closed_loop_train_batch_size": int(cfg.closed_loop_train_batch_size),
                "full_pass_epochs": bool(cfg.closed_loop_full_pass_epochs),
                "minibatches_per_epoch": int(closed_loop_batches_per_epoch),
                "controller_checkpoint": str(pi_path),
                "controller_mode": closed_loop_controller_mode,
                "initialized_from_original_omega": initialized_from_original_omega,
            },
        }
        save_path = artifact_dir / self.system.checkpoints.omega_closed_loop
        torch.save(checkpoint, save_path)
        save_loss_history_artifacts(
            self.run_context,
            artifact_dir,
            stem="omega_closed_loop",
            title="Closed-Loop Omega Training Losses",
            history=loss_history,
        )
        validation_total = int(cfg.closed_loop_validation_points)
        if validation_total > 0:
            val_ellipse, val_uniform, val_rejection = _split_count(
                validation_total,
                (
                    float(counts.closed_loop_constraint_ellipse_points),
                    float(counts.closed_loop_constraint_uniform_points),
                    float(counts.closed_loop_constraint_gd_omega_sublevel_points),
                ),
            )
            validation_states, validation_sampling_stats, validation_source_ids = sample_closed_loop_states(
                ellipse_points=val_ellipse,
                uniform_points=val_uniform,
                optimized_points=val_rejection,
            )
            validation_report = evaluate_closed_loop_decrease_report(
                self.system,
                pi,
                original_omega,
                omega_cl,
                p_matrix,
                validation_states,
                validation_source_ids,
                controller_mode=closed_loop_controller_mode,
                batch_size=int(cfg.closed_loop_validation_batch_size),
                dos_gate=float(cfg.closed_loop_loss_dos_gate),
                eps=float(self.system.verification.eps if self.system.verification is not None else 0.0),
                domain=tuple(zip(self.system.x_lb, self.system.x_hi)),
            )
            validation_report["sampling"] = validation_sampling_stats
            validation_report["requested_points"] = validation_total
            if self.run_context is not None:
                self.run_context.write_json("omega_closed_loop_heldout_decrease_report.json", validation_report)
            (artifact_dir / "omega_closed_loop_heldout_decrease_report.json").write_text(
                json.dumps(validation_report, indent=2),
                encoding="utf-8",
            )
            overall = validation_report["overall"]
            print("Closed-loop held-out decrease report:")
            print(
                "  omega_cl violations = "
                f"{overall['omega_cl_violation_count']} / {overall['count']} "
                f"({overall['omega_cl_violation_fraction']:.3%}), "
                f"max_delta={overall['omega_cl_max_delta']:.6g}, "
                f"p95_delta={overall['omega_cl_p95_delta']:.6g}"
            )
        print(f"Closed-loop omega checkpoint saved to {save_path}")

    def search_thresholds(self) -> None:
        """Sample c2, w1, w2 and record raw/rescaled witness data. NO C2 CAP

        This stage approximates the infima in the threshold definitions by
        throwing many uniform samples at the learned functions over X_v. It is
        intentionally not a formal proof. The resulting rescaled constants are
        written separately so the verification stage can consume the reviewed
        values later.
        """
        if self.lqr_box_artifacts is None:
            self.setup_lqr_box()
        assert self.lqr_box_artifacts is not None
        if self.system.verification is None:
            raise ValueError(f"{self.system.name}: verification domain/eps are required for threshold search.")

        cfg = self.system.training
        artifact_dir = self.artifact_dir()
        artifact_dir.mkdir(parents=True, exist_ok=True)

        pi_path = self.checkpoint_path(self.system.checkpoints.pi, "policy")
        omega_cl_path = self.checkpoint_path(
            self.system.checkpoints.omega_closed_loop,
            "closed-loop omega",
        )
        omega_path = self.checkpoint_path(self.system.checkpoints.omega, "original omega")

        pi = load_pinet(pi_path, device=self.device)
        omega_orig = load_omeganet(omega_path, n_dim=self.system.n_dim, device=self.device)
        omega_cl = load_omeganet(omega_cl_path, n_dim=self.system.n_dim, device=self.device)
        p_matrix = self.lqr_box_artifacts.P.to(device=self.device, dtype=torch.float32)

        print("Running grid threshold search...")
        print(f"  target points = {cfg.threshold_search_samples}")
        print(f"  batch_size    = {cfg.threshold_search_batch_size}")
        print(
            "  scales        = "
            f"c2={cfg.threshold_search_c2_scale if cfg.threshold_search_c2_scale is not None else cfg.threshold_search_scale}, "
            f"w1={cfg.threshold_search_w1_scale if cfg.threshold_search_w1_scale is not None else cfg.threshold_search_scale}, "
            f"w2={cfg.threshold_search_w2_scale if cfg.threshold_search_w2_scale is not None else cfg.threshold_search_scale}"
        )
        print(f"  X_v           = {self.system.verification.domain}")
        print(f"  c1 search gate = {self.lqr_box_artifacts.c_analytic:.9g} (eta-analytic)")
        print(f"  sample cap = {self.lqr_box_artifacts.c_sigmoid_center:.9g}")
        print(f"  C_S            = {self.lqr_box_artifacts.c_s:.9g}")
        c2_hard_cap = min(
            float(self.lqr_box_artifacts.c_input),
            float(self.lqr_box_artifacts.c_box),
        )
        print(f"  c2 containment diagnostic = {c2_hard_cap:.9g} (input/domain)")
        print(f"    c_input      = {self.lqr_box_artifacts.c_input:.9g}")
        print(f"    domain box   = {self.lqr_box_artifacts.c_box:.9g}")
        print(f"    LQR grid     = {self.lqr_box_artifacts.c_nonlinear_decrease_grid:.9g}")
        result = sampled_threshold_search(
            self.system,
            pi,
            omega_cl,
            p_matrix,
            omega_original=omega_orig,
            c1_raw=self.lqr_box_artifacts.c_analytic,
            c_s=self.lqr_box_artifacts.c_s,
            n_samples=cfg.threshold_search_samples,
            batch_size=cfg.threshold_search_batch_size,
            scale=cfg.threshold_search_scale,
            c2_scale=cfg.threshold_search_c2_scale,
            w1_scale=cfg.threshold_search_w1_scale,
            w2_scale=cfg.threshold_search_w2_scale,
            device=self.device,
        )

        full_payload = asdict(result)
        verifier_payload = {
            "eps": float(result.eps),
            "c1": float(result.c1_raw),
            "c2": result.c2_verify,
            "w1": result.w1_verify,
            "w2": result.w2_verify,
            "raw": {
                "c1_search": result.c1_raw,
                "c_analytic": float(self.lqr_box_artifacts.c_analytic),
                "sample_cap": float(self.lqr_box_artifacts.c_sigmoid_center),
                "C_S": result.c_s,
                "c2_cap": result.c2_cap,
                "c2_cap_input": float(self.lqr_box_artifacts.c_input),
                "c2_cap_active_domain": float(self.lqr_box_artifacts.c_box),
                "c2_lqr_grid_decrease_diagnostic": float(self.lqr_box_artifacts.c_nonlinear_decrease_grid),
                "c2": result.c2_raw,
                "w1": result.w1_raw,
                "w2": result.w2_raw,
                "w1_original": result.w1_original_raw,
                "w2_original": result.w2_original_raw,
            },
            "original_omega": {
                "w1": result.w1_original_verify,
                "w2": result.w2_original_verify,
                "raw": {
                    "w1": result.w1_original_raw,
                    "w2": result.w2_original_raw,
                },
                "source": "sampled_threshold_search_original_omega_diagnostics",
            },
            "scale": result.scale,
            "scales": {
                "c2": result.c2_scale,
                "w1": result.w1_scale,
                "w2": result.w2_scale,
            },
            "controller_switch": "clamped_lqr_inside_nu_c2_discontinuous_switch_to_saturated_nn_after_omega_w1",
            "source": "sampled_threshold_search_clamped_lqr_and_saturated_nn",
        }

        rows = []
        for name, witness in (
            ("c2", result.c2_witness),
            ("w1", result.w1_witness),
            ("w2", result.w2_witness),
            ("w1_original", result.w1_original_witness),
            ("w2_original", result.w2_original_witness),
            ("band_bad", result.band_bad_witness),
        ):
            if witness is None:
                continue
            row = {"name": name, "reason": witness["reason"]}
            row["controller_branch"] = witness.get("controller_branch") or ""
            row["c2_raw"] = "" if result.c2_raw is None else result.c2_raw
            row["c2_verify"] = "" if result.c2_verify is None else result.c2_verify
            row["nu_minus_c2_verify"] = (
                ""
                if result.c2_verify is None
                else float(witness["nu"]) - float(result.c2_verify)
            )
            row.update({f"x{i}": value for i, value in enumerate(witness["x"])})
            row.update({f"fx{i}": value for i, value in enumerate(witness["f_pi_x"])})
            for key in (
                "nu",
                "nu_next",
                "delta_nu",
                "omega_cl",
                "omega_cl_next",
                "delta_omega_cl",
                "omega_original",
                "omega_original_next",
                "delta_omega_original",
            ):
                row[key] = witness[key]
            row["f_pi_x_in_domain"] = witness["f_pi_x_in_domain"]
            rows.append(row)

        shell_rows = []
        for record in result.shell_diagnostics:
            row = {
                "label": record["label"],
                "nu_over_c2_lower": record["nu_over_c2_lower"],
                "nu_over_c2_upper": "" if record["nu_over_c2_upper"] is None else record["nu_over_c2_upper"],
                "count": record["count"],
                "min_omega_cl": "" if record["min_omega_cl"] is None else record["min_omega_cl"],
                "min_omega_cl_nu": "",
                "min_omega_cl_nu_over_c2": "",
                "min_omega_original": "" if record["min_omega_original"] is None else record["min_omega_original"],
                "min_omega_original_nu": "",
                "min_omega_original_nu_over_c2": "",
            }
            cl_witness = record.get("min_omega_cl_witness")
            if cl_witness is not None:
                row["min_omega_cl_nu"] = cl_witness["nu"]
                row["min_omega_cl_nu_over_c2"] = cl_witness["nu_over_c2_verify"]
            original_witness = record.get("min_omega_original_witness")
            if original_witness is not None:
                row["min_omega_original_nu"] = original_witness["nu"]
                row["min_omega_original_nu_over_c2"] = original_witness["nu_over_c2_verify"]
            shell_rows.append(row)

        if self.run_context is not None:
            self.run_context.write_json("threshold_search.json", full_payload)
            self.run_context.write_json("verification_thresholds_rescaled.json", verifier_payload)
            self.run_context.write_csv("threshold_witness_points.csv", rows)
            self.run_context.write_csv("threshold_shell_diagnostics.csv", shell_rows)
        else:
            (artifact_dir / "threshold_search.json").write_text(json.dumps(full_payload, indent=2), encoding="utf-8")
            (artifact_dir / "verification_thresholds_rescaled.json").write_text(
                json.dumps(verifier_payload, indent=2),
                encoding="utf-8",
            )
            write_csv(artifact_dir / "threshold_shell_diagnostics.csv", shell_rows)

        print("Grid threshold search complete.")
        print(f"  c2_raw / c2_verify = {result.c2_raw} / {result.c2_verify}")
        print(f"  w1_raw / w1_verify = {result.w1_raw} / {result.w1_verify}")
        print(f"  w2_raw / w2_verify = {result.w2_raw} / {result.w2_verify}")
        print(
            "  original w1_raw / w1_verify = "
            f"{result.w1_original_raw} / {result.w1_original_verify}"
        )
        print(
            "  original w2_raw / w2_verify = "
            f"{result.w2_original_raw} / {result.w2_original_verify}"
        )
        print(f"  grid band good/bad = {result.band_good_count} / {result.band_bad_count}")

    def plot_threshold_diagnostics(self) -> None:
        """Plot supplied verifier thresholds against the learned closed-loop value.

        This stage prefers `verification_thresholds_rescaled.json` from the
        current run when present. Otherwise it falls back to the manually
        supplied `system.verification.c2/w1/w2`.
        """
        if self.lqr_box_artifacts is None:
            self.setup_lqr_box()
        assert self.lqr_box_artifacts is not None

        artifact_dir = self.artifact_dir()

        pi_path = self.checkpoint_path(self.system.checkpoints.pi, "policy")
        omega_cl_path = self.checkpoint_path(
            self.system.checkpoints.omega_closed_loop,
            "closed-loop omega",
        )
        omega_path = self.checkpoint_path(self.system.checkpoints.omega, "original omega")

        threshold_path = (
            self.run_context.run_dir / "verification_thresholds_rescaled.json"
            if self.run_context is not None
            else artifact_dir / "verification_thresholds_rescaled.json"
        )
        thresholds = None
        if threshold_path.exists():
            loaded = json.loads(threshold_path.read_text(encoding="utf-8"))
            if all(loaded.get(name) is not None for name in ("c1", "c2", "w1", "w2")):
                thresholds = {
                    "c1": float(loaded["c1"]),
                    "c2": float(loaded["c2"]),
                    "w1": float(loaded["w1"]),
                    "w2": float(loaded["w2"]),
                }

        pi = load_pinet(pi_path, device=self.device)
        omega_orig = load_omeganet(omega_path, n_dim=self.system.n_dim, device=self.device)
        omega_cl = load_omeganet(omega_cl_path, n_dim=self.system.n_dim, device=self.device)
        p_matrix = self.lqr_box_artifacts.P.to(device=self.device, dtype=torch.float32)
        k_matrix = self.lqr_box_artifacts.K.to(device=self.device, dtype=torch.float32)
        if self.system.n_dim == 2:
            save_threshold_diagnostic_plot_2d(
                self.run_context,
                artifact_dir,
                system=self.system,
                pi=pi,
                omega_cl=omega_cl,
                p_matrix=p_matrix,
                thresholds=thresholds,
            )
        plot_pairs = normalize_plot_pairs(self.system)
        for axis_pair in plot_pairs:
            optimal_grid = prepare_value_heatmap_optimal_control_grid_2d(
                self.run_context,
                artifact_dir,
                system=self.system,
                omega_orig=omega_orig,
                p_matrix=p_matrix,
                k_matrix=k_matrix,
                axis_pair=axis_pair,
            )
            save_value_heatmap_decrease_diagnostic_2d(
                self.run_context,
                artifact_dir,
                system=self.system,
                pi=pi,
                omega_orig=omega_orig,
                omega_cl=omega_cl,
                p_matrix=p_matrix,
                c_analytic=self.lqr_box_artifacts.c_analytic,
                c_sigmoid_center=self.lqr_box_artifacts.c_sigmoid_center,
                c_s=self.lqr_box_artifacts.c_s,
                thresholds=thresholds,
                optimal_grid=optimal_grid,
                axis_pair=axis_pair,
            )
            save_pinet_control_slice_diagnostic_2d(
                self.run_context,
                artifact_dir,
                system=self.system,
                pi=pi,
                p_matrix=p_matrix,
                c_analytic=self.lqr_box_artifacts.c_analytic,
                c_sigmoid_center=self.lqr_box_artifacts.c_sigmoid_center,
                c_s=self.lqr_box_artifacts.c_s,
                thresholds=thresholds,
                optimal_grid=optimal_grid,
                axis_pair=axis_pair,
            )
            save_closed_loop_difference_field_diagnostic_2d(
                self.run_context,
                artifact_dir,
                system=self.system,
                pi=pi,
                omega_orig=omega_orig,
                omega_cl=omega_cl,
                p_matrix=p_matrix,
                thresholds=thresholds,
                axis_pair=axis_pair,
            )
        print("Threshold diagnostic plots saved.")
        print(f"  pi checkpoint      = {pi_path}")
        print(f"  omega checkpoint   = {omega_path}")
        print(f"  omega_cl checkpoint = {omega_cl_path}")
        print(f"  heatmap c_analytic  = {self.lqr_box_artifacts.c_analytic:.9g}")
        print(f"  heatmap sample cap = {self.lqr_box_artifacts.c_sigmoid_center:.9g}")
        print(f"  heatmap C_S         = {self.lqr_box_artifacts.c_s:.9g}")
        print(f"  heatmap plot pairs  = {plot_pairs}")
        print(f"  thresholds         = {threshold_path if thresholds else 'systems.py verification config'}")

    def plot_policy_label_gap_diagnostics(self) -> None:
        """Optional expensive diagnostic for grid-search policy label ambiguity."""
        if self.lqr_box_artifacts is None:
            self.setup_lqr_box()
        assert self.lqr_box_artifacts is not None

        artifact_dir = self.artifact_dir()
        omega_path = self.checkpoint_path(self.system.checkpoints.omega, "original omega")
        thresholds = None
        threshold_path = (
            self.run_context.run_dir / "verification_thresholds_rescaled.json"
            if self.run_context is not None
            else artifact_dir / "verification_thresholds_rescaled.json"
        )
        if threshold_path.exists():
            loaded = json.loads(threshold_path.read_text(encoding="utf-8"))
            if all(loaded.get(name) is not None for name in ("c1", "c2", "w1", "w2")):
                thresholds = {
                    "eps": float(loaded["eps"]),
                    "c1": float(loaded["c1"]),
                    "c2": float(loaded["c2"]),
                    "w1": float(loaded["w1"]),
                    "w2": float(loaded["w2"]),
                }

        omega_orig = load_omeganet(omega_path, n_dim=self.system.n_dim, device=self.device)
        p_matrix = self.lqr_box_artifacts.P.to(device=self.device, dtype=torch.float32)
        plot_pairs = normalize_plot_pairs(self.system)
        for axis_pair in plot_pairs:
            save_policy_label_gap_diagnostic_2d(
                self.run_context,
                artifact_dir,
                system=self.system,
                omega_orig=omega_orig,
                p_matrix=p_matrix,
                c_analytic=self.lqr_box_artifacts.c_analytic,
                c_s=self.lqr_box_artifacts.c_s,
                thresholds=thresholds,
                axis_pair=axis_pair,
            )
        print("Policy label gap diagnostics saved.")
        print(f"  omega checkpoint  = {omega_path}")
        print(f"  plot pairs        = {plot_pairs}")
        print(f"  near-optimal tol  = {self.system.training.policy_label_gap_near_optimal_rel_tol:.9g}")
        print(f"  thresholds        = {threshold_path if thresholds else 'systems.py verification config'}")

    def export(self) -> None:
        # Save only artifacts whose construction has been reviewed:
        #   - raw PyTorch checkpoints with metadata,
        #   - TorchScript/ONNX forms if alpha-beta-CROWN needs them,
        #   - configuration JSON with system name, dt, bounds, P/K, c-levels,
        #     training scales, and checkpoint paths.
        #
        # Do not silently export a model trained with placeholder samplers.
        raise NotImplementedError(
            "Add export once the reviewed training stages produce final models."
        )

    def run(self, stage: str) -> None:
        if stage not in self.STAGES:
            raise ValueError(f"Unknown stage {stage!r}. Choose from {', '.join(self.STAGES)}")
        preflight_stages = set(self.RUNNABLE_TRAINING_STAGES) | {
            "diagnose_training_domain",
            "export",
            "all",
        }
        if stage in preflight_stages and not self.skip_preflight:
            result = check_machine_preflight(device=self.device)
            if not result.ok:
                print("Preflight check failed. Recommended reset before running this stage:")
                for message in result.messages:
                    print(f"  - {message}")
                print("Close stale terminals/apps, restart failed Python/CUDA processes, then rerun.")
                print("Override only if intentional: add --skip-preflight.")
                return
            print("Preflight check passed.")
            for message in result.messages:
                print(f"  - {message}")
        if stage == "all":
            for name in self.RUNNABLE_TRAINING_STAGES:
                getattr(self, name)()
            return
        getattr(self, stage)()


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run a registry-driven training pipeline stage.")
    parser.add_argument("--system", default="2d_bench", help="System key from systems.py")
    parser.add_argument("--stage", default="plan", choices=TrainingPipeline.STAGES)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run folder name under Streamlined/pipeline_runs/<system>. Omit to create a timestamped run.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Run even if the resource preflight would recommend a reset.",
    )
    args = parser.parse_args(argv)

    system = get_system(args.system)
    run_dir = RUNS_ROOT / system.name / (args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    if args.run_name is not None and run_dir.exists():
        system = system_with_existing_run_domain(system, run_dir)
    run = PipelineRun(system, run_name=args.run_name)
    run.write_config()
    print(f"Run directory: {run.run_dir}")
    TrainingPipeline(system, run=run, skip_preflight=args.skip_preflight).run(args.stage)


if __name__ == "__main__":
    main()
