from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(__file__).resolve().parent / "pipeline_runs"
AUTO_LIRPA_ROOT = REPO_ROOT / "auto_LiRPA"
if (AUTO_LIRPA_ROOT / "auto_LiRPA" / "__init__.py").is_file():
    sys.path.insert(0, str(AUTO_LIRPA_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from complete_verifier import ABCrownSolver, ConfigBuilder, VerificationSpec
from models import OmegaNet, hidden_sizes_from_state_dict, load_omeganet, load_pinet
from systems import SystemSpec, get_system
from training_pipeline import pick_device, system_with_existing_run_domain


def apply_threshold_override(system: SystemSpec, threshold_path: Path) -> SystemSpec:
    """Return a copy of `system` with verifier thresholds loaded from JSON."""
    if system.verification is None:
        raise ValueError(f"{system.name} has no verification config to override.")
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    required = ("c1", "c2", "w1", "w2", "eps")
    missing = [key for key in required if payload.get(key) is None]
    if missing:
        raise ValueError(f"{threshold_path} is missing verifier thresholds: {missing}")
    verification = replace(
        system.verification,
        eps=float(payload["eps"]),
        c1=float(payload["c1"]),
        c2=float(payload["c2"]),
        w1=float(payload["w1"]),
        w2=float(payload["w2"]),
    )
    return replace(system, verification=verification)


def discover_threshold_override(run_dir: Path) -> Optional[Path]:
    """Find run-specific verifier thresholds for a named pipeline run."""
    candidate = run_dir / "verification_thresholds_rescaled.json"
    return candidate if candidate.is_file() else None


def resolve_run_dir(system: SystemSpec, run_name: str) -> Path:
    run_dir = RUNS_ROOT / system.name / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    config_path = run_dir / "config.json"
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        configured_system = payload.get("system")
        if configured_system != system.name:
            raise ValueError(
                f"Run/system mismatch: {config_path} says {configured_system!r}, "
                f"but --system was {system.name!r}."
            )
    return run_dir


def describe_thresholds(system: SystemSpec) -> str:
    if system.verification is None:
        return "<none>"
    vcfg = system.verification
    return (
        f"eps={vcfg.eps}, c1={vcfg.c1}, "
        f"c2={vcfg.c2}, w1={vcfg.w1}, w2={vcfg.w2}"
    )


def require_thresholds(system: SystemSpec) -> None:
    if system.verification is None:
        raise ValueError(f"{system.name} has no verification config.")
    vcfg = system.verification
    missing = [
        name
        for name in ("c1", "c2", "w1", "w2")
        if getattr(vcfg, name) is None
    ]
    if missing:
        raise ValueError(
            f"{system.name} is missing verifier thresholds {missing}. "
            "Run search_thresholds first or pass --thresholds with c1/c2/w1/w2/eps."
        )


def describe_verification_domain(system: SystemSpec) -> str:
    if system.verification is None:
        return "<none>"
    return str(system.verification.domain)


def singleton_state_omeganet_for_verifier(omega: OmegaNet, n_dim: int) -> OmegaNet:
    """Return an equivalent singleton-state OmegaNet without verifier-time concat.

    Omega models copied from omega_orig may have input `[x, zeros]`. For
    singleton verification those zero columns never contribute, but leaving the
    concatenation inside the traced graph sends auto_LiRPA through a slice/concat
    interval path that is not CUDA-safe in this local version.
    """
    if omega.input_dim == n_dim:
        return omega
    if omega.input_dim < n_dim:
        raise ValueError(f"omega.input_dim={omega.input_dim} is smaller than n_dim={n_dim}.")

    state_dict = dict(omega.state_dict())
    hidden_sizes = hidden_sizes_from_state_dict(state_dict)
    collapsed = OmegaNet(
        n_dim=n_dim,
        hidden_sizes=hidden_sizes,
        input_dim=n_dim,
        embedding_mode="state_singleton_from_embedded_omega",
    )
    first_weight_key = "net.0.weight"
    state_dict[first_weight_key] = state_dict[first_weight_key][:, :n_dim].clone()
    collapsed.load_state_dict(state_dict)
    collapsed.to(next(omega.parameters()).device)
    collapsed.eval()
    return collapsed


class CombinedVerificationModel(nn.Module):
    """Verifier-facing model with a fixed output layout for all registered systems."""

    def __init__(self, system: SystemSpec, pi: nn.Module, omega_cl: nn.Module):
        super().__init__()
        self.system = system
        self.pi = pi
        self.omega_cl = omega_cl

    def nu(self, x: torch.Tensor) -> torch.Tensor:
        p_matrix = self.pi.P_matrix.to(device=x.device, dtype=x.dtype)
        return (x @ p_matrix * x).sum(dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.system.n_dim:
            raise ValueError(
                f"Expected last input dimension {self.system.n_dim}, got {tuple(x.shape)}."
            )
        leading_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.system.n_dim)

        u_lqr = self.pi.lqr_control(x_flat)
        fx_lqr = self.system.dynamics(x_flat, u_lqr)
        u_nn = self.pi.nn_control(x_flat)
        fx_nn = self.system.dynamics(x_flat, u_nn)

        nu = self.nu(x_flat)
        nu_lqr_next = self.nu(fx_lqr)
        dnu_lqr = nu_lqr_next - nu

        omega_cl = self.omega_cl.query(x_flat).reshape(-1, 1)
        omega_cl_nn_next = self.omega_cl.query(fx_nn).reshape(-1, 1)
        domega_cl_nn = omega_cl_nn_next - omega_cl

        output = torch.cat(
            [
                nu,
                nu_lqr_next,
                dnu_lqr,
                omega_cl,
                omega_cl_nn_next,
                domega_cl_nn,
                fx_lqr,
                fx_nn,
                u_lqr,
                u_nn,
            ],
            dim=1,
        )
        return output.reshape(*leading_shape, output.shape[-1])


def output_clause(rows: Iterable[tuple[int, float, float]], y_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows_tuple = tuple(rows)
    c = torch.zeros((len(rows_tuple), y_dim), dtype=torch.float32)
    rhs = torch.empty((c.shape[0],), dtype=torch.float32)
    for row_idx, (output_idx, coeff, bound) in enumerate(rows_tuple):
        c[row_idx, int(output_idx)] = float(coeff)
        rhs[row_idx] = float(bound)
    return c, rhs


def solve_case(model, lower, upper, clauses, cex_path: Path):
    spec = VerificationSpec.build_from_input_bounds(
        torch.as_tensor(lower, dtype=torch.float32).unsqueeze(0),
        torch.as_tensor(upper, dtype=torch.float32).unsqueeze(0),
        [clauses],
    )
    cfg = ConfigBuilder.from_defaults().set(
        model__with_jacobian=False,
        general__save_adv_example=True,
        attack__cex_path=str(cex_path),
    )
    solver = ABCrownSolver(spec, model, config=cfg)
    t0 = time.perf_counter()
    result = solver.solve()
    return result, time.perf_counter() - t0


def result_is_safe(result) -> bool:
    """Interpret alpha-beta-CROWN result status as a safety verdict.

    `result.success` means the verifier finished with a decisive outcome; it is
    also true for unsafe counterexamples. The Lyapunov proof obligation needs a
    proven/safe status specifically.
    """
    return str(result.status).lower() in {"safe", "verified"}


def result_verdict(result) -> str:
    """Small user-facing label for alpha-beta-CROWN outcomes."""
    if result_is_safe(result):
        return "proved"
    status = str(result.status).lower()
    if status in {"unsafe", "falsified"}:
        return "counterexample_found"
    return "not_proved"


def archive_stale_cex(cex_path: Path) -> Optional[Path]:
    """Move a previous counterexample file out of the active reporting path."""
    if not cex_path.is_file():
        return None
    stale_dir = cex_path.parent / "stale"
    stale_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stale_path = stale_dir / f"{cex_path.stem}_{stamp}{cex_path.suffix}"
    shutil.move(str(cex_path), str(stale_path))
    return stale_path


def run_verification(system: SystemSpec, *, artifact_dir: Path, cex_dir: Path):
    if system.verification is None:
        raise ValueError(f"{system.name} has no verification config.")
    require_thresholds(system)

    device = pick_device()
    pi = load_pinet(artifact_dir / system.checkpoints.pi, device=device)
    omega_cl = load_omeganet(
        artifact_dir / system.checkpoints.omega_closed_loop,
        n_dim=pi.n_dim,
        device=device,
    )
    raw_omega_cl_input_dim = int(omega_cl.input_dim)
    omega_cl = singleton_state_omeganet_for_verifier(omega_cl, system.n_dim)
    if raw_omega_cl_input_dim != int(omega_cl.input_dim):
        print(
            "Verifier collapsed copied embedded omega_cl singleton input: "
            f"{raw_omega_cl_input_dim} -> {omega_cl.input_dim}"
        )
    model = CombinedVerificationModel(
        system,
        pi=pi,
        omega_cl=omega_cl,
    ).to(device).eval()

    y_dim = 6 + 2 * system.n_dim + 2 * system.u_dim
    vcfg = system.verification

    x_lo = [d[0] for d in vcfg.domain]
    x_hi = [d[1] for d in vcfg.domain]
    # Each case is encoded as an unsafe OR-list over one input box. This avoids
    # expression-DNF construction repeating the same input box as a fake batch.
    fx_lqr_start = 6
    fx_nn_start = 6 + system.n_dim
    case_1 = [
        output_clause(
            (
                (0, 1.0, vcfg.c2),      # nu <= c2
                (0, -1.0, -vcfg.c1),    # nu >= c1
                (2, -1.0, vcfg.eps),    # dnu_lqr >= -eps
            ),
            y_dim,
        )
    ]
    for dim, (lo, hi) in enumerate(vcfg.domain):
        case_1.append(
            output_clause(
                (
                    (0, 1.0, vcfg.c2),            # nu <= c2
                    (0, -1.0, -vcfg.c1),          # nu >= c1
                    (fx_lqr_start + dim, 1.0, float(lo)),  # f_lqr(x)_dim <= lo
                ),
                y_dim,
            )
        )
        case_1.append(
            output_clause(
                (
                    (0, 1.0, vcfg.c2),            # nu <= c2
                    (0, -1.0, -vcfg.c1),          # nu >= c1
                    (fx_lqr_start + dim, -1.0, -float(hi)),  # f_lqr(x)_dim >= hi
                ),
                y_dim,
            )
        )
    case_2 = [
        output_clause(
            (
                (3, 1.0, vcfg.w1),      # omega_cl <= w1
                (0, -1.0, -vcfg.c2),    # nu >= c2
            ),
            y_dim,
        )
    ]
    case_3 = [
        output_clause(
            (
                (3, -1.0, -vcfg.w1),    # omega_cl >= w1
                (3, 1.0, vcfg.w2),      # omega_cl <= w2
                (5, -1.0, vcfg.eps),    # domega_cl_nn >= -eps
            ),
            y_dim,
        )
    ]
    for dim, (lo, hi) in enumerate(vcfg.domain):
        case_3.append(
            output_clause(
                (
                    (3, -1.0, -vcfg.w1),           # omega_cl >= w1
                    (3, 1.0, vcfg.w2),             # omega_cl <= w2
                    (fx_nn_start + dim, 1.0, float(lo)),  # f_nn(x)_dim <= lo
                ),
                y_dim,
            )
        )
        case_3.append(
            output_clause(
                (
                    (3, -1.0, -vcfg.w1),           # omega_cl >= w1
                    (3, 1.0, vcfg.w2),             # omega_cl <= w2
                    (fx_nn_start + dim, -1.0, -float(hi)),  # f_nn(x)_dim >= hi
                ),
                y_dim,
            )
        )

    cex_dir.mkdir(parents=True, exist_ok=True)
    case_names = {
        1: "clamped_lqr_nu_decrease",
        2: "omega_cl_w1_contained_in_nu_c2",
        3: "saturated_nn_omega_cl_decrease",
    }
    results = []
    for idx, case in enumerate((case_1, case_2, case_3), start=1):
        cex_path = cex_dir / f"{system.name}_cex{idx}.txt"
        stale_cex = archive_stale_cex(cex_path)
        result, dt = solve_case(model, x_lo, x_hi, case, cex_path)
        results.append((idx, case_names[idx], result, dt, cex_path, stale_cex))
        print(
            f"case_{idx}: status={result.status}, "
            f"solver_success={result.success}, verdict={result_verdict(result)}, "
            f"time={dt:.3f}s"
        )

    overall = all(result_is_safe(result) for _, _, result, _, _, _ in results)
    total = sum(dt for _, _, _, dt, _, _ in results)
    print(f"overall: success={overall}, total_time={total:.3f}s")
    print("\nVerification summary")
    for idx, name, result, dt, cex_path, stale_cex in results:
        cex_display = str(cex_path) if cex_path.is_file() else "none written"
        stale_note = f", stale_cex_archived={stale_cex}" if stale_cex is not None else ""
        print(
            f"  case_{idx} ({name}): "
            f"proved={result_is_safe(result)}, verdict={result_verdict(result)}, "
            f"status={result.status}, "
            f"solver_success={result.success}, "
            f"time={dt:.3f}s, cex={cex_display}{stale_note}"
        )
    print(
        "  thresholds: "
        f"eps={vcfg.eps}, c1={vcfg.c1}, c2={vcfg.c2}, "
        f"w1={vcfg.w1}, w2={vcfg.w2}"
    )
    print(f"  overall_proved={overall}, total_time={total:.3f}s")
    return results


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run alpha-beta-CROWN verification for a registered system.")
    parser.add_argument("--system", default="2d_bench", help="System key from systems.py")
    parser.add_argument(
        "--run-name",
        required=True,
        help="Run folder name under Streamlined/pipeline_runs/<system> containing trained artifacts",
    )
    parser.add_argument("--cex-dir", default=str(Path(__file__).resolve().parent / "counterexamples"))
    parser.add_argument("--thresholds", default=None, help="JSON file with rescaled c1/c2/w1/w2/eps")
    args = parser.parse_args(argv)

    system = get_system(args.system)
    run_dir = resolve_run_dir(system, args.run_name)
    artifact_dir = run_dir / "artifacts"
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"Run artifact directory does not exist: {artifact_dir}")

    base_domain = system.verification.domain if system.verification is not None else None
    system = system_with_existing_run_domain(system, run_dir)
    if base_domain != (system.verification.domain if system.verification is not None else None):
        print("Using verifier input domain from saved run expansion/config")
    else:
        print("Using verifier input domain from systems.py")

    threshold_path = Path(args.thresholds) if args.thresholds else discover_threshold_override(run_dir)
    if threshold_path:
        print(f"Using verifier thresholds from {threshold_path}")
        system = apply_threshold_override(system, threshold_path)
    else:
        print("Using verifier thresholds from systems.py")
        require_thresholds(system)
    print(f"Verifier input domain: {describe_verification_domain(system)}")
    print(f"Verifier thresholds: {describe_thresholds(system)}")

    run_verification(system, artifact_dir=artifact_dir, cex_dir=Path(args.cex_dir))


if __name__ == "__main__":
    main()
