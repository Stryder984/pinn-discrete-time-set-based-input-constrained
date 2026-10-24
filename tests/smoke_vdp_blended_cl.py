"""Run a small real CL stage using existing Van der Pol checkpoints on CUDA."""

import argparse
import contextlib
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import load_omeganet, load_pinet
import torch
from systems import get_system
from training_pipeline import (
    PipelineRun, TrainingPipeline, closed_loop_controller_metadata,
    resolve_data_distribution_counts, system_with_existing_run_domain,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires CUDA; no CPU fallback.")
    torch.set_num_threads(4)
    source = args.source_run.resolve()
    base = system_with_existing_run_domain(get_system("van_der_pol"), source)
    assert base.training.closed_loop_controller_mode == "tanh_blend"
    assert base.training.closed_loop_data_target == "trajectory"
    assert base.training.closed_loop_anchor_filter == "none"
    assert base.training.closed_loop_lambda_range == 0
    profile = replace(base.training.data_profile, global_scale=0.001024,
                      closed_loop_trajectory_total_ratio=0.25,
                      closed_loop_constraint_total_ratio=1.0)
    config = replace(base.training, data_profile=profile, closed_loop_epochs=3,
                     closed_loop_train_batch_size=128, closed_loop_label_batch_size=128,
                     closed_loop_validation_points=256, closed_loop_validation_batch_size=128,
                     closed_loop_log_every=1, closed_loop_full_pass_epochs=False)
    system = replace(base, training=config)
    counts = resolve_data_distribution_counts(system)
    assert (counts.closed_loop_trajectory_ellipse_points
            + counts.closed_loop_trajectory_uniform_points
            + counts.closed_loop_trajectory_gd_omega_sublevel_points) == 256
    assert (counts.closed_loop_constraint_ellipse_points
            + counts.closed_loop_constraint_uniform_points
            + counts.closed_loop_constraint_gd_omega_sublevel_points) == 1024
    name = datetime.now().strftime("blend_cl_smoke_%Y%m%d_%H%M%S")
    output = args.output_root.resolve() / system.name / name
    if output.exists():
        raise FileExistsError(output)
    run = PipelineRun(system, run_root=args.output_root.resolve(), run_name=name)
    run.write_config()
    originals = [source / "artifacts" / filename
                 for filename in (system.checkpoints.pi, system.checkpoints.omega)]
    source_hashes = {str(path): sha256(path) for path in originals}
    for path in originals:
        shutil.copy2(path, run.artifact_dir / path.name)
    print(f"Isolated run: {run.run_dir}", flush=True)
    summary = {"status": "running", "source_run": str(source), "device": "cuda",
               "purpose": "CL integration smoke test, not a DOS quality or formal verification claim",
               "epochs": 3, "horizon": config.closed_loop_horizon,
               "requested_trajectory_points": 256, "requested_constraint_points": 1024,
               "requested_validation_points": 256, "source_hashes": source_hashes}
    run.write_json("smoke_test_summary.json", summary)
    try:
        with (run.run_dir / "stage.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                pipeline = TrainingPipeline(system, device=torch.device("cuda"), run=run)
                pipeline.setup_lqr_box()
                pipeline.train_closed_loop_omega()
        checkpoint_path = run.artifact_dir / system.checkpoints.omega_closed_loop
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        expected_metadata = closed_loop_controller_metadata(
            "tanh_blend", c1=pipeline.lqr_box_artifacts.c_analytic)
        for key, value in expected_metadata.items():
            assert checkpoint["closed_loop_loss"][key] == value, key
        report_path = run.run_dir / "omega_closed_loop_heldout_decrease_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for key, value in expected_metadata.items():
            assert report[key] == value, key
        assert report["overall"]["count"] == 256
        assert checkpoint["sampling"]["trajectory"]["returned_points"] == 256
        assert checkpoint["sampling"]["constraint"]["returned_points"] == 1024
        history = json.loads((run.run_dir / "omega_closed_loop_loss_history.json").read_text(encoding="utf-8"))
        assert len(history) == 3
        assert all(torch.isfinite(torch.tensor(row["total"])) and row["grad_max"] > 0 for row in history)
        restored = load_omeganet(checkpoint_path, n_dim=system.n_dim, device="cuda")
        original = load_omeganet(run.artifact_dir / system.checkpoints.omega, n_dim=system.n_dim, device="cuda")
        assert any(not torch.equal(a, b) for a, b in zip(restored.parameters(), original.parameters()))
        direct = type(restored)(system.n_dim, checkpoint["hidden_sizes"],
                                input_dim=checkpoint["input_dim"], embedding_mode=checkpoint["embedding_mode"]).cuda()
        direct.load_state_dict(checkpoint["model_state_dict"])
        x = torch.tensor([[0, 0], [0.2, -0.1], [-0.4, 0.3]], device="cuda")
        with torch.no_grad():
            torch.testing.assert_close(restored.query(x), direct.query(x), rtol=0, atol=0)
        # Reconstruct the training control using only the saved CL metadata and policy.
        pi = load_pinet(run.artifact_dir / system.checkpoints.pi, device="cuda")
        torch.testing.assert_close(pi.blended_control(x, c1=checkpoint["closed_loop_loss"]["blend_midpoint"]),
                                   pi.blended_control(x, c1=pipeline.lqr_box_artifacts.c_analytic), rtol=0, atol=0)
        for path in originals:
            assert sha256(path) == source_hashes[str(path)]
            assert sha256(run.artifact_dir / path.name) == source_hashes[str(path)]
        summary.update(status="passed", controller=expected_metadata,
                       trajectory_sampling=checkpoint["sampling"]["trajectory"],
                       constraint_sampling=checkpoint["sampling"]["constraint"],
                       heldout=report["overall"], loss_history=history,
                       checkpoint_reload_exact=True, source_and_copied_checkpoints_unchanged=True,
                       plot_files=[str(path) for path in run.plot_dir.glob("*.png")])
    except Exception as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        run.write_json("smoke_test_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
