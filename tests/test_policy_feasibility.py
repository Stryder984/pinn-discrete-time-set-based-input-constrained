import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import training_pipeline as tp
from systems import get_system


class ToyPolicy(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.action = torch.nn.Parameter(torch.tensor([[value]], device="cuda"))

    def nn_control(self, x):
        return self.action.expand(len(x), -1)

    forward = nn_control


class ToyOmega(torch.nn.Module):
    def query(self, x):
        return torch.sigmoid(x[:, 0])


def toy_dynamics(x, u):
    return torch.stack((x[:, 0] + u[:, 0], x[:, 1] + 2 * (x[:, 1] > .5)), dim=-1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PolicyFeasibilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.system = SimpleNamespace(
            n_dim=2, u_dim=1, x_lb=(-1., -1.), x_hi=(1., 1.),
            u_lb=(-2.,), u_hi=(2.,), dynamics=toy_dynamics,
        )
        self.omega = ToyOmega()

    def test_original_losses_and_gradients_unchanged(self):
        pi = ToyPolicy(.2)
        # Include an escaped prediction: feasibility filtering is for labels,
        # not a replacement or penalty in the original loss functions.
        x = torch.tensor([[0., 0.], [.9, 0.]], device="cuda")
        target = torch.tensor([.4, .5], device="cuda")
        expected = ((self.omega.query(toy_dynamics(x, pi.nn_control(x))) - target) ** 2).mean()
        actual = tp.policy_next_omega_mse(self.system, pi, self.omega, x, target)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, pi.action)[0],
                                   torch.autograd.grad(expected, pi.action)[0])
        expected_hinge = torch.relu(self.omega.query(toy_dynamics(x, pi.nn_control(x))) - .9*self.omega.query(x)).mean()
        actual_hinge = tp.policy_omega_decrease_hinge(self.system, pi, self.omega, x, kappa=.1)
        torch.testing.assert_close(actual_hinge, expected_hinge)
        torch.testing.assert_close(torch.autograd.grad(actual_hinge, pi.action)[0],
                                   torch.autograd.grad(expected_hinge, pi.action)[0])

    def test_filter_excludes_placeholders_rechecks_controls_and_preserves_alignment(self):
        labels = tp.PolicyLabelBatch(
            x=torch.tensor([[0., 0.], [0., .75], [0., 0.], [0., 0.], [1., 0.]], device="cuda"),
            u_opt=torch.tensor([[0.], [0.], [1.5], [0.], [0.]], device="cuda"),
            omega_opt_next=torch.ones(5, device="cuda"),
            feasible_control_fraction=torch.tensor([1., 0., 1., 0., 1.], device="cuda"),
            control_weight=torch.ones(5, device="cuda"),
            decisive_mask=torch.tensor([True, False, False, True, False], device="cuda"),
        )
        sources = torch.tensor([2, 1, 0, 2, 1])
        for batch_size in (1, 3, 20):
            kept, ids, report = tp.filter_feasible_policy_labels(self.system, labels, sources, batch_size=batch_size)
            self.assertEqual(ids.tolist(), [2, 1])
            self.assertEqual(kept.decisive_mask.tolist(), [True, False])
            self.assertEqual(report["removed_points"], 3)
            self.assertEqual(report["no_feasible_grid_control_points"], 2)
            self.assertEqual(report["invalid_selected_control_or_successor_or_value_points"], 2)
        with self.assertRaisesRegex(ValueError, "feasibility metadata"):
            tp.filter_feasible_policy_labels(self.system, replace(labels, feasible_control_fraction=None), sources, batch_size=2)
        empty, _, report = tp.filter_feasible_policy_labels(
            self.system, replace(labels, feasible_control_fraction=torch.zeros(5, device="cuda")), sources, batch_size=2)
        self.assertEqual(len(empty.x), 0)
        self.assertEqual(report["removed_points"], 5)

    def test_fresh_and_cached_training_filter_even_without_confidence(self):
        base = get_system("van_der_pol")
        cfg = replace(base.training, pi_epochs=1, pi_batch_size=2, pi_eval_points=2,
                      pi_hidden_sizes=(4,), pi_plot_points=0, pi_log_every=1,
                      policy_decrease_report_points=0, control_grid_points=5, control_grid_batch_size=2,
                      policy_label_use_confidence_weight=False, policy_label_mask_out_of_domain_successors=False,
                      policy_label_cache_path=None, pi_train_max_points=None)
        system = replace(base, x_lb=(-1., -1.), x_hi=(1., 1.), u_lb=(-2.,), u_hi=(2.,),
                         dynamics=toy_dynamics, training=cfg)
        def point_sampler(*args, **kwargs):
            return torch.tensor([[0., 0.], [0., .75]], device="cuda")
        # Grid search may query OOD values before masking; the training loss must not.
        class GridOmega(torch.nn.Module):
            def query(self, x):
                return torch.sigmoid(x[:, 0])
        with TemporaryDirectory() as directory:
            counts = SimpleNamespace(policy_gd_omega_sublevel_points=2, policy_ellipse_points=1, policy_uniform_points=2)
            for name, cache in (("fresh", None), ("cached", Path(directory)/system.name/"fresh/artifacts/policy_grid_labels.pt")):
                current = replace(system, training=replace(cfg, policy_label_cache_path=str(cache) if cache else None))
                run = tp.PipelineRun(current, run_root=Path(directory), run_name=name)
                torch.save({}, run.artifact_dir/current.checkpoints.omega)
                pipeline = tp.TrainingPipeline(current, run=run, device=torch.device("cuda"))
                pipeline.lqr_box_artifacts = SimpleNamespace(P=torch.eye(2, device="cuda"),
                    K=torch.zeros(1, 2, device="cuda"), c_s=.1, c_analytic=.01, C_sample_cap=.1)
                with patch.object(tp, "load_omeganet", return_value=GridOmega()), \
                     patch.object(tp, "resolve_data_distribution_counts", return_value=counts), \
                     patch.object(tp, "sample_omega_rejection_points", return_value=(point_sampler(), {"acceptance_rate": 1.})), \
                     patch.object(tp, "sample_uniform_bounds", side_effect=point_sampler), \
                     patch.object(pipeline, "sample_lqr_c2_band", return_value=torch.zeros(1, 2, device="cuda")), \
                     patch.object(tp, "save_loss_history_artifacts"), \
                     patch.object(tp, "save_policy_control_comparison_plot"):
                    pipeline.train_policy()
                saved = torch.load(run.artifact_dir/current.checkpoints.pi, map_location="cpu", weights_only=False)
                self.assertEqual(saved["policy_training_feasibility"]["kept_points"], 3)
                self.assertEqual(saved["policy_training_feasibility"]["no_feasible_grid_control_points"], 2)
                self.assertEqual(saved["label_source"]["states"], 3)
                summary = json.loads((run.run_dir/"policy_label_dos_filtration.json").read_text())
                self.assertEqual(summary["removed_points"], 2)
                self.assertEqual(summary["policy_uniform_points"], 1)
                self.assertEqual(summary["policy_rejection_omega_sublevel_points"], 1)


if __name__ == "__main__":
    unittest.main()
