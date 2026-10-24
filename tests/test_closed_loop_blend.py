import ast
import inspect
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from models import OmegaNet, PiNet
import torch
import systems
from training_pipeline import (
    closed_loop_controller_metadata,
    closed_loop_omega_loss_terms,
    compute_closed_loop_zubov_targets,
    evaluate_closed_loop_decrease_report,
    policy_control_for_mode,
    validate_blended_controller_artifacts,
)


class ClosedLoopBlendTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        self.c1 = 0.4
        self.system = SimpleNamespace(
            n_dim=2, dt=0.05, x_lb=(-10, -10), x_hi=(10, 10),
            training=SimpleNamespace(zubov_alpha_scale=0.3),
            dynamics=lambda x, u: x + 0.05 * (u + 0.1 * x ** 3),
        )

    def setup_models(self, device):
        pi = PiNet(
            n_dim=2, u_dim=2, hidden_sizes=(4,), activation="relu",
            p_matrix=[[2, 0.25], [0.25, 1]], k_matrix=[[0.2, 0.1], [-0.1, 0.3]],
            u_lb=[-1, -2], u_hi=[1, 2],
        ).to(device)
        original = OmegaNet(2, (4,), input_dim=2).to(device)
        value = OmegaNet(2, (4,), input_dim=2).to(device)
        for model in (pi, original):
            model.eval()
            model.requires_grad_(False)
        x = torch.tensor([[0, 0], [0.2, -0.3], [0.7, 0.1], [-0.1, 0.4]], device=device)
        return pi, original, value, x

    def manual_control(self, pi, x):
        nu = (x @ pi.P_matrix * x).sum(-1, keepdim=True)
        weight = torch.tanh((math.atanh(0.5) / self.c1) * nu)
        return (1 - weight) * pi.lqr_control(x) + weight * pi.nn_control(x)

    def test_shared_default_and_no_system_specific_overrides(self):
        self.assertEqual(systems.TrainingConfig().closed_loop_controller_mode, "tanh_blend")
        for name in systems.SYSTEMS:
            with self.subTest(system=name):
                self.assertEqual(systems.get_system(name).training.closed_loop_controller_mode, "tanh_blend")
        tree = ast.parse(inspect.getsource(systems))
        overrides = [node for node in ast.walk(tree)
                     if isinstance(node, ast.keyword) and node.arg == "closed_loop_controller_mode"]
        self.assertEqual(overrides, [])

    def test_controller_metadata(self):
        metadata = closed_loop_controller_metadata("tanh_blend", c1=self.c1)
        self.assertEqual(metadata["blend_midpoint_source"], "c_analytic")
        self.assertEqual(metadata["blend_midpoint"], self.c1)
        self.assertAlmostEqual(math.tanh(metadata["blend_k"] * self.c1), 0.5)
        self.assertEqual(metadata["nn_weight_formula"], "tanh(blend_k * (x^T P x))")
        for mode in ("nn", "lqr"):
            self.assertEqual(closed_loop_controller_metadata(mode), {"controller_mode": mode})
        for invalid in (None, 0, -1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite, positive"):
                closed_loop_controller_metadata("tanh_blend", c1=invalid)

    def test_selector_and_legacy_modes(self):
        for device in self.devices:
            pi, _, _, x = self.setup_models(device)
            with self.subTest(device=device):
                for mode in ("nn", "neural", "neural_net", "saturated_nn", "tanh_nn"):
                    torch.testing.assert_close(policy_control_for_mode(pi, x, mode), pi.nn_control(x), rtol=0, atol=0)
                for mode in ("lqr", "linear"):
                    torch.testing.assert_close(policy_control_for_mode(pi, x, mode), pi.lqr_control(x), rtol=0, atol=0)
                torch.testing.assert_close(policy_control_for_mode(pi, x, "tanh_blend", c1=self.c1), self.manual_control(pi, x))
                with self.assertRaisesRegex(ValueError, "explicit analytic c1"):
                    policy_control_for_mode(pi, x, "tanh_blend")
                with self.assertRaisesRegex(ValueError, "controller_mode"):
                    policy_control_for_mode(pi, x, "unknown")

    def test_artifact_guard(self):
        for device in self.devices:
            pi, _, _, _ = self.setup_models(device)
            with self.subTest(device=device):
                validate_blended_controller_artifacts(pi, pi.P_matrix, pi.K_matrix, c1=self.c1)
                for name in ("P", "K"):
                    p, k = pi.P_matrix.clone(), pi.K_matrix.clone()
                    (p if name == "P" else k)[0, 0] += 0.01
                    with self.assertRaisesRegex(ValueError, f"checkpoint {name}"):
                        validate_blended_controller_artifacts(pi, p, k, c1=self.c1)
                for c1 in (0, -1, float("inf"), float("nan")):
                    with self.assertRaisesRegex(ValueError, "finite and positive"):
                        validate_blended_controller_artifacts(pi, pi.P_matrix, pi.K_matrix, c1=c1)

    def test_trajectory_labels_match_manual_cost_and_blend(self):
        for device in self.devices:
            pi, _, _, x = self.setup_models(device)
            for horizon in (0, 3):
                with self.subTest(device=device, horizon=horizon):
                    labels = compute_closed_loop_zubov_targets(
                        self.system, pi, x, pi.P_matrix, horizon=horizon,
                        controller_mode="tanh_blend", c1=self.c1, value_input_dim=2,
                    )
                    state = x.clone()
                    cost = torch.zeros(len(x), device=device)
                    for _ in range(horizon + 1):
                        cost += 0.3 * 0.05 * (state @ pi.P_matrix * state).sum(-1)
                        state = self.system.dynamics(state, self.manual_control(pi, state))
                    torch.testing.assert_close(labels.target_w, 1 - torch.exp(-cost))
                    self.assertFalse(labels.target_w.requires_grad)

    def test_loss_and_frozen_controller_optimization(self):
        for device in self.devices:
            pi, original, value, x = self.setup_models(device)
            with self.subTest(device=device):
                frozen = [[p.detach().clone() for p in m.parameters()] for m in (pi, original)]
                initial = [p.detach().clone() for p in value.parameters()]
                labels = compute_closed_loop_zubov_targets(
                    self.system, pi, x, pi.P_matrix, horizon=3,
                    controller_mode="tanh_blend", c1=self.c1, value_input_dim=2,
                )
                step = lambda z: self.system.dynamics(z, policy_control_for_mode(pi, z, "tanh_blend", c1=self.c1))
                optimizer = torch.optim.Adam(value.parameters(), lr=1e-3)
                for _ in range(2):
                    optimizer.zero_grad()
                    losses = closed_loop_omega_loss_terms(
                        value, original, x, labels, step, pi.P_matrix, system=self.system,
                        lambda_zubov=1, lambda_data=1, lambda_decrease=1, lambda_range=0,
                        lambda_boundary=0, dos_gate=1, kappa=0.001, anchor_filter="none",
                    )
                    w = value.query(x)
                    w_next = value.query(self.system.dynamics(x, self.manual_control(pi, x)))
                    xi = 1 - torch.exp(-0.3 * 0.05 * (x @ pi.P_matrix * x).sum(-1))
                    torch.testing.assert_close(losses.zubov_mse, ((w - w_next - xi * (1 - w_next)) ** 2).mean())
                    torch.testing.assert_close(losses.decrease_penalty, torch.relu(w_next - 0.999 * w).mean())
                    torch.testing.assert_close(losses.data_mse, ((w - labels.target_w) ** 2).mean())
                    self.assertTrue(torch.isfinite(losses.total).item())
                    losses.total.backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in value.parameters()))
                    optimizer.step()
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(initial, value.parameters())))
                for model, before in zip((pi, original), frozen):
                    for parameter, saved in zip(model.parameters(), before):
                        self.assertIsNone(parameter.grad)
                        torch.testing.assert_close(parameter, saved, rtol=0, atol=0)

    def test_heldout_report_uses_same_blend_in_every_batch(self):
        for device in self.devices:
            pi, original, value, x = self.setup_models(device)
            with self.subTest(device=device), patch.object(pi, "blended_control", wraps=pi.blended_control) as blend:
                report = evaluate_closed_loop_decrease_report(
                    self.system, pi, original, value, pi.P_matrix, x, torch.zeros(len(x), dtype=torch.long),
                    controller_mode="tanh_blend", c1=self.c1, batch_size=2, dos_gate=1, eps=1e-8,
                )
                for key, expected in closed_loop_controller_metadata("tanh_blend", c1=self.c1).items():
                    self.assertEqual(report[key], expected)
                self.assertEqual(blend.call_count, 2)
                self.assertTrue(all(call.kwargs["c1"] == self.c1 for call in blend.call_args_list))
                for prefix, model in (("omega_orig", original), ("omega_cl", value)):
                    delta = model.query(self.system.dynamics(x, self.manual_control(pi, x))) - model.query(x)
                    self.assertAlmostEqual(report["overall"][prefix + "_max_delta"], float(delta.max()), places=6)
                    self.assertEqual(report["overall"][prefix + "_violation_count"], int((delta > -1e-8).sum()))


if __name__ == "__main__":
    unittest.main()
