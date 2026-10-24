"""Constant control-box vertices augment, not replace, existing label rollouts."""

import itertools
from types import SimpleNamespace
import unittest

import torch

import training_pipeline as tp
from systems import f_sys_radial_clf_2d


class StepwiseVertexRolloutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def system(self, m=1):
        return SimpleNamespace(
            n_dim=m, u_dim=m, u_lb=tuple(-1.0 - j for j in range(m)),
            u_hi=tuple(0.5 + j for j in range(m)), dt=0.1,
            training=SimpleNamespace(zubov_alpha_scale=0.3),
            dynamics=lambda x, u: x + 0.1 * u,
        )

    def test_every_vertex_is_held_for_the_full_horizon(self):
        for device, m in itertools.product(self.devices, (1, 2, 3)):
            with self.subTest(device=device, m=m):
                system = self.system(m)
                x = torch.ones(2, m, device=device, dtype=torch.float64)
                calls, sampler_shapes = [], []

                def dynamics(state, control):
                    calls.append((state.clone(), control.clone()))
                    return state + 0.1 * control

                def sampler(shape, dev, dtype):
                    sampler_shapes.append(shape)
                    return torch.zeros(*shape, m, device=dev, dtype=dtype)

                system.dynamics = dynamics
                tp.compute_stepwise_min_zubov_values(
                    system, x, torch.eye(m, device=device, dtype=x.dtype),
                    k_matrix=torch.zeros(m, m, device=device, dtype=x.dtype),
                    horizon=4, ntraj=3, control_sampler=sampler,
                )
                expected = set(itertools.product(*zip(system.u_lb, system.u_hi)))
                vertex_calls = calls[2::3]
                self.assertEqual(len(calls), 12)
                self.assertEqual(sampler_shapes, [(2, 3)] * 4)
                self.assertEqual(vertex_calls[0][1].shape, (2, 2**m, m))
                self.assertEqual(set(map(tuple, vertex_calls[0][1][0].tolist())), expected)
                for k, (state, control) in enumerate(vertex_calls):
                    torch.testing.assert_close(control, vertex_calls[0][1])
                    torch.testing.assert_close(state, x[:, None, :] + k * 0.1 * control)

    def test_matches_independent_all_candidate_reference_and_keeps_rng(self):
        for device in self.devices:
            with self.subTest(device=device):
                system = self.system(2)
                x = torch.tensor([[0.4, -0.6], [1.0, 0.2]], device=device, dtype=torch.float64)
                p = torch.tensor([[2.0, 0.2], [0.2, 1.0]], device=device, dtype=x.dtype)
                k = torch.eye(2, device=device, dtype=x.dtype) * 0.7
                vertices = torch.tensor(list(itertools.product(*zip(system.u_lb, system.u_hi))),
                                        device=device, dtype=x.dtype)
                random_state = x[:, None, :].expand(-1, 5, -1).clone()
                lqr_state = x.clone()
                vertex_state = x[:, None, :].expand(-1, 4, -1).clone()
                total = torch.zeros(2, device=device, dtype=x.dtype)
                old_total = total.clone()
                torch.manual_seed(731)
                for step in range(8):
                    random_cost = torch.einsum("...i,ij,...j->...", random_state, p, random_state).min(1).values
                    lqr_cost = torch.einsum("...i,ij,...j->...", lqr_state, p, lqr_state)
                    vertex_cost = torch.einsum("...i,ij,...j->...", vertex_state, p, vertex_state).min(1).values
                    old_min = torch.minimum(random_cost, lqr_cost)
                    old_total += 0.03 * old_min
                    total += 0.03 * torch.minimum(old_min, vertex_cost)
                    if step < 7:
                        u = tp.sample_uniform_controls(system, (2, 5), device=x.device, dtype=x.dtype)
                        random_state = system.dynamics(random_state, u)
                        lower = torch.tensor(system.u_lb, device=device, dtype=x.dtype)
                        upper = torch.tensor(system.u_hi, device=device, dtype=x.dtype)
                        lqr_state = system.dynamics(lqr_state, (-lqr_state @ k.T).clamp(lower, upper))
                        vertex_state = system.dynamics(vertex_state, vertices)
                next_random = torch.rand(10, device=device)
                torch.manual_seed(731)
                v, w = tp.compute_stepwise_min_zubov_values(
                    system, x, p, k_matrix=k, horizon=7, ntraj=5,
                )
                torch.testing.assert_close(v, total)
                torch.testing.assert_close(w, 1 - torch.exp(-total))
                torch.testing.assert_close(torch.rand(10, device=device), next_random)
                self.assertTrue(bool((v <= old_total + 1e-14).all()))

    def test_horizon_zero_has_one_cost_and_no_advances(self):
        system = self.system()

        def unexpected(*args):
            self.fail("Horizon zero must not advance dynamics or sample controls")

        system.dynamics = unexpected
        x = torch.tensor([[2.0]], dtype=torch.float64)
        v, w = tp.compute_stepwise_min_zubov_values(
            system, x, torch.ones(1, 1, dtype=x.dtype), k_matrix=torch.zeros(1, 1),
            horizon=0, ntraj=2, control_sampler=unexpected,
        )
        torch.testing.assert_close(v, torch.tensor([0.12], dtype=x.dtype))
        torch.testing.assert_close(w, 1 - torch.exp(-v))

    def test_minimum_is_stepwise_not_over_total_trajectory_costs(self):
        system = self.system()
        system.n_dim = 2
        system.dt = system.training.zubov_alpha_scale = 1.0

        def dynamics(x, u):
            first = torch.where(u < 0, 1.0, torch.where(u > 0, 2.0, 3.0))
            second = torch.where(u < 0, 4.0, torch.where(u > 0, 1.0, 3.0))
            return torch.cat((x[..., :1] + 1, torch.where(x[..., :1] == 0, first, second)), -1)

        system.dynamics = dynamics
        v, _ = tp.compute_stepwise_min_zubov_values(
            system, torch.tensor([[0.0, 1.0]]), torch.diag(torch.tensor([0.0, 1.0])),
            k_matrix=torch.zeros(1, 2), horizon=2, ntraj=1,
            control_sampler=lambda shape, dev, dtype: torch.zeros(*shape, 1, device=dev, dtype=dtype),
        )
        self.assertEqual(v.item(), 3.0)
        self.assertLess(v.item(), 6.0)  # Best complete trajectory costs 1 + 4 + 1.

    def test_nonfinite_vertex_does_not_poison_finite_candidates(self):
        for device in self.devices:
            with self.subTest(device=device):
                system = self.system()
                system.dynamics = lambda x, u: torch.where(u != 0, torch.full_like(x, torch.nan), x)
                v, w = tp.compute_stepwise_min_zubov_values(
                    system, torch.ones(1, 1, device=device), torch.ones(1, 1, device=device),
                    k_matrix=torch.zeros(1, 1), horizon=3, ntraj=1,
                    control_sampler=lambda shape, dev, dtype: torch.zeros(*shape, 1, device=dev, dtype=dtype),
                )
                torch.testing.assert_close(v, torch.tensor([0.12], device=device))
                self.assertTrue(bool(torch.isfinite(w).all()))

    def test_radial_labels_match_exact_finite_horizon_optimum(self):
        for device in self.devices:
            with self.subTest(device=device):
                system = self.system()
                system.n_dim = 2
                system.u_lb, system.u_hi = (-2.0,), (2.0,)
                system.dt = 0.05
                system.training.zubov_alpha_scale = 100.0
                system.dynamics = f_sys_radial_clf_2d
                x = torch.tensor([[0, 0], [0.5, 0], [1.5, 0], [1.0, 1.0], [1.69, 0]],
                                 device=device, dtype=torch.float64)
                p = torch.eye(2, device=device, dtype=x.dtype) * (1e-4 / 0.095)
                s = x.square().sum(-1)
                expected = torch.zeros_like(s)
                for step in range(151):
                    expected += (100 * 0.05 * 1e-4 / 0.095) * s
                    if step < 150:
                        s = s * ((1.05 - 0.2 / (1 + s)).square() + 0.0025)
                v, w = tp.compute_stepwise_min_zubov_values(
                    system, x, p, k_matrix=torch.zeros(1, 2), horizon=150, ntraj=7,
                )
                torch.testing.assert_close(v, expected, atol=1e-12, rtol=1e-11)
                torch.testing.assert_close(w, 1 - torch.exp(-expected), atol=1e-12, rtol=1e-11)
                self.assertEqual(w[0].item(), 0.0)

    def test_label_wrapper_and_batched_boundary_use_vertex_candidates(self):
        for device in self.devices:
            with self.subTest(device=device):
                system = self.system()
                system.n_dim = 2
                system.u_lb, system.u_hi = (-2.0,), (2.0,)
                system.dt = 0.05
                system.dynamics = f_sys_radial_clf_2d
                p = torch.eye(2, device=device)
                k = torch.zeros(1, 2, device=device)
                boundary, v, w = tp.evaluate_boundary_stepwise_min_reachability(
                    system, p, k_matrix=k, bounds=((-0.8, 0.8), (-0.8, 0.8)),
                    points_per_face=5, horizon=20, ntraj=3, batch_size=7,
                    device=torch.device(device),
                )
                s = boundary.square().sum(-1)
                expected = torch.zeros_like(s)
                for step in range(21):
                    expected += 0.3 * 0.05 * s
                    if step < 20:
                        s = s * ((1.05 - 0.2 / (1 + s)).square() + 0.0025)
                torch.testing.assert_close(v, expected)
                torch.testing.assert_close(w, 1 - torch.exp(-expected))
                labels = tp.compute_stepwise_min_zubov_targets(
                    system, boundary.to(device), p, k_matrix=k, horizon=20, ntraj=3,
                )
                torch.testing.assert_close(labels.target_w.cpu(), w)
                torch.testing.assert_close(labels.z_x[:, :2].cpu(), boundary)
                self.assertTrue(bool((labels.z_x[:, 2:] == 0).all()))


if __name__ == "__main__":
    unittest.main()
