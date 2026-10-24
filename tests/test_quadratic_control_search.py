import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import training_pipeline as tp
from systems import get_system


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class QuadraticControlSearchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        base = get_system('van_der_pol')
        self.system = replace(base, x_lb=(-2., -2.), x_hi=(2., 2.),
            u_lb=(-.5,), u_hi=(.5,),
            verification=replace(base.verification, domain=((-2., 2.), (-2., 2.))),
            training=replace(base.training, quadratic_control_search_enabled=True,
                quadratic_control_search_samples=81, quadratic_control_search_points_per_dim=5,
                quadratic_control_search_state_batch_size=7,
                quadratic_control_search_control_batch_size=2),
            dynamics=lambda x, u: torch.stack((1.5*x[:, 0]+u[:, 0], .5*x[:, 1]), dim=-1))
        self.p = torch.eye(2, device='cuda', dtype=torch.float64)

    def test_known_first_failure_and_strict_equality(self):
        result = tp.quadratic_control_grid_search(self.system, self.p)
        self.assertEqual(result['first_violation_nu'], 1.)
        self.assertEqual(result['witness']['min_delta_nu'], 0.)
        self.assertEqual(result['level'], .99)
        self.assertEqual(result['origin_excluded_count'], 1)
        self.assertEqual(result['domain_containment_cap'], 4.)
        json.dumps(result, allow_nan=False)

    def test_batch_invariance(self):
        a = tp.quadratic_control_grid_search(self.system, self.p)
        other = replace(self.system, training=replace(self.system.training,
            quadratic_control_search_state_batch_size=81,
            quadratic_control_search_control_batch_size=5))
        b = tp.quadratic_control_grid_search(other, self.p)
        for key in ('first_violation_nu', 'level', 'witness', 'violation_count', 'all_controls_exit_count'):
            self.assertEqual(a[key], b[key])

    def test_c1_exclusion_and_boundary(self):
        system = replace(self.system, dynamics=lambda x, u:
            torch.where((x.square().sum(dim=1) < .3)[:, None], x, .5*x))
        ungated = tp.quadratic_control_grid_search(system, self.p)
        gated = tp.quadratic_control_grid_search(system, self.p, c1=.3)
        self.assertEqual(ungated['first_violation_nu'], .25)
        self.assertIsNone(gated['first_violation_nu'])
        self.assertEqual(gated['analytic_core_excluded_count'], 4)
        self.assertEqual(gated['evaluated_states'], 76)
        boundary = tp.quadratic_control_grid_search(system, self.p, c1=.25)
        self.assertEqual(boundary['first_violation_nu'], .25)

    def test_near_zero_roundoff_is_inside_analytic_core(self):
        iterator = tp.iter_threshold_grid_batches
        def batches(bounds, **kwargs):
            if bounds == self.system.verification.domain:
                yield self.p.new_tensor([[2.463307335887066e-16, 1.0651202142497596e-15], [1., 0.]])
            else:
                yield from iterator(bounds, **kwargs)
        with patch.object(tp, 'iter_threshold_grid_batches', side_effect=batches):
            result = tp.quadratic_control_grid_search(self.system, self.p, c1=1e-4)
        self.assertEqual(result['analytic_core_excluded_count'], 1)
        self.assertEqual(result['evaluated_states'], 1)
        self.assertEqual(result['first_violation_nu'], 1.)

    def test_no_failure_uses_domain_cap_not_lqr_input_cap(self):
        system = replace(self.system, dynamics=lambda x, u: .5*x)
        p = self.p.new_tensor([[2., .4], [.4, 1.]])
        result = tp.quadratic_control_grid_search(system, p)
        self.assertIsNone(result['first_violation_nu'])
        self.assertIsNone(result['witness'])
        expected = float((4/torch.linalg.inv(p).diagonal()).min())
        self.assertAlmostEqual(result['level'], .99*expected)
        self.assertEqual(result['limited_by'], 'domain')

    def test_all_controls_exit_has_no_placeholder_control(self):
        result = tp.quadratic_control_grid_search(
            replace(self.system, dynamics=lambda x, u: torch.full_like(x, 3.)), self.p)
        self.assertEqual(result['all_controls_exit_count'], 80)
        self.assertEqual(result['first_violation_nu'], .25)
        self.assertIsNone(result['witness']['best_control'])
        self.assertIsNone(result['witness']['successor'])
        self.assertEqual(result['witness']['reason'], 'all_controls_exit_domain')
        json.dumps(result, allow_nan=False)

    def test_brute_force_two_inputs_and_containment(self):
        system = replace(self.system, u_dim=2, u_lb=(-.5, -.5), u_hi=(.5, .5),
                         dynamics=lambda x, u: 1.5*x+u)
        result = tp.quadratic_control_grid_search(system, self.p)
        axis = torch.linspace(-2, 2, 9, device='cuda', dtype=torch.float64)
        x = torch.cartesian_prod(axis, axis)
        controls = tp.make_control_grid(system, 5, device=x.device, dtype=x.dtype)
        minima = []
        for state in x:
            fx = system.dynamics(state.expand(len(controls), -1), controls)
            value = tp.nu_from_p(fx, self.p).masked_fill(
                ~tp.points_in_bounds(fx, system.verification.domain), float('inf'))
            minima.append(value.min())
        nu = tp.nu_from_p(x, self.p)
        bad = (torch.stack(minima) >= nu) & (x != 0).any(dim=1)
        self.assertEqual(result['controls'], 25)
        self.assertEqual(result['violation_count'], int(bad.sum()))
        self.assertEqual(result['first_violation_nu'], float(nu[bad].min()))

    def test_outside_minimizer_does_not_hide_contained_option(self):
        # With a tilted P, a smaller quadratic value can still leave the box.
        def dynamics(x, u):
            escaped = x.new_tensor([2.1, -1.9]).expand_as(x)
            contained = x.new_tensor([1.5, 1.5]).expand_as(x)
            return torch.where((u[:, :1] < 0), escaped, contained)
        system = replace(self.system, dynamics=dynamics)
        p = self.p.new_tensor([[1., .9], [.9, 1.]])
        result = tp.quadratic_control_grid_search(system, p)
        self.assertEqual(result['all_controls_exit_count'], 0)
        self.assertEqual(result['witness']['successor'], [1.5, 1.5])
        self.assertGreaterEqual(result['witness']['best_control'][0], 0.)

    def test_invalid_and_nonfinite_fail_explicitly(self):
        with self.assertRaises(ValueError):
            tp.quadratic_control_grid_search(self.system, self.p, device=torch.device('cpu'))
        with self.assertRaises(ValueError):
            tp.quadratic_control_grid_search(replace(self.system, training=replace(
                self.system.training, quadratic_control_search_scale=1.)), self.p)
        with self.assertRaises(ValueError):
            tp.quadratic_control_grid_search(replace(self.system,
                dynamics=lambda x, u: x*float('nan')), self.p)
        with self.assertRaises(ValueError):
            tp.quadratic_control_grid_search(self.system, -self.p)

    def test_optional_plot_and_stale_matrix(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        result = tp.quadratic_control_grid_search(self.system, self.p)
        axis = np.linspace(-2, 2, 21)
        xx, yy = np.meshgrid(axis, axis)
        fig, axes = plt.subplots(1, 2)
        with TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'quadratic_control_search.json').write_text(json.dumps(result))
            disabled = replace(self.system, training=replace(self.system.training,
                quadratic_control_search_enabled=False))
            with patch.object(Path, 'read_text', side_effect=AssertionError('disabled must not read')):
                self.assertEqual(tp.quadratic_control_plot_overlay(
                    disabled, None, path, self.p, axes, xx, yy, xx**2+yy**2), [])
            before = [len(ax.collections) for ax in axes]
            handles = tp.quadratic_control_plot_overlay(
                self.system, None, path, self.p, axes, xx, yy, xx**2+yy**2)
            self.assertEqual(len(handles), 1)
            self.assertEqual(handles[0].get_color(), 'yellow')
            self.assertIn('empirical', handles[0].get_label())
            self.assertTrue(all(len(ax.collections) > n for ax, n in zip(axes, before)))
            with self.assertRaises(ValueError):
                tp.quadratic_control_plot_overlay(
                    self.system, None, path, self.p*2, axes, xx, yy, xx**2+yy**2)
        plt.close(fig)

    def test_threshold_stage_preserves_verifier_payload(self):
        system = replace(self.system, training=replace(self.system.training,
            threshold_search_samples=81, threshold_search_batch_size=16))
        pi = tp.PiNet(n_dim=2, u_dim=1, hidden_sizes=(4,), p_matrix=self.p,
            k_matrix=[[1., 0.]], u_lb=system.u_lb, u_hi=system.u_hi).cuda()
        omega = tp.OmegaNet(2, (4,)).cuda()
        with TemporaryDirectory() as folder:
            path = Path(folder)
            payloads = []
            for enabled in (False, True):
                candidate = replace(system, training=replace(system.training,
                    quadratic_control_search_enabled=enabled))
                run = tp.PipelineRun(candidate, run_root=path, run_name=str(enabled))
                pipeline = tp.TrainingPipeline(candidate, device=torch.device('cuda'), run=run)
                pipeline.lqr_box_artifacts = SimpleNamespace(P=self.p, c_analytic=.01,
                    C_sample_cap=.02, c_s=.03, c_input=4., c_box=4., c_nonlinear_decrease_grid=.04)
                with patch.object(pipeline, 'checkpoint_path', return_value=path/'unused'), \
                     patch.object(tp, 'load_pinet', return_value=pi), \
                     patch.object(tp, 'load_omeganet', return_value=omega):
                    pipeline.search_thresholds()
                self.assertEqual((run.run_dir/'quadratic_control_search.json').exists(), enabled)
                payloads.append(json.loads((run.run_dir/'verification_thresholds_rescaled.json').read_text()))
                full = json.loads((run.run_dir/'threshold_search.json').read_text())
                self.assertEqual('quadratic_control_diagnostic' in full, enabled)
            self.assertEqual(payloads[0], payloads[1])


if __name__ == '__main__':
    unittest.main()
