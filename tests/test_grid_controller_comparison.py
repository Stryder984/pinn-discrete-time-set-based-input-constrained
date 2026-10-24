import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import sample_initial_sets as sis
from systems import f_sys_radial_clf_2d


class Value:
    def __init__(self, fn):
        self.fn = fn

    def query(self, x):
        return self.fn(x)


class GridControllerComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def fixture(self, device="cpu", n=1):
        system = SimpleNamespace(name="toy", n_dim=n, u_dim=1, u_lb=(-1.0,), u_hi=(1.0,),
                                 dt=0.1, dynamics=lambda x, u: x + 0.25 * u)
        value = Value(lambda x: x.square().sum(-1))
        models = dict(omega=value, omega_closed_loop=value,
                      pi=SimpleNamespace(P_matrix=torch.eye(n, device=device)))
        grid = torch.tensor([[-1.0], [0.0], [1.0]], device=device)
        domain = np.array([[-2.0, 2.0]] * n, dtype=np.float32)
        return system, models, grid, domain

    def rollout(self, objective="nu", points=None, steps=5, device="cpu", **overrides):
        system, models, grid, domain = self.fixture(device)
        if points is None:
            points = np.array([[0.1], [0.75], [1.75]], dtype=np.float32)
        args = dict(system=system, models=models, points=points, control_grid=grid,
                    steps=steps, objective=objective, roa_threshold=0.0625, domain=domain,
                    device=torch.device(device), control_chunk_size=2)
        args.update(overrides)
        return sis.simulate_grid_optimal_trajectories(**args)

    def test_objectives_match_exhaustive_grid_and_are_distinct(self):
        for device in self.devices:
            with self.subTest(device=device):
                system, models, grid, domain = self.fixture(device, n=2)
                system.dynamics = lambda x, u: torch.cat((u, 1 - u), -1)
                grid = torch.tensor([[0.0], [0.5], [1.0]], device=device)
                p = torch.diag(torch.tensor([10.0, 1.0], device=device))
                models["omega"] = Value(lambda x: (x[:, 0] - 1).square())
                x = torch.zeros(2, 2, device=device)
                successors = system.dynamics(x[:1].expand(3, -1), grid)
                for objective, expected_index in (("omega_orig", 2), ("norm2", 1), ("nu", 0)):
                    for chunk in (1, 2, 10):
                        result = sis.grid_optimal_controls(
                            system=system, omega_orig=models["omega"], x=x, control_grid=grid,
                            control_chunk_size=chunk, objective=objective, p_matrix=p,
                            domain=torch.tensor(domain, device=device), candidate_batch_size=4,
                        )
                        scores = {"omega_orig": models["omega"].query(successors),
                                  "norm2": successors.square().sum(-1),
                                  "nu": (successors @ p * successors).sum(-1)}[objective]
                        self.assertEqual(scores.argmin().item(), expected_index)
                        self.assertTrue(bool((result["grid_indices"] == expected_index).all()))
                        torch.testing.assert_close(result["successor_objective"], scores[expected_index].expand(2))
                        self.assertTrue(bool((result["feasible_count"] == 3).all()))

    def test_masks_nonfinite_and_out_of_domain_before_value_query(self):
        for device in self.devices:
            with self.subTest(device=device):
                system, models, _, _ = self.fixture(device)
                system.u_lb, system.u_hi = (0.0,), (3.0,)
                grid = torch.arange(4.0, device=device)[:, None]
                system.dynamics = lambda x, u: torch.where(u == 0, torch.nan, torch.where(u == 1, 2.0, u - 2.0))

                def query(x):
                    self.assertTrue(bool(torch.isfinite(x).all()))
                    self.assertTrue(bool((x.abs() <= 1).all()))
                    return torch.where(x[:, 0] == 0, torch.nan, x[:, 0])

                result = sis.grid_optimal_controls(
                    system=system, omega_orig=Value(query), x=torch.zeros(1, 1, device=device),
                    control_grid=grid, control_chunk_size=2, objective="omega_orig",
                    p_matrix=torch.ones(1, 1, device=device), domain=torch.tensor([[-1.0, 1.0]], device=device),
                )
                self.assertEqual(result["grid_indices"].item(), 3)
                for key in ("feasible_count", "out_of_domain_count", "nonfinite_successor_count", "nonfinite_objective_count"):
                    self.assertEqual(result[key].item(), 1)

    def test_ties_keep_first_index_across_chunks(self):
        system, models, grid, domain = self.fixture()
        result = sis.grid_optimal_controls(
            system=system, omega_orig=Value(lambda x: torch.zeros(len(x))), x=torch.zeros(1, 1),
            control_grid=grid, control_chunk_size=1, objective="omega_orig",
            p_matrix=torch.eye(1), domain=torch.tensor(domain),
        )
        self.assertEqual(result["grid_indices"].item(), 0)

    def test_c1_entry_lengths_terminal_states_and_padding(self):
        for device in self.devices:
            for objective in sis.GRID_OBJECTIVES:
                with self.subTest(device=device, objective=objective):
                    data = self.rollout(device=device, objective=objective)
                    np.testing.assert_array_equal(data["stop_reason"], ["reached_roa", "reached_roa", "horizon"])
                    np.testing.assert_array_equal(data["stop_step"], [0, 2, 5])
                    np.testing.assert_array_equal(data["state_lengths"], [1, 3, 6])
                    np.testing.assert_array_equal(data["transition_lengths"], [0, 2, 5])
                    np.testing.assert_array_equal(data["attempt_lengths"], [0, 2, 5])
                    np.testing.assert_allclose(data["terminal_states"][:, 0], [0.1, 0.25, 0.5])
                    self.assertTrue(np.isnan(data["states"][0, 1:]).all())
                    self.assertTrue(np.isnan(data["controls"][0]).all())
                    self.assertTrue((data["grid_indices"][0] == -1).all())
                    np.testing.assert_allclose(data["stop_time"], [0, 0.2, 0.5])
                    for i in (1, 2):
                        length = data["transition_lengths"][i]
                        np.testing.assert_allclose(data["objective_delta"][i, :length], np.diff(data[objective][i, :length + 1]))

    def test_no_feasible_control_retains_failed_search_without_advancing(self):
        system, models, grid, domain = self.fixture()
        system.dynamics = lambda x, u: torch.full_like(x, 3.0)
        data = self.rollout(points=np.array([[0.75]], dtype=np.float32), system=system)
        self.assertEqual(data["stop_reason"][0], "no_feasible_control")
        np.testing.assert_array_equal(data["state_lengths"], [1])
        np.testing.assert_array_equal(data["attempt_lengths"], [1])
        np.testing.assert_array_equal(data["transition_lengths"], [0])
        self.assertEqual(data["out_of_domain_count"][0, 0], 3)
        self.assertEqual(data["grid_indices"][0, 0], -1)
        self.assertTrue(np.isnan(data["controls"]).all())
        np.testing.assert_allclose(data["terminal_states"], [[0.75]])

    def test_horizon_zero_nonfinite_initial_state_and_empty_points(self):
        data = self.rollout(points=np.array([[0.1], [0.75], [np.nan]], dtype=np.float32), steps=0)
        np.testing.assert_array_equal(data["stop_reason"], ["reached_roa", "horizon", "nonfinite_state"])
        self.assertEqual(data["controls"].shape[1], 0)
        empty = self.rollout(points=np.empty((0, 1), dtype=np.float32))
        self.assertEqual(empty["states"].shape, (0, 0, 1))

    def test_nondecreasing_step_is_not_an_early_stop(self):
        system, _, _, _ = self.fixture()
        system.dynamics = lambda x, u: x + 0.1
        data = self.rollout(points=np.array([[0.75]], dtype=np.float32), system=system, steps=2)
        self.assertEqual(data["stop_reason"][0], "horizon")
        self.assertTrue((data["objective_delta"] > 0).all())
        self.assertEqual(data["transition_lengths"][0], 2)

    def test_entry_on_last_step_and_failed_attempt_after_advances(self):
        data = self.rollout(points=np.array([[0.75]], dtype=np.float32), steps=2)
        self.assertEqual(data["stop_reason"][0], "reached_roa")
        system, _, _, _ = self.fixture()
        system.dynamics = lambda x, u: torch.where(x > 0.5, x - 0.25, torch.full_like(x, 3.0))
        data = self.rollout(system=system, points=np.array([[1.0]], dtype=np.float32))
        self.assertEqual(data["stop_reason"][0], "no_feasible_control")
        np.testing.assert_array_equal(data["state_lengths"], [3])
        np.testing.assert_array_equal(data["attempt_lengths"], [3])
        np.testing.assert_array_equal(data["transition_lengths"], [2])
        np.testing.assert_allclose(data["states"][0, :, 0], [1.0, 0.75, 0.5])
        self.assertTrue(np.isnan(data["controls"][0, 2]).all())

    def test_invalid_roa_threshold_is_rejected(self):
        for value in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(roa_threshold=value), self.assertRaises(ValueError):
                self.rollout(roa_threshold=value)

    def test_saved_threshold_selection_never_falls_back(self):
        saved = {"c1": 0.0625, "c2": 0.25}
        for name in ("c1", "c2"):
            self.assertEqual(sis.select_grid_roa_threshold(saved, name), saved[name])
            for bad in (None, 0, -1, np.nan, np.inf, "bad"):
                with self.subTest(name=name, value=bad), self.assertRaisesRegex(ValueError, name):
                    sis.select_grid_roa_threshold(dict(saved, **{name: bad}), name)
        with self.assertRaisesRegex(ValueError, "c2"):
            sis.select_grid_roa_threshold({"c1": 0.0625}, "c2")
        with self.assertRaises(ValueError):
            sis.select_grid_roa_threshold(saved, "w2")

    def test_c2_entry_including_boundary_and_initial_points_for_all_objectives(self):
        points = np.array([[0.5], [0.75], [1.0]], dtype=np.float32)
        for device in self.devices:
            for objective in sis.GRID_OBJECTIVES:
                with self.subTest(device=device, objective=objective):
                    small = self.rollout(points=points, device=device, objective=objective, roa_threshold=0.0625)
                    expanded = self.rollout(points=points, device=device, objective=objective, roa_threshold=0.25)
                    np.testing.assert_array_equal(small["stop_step"], [1, 2, 3])
                    np.testing.assert_array_equal(expanded["stop_step"], [0, 1, 2])
                    np.testing.assert_array_equal(expanded["stop_reason"], ["reached_roa"] * 3)
                    np.testing.assert_array_equal(expanded["attempt_lengths"], [0, 1, 2])
                    self.assertTrue(np.isnan(expanded["controls"][0]).all())
                    for i, length in enumerate(expanded["state_lengths"]):
                        np.testing.assert_array_equal(expanded["states"][i, :length], small["states"][i, :length])
                        self.assertLessEqual(expanded["nu"][i, length - 1], 0.25)

    def test_failure_manifest_preserves_completed_shards(self):
        system, models, grid, domain = self.fixture()
        real_rollout = sis.simulate_grid_optimal_trajectories
        calls = 0

        def fail_second(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic failure")
            return real_rollout(**kwargs)

        with TemporaryDirectory() as directory:
            output = Path(directory) / "failed"
            with patch.object(sis, "simulate_grid_optimal_trajectories", side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    sis.run_grid_controller_comparison(
                        system=system, models=models, points=np.array([[0.5], [0.75]], dtype=np.float32),
                        inside_mask=np.array([True, False]), classification_values=np.array([0.25, 0.5625]),
                        control_grid=grid, objectives=["nu"], roa_threshold=0.0625, domain=domain, steps=2,
                        device=torch.device("cpu"), rollout_batch_size=1, candidate_batch_size=8,
                        control_chunk_size=2, output_dir=output, metadata={},
                    )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            shards = manifest["controllers"]["nu"]["shards"]
            self.assertEqual(len(shards), 1)
            self.assertTrue((output / shards[0]["file"]).is_file())

    def test_archive_round_trip_all_points_and_objectives_no_overwrite(self):
        system, models, grid, domain = self.fixture()
        points = np.array([[0.1], [0.75], [1.75]], dtype=np.float32)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "comparison"
            args = dict(system=system, models=models, points=points, inside_mask=np.array([True, True, False]),
                        classification_values=points[:, 0] ** 2, control_grid=grid, objectives=sis.GRID_OBJECTIVES,
                        roa_threshold=0.0625, domain=domain, steps=5, device=torch.device("cpu"), rollout_batch_size=2,
                        candidate_batch_size=4, control_chunk_size=2, output_dir=output, metadata={"seed": 7})
            result = sis.run_grid_controller_comparison(**args)
            manifest = json.loads(Path(result["manifest"]).read_text())
            self.assertEqual(manifest["status"], "complete")
            with np.load(output / "shared.npz", allow_pickle=False) as shared:
                np.testing.assert_array_equal(shared["initial_states"], points)
            for objective in sis.GRID_OBJECTIVES:
                entry = manifest["controllers"][objective]
                self.assertEqual(entry["summary"]["inside"]["reached_roa"], 2)
                self.assertEqual(entry["summary"]["outside"]["termination_counts"]["horizon"], 1)
                ids = []
                for shard in entry["shards"]:
                    with np.load(output / shard["file"], allow_pickle=False) as data:
                        ids.extend(data["point_ids"].tolist())
                        self.assertEqual(sis.file_sha256(output / shard["file"]), shard["sha256"])
                self.assertEqual(ids, [0, 1, 2])
            with self.assertRaises(FileExistsError):
                sis.run_grid_controller_comparison(**args)

    def test_radial_norm_and_quadratic_agree_and_select_lower_bound(self):
        for device in self.devices:
            with self.subTest(device=device):
                system, models, _, domain = self.fixture(device, n=2)
                system.dynamics = f_sys_radial_clf_2d
                system.u_lb, system.u_hi = (-2.0,), (2.0,)
                models["pi"].P_matrix *= 1e-4 / 0.095
                grid = torch.linspace(-2, 2, 51, device=device)[:, None]
                points = np.array([[1.5, 0.0], [0.8, 0.8]], dtype=np.float32)
                results = [self.rollout(
                    objective=objective, system=system, models=models, control_grid=grid, domain=domain,
                    points=points, device=device, roa_threshold=0.0002631578, steps=150,
                ) for objective in ("norm2", "nu")]
                np.testing.assert_array_equal(results[0]["states"], results[1]["states"])
                for data in results:
                    self.assertTrue((data["stop_reason"] == "reached_roa").all())
                    for i, length in enumerate(data["transition_lengths"]):
                        self.assertTrue((data["controls"][i, :length] == -2).all())


if __name__ == "__main__":
    unittest.main()
