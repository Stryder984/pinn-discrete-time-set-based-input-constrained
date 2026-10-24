import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

import plot_grid_controller_comparison as report
import sample_initial_sets as sis
import test_grid_controller_comparison as fixtures


class ComparisonReportingTests(unittest.TestCase):
    def test_state_normalization_preserves_objective_data_and_aligns_segments(self):
        domain = np.array([[-2., 6.], [-30., 10.]])
        states = np.array([[-2., -30.], [2., -10.], [6., 10.]])
        original = states.copy()
        normalized = report.normalized_states(states, domain)
        np.testing.assert_allclose(normalized, [[-1, -1], [0, 0], [1, 1]])
        np.testing.assert_array_equal(states, original)
        mask = np.array([False, True])
        np.testing.assert_allclose(report.increasing_segments(normalized, mask),
                                   report.normalized_states(report.increasing_segments(states, mask), domain))
        with self.assertRaisesRegex(ValueError, "positive domain widths"):
            report.normalized_states(states, [[0, 0], [-1, 1]])

    def test_outcomes_do_not_mislabel_nonfinite_candidates_as_domain_exits(self):
        data = dict(stop_reason=np.array(["no_feasible_control"] * 3 + ["nonfinite_state", "horizon", "reached_roa"]),
                    attempt_lengths=np.ones(6, dtype=int), nonfinite_successor_count=np.array([[0], [1], [0], [0], [0], [0]]),
                    nonfinite_objective_count=np.array([[0], [0], [1], [0], [0], [0]]),
                    out_of_domain_count=np.array([[3], [2], [2], [0], [0], [0]]))
        self.assertEqual([report.outcome_for_row(data, i, 3) for i in range(6)],
                         ["no_in_domain_successor", "numerical_error", "numerical_error", "numerical_error", "unconverged_in_domain", "reached_roa"])

    def test_denominators_pairing_and_times(self):
        shared = dict(point_ids=np.array([10, 20, 30, 40, 50, 60]), inside_mask=np.array([True, True, True, False, False, False]),
                      roa_threshold_name="c2", roa_threshold=0.25)

        def row(outcome, time=2, initial=False, increase=False):
            return dict(outcome=outcome, stop_time=time, already_inside=initial, increase_steps=int(increase),
                        largest_increase=0.2 if increase else 0, longest_increase_streak=int(increase), nonfinite_increments=0)

        omega = {10: row("reached_roa", 0, True), 20: row("reached_roa", 2, increase=True), 30: row("reached_roa", 4),
                 40: row("unconverged_in_domain"), 50: row("no_in_domain_successor"), 60: row("numerical_error")}
        norm = {60: row("numerical_error"), 50: row("unconverged_in_domain"), 40: row("reached_roa", 5),
                30: row("unconverged_in_domain"), 20: row("reached_roa", 3), 10: row("reached_roa", 0, True)}
        outcomes, pairs, increases = report.comparison_tables(shared, dict(omega_orig=omega, norm2=norm))
        first = outcomes[0]
        self.assertEqual(first["initially_inside_roa"], 1)
        self.assertEqual(first["tested_outside_roa"], 5)
        self.assertEqual(first["success_percent"], 40)
        self.assertEqual(first["median_entry_time_s"], 3)
        self.assertEqual(sum(first[key] for key in report.OUTCOMES), 5)
        self.assertEqual([pairs[0][key] for key in ("both_reach_roa", "only_omega_reaches", "only_alternative_reaches", "neither_reaches")], [1, 1, 1, 2])
        self.assertEqual(pairs[0]["median_omega_minus_alternative_time_s"], -1)
        self.assertEqual(increases[0]["successful_with_increases"], 1)
        self.assertEqual(increases[0]["largest_positive_increment"], 0.2)
        self.assertEqual(outcomes[2]["group"], "Inside sublevel")
        self.assertEqual(outcomes[2]["tested_outside_roa"], 2)
        self.assertEqual(outcomes[2]["success_percent"], 100)

    def test_empty_groups_are_not_zero_percent_success(self):
        shared = dict(point_ids=np.array([], dtype=int), inside_mask=np.array([], dtype=bool),
                      roa_threshold_name="c1", roa_threshold=0.01)
        outcomes, pairs, increases = report.comparison_tables(shared, {"nu": {}})
        self.assertIsNone(outcomes[0]["success_percent"])
        self.assertIsNone(outcomes[0]["median_entry_time_s"])
        self.assertEqual(pairs, [])

    def test_yellow_segments_and_streaks_are_transition_aligned(self):
        states = np.array([[0, 0], [1, 0], [1, 1], [2, 1]])
        np.testing.assert_array_equal(report.increasing_segments(states, np.array([False, True, False])),
                                      np.array([[[1, 0], [1, 1]]]))
        self.assertEqual(report.longest_true_streak([False, True, True, False, True]), 2)
        self.assertEqual(report.increasing_segments(states[:1], np.array([], dtype=bool)).shape, (0, 2, 2))

    def test_archive_reports_each_objectives_own_increases_and_all_points(self):
        fixture = fixtures.GridControllerComparisonTests()
        system, models, grid, domain = fixture.fixture(n=2)
        # State moves right regardless of control. Omega decreases while both
        # norm and quadratic increase: their yellow masks must therefore differ.
        system.dynamics = lambda x, u: x + torch.tensor([0.1, 0.0], device=x.device)
        models["omega"].fn = lambda x: -x[:, 0]
        with TemporaryDirectory() as directory:
            root = Path(directory) / "comparison"
            sis.run_grid_controller_comparison(
                system=system, models=models, points=np.array([[0.5, 0], [0.75, 0]], dtype=np.float32),
                inside_mask=np.array([True, False]), classification_values=np.array([0.1, 0.8]),
                control_grid=grid, objectives=sis.GRID_OBJECTIVES, roa_threshold=0.01, domain=domain, steps=3,
                device=torch.device("cpu"), rollout_batch_size=1, candidate_batch_size=8,
                control_chunk_size=3, output_dir=root, metadata={},
            )
            manifest, shared, metrics, histories, selected = report.load_comparison(root, max_trajectories=1)
            self.assertEqual(len(selected), 1)
            for objective in sis.GRID_OBJECTIVES:
                self.assertEqual(len(metrics[objective]), 2)
                self.assertEqual(list(histories[objective]), selected)
            self.assertFalse(histories["omega_orig"][selected[0]]["increase"].any())
            self.assertTrue(histories["norm2"][selected[0]]["increase"].all())
            self.assertTrue(histories["nu"][selected[0]]["increase"].all())
            with patch.object(report, "load_original_for_contours", return_value=models["omega"]):
                with patch.object(system, "dynamics", side_effect=AssertionError("Reporting must not simulate")):
                    result = report.generate_report(root, device="cpu", contour_resolution=21, max_trajectories=1)
            output = Path(result["report_dir"])
            for name in ("tables.png", "trajectories.png", "outcomes.csv", "paired_outcomes.csv", "increases.csv", "report.json"):
                self.assertGreater((output / name).stat().st_size, 0)
            self.assertEqual(result["outcomes"][0]["tested_outside_roa"], 2)
            self.assertEqual(result["selected_point_ids"], selected)
            (root / "shared.npz").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                report.load_comparison(root)

    def test_higher_dimensional_archives_produce_tables_without_projected_contours(self):
        fixture = fixtures.GridControllerComparisonTests()
        system, models, grid, domain = fixture.fixture(n=3)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "comparison"
            sis.run_grid_controller_comparison(
                system=system, models=models, points=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
                inside_mask=np.array([True]), classification_values=np.array([0.4]), control_grid=grid,
                objectives=["nu"], roa_threshold=0.01, domain=domain, steps=1, device=torch.device("cpu"),
                rollout_batch_size=1, candidate_batch_size=8, control_chunk_size=3, output_dir=root, metadata={},
            )
            with patch.object(report, "load_original_for_contours", side_effect=AssertionError("No 3D contours")):
                result = report.generate_report(root, device="cpu")
            self.assertIsNone(result["trajectories"])
            self.assertTrue((Path(result["report_dir"]) / "tables.png").is_file())
            self.assertEqual(result["outcomes"][0]["tested_outside_roa"], 1)

    def test_unknown_candidate_cost_does_not_claim_unavoidable_increase(self):
        fixture = fixtures.GridControllerComparisonTests()
        system, models, grid, domain = fixture.fixture()
        system.dynamics = lambda x, u: torch.where(u > 0, torch.full_like(x, torch.nan), x + 0.1)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "comparison"
            sis.run_grid_controller_comparison(
                system=system, models=models, points=np.array([[0.5]], dtype=np.float32),
                inside_mask=np.array([True]), classification_values=np.array([0.25]), control_grid=grid,
                objectives=["norm2"], roa_threshold=0.01, domain=domain, steps=1, device=torch.device("cpu"),
                rollout_batch_size=1, candidate_batch_size=8, control_chunk_size=3, output_dir=root, metadata={},
            )
            _, _, metrics, histories, _ = report.load_comparison(root)
            self.assertFalse(histories["norm2"][0]["increase"].any())
            self.assertEqual(metrics["norm2"][0]["unresolved_positive_steps"], 1)

    def test_c2_archive_reports_selected_threshold_everywhere(self):
        fixture = fixtures.GridControllerComparisonTests()
        system, models, grid, domain = fixture.fixture(n=2)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "c2_comparison"
            sis.run_grid_controller_comparison(
                system=system, models=models, points=np.array([[0, 0], [0.75, 0.75]], dtype=np.float32),
                inside_mask=np.array([True, False]), classification_values=np.array([0, 1.125]),
                control_grid=grid, objectives=sis.GRID_OBJECTIVES, roa_threshold=0.5, roa_threshold_name="c2",
                domain=domain, steps=3, device=torch.device("cpu"), rollout_batch_size=2,
                candidate_batch_size=8, control_chunk_size=3, output_dir=root, metadata={},
            )
            manifest, shared, _, _, _ = report.load_comparison(root)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["roa_threshold_name"], "c2")
            self.assertEqual(shared["roa_threshold"], 0.5)
            self.assertIn("<=c2", manifest["stopping_criterion"])
            with patch.object(report, "load_original_for_contours", return_value=models["omega"]):
                with patch.object(report.plt, "close"):
                    result = report.generate_report(root, device="cpu", contour_resolution=31, max_trajectories=0)
                    figures = [report.plt.figure(num) for num in report.plt.get_fignums()]
                    texts = [obj.get_text() for fig in figures for obj in fig.findobj()
                             if isinstance(obj, report.matplotlib.text.Text)]
                    texts.extend(cell.get_text().get_text() for fig in figures for ax in fig.axes
                                 for table in ax.tables for cell in table.get_celld().values())
                    self.assertIn("nu = c2", texts)
                    self.assertIn("Reached expanded ROA (c2)", texts)
                    self.assertTrue(any("outside c2" in text for text in texts))
                    self.assertFalse(any("c1" in text for text in texts))
                report.plt.close("all")
            output = Path(result["report_dir"])
            self.assertEqual(result["roa_threshold_name"], "c2")
            for row in result["outcomes"]:
                self.assertEqual(row["roa_threshold_name"], "c2")
                self.assertEqual(row["roa_threshold"], 0.5)
            self.assertEqual(result["outcomes"][0]["initially_inside_roa"], 1)
            self.assertEqual(result["outcomes"][0]["tested_outside_roa"], 1)
            self.assertEqual(result["outcomes"][0]["reached_roa"], 1)
            self.assertIn("nu <= c2 = 0.5", (output / "README.md").read_text())
            for filename in ("outcomes.csv", "paired_outcomes.csv", "increases.csv"):
                text = (output / filename).read_text()
                self.assertIn("roa_threshold_name,roa_threshold", text)
                self.assertIn("c2,0.5", text)
                self.assertNotIn("c1", text)

    def test_legacy_c1_archive_remains_readable_without_reinterpreting_stops(self):
        fixture = fixtures.GridControllerComparisonTests()
        system, models, grid, domain = fixture.fixture()
        with TemporaryDirectory() as directory:
            root = Path(directory) / "legacy"
            sis.run_grid_controller_comparison(
                system=system, models=models, points=np.array([[0.1], [0.75]], dtype=np.float32),
                inside_mask=np.array([True, False]), classification_values=np.array([0.01, 0.5625]),
                control_grid=grid, objectives=["nu"], roa_threshold=0.0625, domain=domain, steps=1,
                device=torch.device("cpu"), rollout_batch_size=2, candidate_batch_size=8,
                control_chunk_size=3, output_dir=root, metadata={},
            )
            # Recreate the old schema, including its raw stop code and hashes.
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["schema_version"] = 1
            manifest["c1"] = manifest.pop("roa_threshold")
            manifest.pop("roa_threshold_name")
            with np.load(root / "shared.npz") as saved:
                shared = dict(saved)
            shared["c1"] = shared.pop("roa_threshold")
            shared.pop("roa_threshold_name")
            np.savez_compressed(root / "shared.npz", **shared)
            manifest["shared_sha256"] = sis.file_sha256(root / "shared.npz")
            shard = manifest["controllers"]["nu"]["shards"][0]
            with np.load(root / shard["file"]) as saved:
                data = dict(saved)
            data["stop_reason"][data["stop_reason"] == "reached_roa"] = "reached_c1"
            np.savez_compressed(root / shard["file"], **data)
            shard["sha256"] = sis.file_sha256(root / shard["file"])
            manifest_path.write_text(json.dumps(manifest))
            _, shared, metrics, _, _ = report.load_comparison(root)
            self.assertEqual(shared["roa_threshold_name"], "c1")
            self.assertEqual(shared["roa_threshold"], 0.0625)
            self.assertEqual(metrics["nu"][0]["outcome"], "reached_roa")
            self.assertTrue(metrics["nu"][0]["already_inside"])
            self.assertEqual(metrics["nu"][1]["outcome"], "unconverged_in_domain")

    def test_mismatched_archive_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            report.archive_roa_threshold(
                dict(schema_version=2, roa_threshold_name="c2", roa_threshold=0.5),
                dict(roa_threshold_name=np.array("c1"), roa_threshold=np.array(0.5)))


if __name__ == "__main__":
    unittest.main()
