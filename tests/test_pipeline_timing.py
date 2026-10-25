import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from pipeline_timing import PipelineTiming, timed_stage


class PipelineTimingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.timer = PipelineTiming(self.root, system="test", device="cpu")

    def read(self):
        return json.loads(self.timer.path.read_text(encoding="utf-8"))

    def test_phase_partition_and_final_loss(self):
        history = [{"epoch": 0, "total": 0.125, "data_mse": 0.25}]
        with patch("pipeline_timing.time.perf_counter", side_effect=[10., 12., 17., 18.]), \
                patch("pipeline_timing.torch.cuda.synchronize") as sync:
            with self.timer.measure("train"):
                self.timer.phase("optimization")
                self.timer.phase("plotting")
                self.timer.final_loss(history, filename="omega_loss_history.json")
                self.assertEqual(self.read()["status"], "running")
        event = self.read()["events"][0]
        self.assertEqual(event["elapsed_seconds"], 8.)
        self.assertEqual(event["phase_seconds"], {"setup": 2., "optimization": 5., "plotting": 1.})
        self.assertEqual(event["final_training_loss"]["last_epoch"], history[-1])
        self.assertEqual(event["status"], "complete")
        sync.assert_not_called()

    def test_nested_inclusive_times_and_parent_ids(self):
        with patch("pipeline_timing.time.perf_counter", side_effect=[1., 2., 4., 6.]):
            with self.timer.measure("run", kind="invocation"):
                with self.timer.measure("train"):
                    pass
        root, child = self.read()["events"]
        self.assertEqual(root["elapsed_seconds"], 5.)
        self.assertEqual(child["elapsed_seconds"], 2.)
        self.assertEqual(child["parent_id"], root["id"])
        self.assertEqual(self.timer.stack, [])

    def test_reruns_do_not_overwrite_or_reuse_loss(self):
        with self.timer.measure("train"):
            self.timer.final_loss([{"epoch": 0, "total": 1.}], filename="loss.json")
        first = self.timer.path
        original = first.read_bytes()
        with self.timer.measure("train"):
            pass
        self.assertNotEqual(first, self.timer.path)
        self.assertEqual(first.read_bytes(), original)
        self.assertIsNone(self.read()["events"][0]["final_training_loss"])
        self.assertEqual(len(list(self.root.glob("*.json"))), 2)

    def test_failures_and_interruptions_are_recorded_and_reraised(self):
        for error, status in ((ValueError("bad data"), "failed"), (KeyboardInterrupt(), "interrupted")):
            with self.subTest(status=status):
                with self.assertRaises(type(error)) as caught:
                    with self.timer.measure("run", kind="invocation"):
                        with self.timer.measure("train"):
                            raise error
                self.assertIs(caught.exception, error)
                report = self.read()
                self.assertEqual(report["status"], status)
                self.assertTrue(all(e["status"] == status for e in report["events"]))
                self.assertEqual(self.timer.stack, [])

    def test_skipped_is_not_success(self):
        with self.timer.measure("run"):
            self.timer.skip("preflight")
        self.assertEqual(self.read()["status"], "skipped")

    def test_cuda_synchronizes_boundaries_only(self):
        timer = PipelineTiming(self.root, system="test", device="cuda:0")
        with patch("pipeline_timing.torch.cuda.synchronize") as sync, \
                patch("pipeline_timing.torch.cuda.get_device_name", return_value="test GPU"):
            with timer.measure("train"):
                timer.phase("optimization")
                for _ in range(10):
                    pass
                timer.final_loss([], filename="loss.json")
        self.assertEqual(sync.call_count, 3)
        self.assertTrue(all(c.args == (torch.device("cuda:0"),) for c in sync.call_args_list))

    def test_sync_failure_does_not_hide_training_failure(self):
        timer = PipelineTiming(self.root, system="test", device="cuda")
        with patch("pipeline_timing.torch.cuda.synchronize", side_effect=[None, RuntimeError("sync failed")]), \
                patch("pipeline_timing.torch.cuda.get_device_name", return_value="test GPU"):
            with self.assertRaisesRegex(ValueError, "original error"):
                with timer.measure("train"):
                    raise ValueError("original error")
        report = json.loads(timer.path.read_text())
        self.assertEqual(report["events"][0]["synchronization_error"], "sync failed")

    def test_logging_io_failure_does_not_replace_training_error(self):
        with patch.object(Path, "write_text", side_effect=OSError("disk full")), \
                self.assertWarnsRegex(RuntimeWarning, "Unable to save pipeline timings"):
            with self.assertRaisesRegex(ValueError, "training failed"):
                with self.timer.measure("train"):
                    raise ValueError("training failed")

    def test_nonfinite_and_empty_loss_are_explicit(self):
        history = [{"epoch": 0, "total": float("nan"), "component": float("inf")}]
        with self.timer.measure("train"):
            self.timer.final_loss(history, filename="loss.json")
        loss = self.read()["events"][0]["final_training_loss"]
        self.assertIsNone(loss["total"])
        self.assertEqual(loss["nonfinite_fields"], ["total", "component"])
        self.assertTrue(torch.isnan(torch.tensor(history[0]["total"])))
        with self.timer.measure("train"):
            self.timer.final_loss([], filename="loss.json")
        loss = self.read()["events"][0]["final_training_loss"]
        self.assertEqual(loss["epochs_recorded"], 0)
        self.assertIsNone(loss["last_epoch"])

    def test_decorator_preserves_result_rng_and_updates(self):
        class Example:
            timing = self.timer

            @timed_stage()
            def train(self):
                model = torch.nn.Linear(2, 1)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                self.timing.phase("optimization")
                for _ in range(2):
                    optimizer.zero_grad()
                    model(torch.randn(3, 2)).square().mean().backward()
                    optimizer.step()
                return {key: value.clone() for key, value in model.state_dict().items()}

        obj = Example()
        torch.manual_seed(9)
        expected = Example.train.__wrapped__(obj)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(9)
        actual = obj.train()
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]))

    def test_pipeline_preflight_skip(self):
        from types import SimpleNamespace
        from training_pipeline import PipelineRun, TrainingPipeline
        from systems import get_system

        system = get_system("van_der_pol")
        run = PipelineRun(system, run_root=self.root, run_name="preflight")
        pipeline = TrainingPipeline(system, run=run, device=torch.device("cpu"))
        with patch("training_pipeline.check_machine_preflight", return_value=SimpleNamespace(ok=False, messages=[])), \
                patch.object(pipeline, "train_omega") as train:
            pipeline.run("train_omega")
        train.assert_not_called()
        report = json.loads(pipeline.timing.path.read_text())
        self.assertEqual(report["status"], "skipped")
        self.assertEqual(report["events"][0]["details"]["requested_stage"], "train_omega")


if __name__ == "__main__":
    unittest.main()
