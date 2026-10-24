import math
import tempfile
import unittest
from pathlib import Path

from models import PiNet, load_pinet
import torch


class BlendedControllerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def make_policy(self, device="cpu", dtype=torch.float32):
        return PiNet(
            n_dim=2, u_dim=2, hidden_sizes=(4,), activation="relu",
            p_matrix=[[2.0, 0.25], [0.25, 1.0]],
            k_matrix=[[0.2, 0.1], [-0.1, 0.3]],
            u_lb=[-1.0, -2.0], u_hi=[1.0, 2.0],
        ).to(device=device, dtype=dtype)

    def test_gate_levels_and_origin(self):
        for device in self.devices:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(device=device, dtype=dtype):
                    pi = self.make_policy(device, dtype)
                    # With P[0,0]=2 and c1=2, these states have nu/c1=0,1,2,100.
                    x = torch.tensor([[0, 0], [1, 0], [math.sqrt(2), 0], [10, 0]],
                                     device=device, dtype=dtype)
                    weight = torch.tensor([[0], [0.5], [0.8], [1]], device=device, dtype=dtype)
                    expected = (1 - weight) * pi.lqr_control(x) + weight * pi.nn_control(x)
                    torch.testing.assert_close(pi.blended_control(x, c1=2), expected)
                    torch.testing.assert_close(pi.blended_control(x, c1=2)[0], torch.zeros_like(x[0]))

    def test_shapes_bounds_and_unchanged_forward(self):
        for device in self.devices:
            pi = self.make_policy(device)
            for shape in ((2,), (0, 2), (31, 2), (3, 7, 2)):
                with self.subTest(device=device, shape=shape):
                    x = 10 * torch.randn(shape, device=device)
                    before = pi(x).detach().clone()
                    u = pi.blended_control(x, c1=0.1)
                    self.assertEqual(u.shape, x.shape)
                    self.assertTrue(torch.isfinite(u).all().item())
                    self.assertTrue((u >= pi.u_lb - 1e-6).all().item())
                    self.assertTrue((u <= pi.u_hi + 1e-6).all().item())
                    torch.testing.assert_close(pi(x), before, rtol=0, atol=0)
                    torch.testing.assert_close(pi(x), pi.nn_control(x), rtol=0, atol=0)

    def test_invalid_midpoints(self):
        pi = self.make_policy()
        for c1 in (0, -1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(c1=c1), self.assertRaisesRegex(ValueError, "finite and positive"):
                pi.blended_control(torch.zeros(1, 2), c1=c1)

    def test_gradients_and_lqr_linearization(self):
        for device in self.devices:
            with self.subTest(device=device):
                pi = self.make_policy(device, torch.float64)
                x = torch.zeros(2, device=device, dtype=torch.float64, requires_grad=True)
                jacobian = torch.autograd.functional.jacobian(lambda z: pi.blended_control(z, c1=0.1), x)
                torch.testing.assert_close(jacobian, -pi.K_matrix, rtol=0, atol=0)
                y = torch.tensor([[0.2, -0.1]], device=device, dtype=torch.float64, requires_grad=True)
                self.assertTrue(torch.autograd.gradcheck(lambda z: pi.blended_control(z, c1=0.1), (y,)))

    def test_current_and_legacy_checkpoint_loading(self):
        pi = self.make_policy()
        x = torch.randn(11, 2)
        metadata = dict(n_dim=2, u_dim=2, hidden_sizes=[4], activation="relu",
                        P=pi.P_matrix.tolist(), K=pi.K_matrix.tolist(),
                        u_lb=pi.u_lb.tolist(), u_hi=pi.u_hi.tolist())
        self.assertNotIn("c_sigmoid_center_tensor", pi.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            for legacy in (False, True):
                with self.subTest(legacy=legacy):
                    checkpoint = dict(metadata, model_state_dict=dict(pi.state_dict()))
                    if legacy:
                        checkpoint["c_sigmoid_center"] = 123.0
                        checkpoint["model_state_dict"]["c_sigmoid_center_tensor"] = torch.tensor(123.0)
                    torch.save(checkpoint, path)
                    loaded = load_pinet(path)
                    for method in ("nn_control", "lqr_control", "forward"):
                        torch.testing.assert_close(getattr(loaded, method)(x), getattr(pi, method)(x), rtol=0, atol=0)
                    torch.testing.assert_close(loaded.blended_control(x, c1=0.2), pi.blended_control(x, c1=0.2), rtol=0, atol=0)
            checkpoint["model_state_dict"]["unexpected_buffer"] = torch.tensor(1.0)
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(RuntimeError, "checkpoint mismatch"):
                load_pinet(path)


if __name__ == "__main__":
    unittest.main()
