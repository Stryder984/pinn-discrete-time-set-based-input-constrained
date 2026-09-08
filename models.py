from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn

from systems import SystemSpec


def _as_float_tensor(value, *, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).clone()
    if shape is not None:
        tensor = tensor.view(*shape)
    return tensor


def _state_dict_from_checkpoint(obj) -> Mapping[str, torch.Tensor]:
    if isinstance(obj, Mapping) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, Mapping):
        return obj
    raise TypeError(f"Unsupported checkpoint object: {type(obj)!r}")


def infer_linear_sizes(state_dict: Mapping[str, torch.Tensor]) -> List[Tuple[int, int]]:
    sizes = []
    linear_indices = sorted(
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("net.") and key.endswith(".weight")
    )
    for idx in linear_indices:
        weight = state_dict[f"net.{idx}.weight"]
        sizes.append((int(weight.shape[1]), int(weight.shape[0])))
    if not sizes:
        raise ValueError("No net.*.weight tensors found in checkpoint.")
    return sizes


def hidden_sizes_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> List[int]:
    sizes = infer_linear_sizes(state_dict)
    return [out_features for _, out_features in sizes[:-1]]


def embed_singleton(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.zeros_like(x)], dim=-1)


def embed_interval(lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    return torch.cat([(lb + ub) / 2.0, ub - lb], dim=-1)


class OmegaNet(nn.Module):
    """MLP over reachable-set embeddings with a sigmoid-bounded value output."""

    def __init__(
        self,
        n_dim: int,
        hidden_sizes: Iterable[int],
        *,
        input_dim: Optional[int] = None,
        embedding_mode: str = "reachable_interval_affine_control",
    ):
        super().__init__()
        self.n_dim = int(n_dim)
        self.input_dim = int(input_dim) if input_dim is not None else 2 * self.n_dim
        if self.input_dim < self.n_dim:
            raise ValueError(f"OmegaNet input_dim={self.input_dim} must be at least n_dim={self.n_dim}.")
        layers: List[nn.Module] = []
        prev = self.input_dim
        for hidden in hidden_sizes:
            layers += [nn.Linear(prev, int(hidden)), nn.Tanh()]
            prev = int(hidden)
        layers.append(nn.Linear(prev, 1))
        self.embedding_mode = str(embedding_mode)
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z).squeeze(-1))

    def query(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate a singleton state in this model's reachable-set input space."""
        if x.shape[-1] != self.n_dim:
            raise ValueError(f"x last dimension must be n_dim={self.n_dim}.")
        if self.input_dim == self.n_dim:
            return self.forward(x)
        zeros = torch.zeros(*x.shape[:-1], self.input_dim - self.n_dim, device=x.device, dtype=x.dtype)
        return self.forward(torch.cat([x, zeros], dim=-1))


class PiNet(nn.Module):
    """Controller container with explicit saturated-NN and clamped-LQR branches."""

    def __init__(
        self,
        *,
        n_dim: int,
        u_dim: int,
        hidden_sizes: Iterable[int],
        p_matrix,
        k_matrix,
        u_lb,
        u_hi,
        activation: str = "tanh",
        c_nn_off: Optional[float] = None,
        c_s: Optional[float] = None,
        c_sigmoid_center: Optional[float] = None,
        sharpness: Optional[float] = None,
        gate_type: Optional[str] = None,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = int(n_dim)
        activation_name = str(activation).strip().lower()
        activation_factories = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "swish": nn.SiLU,
        }
        if activation_name not in activation_factories:
            known = ", ".join(sorted(activation_factories))
            raise ValueError(f"Unknown PiNet activation {activation!r}; expected one of: {known}.")
        for hidden in hidden_sizes:
            layers += [nn.Linear(prev, int(hidden)), activation_factories[activation_name]()]
            prev = int(hidden)
        layers.append(nn.Linear(prev, int(u_dim)))

        self.n_dim = int(n_dim)
        self.u_dim = int(u_dim)
        self.activation_name = activation_name
        self.net = nn.Sequential(*layers)
        self.register_buffer("P_matrix", _as_float_tensor(p_matrix))
        self.register_buffer("K_matrix", _as_float_tensor(k_matrix))
        self.register_buffer("u_lb", _as_float_tensor(u_lb, shape=(self.u_dim,)))
        self.register_buffer("u_hi", _as_float_tensor(u_hi, shape=(self.u_dim,)))
        normalized_gate = str(gate_type or "").strip().lower()
        use_branch_only = normalized_gate in {"branch_only", "nn", "saturated_nn", "tanh_nn"}
        use_relu_gate = normalized_gate in {"saturated_relu", "relu", "relu_ramp"}
        use_sigmoid_gate = not (use_branch_only or use_relu_gate)
        if use_branch_only:
            c_s = float(c_s if c_s is not None else 0.0)
            c_nn_off = float(c_nn_off if c_nn_off is not None else c_s)
            c_sigmoid_center = float(
                c_sigmoid_center if c_sigmoid_center is not None else c_s
            )
            sharpness = float(sharpness if sharpness is not None else 0.0)
        elif use_sigmoid_gate:
            if c_sigmoid_center is None:
                if c_s is None:
                    raise ValueError("Sigmoid PiNet requires c_sigmoid_center or c_s as the switch center.")
                c_sigmoid_center = float(c_s)
            if sharpness is None:
                raise ValueError("Sigmoid PiNet requires sharpness.")
            c_s = float(c_s if c_s is not None else c_sigmoid_center)
            c_nn_off = float(c_nn_off if c_nn_off is not None else c_sigmoid_center)
        else:
            if c_nn_off is None or c_s is None:
                raise ValueError("ReLU PiNet requires c_nn_off and c_s.")
            c_nn_off = float(c_nn_off)
            c_s = float(c_s)
            if not c_s > c_nn_off:
                raise ValueError(
                    f"PiNet requires c_s > c_nn_off, got c_s={c_s} and "
                    f"c_nn_off={c_nn_off}."
                )
            c_sigmoid_center = float(
                c_sigmoid_center if c_sigmoid_center is not None else c_nn_off
            )
            sharpness = float(sharpness if sharpness is not None else 100.0)

        self.use_branch_only = bool(use_branch_only)
        self.use_sigmoid_gate = bool(use_sigmoid_gate)
        self.register_buffer("c_nn_off_tensor", torch.tensor(c_nn_off, dtype=torch.float32))
        self.register_buffer("c_s_tensor", torch.tensor(c_s, dtype=torch.float32))
        self.register_buffer(
            "c_sigmoid_center_tensor",
            torch.tensor(
                float(c_sigmoid_center) if c_sigmoid_center is not None else 0.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "sharpness_tensor",
            torch.tensor(float(sharpness) if sharpness is not None else 0.0, dtype=torch.float32),
        )

    def nn_control(self, x: torch.Tensor) -> torch.Tensor:
        """Return the zero-at-origin saturated neural controller branch."""
        u_lb = self.u_lb.to(device=x.device, dtype=x.dtype)
        u_hi = self.u_hi.to(device=x.device, dtype=x.dtype)
        mid = (u_hi + u_lb) / 2.0
        half = (u_hi - u_lb) / 2.0
        origin = torch.zeros(1, self.n_dim, device=x.device, dtype=x.dtype)
        origin_output = self.net(origin).squeeze(0)
        return torch.clamp(mid + half * torch.tanh(self.net(x) - origin_output), min=u_lb, max=u_hi)

    def lqr_control(self, x: torch.Tensor) -> torch.Tensor:
        """Saturated origin-linearized LQR branch."""
        k_matrix = self.K_matrix.to(device=x.device, dtype=x.dtype)
        u_lb = self.u_lb.to(device=x.device, dtype=x.dtype)
        u_hi = self.u_hi.to(device=x.device, dtype=x.dtype)
        return torch.clamp(-(x @ k_matrix.T), min=u_lb, max=u_hi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_branch_only:
            return self.nn_control(x)

        p_matrix = self.P_matrix.to(device=x.device, dtype=x.dtype)
        u_lb = self.u_lb.to(device=x.device, dtype=x.dtype)
        u_hi = self.u_hi.to(device=x.device, dtype=x.dtype)

        nu = (x @ p_matrix * x).sum(dim=-1, keepdim=True)
        u_nn = self.nn_control(x)
        u_lin = self.lqr_control(x)
        if self.use_sigmoid_gate:
            center = self.c_sigmoid_center_tensor.to(device=x.device, dtype=x.dtype)
            sharpness = self.sharpness_tensor.to(device=x.device, dtype=x.dtype)
            nn_gate = torch.sigmoid(sharpness * (nu - center))
        else:
            c_nn_off = self.c_nn_off_tensor.to(device=x.device, dtype=x.dtype)
            c_s = self.c_s_tensor.to(device=x.device, dtype=x.dtype)
            gate_width = c_s - c_nn_off
            nn_gate = (F.relu(nu - c_nn_off) - F.relu(nu - c_s)) / gate_width
        lqr_gate = 1.0 - nn_gate
        return torch.clamp(nn_gate * u_nn + lqr_gate * u_lin, min=u_lb, max=u_hi)


def load_pinet(checkpoint_path: os.PathLike | str, device: str | torch.device = "cpu") -> PiNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise ValueError("PiNet checkpoint must be a dict with model_state_dict and metadata.")

    gate_type = str(checkpoint.get("gate_type", "")).strip().lower()
    has_branch_only_gate = gate_type in {"branch_only", "nn", "saturated_nn", "tanh_nn"}
    has_relu_gate = gate_type in {"saturated_relu", "relu", "relu_ramp"} or (
        not gate_type
        and ("c_nn_off" in checkpoint or "C_NN_OFF" in checkpoint)
        and ("c_s" in checkpoint or "C_S" in checkpoint)
        and "sharpness" not in checkpoint
        and "c1_eff" not in checkpoint
    )
    model = PiNet(
        n_dim=checkpoint["n_dim"],
        u_dim=checkpoint["u_dim"],
        hidden_sizes=checkpoint["hidden_sizes"],
        p_matrix=checkpoint["P"],
        k_matrix=checkpoint["K"],
        u_lb=checkpoint["u_lb"],
        u_hi=checkpoint["u_hi"],
        activation=str(checkpoint.get("activation", checkpoint.get("pi_activation", "tanh"))),
        c_nn_off=checkpoint.get("c_nn_off", checkpoint.get("C_NN_OFF")) if has_relu_gate else None,
        c_s=checkpoint.get("c_s", checkpoint.get("C_S")),
        c_sigmoid_center=(
            checkpoint.get("c_sigmoid_center", checkpoint.get("c1_eff"))
            if not (has_branch_only_gate or has_relu_gate)
            else None
        ),
        sharpness=checkpoint.get("sharpness") if not (has_branch_only_gate or has_relu_gate) else None,
        gate_type=(
            "branch_only"
            if has_branch_only_gate
            else ("saturated_relu" if has_relu_gate else "sigmoid_blend")
        ),
    )
    state_dict = dict(checkpoint["model_state_dict"])
    state_dict.pop("u_lb", None)
    state_dict.pop("u_hi", None)
    state_dict.pop("c1_eff_tensor", None)
    state_dict.pop("sharpness_tensor", None)
    state_dict.pop("legacy_c1_eff_tensor", None)
    state_dict.pop("legacy_sharpness_tensor", None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "P_matrix",
        "K_matrix",
        "u_lb",
        "u_hi",
        "c_nn_off_tensor",
        "c_s_tensor",
        "c_sigmoid_center_tensor",
        "sharpness_tensor",
    }
    if unexpected or set(missing) - allowed_missing:
        raise RuntimeError(
            "Unexpected PiNet checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.to(device)
    model.eval()
    return model


def load_omeganet(
    checkpoint_path: os.PathLike | str,
    *,
    n_dim: Optional[int] = None,
    device: str | torch.device = "cpu",
) -> OmegaNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _state_dict_from_checkpoint(checkpoint)
    if n_dim is None:
        n_dim = int(checkpoint.get("n_dim", 2)) if isinstance(checkpoint, Mapping) else 2
    first_weight = next(
        tensor
        for key, tensor in state_dict.items()
        if key.startswith("net.") and key.endswith(".weight")
    )
    input_dim = int(checkpoint.get("input_dim", first_weight.shape[1])) if isinstance(checkpoint, Mapping) else int(first_weight.shape[1])
    embedding_mode = (
        str(checkpoint.get("reachable_set_embedding_mode", checkpoint.get("embedding_mode", "reachable_interval_affine_control")))
        if isinstance(checkpoint, Mapping)
        else "reachable_interval_affine_control"
    )

    hidden_sizes = (
        checkpoint.get("hidden_sizes")
        if isinstance(checkpoint, Mapping) and "hidden_sizes" in checkpoint
        else hidden_sizes_from_state_dict(state_dict)
    )
    model = OmegaNet(
        n_dim=n_dim,
        hidden_sizes=hidden_sizes,
        input_dim=input_dim,
        embedding_mode=embedding_mode,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_models(
    system: SystemSpec,
    checkpoint_dir: os.PathLike | str,
    device: str | torch.device = "cpu",
) -> Dict[str, nn.Module]:
    checkpoint_dir = Path(checkpoint_dir)
    pi = load_pinet(checkpoint_dir / system.checkpoints.pi, device=device)
    omega = load_omeganet(checkpoint_dir / system.checkpoints.omega, n_dim=pi.n_dim, device=device)
    omega_closed_loop = load_omeganet(
        checkpoint_dir / system.checkpoints.omega_closed_loop,
        n_dim=pi.n_dim,
        device=device,
    )
    return {"pi": pi, "omega": omega, "omega_closed_loop": omega_closed_loop}
