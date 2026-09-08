from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


TensorDynamics = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CheckpointNames:
    pi: str
    omega: str
    omega_closed_loop: str
    pi_script: Optional[str] = None
    omega_script: Optional[str] = None
    omega_closed_loop_script: Optional[str] = None
    pi_onnx: Optional[str] = None
    omega_onnx: Optional[str] = None
    omega_closed_loop_onnx: Optional[str] = None


@dataclass(frozen=True)
class DataDistributionCounts:
    omega_trajectory_uniform_points: int = 10_000
    omega_trajectory_ellipse_points: int = 200
    omega_collocation_uniform_points: int = 50_000
    omega_collocation_ellipse_points: int = 5_000

    policy_ellipse_points: int = 25_000
    policy_uniform_points: int = 100_000
    policy_gd_omega_sublevel_points: int = 25_000
    closed_loop_trajectory_ellipse_points: int = 20_000
    closed_loop_trajectory_uniform_points: int = 10_000
    closed_loop_trajectory_gd_omega_sublevel_points: int = 20_000
    closed_loop_constraint_ellipse_points: int = 20_000
    closed_loop_constraint_uniform_points: int = 10_000
    closed_loop_constraint_gd_omega_sublevel_points: int = 20_000


@dataclass(frozen=True)
class DataDistributionProfile:
    """Ratio-based sampling profile resolved into explicit point counts.

    The base count is sub-exponential in the state dimension:

        global_scale * base_points_per_dim**base_reference_dim
        * base_dimension_growth**max(0, n_dim - base_reference_dim)

    Dataset totals are ratios of that base, then source fractions split each
    total across samplers. `base_exponent` is kept only as a legacy exact
    exponent override for old experiments.
    """

    base_points_per_dim: int = 1000
    base_exponent: Optional[int] = None
    base_reference_dim: int = 2
    base_dimension_growth: float = 2.0
    global_scale: float = 1.0

    # Supervised omega_orig labels require expensive finite-horizon target
    # generation, so scale them down with dimension separately from the other
    # fixed datasets. The effective label total is
    #   base * omega_label_total_ratio / omega_label_dimension_scale_base**n
    # where n is the state dimension.
    omega_label_total_ratio: float = 1e-3
    omega_label_dimension_scale_base: float = 1.0
    omega_label_ellipse_fraction: float = 0.2
    omega_label_uniform_fraction: float = 0.8

    omega_collocation_total_ratio: float = 1.0
    omega_collocation_ellipse_fraction: float = 0.2
    omega_collocation_uniform_fraction: float = 0.8

    # PiNet labels require control-grid minimization, so keep the policy total
    # ratio independent but apply its own dimension discount:
    #   base * policy_total_ratio / policy_dimension_scale_base**n
    policy_total_ratio: float = 1.0
    policy_dimension_scale_base: float = 1.0
    policy_ellipse_fraction: float = 0.20
    policy_uniform_fraction: float = 0.20
    policy_gd_fraction: float = 0.6

    # Closed-loop trajectory labels also require finite-horizon rollout target
    # construction, so downscale them independently from constraint points.
    closed_loop_trajectory_total_ratio: float = 1e-2
    closed_loop_trajectory_dimension_scale_base: float = 1.0
    closed_loop_trajectory_ellipse_fraction: float = 0.2
    closed_loop_trajectory_uniform_fraction: float = 0.2
    closed_loop_trajectory_gd_fraction: float = 0.6

    closed_loop_constraint_total_ratio: float = 1.0
    closed_loop_constraint_ellipse_fraction: float = 0.2
    closed_loop_constraint_uniform_fraction: float = 0.2
    closed_loop_constraint_gd_fraction: float = 0.6


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 103289
    q_scale: float = 1e-2
    r_scale: float = 1e-2
    c1_scale: float = 1.2
    lqr_decrease_grid_points_per_dim: int = 101
    lqr_decrease_grid_batch_size: int = 50_000
    zubov_alpha_scale: float = 1.0
    omega_hidden_sizes: Tuple[int, ...] = (32, 64,64, 32)
    pi_hidden_sizes: Tuple[int, ...] = (32,32,32,32)
    pi_activation: str = "tanh"
    omega_rollout_horizon: int = 50
    omega_rollouts_per_state: int = 30_000
    data_profile: DataDistributionProfile = field(default_factory=DataDistributionProfile)
    omega_boundary_noise_std: float = 1.2
    omega_label_batch_size: Optional[int] = None
    omega_batch_size: int = 1_000
    omega_full_pass_epochs: bool = True
    omega_epochs: int = 3000
    omega_lr: float = 5e-3
    omega_lambda_data: float = 0.01
    omega_lambda_bellman: float = 1.0
    omega_lambda_boundary: float = 0.01
    control_grid_points: int = 500
    control_grid_batch_size: int = 2000
    domain_expansion_strategy: str = "reachability_stepwise_min"
    reachability_expansion_max_iters: int = 100
    reachability_expansion_factor: float = 1.25
    reachability_expansion_max_radius_scale: float = 100.0
    policy_label_dos_sublevel: float = 0.98
    optimized_sampler_steps: int = 0
    optimized_sampler_lr: float = 5e-2
    optimized_sampler_max_attempts: int = 100
    optimized_sampler_strict_sublevel: bool = False
    closed_loop_optimized_sampler_strict_sublevel: bool = False
    policy_label_cache_path: Optional[str] = None
    pi_epochs: int = 3000
    pi_lr: float = 1e-2
    pi_lr_min: float = 1e-5
    pi_batch_size: int = 1000
    pi_train_max_points: Optional[int] = None
    pi_full_pass_epochs: bool = True
    pi_lambda_control: float = 1e-4
    pi_lambda_decrease: float = 0.0
    sharpness: float = 100.0
    grad_clip_norm: float = 5.0
    pi_log_every: int = 200
    pi_eval_points: int = 2_000
    pi_plot_points: int = 5_000
    policy_decrease_report_points: int = 200_000
    policy_decrease_report_batch_size: int = 50_000
    policy_label_selection_mode: str = "argmin"
    policy_label_use_confidence_weight: bool = False
    policy_label_mask_out_of_domain_successors: bool = False
    policy_label_near_optimal_rel_tol: float = 1e-2
    policy_label_lqr_near_optimal_rel_tol: float = 1e-2
    policy_label_lqr_min_clip_fraction: float = 1e-3
    diagnostic_plot_grid_points: int = 151
    policy_label_gap_near_optimal_rel_tol: float = 1e-2
    # Diagnostic heatmaps are 2D coordinate slices through the origin. Each pair
    # chooses the two varying state axes; all other state coordinates are fixed
    # at zero. Axis indices are zero-based and refer to `state_labels` when set.
    plot_pairs: Tuple[Tuple[int, int], ...] = ((0, 1),)
    closed_loop_hidden_sizes: Tuple[int, ...] = (32, 64,64, 32)
    closed_loop_initialize_from_original_omega: bool = True
    closed_loop_dos_sublevel: float = 0.98
    closed_loop_horizon: int = 50
    closed_loop_epochs: int = 1000
    closed_loop_label_batch_size: int = 50_000
    closed_loop_train_batch_size: int = 10_000
    closed_loop_full_pass_epochs: bool = True
    closed_loop_lr: float = 1e-3
    closed_loop_lambda_zubov: float = 0.0
    closed_loop_data_target: str = "trajectory"
    closed_loop_anchor_filter: str = "none"
    closed_loop_lambda_data: float = 0.01
    closed_loop_lambda_decrease: float = 10.0
    closed_loop_lambda_range: float = 0.01
    closed_loop_lambda_boundary: float = 0.0
    closed_loop_loss_dos_gate: float = 0.98
    # Held-out closed-loop decrease diagnostics after omega_cl training.
    closed_loop_validation_points: int = 200_000
    closed_loop_validation_batch_size: int = 50_000
    # Closed-loop omega is trained against the saturated NN branch. Verification
    # separately checks the clamped LQR branch inside nu <= c2.
    closed_loop_controller_mode: str = "nn"
    closed_loop_log_every: int = 500
    kappa: float = 1e-4
    boundary_diagnostic_points_per_face: int = 64
    boundary_diagnostic_horizon: int = 100
    boundary_diagnostic_rollouts_per_state: int = 20_000
    boundary_diagnostic_batch_size: int = 32
    boundary_diagnostic_w_threshold: float = 0.99
    threshold_search_samples: int = 10_000_000
    threshold_search_batch_size: int = 20_000
    # Legacy fallback used when a per-threshold scale below is left unset.
    threshold_search_scale: float = 0.98
    threshold_search_c2_scale: Optional[float] = 0.90
    threshold_search_w1_scale: Optional[float] = 0.90
    threshold_search_w2_scale: Optional[float] = None


@dataclass(frozen=True)
class VerificationConfig:
    domain: Tuple[Tuple[float, float], ...]
    eps: float = 1e-8
    c1: Optional[float] = None
    c2: Optional[float] = None
    w1: Optional[float] = None
    w2: Optional[float] = None


@dataclass(frozen=True)
class LocalDecreaseConfig:
    """Required inputs for the analytic LQR switching threshold.

    `box_radii` defines the origin-centered rectangle B=[-R_B,R_B].
    `eta_bound` is the supplied vector eta_B satisfying the local nonlinear
    remainder bound used by the Serry-Liu c1 estimate.
    """

    box_radii: Tuple[float, ...]
    eta_bound: Tuple[float, ...]
    epsilon: float = 1e-8
    source: str = "system supplied"


@dataclass(frozen=True)
class SystemSpec:
    name: str
    n_dim: int
    u_dim: int
    x_lb: Tuple[float, ...]
    x_hi: Tuple[float, ...]
    u_lb: Tuple[float, ...]
    u_hi: Tuple[float, ...]
    dt: float
    dynamics: TensorDynamics
    local_decrease: LocalDecreaseConfig
    checkpoints: CheckpointNames
    training: TrainingConfig = field(default_factory=TrainingConfig)
    verification: Optional[VerificationConfig] = None
    state_labels: Optional[Tuple[str, ...]] = None

    def validate(self) -> None:
        if len(self.x_lb) != self.n_dim or len(self.x_hi) != self.n_dim:
            raise ValueError(f"{self.name}: state bounds must have length n_dim")
        if len(self.u_lb) != self.u_dim or len(self.u_hi) != self.u_dim:
            raise ValueError(f"{self.name}: control bounds must have length u_dim")
        if self.local_decrease is None:
            raise ValueError(
                f"{self.name}: local_decrease is required. Provide box_radii, "
                "eta_bound, and epsilon in systems.py."
            )
        if len(self.local_decrease.box_radii) != self.n_dim:
            raise ValueError(f"{self.name}: local_decrease.box_radii must have length n_dim")
        if len(self.local_decrease.eta_bound) != self.n_dim:
            raise ValueError(f"{self.name}: local_decrease.eta_bound must have length n_dim")
        if any(lo >= hi for lo, hi in zip(self.x_lb, self.x_hi)):
            raise ValueError(f"{self.name}: each x lower bound must be below upper bound")
        if any(lo >= hi for lo, hi in zip(self.u_lb, self.u_hi)):
            raise ValueError(f"{self.name}: each u lower bound must be below upper bound")
        if any(r <= 0.0 for r in self.local_decrease.box_radii):
            raise ValueError(f"{self.name}: local_decrease.box_radii must be positive")
        if not all(math.isfinite(r) for r in self.local_decrease.box_radii):
            raise ValueError(f"{self.name}: local_decrease.box_radii must be finite")
        if any(v < 0.0 for v in self.local_decrease.eta_bound):
            raise ValueError(f"{self.name}: local_decrease.eta_bound must be nonnegative")
        if not all(math.isfinite(v) for v in self.local_decrease.eta_bound):
            raise ValueError(
                f"{self.name}: local_decrease.eta_bound must be finite. "
                "Replace the placeholder eta_bound in systems.py before running this system."
            )
        if self.local_decrease.epsilon <= 0.0:
            raise ValueError(f"{self.name}: local_decrease.epsilon must be positive")
        if not math.isfinite(self.local_decrease.epsilon):
            raise ValueError(f"{self.name}: local_decrease.epsilon must be finite")
        if not (0.0 <= self.training.c1_scale <= 1.5):
            raise ValueError(f"{self.name}: training.c1_scale must lie in [0, 1.5]")
def f_sys_2d_bench(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Two-stage pendulum_bigtorque dynamics, forward-Euler discretized."""
    max_torque = 6.0
    dt = 0.001
    gravity = 9.81
    mass = 0.15
    length = 0.5

    force = torch.clamp(action, min=-max_torque, max=max_torque)
    theta = state[..., 0:1]
    theta_dot = state[..., 1:2]
    thetaacc = (mass * gravity * length * torch.sin(theta) + force - 0.1 * theta_dot) / (
        mass * length * length
    )
    theta_next = theta + dt * theta_dot
    theta_dot_next = theta_dot + dt * thetaacc
    return torch.cat([theta_next, theta_dot_next], dim=-1)


def f_sys_3d_torque_rate_pendulum(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """pendulum with torque as a state and torque-rate input.
    taken from safe&robust DOA
    """
    max_torque = 5.0
    max_torque_rate = 5.0
    dt = 0.05

    theta = state[..., 0:1]
    theta_dot = state[..., 1:2]
    torque = torch.clamp(state[..., 2:3], min=-max_torque, max=max_torque)
    torque_rate = torch.clamp(action[..., 0:1], min=-max_torque_rate, max=max_torque_rate)

    thetaacc = (-4.905* torch.sin(theta) + torque) 
    
    theta_next = theta + dt * theta_dot
    theta_dot_next = theta_dot + dt * 3*thetaacc
    torque_next = state[..., 2:3] + dt * torque_rate
    return torch.cat([theta_next, theta_dot_next, torque_next], dim=-1)


def f_sys_cartpole(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """4D cart-pole dynamics from two_stage_dynamics.py, forward-Euler discretized.

    State layout is (cart_position, pole_angle, cart_velocity, pole_angular_velocity).
    The upright equilibrium is the origin and the zero action is the equilibrium
    force, matching the two-stage cartpole example.
    """
    dt = 0.001
    cart_mass = 1.0
    pole_mass = 0.1
    length = 1.0
    gravity = 9.81

    theta = state[..., 1:2]
    cart_velocity = state[..., 2:3]
    theta_dot = state[..., 3:4]
    force = action[..., 0:1]

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    denominator = pole_mass * sin_theta * sin_theta + cart_mass

    cart_acc = (
        force + pole_mass * sin_theta * (length * theta_dot * theta_dot - gravity * cos_theta)
    ) / denominator
    theta_acc = (
        -force * cos_theta
        - pole_mass * length * theta_dot * theta_dot * cos_theta * sin_theta
        + (cart_mass + pole_mass) * gravity * sin_theta
    ) / (length * denominator)

    return torch.cat(
        [
            state[..., 0:1] + dt * cart_velocity,
            theta + dt * theta_dot,
            cart_velocity + dt * cart_acc,
            theta_dot + dt * theta_acc,
        ],
        dim=-1,
    )


def f_sys_van_der_pol(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Controlled Van der Pol oscillator, discretized by forward Euler.

    Continuous-time model assumed here:

        x1_dot = x2
        x2_dot = mu * (1 - x1^2) * x2 - x1 + u

    with mu=1.0 and dt=0.01, matching two_stage_training_config.py. The
    control enters affinely, matching the interval reachable-set assumption used
    by the training pipeline.
    """
    dt = 0.01
    mu = 1.0

    x1 = state[..., 0:1]
    x2 = state[..., 1:2]
    u = action[..., 0:1]

    x1_next = x1 + dt * x2
    x2_next = x2 + dt * (mu * (1.0 - x1 * x1) * x2 - x1 + u)
    return torch.cat([x1_next, x2_next], dim=-1)


def f_sys_path_tracking(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Path-tracking dynamics from two_stage_dynamics.py.

    The source model uses physical steering input with equilibrium L/R. The
    streamlined pipeline assumes the origin has zero input, so `action` is the
    steering deviation and L/R is added internally.
    """
    dt = 0.01
    speed = 2.0
    length = 1.0
    radius = 10.0

    lateral_error = state[..., 0:1]
    theta_error = state[..., 1:2]
    steering = action[..., 0:1] + length / radius

    sin_theta = torch.sin(theta_error)
    cos_theta = torch.cos(theta_error)
    coef = radius / speed

    lateral_error_dot = speed * sin_theta
    theta_error_dot = speed * steering / length - cos_theta / (coef - sin_theta)

    lateral_error_next = lateral_error + dt * lateral_error_dot
    theta_error_next = theta_error + dt * theta_error_dot
    return torch.cat([lateral_error_next, theta_error_next], dim=-1)


def f_sys_ducted_fan(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Ducted-fan dynamics from two_stage_dynamics.py.

    `action` is a thrust-deviation coordinate. The physical fan input is
    [action_0, action_1 + m*g], matching the source equilibrium [0, m*g] while
    keeping zero action as the streamlined equilibrium.
    """
    dt = 0.0005
    gravity = 0.28
    mass = 11.2
    inertia = 0.0462
    radius = 0.156
    damping = 0.1

    q_dot = state[..., 3:6]
    theta = state[..., 2:3]
    u0 = action[..., 0:1]
    u1 = action[..., 1:2] + mass * gravity

    qddot_x = (-damping * q_dot[..., 0:1] + u0 * torch.cos(theta) - u1 * torch.sin(theta)) / mass
    qddot_y = (-damping * q_dot[..., 1:2] + u0 * torch.sin(theta) + u1 * torch.cos(theta)) / mass - gravity
    qddot_theta = radius * u0 / inertia
    state_dot = torch.cat([q_dot, qddot_x, qddot_y, qddot_theta], dim=-1)
    return state + dt * state_dot


def f_sys_pvtol_6d(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """6D planar VTOL dynamics from two_stage_dynamics.py/config.

    State layout is (x, y, pitch, body_vx, body_vy, pitch_rate). Controls are
    thrust deviations from hover, so the origin with zero action is an
    equilibrium.
    """
    tau = 0.0005
    gravity = 9.8
    mass = 4.0
    arm_length = 0.25
    inertia = 0.0475

    x1 = state[..., 0:1]
    x2 = state[..., 1:2]
    x3 = state[..., 2:3]
    x4 = state[..., 3:4]
    x5 = state[..., 4:5]
    x6 = state[..., 5:6]

    u1 = action[..., 0:1]
    u2 = action[..., 1:2]
    hover_thrust = mass * gravity / 2.0
    u1_phys = u1 + hover_thrust
    u2_phys = u2 + hover_thrust

    x1_next = x1 + tau * (x4 * torch.cos(x3) - x5 * torch.sin(x3))
    x2_next = x2 + tau * (x4 * torch.sin(x3) + x5 * torch.cos(x3))
    x3_next = x3 + tau * x6
    x4_next = x4 + tau * (x5 * x6 - gravity * torch.sin(x3))
    x5_next = x5 + tau * (-x4 * x6 - gravity * torch.cos(x3) + (u1_phys + u2_phys) / mass)
    x6_next = x6 + tau * arm_length * (u1_phys - u2_phys) / inertia
    return torch.cat([x1_next, x2_next, x3_next, x4_next, x5_next, x6_next], dim=-1)

# System registry
# Per-system overrides are intentionally narrow:
# - LQR weights: q_scale, r_scale
# - neural-network widths: omega/pi/closed-loop hidden sizes
# - loss weights: omega/pi/closed-loop lambda terms
# Sampling counts are derived from TrainingConfig.data_profile.
# Verification thresholds c1/c2/w1/w2 are produced by search_thresholds.
SYSTEMS: Dict[str, SystemSpec] = {
    '2d_bench': SystemSpec(
        name='2d_bench',
        n_dim=2,
        u_dim=1,
        x_lb=(-6.0, -6.0),
        x_hi=(6.0, 6.0),
        u_lb=(-6.0,),
        u_hi=(6.0,),
        dt=0.001,
        dynamics=f_sys_2d_bench,
        local_decrease=LocalDecreaseConfig(box_radii=(6.0, 6.0), eta_bound=(0.0, 0.01962), epsilon=1e-08, source='analytic sine remainder bound for dt=0.001 pendulum over |theta|<=6'),
        checkpoints=CheckpointNames(pi='2Dbench_pi.pt', omega='2Dbenchomega.pt', omega_closed_loop='2Dbench_omegaCL.pt', pi_script='2Dbench_pi_script.pt', omega_script='2DBenchOMegascript.pt', omega_closed_loop_script='2DBenchCLomegascript.pt', pi_onnx='2Dbench_pi.onnx', omega_onnx='2DBenchOmega.onnx', omega_closed_loop_onnx='2DBenchOMega_CL.onnx'),
        training=TrainingConfig(
                    q_scale=3e-4,
                    r_scale=1e-3,
                    omega_hidden_sizes=(32, 32, 32, 32, 32),
                    pi_hidden_sizes=(32, 32, 32, 32),
                    omega_lambda_data=0.1,
                    omega_lambda_bellman=1.0,
                    omega_lambda_boundary=0.01,
                    omega_batch_size=50_000,
                    omega_full_pass_epochs=True,
                    omega_epochs=50,
                    pi_lambda_control=1e-4,
                    pi_lambda_decrease=0.0,
                    pi_batch_size=50_000,
                    pi_full_pass_epochs=True,
                    pi_epochs=30,
                    closed_loop_hidden_sizes=(32, 32, 32, 32, 32),
                    closed_loop_train_batch_size=50_000,
                    closed_loop_full_pass_epochs=True,
                    closed_loop_epochs=50,
                    closed_loop_lambda_zubov=0.0,
                    closed_loop_data_target="omega_orig",
                    closed_loop_anchor_filter="omega_orig_nonviolating",
                    closed_loop_lambda_data=0.01,
                    closed_loop_lambda_decrease=10.0,
                    closed_loop_lambda_range=0.0,
                    closed_loop_lambda_boundary=0.0,
                ),
        verification=VerificationConfig(
                    domain=((-12.0, 12.0), (-12.0, 12.0)),
                    eps=1e-08,
                ),
        state_labels=('theta', 'theta_dot'),
    ),
    '3d_torque_rate_pendulum': SystemSpec(
        name='3d_torque_rate_pendulum',
        n_dim=3,
        u_dim=1,
        x_lb=(-math.pi / 4, -1.0, -1.0),
        x_hi=(math.pi / 4, 1.0, 1.0),
        u_lb=(-5.0,),
        u_hi=(5.0,),
        dt=0.05,
        dynamics=f_sys_3d_torque_rate_pendulum,
        local_decrease=LocalDecreaseConfig(box_radii=(math.pi / 4, 1.0, 1.0), eta_bound=(0.0, 0.981, 0.0), epsilon=1e-08, source='analytic sine remainder bound over |theta|<=12 plus affine torque state'),
        checkpoints=CheckpointNames(pi='3Dtorquerate_pi.pt', omega='3Dtorquerate_omega.pt', omega_closed_loop='3Dtorquerate_omegaCL.pt', pi_script='3Dtorquerate_pi_script.pt', omega_script='3Dtorquerate_omega_script.pt', omega_closed_loop_script='3Dtorquerate_omegaCL_script.pt', pi_onnx='3Dtorquerate_pi.onnx', omega_onnx='3Dtorquerate_omega.onnx', omega_closed_loop_onnx='3Dtorquerate_omegaCL.onnx'),
        training=TrainingConfig(
                    q_scale=1e-3,
                    r_scale=1e-3,
                    c1_scale=1.2,
                    data_profile=DataDistributionProfile(
                                            base_points_per_dim=1000,
                                            base_exponent=None,
                                            base_reference_dim=2,
                                            base_dimension_growth=2.0,
                                            global_scale=1.0,
                                            omega_label_total_ratio=1e-2,
                                            omega_label_dimension_scale_base=1.0,
                                            omega_label_ellipse_fraction=0.2,
                                            omega_label_uniform_fraction=0.8,
                                            omega_collocation_total_ratio=1.0,
                                            omega_collocation_ellipse_fraction=0.2,
                                            omega_collocation_uniform_fraction=0.8,
                                            policy_total_ratio=1.0,
                                            policy_dimension_scale_base=1.0,
                                            policy_ellipse_fraction=0.25,
                                            policy_uniform_fraction=0.0,
                                            policy_gd_fraction=0.75,
                                            closed_loop_trajectory_total_ratio=1e-2,
                                            closed_loop_trajectory_dimension_scale_base=1.0,
                                            closed_loop_trajectory_ellipse_fraction=0.20,
                                            closed_loop_trajectory_uniform_fraction=0.20,
                                            closed_loop_trajectory_gd_fraction=0.6,
                                            closed_loop_constraint_total_ratio=2.0,
                                            closed_loop_constraint_ellipse_fraction=0.20,
                                            closed_loop_constraint_uniform_fraction=0.0,
                                            closed_loop_constraint_gd_fraction=0.6,
                                        ),
                    omega_hidden_sizes=(64, 64,64,64),
                    pi_hidden_sizes=(64, 64,64,64),
                    omega_label_batch_size=1_000,
                    omega_batch_size=100_000,
                    omega_full_pass_epochs=True,
                    omega_epochs=40,
                    omega_lambda_data=0.01,
                    kappa=1e-4,
                    omega_lambda_bellman=1.0,
                    omega_lambda_boundary=0.01,
                    pi_lambda_control=1e-5,
                    pi_batch_size=100_000,
                    pi_lambda_decrease=0.0,
                    pi_full_pass_epochs=True,
                    pi_epochs=30,
                    closed_loop_hidden_sizes=(64, 64, 64,64,64),
                    closed_loop_train_batch_size=100_000,
                    closed_loop_full_pass_epochs=True,
                    closed_loop_epochs=40,
                    closed_loop_lambda_data=0.01,
                    closed_loop_lambda_zubov=0.0,
                    closed_loop_data_target="omega_orig",
                    closed_loop_anchor_filter="omega_orig_nonviolating",
                    closed_loop_lambda_decrease=30.0,
                    closed_loop_lambda_range=0.0,
                    closed_loop_lambda_boundary=0.0,
                    plot_pairs =((0,1),(1,2),(0,2)),
                    threshold_search_c2_scale=0.99,
                    threshold_search_w1_scale=0.99
                ),
        verification=VerificationConfig(
                    domain=((-math.pi / 4, math.pi / 4), (-1.0, 1.0), (-1.0, 1.0)),
                    eps=1e-08,
                ),
        state_labels=('theta', 'theta_dot', 'torque'),
    ),
    'cartpole': SystemSpec(
        name='cartpole',
        n_dim=4,
        u_dim=1,
        x_lb=(-1.0, -1.0, -1.0, -1.0),
        x_hi=(1.0, 1.0, 1.0, 1.0),
        u_lb=(-30.0,),
        u_hi=(30.0,),
        dt=0.001,
        dynamics=f_sys_cartpole,
        local_decrease=LocalDecreaseConfig(box_radii=(0.4, 0.4, 0.4, 0.4), eta_bound=(0.0, 0.0, 0.02, 0.08), epsilon=1e-07, source='conservative cart-pole Hessian envelope over the configured local box'),
        checkpoints=CheckpointNames(pi='cartpole_pi.pt', omega='cartpole_omega.pt', omega_closed_loop='cartpole_omegaCL.pt', pi_script='cartpole_pi_script.pt', omega_script='cartpole_omega_script.pt', omega_closed_loop_script='cartpole_omegaCL_script.pt', pi_onnx='cartpole_pi.onnx', omega_onnx='cartpole_omega.onnx', omega_closed_loop_onnx='cartpole_omegaCL.onnx'),
        training=TrainingConfig(
                    q_scale=3e-4,
                    r_scale=1e-3,
                    omega_hidden_sizes=(64, 64, 64, 64, 64, 64),
                    pi_hidden_sizes=(32, 32, 32, 32, 32),
                    omega_lambda_data=0.01,
                    omega_lambda_bellman=1.0,
                    omega_lambda_boundary=0.01,
                    pi_lambda_control=1e-6,
                    closed_loop_hidden_sizes=(64, 64, 64, 64, 64, 64),
                    closed_loop_lambda_data=0.01,
                    closed_loop_lambda_decrease=10.0,
                    closed_loop_lambda_range=0.01,
                ),
        verification=VerificationConfig(
                    domain=((-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4)),
                    eps=1e-08,
                ),
        state_labels=('cart_position', 'pole_angle', 'cart_velocity', 'pole_angular_velocity'),
    ),
    'van_der_pol': SystemSpec(
        name='van_der_pol',
        n_dim=2,
        u_dim=1,
        x_lb=(-5.0, -5.0),
        x_hi=(5.0, 5.0),
        u_lb=(-1.0,),
        u_hi=(1.0,),
        dt=0.01,
        dynamics=f_sys_van_der_pol,
        local_decrease=LocalDecreaseConfig(box_radii=(0.5, 0.5), eta_bound=(0.0, 0.02), epsilon=1e-07),
        checkpoints=CheckpointNames(pi='vanderpol_pi.pt', omega='vanderpol_omega.pt', omega_closed_loop='vanderpol_omegaCL.pt', pi_script='vanderpol_pi_script.pt', omega_script='vanderpol_omega_script.pt', omega_closed_loop_script='vanderpol_omegaCL_script.pt', pi_onnx='vanderpol_pi.onnx', omega_onnx='vanderpol_omega.onnx', omega_closed_loop_onnx='vanderpol_omegaCL.onnx'),
        training=TrainingConfig(
                    q_scale=1e-4,
                    r_scale=1e-3,
                    data_profile=DataDistributionProfile(
                        base_points_per_dim=1000,
                        base_exponent=None,
                        base_reference_dim=2,
                        base_dimension_growth=2.0,
                        global_scale=1.0,
                        omega_label_total_ratio=1e-3,
                        omega_label_dimension_scale_base=1.0,
                        omega_label_ellipse_fraction=0.2,
                        omega_label_uniform_fraction=0.8,
                        omega_collocation_total_ratio=1.0,
                        omega_collocation_ellipse_fraction=0.2,
                        omega_collocation_uniform_fraction=0.8,
                        policy_total_ratio=1.0,
                        policy_dimension_scale_base=1.0,
                        policy_ellipse_fraction=0.2,
                        policy_uniform_fraction=0.2,
                        policy_gd_fraction=0.6,
                        closed_loop_trajectory_total_ratio=1e-2,
                        closed_loop_trajectory_dimension_scale_base=1.0,
                        closed_loop_trajectory_ellipse_fraction=0.2,
                        closed_loop_trajectory_uniform_fraction=0.2,
                        closed_loop_trajectory_gd_fraction=0.6,
                        closed_loop_constraint_total_ratio=1.0,
                        closed_loop_constraint_ellipse_fraction=0.2,
                        closed_loop_constraint_uniform_fraction=0.2,
                        closed_loop_constraint_gd_fraction=0.6,
                    ),
                    omega_hidden_sizes=(20, 20, 20, 20, 20,20,20),
                    pi_hidden_sizes=(60, 60,60,60),
                    pi_lambda_control=1e-8,
                    omega_batch_size=50_000,
                    omega_rollout_horizon=75,
                    closed_loop_horizon=75,
                    omega_full_pass_epochs=True,
                    omega_epochs=50,
                    omega_lambda_data=0.01,
                    omega_lambda_bellman=1.0,
                    omega_lambda_boundary=0.01,
                    pi_batch_size=50_000,
                    pi_full_pass_epochs=True,
                    pi_epochs=30,
                    pi_lambda_decrease=0.0,
                    closed_loop_hidden_sizes=(20,20,20,20,20,20,20),
                    closed_loop_train_batch_size=50_000,
                    closed_loop_full_pass_epochs=True,
                    closed_loop_epochs=50,
                    closed_loop_lambda_zubov=0.0,
                    closed_loop_data_target="omega_orig",
                    closed_loop_anchor_filter="omega_orig_nonviolating",
                    closed_loop_lambda_data=0.01,
                    closed_loop_lambda_decrease=30.0,
                    closed_loop_lambda_range=0.0,
                    closed_loop_lambda_boundary=0.0,
                    threshold_search_w1_scale=0.95
                ),
        verification=VerificationConfig(
                    domain=((-0.5, 0.5), (-0.5, 0.5)),
                    eps=1e-08,
                ),
        state_labels=('x1', 'x2'),
    ),
    'path_tracking_bigtorque': SystemSpec(
        name='path_tracking_bigtorque',
        n_dim=2,
        u_dim=1,
        x_lb=(-2.0, -2.0),
        x_hi=(2.0, 2.0),
        u_lb=(-0.84,),
        u_hi=(0.84,),
        dt=0.01,
        dynamics=f_sys_path_tracking,
        local_decrease=LocalDecreaseConfig(box_radii=(2.0, 2.0), eta_bound=(0.02, 0.005), epsilon=1e-07, source='analytic path-tracking Hessian bound with c=R/v=5'),
        checkpoints=CheckpointNames(pi='path_tracking_bigtorque_pi.pt', omega='path_tracking_bigtorque_omega.pt', omega_closed_loop='path_tracking_bigtorque_omegaCL.pt', pi_script='path_tracking_bigtorque_pi_script.pt', omega_script='path_tracking_bigtorque_omega_script.pt', omega_closed_loop_script='path_tracking_bigtorque_omegaCL_script.pt', pi_onnx='path_tracking_bigtorque_pi.onnx', omega_onnx='path_tracking_bigtorque_omega.onnx', omega_closed_loop_onnx='path_tracking_bigtorque_omegaCL.onnx'),
        training=TrainingConfig(
                    q_scale=0.01,
                    r_scale=0.01,
                    omega_hidden_sizes=(40, 40),
                    pi_hidden_sizes=(10, 10),
                    closed_loop_hidden_sizes=(40, 40),
                ),
        verification=VerificationConfig(
                    domain=((-2.0, 2.0), (-2.0, 2.0)),
                    eps=1e-06,
                ),
        state_labels=('lateral_error', 'theta_error'),
    ),
    'path_tracking_smalltorque': SystemSpec(
        name='path_tracking_smalltorque',
        n_dim=2,
        u_dim=1,
        x_lb=(-2.0, -2.0),
        x_hi=(2.0, 2.0),
        u_lb=(-0.5,),
        u_hi=(0.5,),
        dt=0.01,
        dynamics=f_sys_path_tracking,
        local_decrease=LocalDecreaseConfig(box_radii=(2.0, 2.0), eta_bound=(0.02, 0.005), epsilon=1e-07, source='analytic path-tracking Hessian bound with c=R/v=5'),
        checkpoints=CheckpointNames(pi='path_tracking_smalltorque_pi.pt', omega='path_tracking_smalltorque_omega.pt', omega_closed_loop='path_tracking_smalltorque_omegaCL.pt', pi_script='path_tracking_smalltorque_pi_script.pt', omega_script='path_tracking_smalltorque_omega_script.pt', omega_closed_loop_script='path_tracking_smalltorque_omegaCL_script.pt', pi_onnx='path_tracking_smalltorque_pi.onnx', omega_onnx='path_tracking_smalltorque_omega.onnx', omega_closed_loop_onnx='path_tracking_smalltorque_omegaCL.onnx'),
        training=TrainingConfig(
                    q_scale=1e-4,
                    r_scale=1e-3,
                    omega_hidden_sizes=(32, 32,32,32, 32, 32),
                    pi_hidden_sizes=(32,32, 32,32, 32),
                    closed_loop_hidden_sizes=(32, 32,32,32, 32, 32),
                    closed_loop_lambda_data=0.01,
                    closed_loop_lambda_decrease=1.0,
                    closed_loop_lambda_range=0.01,
                ),
        verification=VerificationConfig(
                    domain=((-2.0, 2.0), (-2.0, 2.0)),
                    eps=1e-06,
                ),
        state_labels=('lateral_error', 'theta_error'),
    ),
    'ducted_fan': SystemSpec(
        name='ducted_fan',
        n_dim=6,
        u_dim=2,
        x_lb=(-0.4, -0.4, -0.4, -0.4, -0.4, -0.4),
        x_hi=(0.4, 0.4, 0.4, 0.4, 0.4, 0.4),
        u_lb=(-10.0, -10.0),
        u_hi=(10.0, 10.0),
        dt=0.0005,
        dynamics=f_sys_ducted_fan,
        local_decrease=LocalDecreaseConfig(box_radii=(0.4, 0.4, 0.4, 0.4, 0.4, 0.4), eta_bound=(0.0, 0.0, 0.0, 0.001033, 0.001033, 0.0), epsilon=1e-07, source='analytic ducted-fan open-loop state Hessian bound over configured input box'),
        checkpoints=CheckpointNames(pi='ducted_fan_pi.pt', omega='ducted_fan_omega.pt', omega_closed_loop='ducted_fan_omegaCL.pt', pi_script='ducted_fan_pi_script.pt', omega_script='ducted_fan_omega_script.pt', omega_closed_loop_script='ducted_fan_omegaCL_script.pt', pi_onnx='ducted_fan_pi.onnx', omega_onnx='ducted_fan_omega.onnx', omega_closed_loop_onnx='ducted_fan_omegaCL.onnx'),
        training=TrainingConfig(
                    q_scale=1e-3,
                    r_scale=1e-3,
                    omega_hidden_sizes=(120, 120, 120),
                    pi_hidden_sizes=(64, 64, 64),
                    closed_loop_hidden_sizes=(120, 120, 120),
                ),
        verification=VerificationConfig(
                    domain=((-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4)),
                    eps=1e-08,
                ),
        state_labels=('x', 'y', 'theta', 'x_dot', 'y_dot', 'theta_dot'),
    ),
    'pvtol_6d': SystemSpec(
        name='pvtol_6d',
        n_dim=6,
        u_dim=2,
        x_lb=(-4.0, -4.0, -math.pi , -4.0, -4.0, -4*1.5 * math.pi),
        x_hi=(4.0, 4.0, math.pi , 4.0, 4.0, 4*1.5 * math.pi),
        u_lb=(-4.0 * 9.8 / 2.0, -4.0 * 9.8 / 2.0),
        u_hi=(4.0 * 9.8 / 2.0, 4.0 * 9.8 / 2.0),
        dt=0.0005,
        dynamics=f_sys_pvtol_6d,
        local_decrease=LocalDecreaseConfig(box_radii=(1.0, 1.0, math.pi / 4.0, 1.5, 2.0, 1.5 * math.pi), eta_bound=(0.0101, 0.0101, 0.0, 0.00495, 0.00495, 0.0), epsilon=1e-07, source='analytic PVTOL Hessian remainder bound over configured local box'),
        checkpoints=CheckpointNames(pi='pvtol_6d_pi.pt', omega='pvtol_6d_omega.pt', omega_closed_loop='pvtol_6d_omegaCL.pt', pi_script='pvtol_6d_pi_script.pt', omega_script='pvtol_6d_omega_script.pt', omega_closed_loop_script='pvtol_6d_omegaCL_script.pt', pi_onnx='pvtol_6d_pi.onnx', omega_onnx='pvtol_6d_omega.onnx', omega_closed_loop_onnx='pvtol_6d_omegaCL.onnx'),
        training=TrainingConfig(
                    q_scale=1e-4,
                    r_scale=1e-3,
                    c1_scale=0.5,
                    boundary_diagnostic_points_per_face=50,
                     data_profile=DataDistributionProfile(
                                            base_points_per_dim=1000,
                                            base_exponent=None,
                                            base_reference_dim=2,
                                            base_dimension_growth=2.0,
                                            global_scale=1.0,
                                            omega_label_total_ratio=1e-3,
                                            omega_label_dimension_scale_base=1.0,
                                            omega_label_ellipse_fraction=0.2,
                                            omega_label_uniform_fraction=0.8,
                                            omega_collocation_total_ratio=1.0,
                                            omega_collocation_ellipse_fraction=0.2,
                                            omega_collocation_uniform_fraction=0.8,
                                            policy_total_ratio=0.01,
                                            policy_dimension_scale_base=1.0,
                                            policy_ellipse_fraction=0.2,
                                            policy_uniform_fraction=0.1,
                                            policy_gd_fraction=0.70,
                                            closed_loop_trajectory_total_ratio=1e-2,
                                            closed_loop_trajectory_dimension_scale_base=1.0,
                                            closed_loop_trajectory_ellipse_fraction=0.05,
                                            closed_loop_trajectory_uniform_fraction=0.05,
                                            closed_loop_trajectory_gd_fraction=0.9,
                                            closed_loop_constraint_total_ratio=1.0,
                                            closed_loop_constraint_ellipse_fraction=0.05,
                                            closed_loop_constraint_uniform_fraction=0.05,
                                            closed_loop_constraint_gd_fraction=0.9,
                                        ),
                    omega_hidden_sizes=(64,64,64,64,64,64,64,64),
                    omega_full_pass_epochs=True,
                    omega_epochs=12,
                    pi_hidden_sizes=(256,256,256,256,256,256,256,256,256,256),
                    pi_activation="relu",
                    omega_lambda_data=0.01,
                    omega_boundary_noise_std=1.0,
                    pi_lambda_control=1e-4,
                    pi_lambda_decrease=0.0,
                    policy_label_cache_path=None,
                    policy_label_selection_mode="argmin",
                    policy_label_use_confidence_weight=False,
                    policy_label_mask_out_of_domain_successors=True,
                    policy_label_near_optimal_rel_tol=1e-2,
                    policy_label_lqr_near_optimal_rel_tol=1e-2,
                    policy_label_lqr_min_clip_fraction=1e-3,
                    pi_lr=3e-3,
                    optimized_sampler_max_attempts=200,
                    pi_batch_size=8000,
                    pi_train_max_points=None,
                    pi_full_pass_epochs=True,
                    pi_epochs=12,
                    pi_log_every=1,
                    policy_decrease_report_points=200_000,
                    closed_loop_hidden_sizes=(64,64,64,64,64,64,64,64),
                    closed_loop_train_batch_size=100_000,
                    closed_loop_full_pass_epochs=True,
                    closed_loop_epochs=12,
                    closed_loop_lr=3e-4,
                    closed_loop_lambda_zubov=0.0,
                    closed_loop_data_target="trajectory",
                    closed_loop_anchor_filter="none",
                    closed_loop_lambda_data=0.1,
                    closed_loop_lambda_range=0.0,
                    closed_loop_lambda_decrease=30.0,
                    closed_loop_lambda_boundary=0.0001,
                    control_grid_points = 20,
                    plot_pairs=((0,1),(0,2),(1,4),(2,5)),
                    closed_loop_dos_sublevel=0.95,
                    closed_loop_loss_dos_gate=0.95,
                    policy_label_dos_sublevel=0.95,
                    boundary_diagnostic_w_threshold=0.99,
                    threshold_search_samples=1_000_000
                ),
        verification=VerificationConfig(
                    domain=((-1.2, 1.2), (-1.2, 1.2), (-math.pi / 2.0, math.pi / 2.0), (-1.5, 1.5), (-1.5, 1.5), (-math.pi, math.pi)),
                    eps=1e-08,
                ),
        state_labels=('x', 'y', 'pitch', 'body_vx', 'body_vy', 'pitch_rate'),
    ),
}
def get_system(name: str) -> SystemSpec:
    try:
        system = SYSTEMS[name]
    except KeyError as exc:
        known = ", ".join(sorted(SYSTEMS))
        raise KeyError(f"Unknown system {name!r}. Known systems: {known}") from exc
    system.validate()
    return system


def tensor_bounds(values: Sequence[float], *, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype)
