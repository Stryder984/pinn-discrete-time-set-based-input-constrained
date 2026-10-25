# ROA Training and Verification Pipeline

This repository contains the training and verification pipeline used for neural value-function and policy experiments for discrete-time controlled dynamical systems.

The core workflow is:

```powershell
python run_pipeline.py --system van_der_pol --stage all
python verification.py --system van_der_pol --run-name <run_name>
```

The pipeline stages are:

1. `setup_lqr_box`: compute the local LQR controller, Lyapunov matrix, and local certified sublevel.
2. `train_omega`: train the original value function using finite-horizon labels and Bellman/Zubov-style collocation losses.
3. `train_policy`: train the saturated neural policy branch from grid-search labels.
4. `train_closed_loop_omega`: train a closed-loop value function under the learned controller.
5. `search_thresholds`: estimate candidate verification thresholds.
6. `plot_threshold_diagnostics`: generate diagnostic plots for the selected thresholds.
7. `verification.py`: run alpha-beta-CROWN against the exported verification model.

The historical `export` stage is currently reserved for future TorchScript/ONNX packaging work and is not part of the reviewed workflow.

Generated runs are written under `pipeline_runs/`. They are intentionally ignored by Git because they can contain large checkpoints, plots, CSVs, and search artifacts.

## Important Files

- `systems.py`: system definitions and training/verification configuration.
- `models.py`: neural value-function and policy model definitions.
- `training_pipeline.py`: staged training, threshold search, and plotting logic.
- `verification.py`: alpha-beta-CROWN verification wrapper.
- `run_pipeline.py`: command-line entry point for the standard pipeline.
- `run_variant.py`: command-line entry point for one-off configuration overrides.
- `sample_initial_sets.py`: trajectory sampling and convergence diagnostics.

## Dependencies

The training scripts require PyTorch, NumPy, Matplotlib, and SciPy. Verification additionally requires the alpha-beta-CROWN / auto_LiRPA stack to be installed or available on `PYTHONPATH`.

See `requirements.txt` for the lightweight Python package list. The verifier backend itself is not vendored in this repository.

## Notes

### Training Times and Final Losses

New runs automatically write `timings/<UTC timestamp>_<unique ID>.json`.
Each top-level pipeline invocation (including a direct stage-method call) gets
a separate journal, so reruns never overwrite earlier timings. No historical
timings are inferred from file timestamps.

The journal records hardware/software metadata, start/end UTC timestamps,
wall-clock seconds, stage status, and nested stage IDs. Status is `complete`,
`failed`, `interrupted`, or `skipped`; a forcibly terminated process can leave
`running`, which must not be treated as completed. Journals are atomically
updated at coarse phase boundaries, not every epoch. Failed writes emit a
warning and do not intentionally abort training.

Each training stage separates `data_preparation` (sampling, labels, model and
batch setup), `optimization` (the existing epoch loop, including its logging
and PiNet's periodic evaluations), and checkpoint/report/plot work. Original
omega additionally separates `domain_expansion`. Target plotting and combined
loss-history writing/plotting are recorded separately from optimization.
The threshold stage separates `threshold_search`, optional
`quadratic_control_search`, and result serialization. Plot-stage time includes
the diagnostic computations needed to construct those figures.

CUDA is synchronized at timing boundaries, never per optimizer step. Times
include CPU work, I/O and any competing machine load; they are not GPU-kernel
times. The outer `run` event includes preflight and all dispatched stages, but
not interpreter startup, imports or construction of the pipeline object.
Parent times include nested stages: do not sum parent and child durations.
Within each event, `phase_seconds` partitions its measured duration.

Training events also copy `final_training_loss` from the last existing epoch
record, with the total, components, epoch count and history filename. This is
the batch-averaged weighted training loss, not an additional evaluation of the
final checkpoint. Empty histories yield null loss; nonfinite values are null
with their field names listed. Existing `*_loss_history.csv/json` files remain
unchanged. Loss totals across different weights are not directly comparable.

The instrumentation changes no dynamics, samples, losses, optimizer steps,
controller behavior or thresholds. No CPU fallback is introduced.

### Optional Quadratic Control Diagnostic

Set `quadratic_control_search_enabled=True` in the system's `TrainingConfig`
to include this diagnostic in `search_thresholds`. It is disabled by default.
No training loss, controller, sampling distribution, c2/w1/w2, or verifier
input changes when it is enabled.

For `nu(x)=x^T P x`, the diagnostic exhaustively evaluates a tensor-product
state grid and, at each nonzero state with `nu >= c1`, a tensor-product control grid. It asks
whether a control has a successor inside the searched box and
`nu(f(x,u))-nu(x) < -quadratic_control_search_decrease_eps`. The supplied analytic
`c1` region covers the excluded core; there is no c2 exclusion. The default
margin is zero (strict decrease). Standalone helper calls default to `c1=0`
and exempt only the exact origin. A positive fixed margin can artificially
limit the result near the origin and should be chosen with care.

The reported `C_quad_opt` is `scale * min(first_violation_nu, box_cap)`.
If no violation is found, the box cap alone is used. The box is the intersection
of the training and verification domains. Its exact quadratic containment cap
is `min_i min(-lower_i, upper_i)^2 / (P^-1)[i,i]`, requiring the origin inside
the box. This cap prevents claiming an ellipsoid that extends into unsearched
states. It is not the LQR input cap: each point may use a different admissible
control.

Configuration knobs and defaults:

| Field | Default | Meaning |
|---|---:|---|
| `quadratic_control_search_enabled` | `False` | Enable search and plot overlay |
| `quadratic_control_search_samples` | `None` | State-grid target; inherit `threshold_search_samples` |
| `quadratic_control_search_points_per_dim` | `1001` | Control points per input dimension, including endpoints |
| `quadratic_control_search_state_batch_size` | `1024` | States per batch |
| `quadratic_control_search_control_batch_size` | `128` | Controls per batch |
| `quadratic_control_search_scale` | `0.99` | Shrink the limiting quadratic level by 1% |
| `quadratic_control_search_decrease_eps` | `0.0` | Strict-decrease margin |

Search uses float64 CUDA without CPU fallback. Both grid products are streamed;
memory is bounded by the batch product, but runtime still scales as the total
number of states times `points_per_dim ** u_dim`. An odd control-grid count
includes zero for symmetric input bounds. State-grid rounding and actual
control counts are recorded, not inferred from the requested sample total.

Results are saved to `quadratic_control_search.json` and nested under
`quadratic_control_diagnostic` in `threshold_search.json`. They include the
matrix, domains, resolutions, counts, first violation, its best feasible control
and successor, raw level, containment cap and scaled level. When all controls
exit, control/successor fields are null, never neutral placeholders.
`verification_thresholds_rescaled.json` is unchanged by this diagnostic.

The main threshold, value/decrease, controller/search, difference-field, and
policy-label-gap plots show a yellow dash-dot `nu=C_quad_opt (empirical)` contour
only when enabled and a compatible saved result exists. Disabling it hides
even a previously saved result. Existing images require regeneration.

This is the largest sublevel suggested by the sampled states and controls,
not a formally verified or globally optimal ROA. Spatial grids can miss
violations and finite control grids can miss decreasing controls. Shrinkage
alone does not resolve either limitation.

This is a cleaned source snapshot. It intentionally excludes historical experiment folders and generated run outputs.
