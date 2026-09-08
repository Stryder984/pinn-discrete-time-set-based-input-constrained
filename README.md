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

This is a cleaned source snapshot. It intentionally excludes historical experiment folders and generated run outputs.
