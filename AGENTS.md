# Repository Guidelines

## Project Structure & Module Organization

- `Inertiaevaluation.py`: primary experiment script (PPO training + final evaluation) and the `TrainingConfig` dataclass.
- `run_multi_seeds.py`: runs multiple seeds and aggregates results/plots under a shared experiment directory.
- `multiquick.py`: quick vs full multi-seed runner.
- `README.md`: brief run notes.
- Runtime outputs are written under `./logs/` (created as needed). Avoid committing generated `logs/` artifacts.

## Build, Test, and Development Commands

This folder is designed to be run inside the broader CARLProject Python environment (it imports `carl.*`, `gymnasium`, `stable_baselines3`, `torch`, `matplotlib`).

- `python Inertiaevaluation.py`: single run; writes a timestamped run folder under `./logs/mario_benchmark/` (model, JSON summaries, TensorBoard events).
- `python run_multi_seeds.py`: default multi-seed experiment; writes aggregated JSON + plots under `./logs/mario_inertia_multi/`.
- `python multiquick.py`: quick smoke run (toggle `QUICK_TEST` at the top); writes under `./logs/mario_inertia_quick/` or `./logs/mario_inertia_full/`.
- `tensorboard --logdir ./logs/mario_benchmark`: view TensorBoard output for single runs.

## Coding Style & Naming Conventions

- Python: 4-space indentation, type hints encouraged (new files should follow the existing `from __future__ import annotations` pattern).
- Naming: `snake_case` for functions/variables, `PascalCase` for classes.
- Put experiment knobs in `TrainingConfig` (or small helper functions) instead of scattering constants.

## Testing Guidelines

No formal test suite is present. Treat “tests” as smoke runs:

- For quick validation, reduce `TrainingConfig.total_timesteps` (or use `multiquick.py` with `QUICK_TEST=True`) and confirm a run directory is created with `final_results.json`/`training_curve.json`.

## Commit & Pull Request Guidelines

- Commit history uses short, descriptive subjects (e.g., “Added …”, “Update …”). Prefer `Add/Update/Fix <what>` and keep the first line ≤72 chars.
- PRs should include: intent (what hypothesis/experiment), config changes (which `TrainingConfig` fields), and where to find outputs (e.g., `./logs/.../aggregate_results.json`). Avoid committing large `logs/` trees; consider adding `logs/` to `.gitignore` for local work.
