# Draft

## Resuming multi-seed runs after a crash

If WSL shuts down mid multi-seed run, you can resume the latest run directory and only re-run missing seeds:

- Default full runner: `python run_multi_seeds.py --resume latest`
- Resume latest run (skips seeds that already have a `final_results.json`): `python multiquick.py --resume latest`
- Resume a specific run directory: `python multiquick.py --resume ./logs/mario_inertia_full/multi_MMDD_HHMMSS`
- Progress is tracked in `seed_progress.json` inside the run directory, and aggregates are updated after every seed.

## Context fusion (concat vs Hadamard)

`Inertiaevaluation.py` supports a cGate/Hadamard-style fusion of vision + context.

- Enable context: set `config.use_context = True` (by default the only context variable used is `mario_inertia`).
- Choose fusion:
  - `config.feature_fusion = "concat"` (SB3 `CombinedExtractor`, default)
  - `config.feature_fusion = "hadamard"` (element-wise Hadamard product via `HadamardGateExtractor`)

## Brief instructions for running experiment with visualization

Info: experiment with some extra for visualization.py goes together with visualize_mario_results.py

1. Run experiments first and then

2. python visualize_mario_results.py --run_dir logs/mario_benchmark/inertia_MMDD_HHMM

where replace MMDD_HHMM with your own folder name values

3. View visualizations from logs/mario_becnhmark/intertia../figs folder

## Brief instructions for running the game and testing intertia values as player

Goal: See how interia works in the game
Findings: Intertia only applied to movement not jumping

Instructions:

1. Run it
2. Press A to decrease (it also resets and applies new inertia)
3. Press B to increase
4. If B pressed then press A again, to see changes press at least 2 times B and 1 time A
5. See how it works
 
