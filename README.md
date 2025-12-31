# Draft

## Setup

Minimal WSL (Ubuntu) setup from a clean shell:

- `sudo apt update && sudo apt install -y git python3 python3-venv python3-pip xvfb`
- `mkdir -p ~/CARLProject && cd ~/CARLProject`
- `git clone --recursive https://github.com/automl/CARL.git`
- `git clone https://github.com/DemoMan444/CARLExperiments.git`
- `cd ~/CARLProject/CARLExperiments`
- `python3 -m venv .venv && source .venv/bin/activate`
- `python -m pip install -U pip && python -m pip install -r requirements.txt`

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
 
