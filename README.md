# FluidRender

Synchronization-aware scheduling for cloud-rendering backends. Two actors observe
the same global state, learn latency and load-stability objectives, and combine
their node preferences into one worker assignment.

This reference implementation follows the revised manuscript of *FluidRender:
Synchronization-Aware Cooperative Scheduling for Cloud Rendering Services*. The
original experiment datasets and trained checkpoints are not included. Default
runs use generated workloads; reproduction of the published numerical results
has not been verified with this implementation.

## Installation

Python 3.10 or later is required. Training uses NumPy and runs on the CPU.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Install the optional plotting dependency with:

```bash
python -m pip install -e '.[plots]'
```

## Quick start

Run the heuristic schedulers:

```bash
python scripts/run_experiment.py --seconds 10 --warmup 1 --seeds 2 \
  --methods RR,SAG,A-LALL,K8s-FS --out results/heuristics
```

Train and evaluate a small policy:

```bash
python scripts/train.py --steps 96 --pretrain-total 64 --batch 16 \
  --checkpoint-every 0 --out checkpoints/example.npz
python scripts/run_experiment.py --seconds 10 --warmup 1 --methods FluidRender,A-LALL \
  --checkpoint FluidRender=checkpoints/example.npz --out results/learned
```

The short run checks the training pipeline and does not produce a converged
policy. Learned schedulers require a checkpoint. Online placement evaluates only
the actors; A-LALL handles numerical inference failures.

## Training

The default budget is 3,600,000 interactions: 1,000,000 for each actor's pre-training
stage, followed by 1,600,000 joint interactions. Rejected requests count toward
this budget. Validation interactions are counted separately. `--steps` specifies
the total target, including interactions stored in a resumed checkpoint.

```bash
python scripts/train.py --method FluidRender --steps 3600000 --pretrain-total 2000000 \
  --checkpoint-every 200000 --out checkpoints/fluidrender.npz
python scripts/train.py --method JointFromScratch --steps 3600000 --pretrain-total 0 \
  --checkpoint-every 200000 --out checkpoints/joint.npz
python scripts/train.py --resume checkpoints/joint.npz --steps 7200000 \
  --pretrain-total 0 --out checkpoints/joint_2B.npz
```

Checkpoints contain actor and critic parameters, Adam state, random-generator
state, configuration and the interaction counter.

## Policy comparison

```bash
python scripts/run_matched_tuning.py --model reference --out results/reference
python scripts/run_matched_tuning.py --model backend --out results/backend_tuning
```

Each method searches 12 configurations using two training seeds and five shared
validation workloads. Selection uses final-budget mean team cost. Final evaluation
uses five new training seeds and ten held-out workloads. JointFromScratch-2B
continues the selected optimizer without retuning. The reduced reference model
uses a budget of 147,456 interactions. Add `--smoke` for a short pipeline check.

Trainable policies include FluidRender, JointFromScratch, 1A-Scalar, 1A-MO,
LatencyOnly, LoadStabilityOnly, 2A-CacheLoad and 3A-SQL. `MAPPO` provides a local
centralized scalar PPO control. MADDPG, DANIEL and the original 2SGA window optimizer
are not implemented. `2SGA-Proxy` is a separate single-task genetic heuristic.

## Experiment scripts

| Experiment | Command |
|---|---|
| Link degradation, QoS, tail components, freshness and recovery | `python scripts/run_replay.py --suite all --seeds 10` |
| Short replay check | `python scripts/run_replay.py --suite all --smoke --seeds 2` |
| Active-session load | `python scripts/run_sweeps.py --suite sessions` |
| Working-set complexity | `python scripts/run_sweeps.py --suite complexity` |
| Cache-reporting interval | `python scripts/run_sweeps.py --suite reporting` |
| Adaptation | `python scripts/run_adaptation.py --checkpoint checkpoints/fluidrender.npz` |
| Authentication and hashing cost | `python scripts/run_practicality.py --mode crypto` |
| Placement overhead | `python scripts/run_practicality.py --mode serving --checkpoint checkpoints/fluidrender.npz` |

Sweeps accept `--methods` and `--checkpoint METHOD=path.npz`. Each adaptation
condition starts from the supplied checkpoint and optimizer. Backend adaptation
uses 400,000 joint interactions; reference-model adaptation uses 32,768. Repeat
adaptation for each training seed to obtain independent policy replicates.

## Plotting

Plots read the CSV files produced by the experiment scripts:

```bash
python scripts/plot_results.py results/replay/network/summary.csv \
  --out results/network.pdf
python scripts/plot_results.py checkpoints/fluidrender.validation.csv \
  --x interactions --y validation_cost --error validation_cost_sd \
  --ylabel 'Validation cost, mean and workload SD' --out results/convergence.pdf
```

## Output and units

Each experiment saves its configuration, per-run measurements and summary
statistics. Source and checkpoint hashes identify the code and policy used.
P99 is calculated within each workload run, then averaged across runs. Miss rates
include rejections, terminal failures and late completions. Requests are followed
to completion or failure after the final arrival. An empty completed-request set
has no latency quantile and is recorded as `null` or an empty CSV cell.

Simulation units are decimal MB/GB, milliseconds, seconds and Gbps. Cryptographic
buffer sizes use MiB. Transfer volume counts payload bytes once, independently
of path length. CPU benchmarks measure the local environment and exclude rendering
and media delivery.

## Project layout

- `src/fluidrender/`: policies, training, simulation, networking and statistics.
- `configs/`: default backend and controlled-replay settings.
- `scripts/`: experiment and plotting entry points.
- `tests/`: numerical, protocol and integration checks.
