# MARVEL: Multi-Agent Reinforcement Learning for constrained field-of-View multi-robot Exploration in Large-scale environments

This repository hosts the code for [**MARVEL**](https://arxiv.org/abs/2502.20217), accepted for [ICRA 2025](https://2025.ieee-icra.org/).

<div>
   <img src="utils/media/MARVEL.gif" height="250"/>
   <img src="utils/media/Hardware_validation.gif" height="250"/>
</div>

**Supplementary video link:** [YouTube](https://youtu.be/uwQqkruBvwY)

MARVEL is a neural framework that leverages graph attention networks, together with novel frontiers and orientation features fusion technique, to develop a collaborative, decentralized policy using multi-agent reinforcement learning (MARL) for robots with constrained FoV. 

## Setup instructions

Use conda to setup your environment:

```bash
conda env create -f marvel.yml
conda activate marvel
```

## CPU acceleration dependencies

After creating the Conda environment, install the pinned CPU acceleration packages:

```bash
python -m pip install -r requirements-speedups.txt
```

The first ray-kernel call compiles CPU code; later processes can reuse the on-disk cache. Graph guide paths still use the original A* implementation.

## Evaluation

First, set the appropriate parameters in `test_parameter.py` and adjust testing configurations within `test_driver.py`. Run `test_driver.py` to evaluate.

## Training

Set appropriate parameters in `parameter.py` and run `driver.py` to train the model.


## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@INPROCEEDINGS{chiun2025marvel,
  author={Chiun, Jimmy and Zhang, Shizhe and Wang, Yizhuo and Cao, Yuhong and Sartoretti, Guillaume},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={MARVEL: Multi-Agent Reinforcement Learning for Constrained Field-of-View Multi-Robot Exploration in Large-Scale Environments}, 
  year={2025},
  pages={11392-11398},
  keywords={Training;Three-dimensional displays;Robot kinematics;Robot vision systems;Reinforcement learning;Reliability engineering;Sensors;Planning;Complexity theory;Drones},
  doi={10.1109/ICRA55743.2025.11127700}}
```

Authors:
[Jimmy Chiun](https://jimmychiun.me),
Shizhe Zhang,
[Yizhuo Wang](https://www.yizhuo-wang.com/),
[Yuhong Cao](https://www.yuhongcao.online/),
[Guillaume Sartoretti](https://cde.nus.edu.sg/me/staff/sartoretti-guillaume-a/)

License:
MIT


## V1 heterogeneous training

The V1 entry point keeps the legacy training path unchanged. On a single
64 GB / RTX 3090 / 8-core host, use six single-threaded sampling workers and
a GPU learner. The default joint batch is 64 (256 agent samples); warm-up
remains 2,000 agent samples (500 joint transitions). Keep each run in its
own output directory.

```bash
python driver_v1.py --episodes 20000 --ray-workers 6 --device cuda \
    --checkpoint-out model/v1_run1/latest.pth --checkpoint-every 100 \
    --tensorboard-logdir train/v1_run1
```

Resume the learner, optimizers, update counter and next episode number:

```bash
python driver_v1.py --resume model/v1_run1/latest.pth --episodes 20000 \
    --ray-workers 6 --device cuda --checkpoint-out model/v1_run1/latest.pth \
    --checkpoint-every 100 --tensorboard-logdir train/v1_run1
```

Replay is not persisted and must warm up again. `--episodes` counts new
attempts, including failed episodes. In asynchronous sampling, `next_episode`
is the next *unlaunched* ID; in-flight episodes abandoned at shutdown are not
retried. Checkpoints are enabled by `--checkpoint-out`; atomic `latest.pth`
is saved at shutdown and every configured interval, alongside `ep{episode}.pth`.
Individual episode failures are logged as `V1_EPISODE_ERROR`; the default
consecutive-failure limit is five. Learner numerical errors remain fatal.
Reproduce an environment with `--episodes 1 --start-episode E --local-worker-id W`
using the same map directory and model. Shared Replay tensors are read-only.

```bash
python -m pytest tests -q
python driver_v1.py --smoke
```

Smoke always uses one joint sample, CPU, two macro steps and one SAC update.
Optional `--stagnation-recovery` enables behavior recovery for the whole run.
Alternatively, `--stagnation-recovery-episodes N` enables it only for training
episode IDs below N; the two options are mutually exclusive. Evaluation ignores
the training cutoff and needs explicit `--stagnation-recovery` to enable it.
Pure Actor behavior is the default. Existing network, reward, SAC, 128-step and 99% success
semantics are unchanged.

### Independent V1 evaluation

Use `--eval-every 100 --eval-maps maps_test --eval-num-maps 20 --eval-seed 0`
to evaluate a fixed sorted-and-seeded subset with the current Actor, greedy
selection, recovery off and no Replay collection. Evaluation map k always uses
worker ID 100000+k and episode k. Evaluation runs on the existing Ray runner
pool between training work; one evaluation round finishes before another starts.
It never advances training episode IDs, processed/failure counts, Replay or SAC.
CPU/CUDA Torch, NumPy, Python and Replay RNG states are saved and restored.
Ray asynchronous scheduling itself does not guarantee bitwise reproducibility;
use `--ray-workers 0 --device cpu` for bitwise training reproduction. Timing and
RSS measurements are not deterministic trajectory values.

`V1_EVAL` and `V1/Eval/*` use processed training episode count as their axis.
The first fixed evaluation map produces a GIF each round under the run's unique
GIF directory, with `eval_trainNNNNNN_` in its filename. Per-map results and errors
are retained in the JSON log; evaluation errors do not count as training failures.

### V1 GIFs and final snapshots

`--save-gif-every 64 --gif-dir gifs/v1_run1` matches MARVEL's
`SAVE_IMG_GAP=64` episode interval; GIF recording remains off by default.
Each invocation creates a unique run directory:
`gifs/v1_run1/run_<UTC timestamp>_seed<seed>_<unique>/`.
Each selected episode keeps its own `episode_<id>_worker_<id>_<unique>/`
subdirectory with the GIF, `frames.json`, and final frame PNG (including its substep suffix).
`make_gif` deletes intermediate PNGs but explicitly retains the final PNG.
After successful encoding, both the GIF and final snapshot are also copied to
`<unique run directory>/summary/ep000064_w0_cov99.12.gif` and the matching `.png`.
The `cov` value is coverage in percent. Original episode directories are retained.
Evaluation copies have an additional `eval_trainNNNNNN_` prefix so rounds on the
same fixed map cannot overwrite each other.

V1 now defaults to `--gif-frame-mode substep`, matching MARVEL's six frames per
macro step, plus one initial frame. Every frame is rendered after all four UAVs
finish that substep's existing scans, using their computed substep poses and trails.
`--gif-frame-mode macro` retains the previous one-frame-per-macro-step behavior.
`--gif-frame-every N` still selects macro steps: every selected macro step renders
all six substeps. The final macro step is always retained, even off the interval;
at most six read-only snapshots are buffered to preserve it without extra scans.
`--gif-frame-duration` is in seconds: omitted means 0.1 in substep mode or 0.5 in
macro mode. These options apply to training GIFs, periodic evaluation GIFs, and
`tools/v1_explore_check.py` (enable diagnostic GIFs with `--gif-dir PATH`).
Rendering never modifies scan order, poses, graph caches, RNG, rewards or training.

### Heterogeneous-sensing maps

`maps_medium` contains narrow 5 m corridors and small rooms, where a short
wide sensor dominates everywhere and capability allocation barely matters.
`tools/gen_hetero_maps.py` generates 80 m x 80 m maps in the same PNG format
(127 obstacle, 195 free, 208 start; 2 px per 0.4 m cell) that combine two
zone families:

| Family | Zones | Favours |
|---|---|---|
| narrow | open halls with sparse pillars; warehouse aisles | 45 deg / 12 m sensor |
| wide | cubicle clutter (furniture, partitions, pillars); room suites | 180 deg / 6 m sensor |

The two sensors of the `compact` profile have the same nominal sector area
(~56.5 m^2), so which one sees more depends only on the scene. Default mix is
70% mixed maps (each family 30-70% of the nodes, with a minimum per-map
contrast) and 15% of each pure family. Layout follows the 4 m node lattice:
Env anchors nodes at the start cell, which is a block centre, so each free
block is exactly one node (146-240 per map, below `NODE_PADDING_SIZE`).
Every map is checked for a single connected free component, at least six
initial nodes inside the 6 m start scan, and full visibility of every free
cell from some node with the 6 m range (so 99% coverage is reachable).

```bash
python tools/gen_hetero_maps.py --out maps_hetero_train --num 4000 --seed 0 --workers 8
python tools/gen_hetero_maps.py --out maps_hetero_test --num 100 --seed 1 --workers 8
MARVEL_SENSOR_PROFILE=compact python driver_v1.py --map-directory maps_hetero_train/maps \
    --eval-maps maps_hetero_test/maps ...
```

`OUT/maps` holds only PNGs (Env lists the directory). `OUT/zones` has
per-cell zone labels at Env resolution (0 solid, 1 hall, 2 aisles, 3 clutter,
4 rooms, 5 connector) for allocation analysis, e.g. the share of time each
sensor type spends in its favoured zones. `OUT/meta.jsonl` records node
counts, zone shares and per-family sight lines and visible areas.

`MARVEL_SENSOR_PROFILE=compact` (or `SENSOR_PROFILE` in `parameter.py`)
switches to sensors (180 deg, 6 m) x2 and (45 deg, 12 m) x2, `SENSOR_RANGE` 12,
`INITIAL_SCAN_RANGE` 6 and `FRONTIER_REPRESENTATION_RANGE` 16. The default
`marvel` profile is unchanged. Before long runs, calibrate on the new maps:
`TEAM_FRONTIER_NORMALIZER` (39, from the original 10 m range) should match the
mean frontier cells removed per macro step, and `tools/v1_explore_check.py`
with the guidepost/random policies should show the maps are completable
within 128 macro steps.
