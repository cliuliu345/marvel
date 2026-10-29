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
Optional `--stagnation-recovery` enables behavior recovery; pure Actor behavior
is the default. Existing network, reward, SAC, 128-step and 99% success
semantics are unchanged.
