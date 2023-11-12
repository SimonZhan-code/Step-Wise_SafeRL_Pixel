CBF Dreamer implementation in PyTorch
======

## Samples
Shown here are videos of example after BF training on some of the SafetyGym Environments. The left-hand-side show the observation and the right-hand-side show image after reconstruction.

<img height="135" src="./imgs/PointGoal1.gif"><img height="135" src="./imgs/CarGoal1.gif"><img height="135" src="./imgs/PointGoal2.gif">
<img height="135" src="./imgs/PointPush1.gif"><img height="135" src="./imgs/DoggoGoal1.gif"><img height="135" src="./imgs/PointButton1.gif">


## ArXiv
The arXiv version of the paper can be accessed from [this link](https://arxiv.org/abs/2311.02227). 
```bibtex
@article{zhan2023state,
  title={State-wise Safe Reinforcement Learning With Pixel Observations},
  author={Zhan, Simon Sinong and Wang, Yixuan and Wu, Qingyuan and Jiao, Ruochen and Huang, Chao and Zhu, Qi},
  journal={arXiv preprint arXiv:2311.02227},
  year={2023}
}
```

## Installation
To install all dependencies with Anaconda run using the following commands. 

`conda create -n CBF-dreamer python==3.8` 

`pip install -r requirements.txt` 

To install the safety_gym library, please use the accomodated safety_gym package provided.

## Training (e.g. Safexp-PointGoal1)
To run CBF-dreamer
```bash
python bc.py --algo CBF-dreamer --env Safexp-PointGoal1-v0 --action-repeat 2 --id {name_of_exp}
```

## Testing (e.g. Safexp-PointGoal1)
```bash
python bc.py --algo CBF-dreamer --env Safexp-PointGoal1-v0 --action-repeat 2 --test --render --models {.pth_file_load} --id {name_of_exp}
```

<!-- For best performance with DeepMind Control Suite, try setting environment variable `MUJOCO_GL=egl` (see instructions and details [here](https://github.com/deepmind/dm_control#rendering)). -->

Use Tensorboard to monitor the training.

`tensorboard --logdir results`



## Links
- [pytorch implementation of Dreamer](https://github.com/yusukeurakami/dreamer-pytorch)
- [RSSM elaboration](https://arxiv.org/abs/1811.04551)
