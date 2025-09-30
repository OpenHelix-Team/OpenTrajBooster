<br>
<p align="center">
<h1 align="center"><strong>TrajBooster: Boosting Humanoid Whole-Body Manipulation via Trajectory-Centric Learning</strong></h1>
  <p align="center">
    <a href='https://jiachengliu3.github.io/' target='_blank'>Jiacheng Liu*</a>, <a href='https://dingpx.github.io/' target='_blank'>Pengxiang Ding*</a>, <a href='https://github.com/litlewhitte/litlewhitte' target='_blank'>Qihang Zhou</a>, <a href='https://github.com/FurryGreen' target='_blank'>Yuxuan Wu</a>, <a href='https://github.com/HDDD16988/Introduction' target='_blank'>Da Huang</a>, <a href='https://github.com/JimmyPang02' target='_blank'>Zimian Peng</a>, <a href='https://xiaowei-i.github.io/' target='_blank'>Wei Xiao</a>, <a href='https://wnzhang.net/' target='_blank'>Weinan Zhang</a>, <a href='https://lixiny.github.io/' target='_blank'>Lixin Yang</a>, <a href='https://www.mvig.org/' target='_blank'>Cewu Lu† </a>, <a href='https://milab.westlake.edu.cn/' target='_blank'>Donglin Wang† </a>
    <br>
    * Equal Controlbution, † Corresponding Authors 
    <br>
    Zhejiang University, Westlake University, Shanghai Jiao Tong University & Shanghai Innovation Institute
    <br>
  </p>
</p>

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2509.11839-orange)](https://arxiv.org/abs/2509.11839)
[![](https://img.shields.io/badge/Project-%F0%9F%9A%80-pink)](https://jiachengliu3.github.io/TrajBooster/)

<p align="center">
  <img src="../Hello_teaser.gif" alt="Hello teaser" width="100%">
</p>

</div>



## 📋 Contents

- [📋 Contents](#-contents)
- [🏠 Description](#-description)
- [📚 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Train](#train)
  - [Play](#play)
- [🔗 Citation](#-citation)
- [📄 License](#-license)
- [👏 Acknowledgements](#-acknowledgements)

## 🏠 Description
<a name="-description"></a>
A manager policy based on the harmonized online DAgger algorithm that outputs control commands (v_x, v_y, v_yaw, height) to direct the worker policy.

## 📚 Getting Started
<a name="-start"></a>
### Prerequisites

We recommend to use our code under the following environment:

- Ubuntu 20.04/22.04 Operating System
- IsaacGym Preview 4.0
  - NVIDIA GPU (RTX 2070 or higher)
  - NVIDIA GPU Driver (recommended version 535.183)
- Conda
  - Python 3.8

### Installation
A. Create a virtual environment and install Isaac Gym:
```
1. conda create -n manager python=3.8
2. conda activate manager

3. conda install mamba -n base -c conda-forge -y
4. mamba install -c conda-forge pinocchio
5. pip install meshcat
6. pip install casadi

# Install Isaac Gym
7. cd path_to_downloaded_isaac_gym/python
8. pip install -e .
```
B. Install this repository:
```
1. cd Manager && pip install -r requirements.txt
2. cd rsl_rl && pip install -e .
3. cd ../legged_gym && pip install -e .
```

### Train
You can train your own policy with our code by running the command below.
```
python legged_gym/legged_gym/scripts/reach_train.py --task g1_reach --num_envs 512 --headless --run_name manager_policy --rl_device cuda:0 --sim_device cuda:0  --resume
```
The meanings of the parameters in this command are listed below:
* `--task`: the training task
* `--num_envs`: the number of parallel environments used for training
* `--headless`: don't use the visualization window; you cannot use it to visualize the training process
* `--run_name`: name of this training
* `--rl_device` & `--sim_device`: which device is used for training

The default logging method is [wandb](https://wandb.ai/), and you have to set the values of ***run_name***, ***experiment_name***, ***wandb_project***, and ***wandb_user*** to yours in `legged_gym/legged_gym/envs/g1/g1_29dof_config.py`. You can also change the ***logger*** to **tensorboard**. The training results will be saved in `legged_gym/logs/`.

If you encounter the error: ***"ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory"***, you can run this command to solve it, where path_to_miniconda3 is the absolute path of your miniconda directory:
```
export LD_LIBRARY_PATH=path_to_miniconda3/envs/manager/lib:$LD_LIBRARY_PATH
```
### Play
Once you train a policy, you can first set the [resume_path]() to the path of your checkpoint, and run the command below:
```
python legged_gym/legged_gym/scripts/reach_play.py   --num_envs 1 --task g1_reach --resume --resume2
```
Then you can view the performance of your trained policy.


Please note! In this paper, the retargeting in this section is only implemented in simulation environments and has not been tested in real-world environments. Please exercise caution when deploying this model to actual Unitree G1.

<!-- ### Export Policy
We provide a script for you to export you `.pt` checkpoint to `.onnx`, which can be used by our [deployment code](). You can set the [pt_path]() and [export_path]() to what you need, and run
```
python legged_gym/legged_gym/scripts/export_onnx.py
``` -->

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@article{liu2025trajbooster,
  title={TrajBooster: Boosting Humanoid Whole-Body Manipulation via Trajectory-Centric Learning},
  author={Liu, Jiacheng and Ding, Pengxiang and Zhou, Qihang and Wu, Yuxuan and Huang, Da and Peng, Zimian and Xiao, Wei and Zhang, Weinan and Yang, Lixin and Lu, Cewu and Wang, Donglin},
  journal={arXiv preprint arXiv:2509.11839},
  year={2025}
}
```

</details>

## 📄 License

This project is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a> <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>.


## 👏 Acknowledgements


- [Walk-These-Ways](https://github.com/leggedrobotics/rsl_rl): Our robot deployment code is based on `walk-these-ways`.
- [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2): We use `Unitree SDK2` library to control the robot.
- [OpenHomie](https://github.com/InternRobotics/OpenHomie): This work is developed based on `OpenHomie` codebase.

