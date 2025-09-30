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
  <img src="./Hello_teaser.gif" alt="Hello teaser" width="100%">
</p>

</div>

## 📋 Contents

- [📋 Contents](#-contents)
- [🏠 About](#-about)
- [📚 Usage](#-usage)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [🔗 Citation](#-citation)
- [📄 License](#-license)
- [👏 Acknowledgements](#-acknowledgements)

## 🏠 About

<a name="-about"></a>

This sub-repository provides the official implementation of the retargeting model module for "TrajBooster: Boosting Humanoid Whole-Body Manipulation via Trajectory-Centric Learning". Given only the 6D coordinates of both wrists, our system enables humanoids to achieve full-body locomotion-manipulation by tracking these two coordinate targets.

This repository contains two key components of our Retargeting model:

* **Worker**: A Homie-style worker policy with the following enhancements: (1) modifications based on [this issue](https://github.com/InternRobotics/OpenHomie/issues/16), and (2) addition of locomotion training in semi-crouched poses, expanding beyond standing-only movement training.

* **Manager**: A manager policy based on the harmonized online DAgger algorithm that outputs control commands (v_x, v_y, v_yaw, height) to direct the worker policy.

These components are organized into separate sub-directories that can be treated as independent repositories. Each sub-directory contains its own README with detailed usage instructions and functional descriptions.

## 📚 Usage

<a name="-usage"></a>

### Prerequisites

We recommend using our code under the following environment:

- **Operating System**: Ubuntu 20.04/22.04
- **Simulation**: IsaacGym Preview 4.0
- **GPU**: NVIDIA GPU (RTX 2070 or higher)
- **Driver**: NVIDIA GPU Driver (recommended version 535.183)
- **Environment Manager**: Conda
- **Python**: Python 3.8

**Hardware Requirements:**
- Unitree G1 with Dex3 Hands
- Head RGB camera × 1
- Wrist RGB cameras × 2

### Installation

First, clone this repository to your Ubuntu system:

```bash
git clone https://github.com/jiachengliu3/OpenTrajBooster.git
```

Then follow the README.md files in each sub-repository to install all components or specific modules as needed.

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


