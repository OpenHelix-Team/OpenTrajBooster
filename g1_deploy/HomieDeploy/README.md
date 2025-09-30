<br>
<p align="center">
<h1 align="center"><strong>TrajBooster: Boosting Humanoid Whole-Body Manipulation via Trajectory-Centric Learning (Deployment)</strong></h1>
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

<!-- <img src="./deploy.png" alt="cross" width="100%" style="position: relative;"> -->

</div>

## 📋 Contents

- [📋 Contents](#-contents)
- [🏠 Description](#-description)
- [📚 Usage](#-usage)
  - [Unitree G1](#unitree-g1)
  - [Deployment](#deployment)
- [🔗 Citation](#-citation)
- [📄 License](#-license)
- [👏 Acknowledgements](#-acknowledgements)

## 🏠 Description
<a name="-description"></a>
This repository is an official implementation of the deployment of "TrajBooster: Boosting Humanoid Whole-Body Manipulation via Trajectory-Centric Learning". It requires an `Unitree G1` with `Dex-3 hands` and a personal computer. All communications between the robot and the PC are via Wi-Fi. Our code is based on [Walk-These-Ways](https://github.com/Improbable-AI/walk-these-ways) and [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2). Our [hardware system](https://github.com/jiachengliu3/OpenTrajBooster/tree/main/g1_deploy/Hardware) is also open-sourced, you can first refer to it to reimplement the teleoperation system. Once you successfully build a system, you can follow the instructions in the [Usage](#-use). For simple usage, we just provide an example checkpoint named `deploy.onnx`.

## 📚 Usage
<a name="-use"></a>

### Unitree G1
**NOTE:** It is recommended that you connect a screen, a keyboard, and a mouse to the Unitree G1 to use the board on it as a computer.

First of all, you should install `PyTorch` on the Nvidia Jetson Orin of the Unitree G1, which is different from the process on a normal Ubuntu PC. For this step, please refer to the official [instruction](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html).

Then, you can use any ways to put this code to the Orin and enter the directory. Install the required packages by running:
```
pip install -r requirements.txt
```
For controlling robot, we use the cpp-based Unitree SDK2, which means you should first compile the `g1_control.cpp` for `Unitree G1` and `hand_control.cpp` for `Dex-3`. We have prepared the required `CMakeLists.txt` for you, so you only need to run the following command lines:
```
cd unitree_sdk2
rm -rf build
mkdir build && cd build
cmake ..
make
```
Then the runnable binary files will be in `unitree_sdk2/build/bin`.
You also need to install the  `g1_gym_deploy` by running:
```
cd g1_gym_deploy && pip install -e .
```

### Deployment
🎯 **Before deployment, please run L1+A L2+R2 L2+A L2+B to close G1's initial control process, if successful, you will see the robot hang up its arm after L2+A and lose efforts after L2+B.**

For TCP communications, you should determine the IP address of your PC and robot by running:
```
ifconfig | grep inet
```
Set the IP addresses in the code to the correct value.

A. Run the robot control program on `robot` (robot terminal 2):
```
cd unitree_sdk2/build/bin && ./g1_control eth0 (or eth1)
```
B. Run the inference thread to make RL policy control robot on `robot` (robot terminal 3):

 Please select **one** of the following control modes:

1) For Teleoperation Mode:
```
python g1_gym_deploy/scripts/deploy_policy.py 
```

2) For Autonomous Control Mode:
```
python g1_gym_deploy/scripts/deploy_policy_infer.py
```
C. After putting the robot on the ground, push the `R2` button of the joysticker, make the robot stand on the ground, and push `R2` again.

***NOTE:*** We strongly recommend you to really deploy the system after you really understand function of all files, otherwise there can be some troubles.

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


