# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger#, export_DAgger_policy_as_jit

import numpy as np
import torch
import torch.onnx
import copy 
import pinocchio as pin
from rsl_rl.ik_modules.robot_arm_ik import G1_29_ArmIK
import onnxruntime as ort


def load_policy(resume_path):
    body = torch.jit.load(resume_path, map_location="cuda:0")
    def policy(obs):
        action = body.forward(obs)
        return action
    return policy

def load_onnx_policy():
    worker_path = os.path.join(LEGGED_GYM_ROOT_DIR, "worker_model/deploy0929.onnx")
    model = ort.InferenceSession(worker_path)
    def run_inference(input_tensor):
        ort_inputs = {model.get_inputs()[0].name: input_tensor.detach().cpu().numpy()}
        # ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device="cuda:0")
    return run_inference


def play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0, height=0.73):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.randomize_body_displacement = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.use_random = False
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.asset.self_collision = 0
    env_cfg.env.upper_teleop = True
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel
    env.commands[:, 4] = height
    # lin_vel_x = [-0.8, 0.8] # min max [m/s]
    # lin_vel_y = [-0.5, 0.5]   # min max [m/s]
    lin_vel_x = [-0.8, 0.8] # min max [m/s]
    lin_vel_y = [-0.5, 0.5]   # min max [m/s]
    env.action_curriculum_ratio = 0.0
    
    obs = env.get_observations()

    resume_path = "" # YOUR WORKER POLICY PATH
    lower_body_policy = load_policy(resume_path)
    # lower_body_policy = load_onnx_policy()

    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.resume2 = True
    bc_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = bc_runner.get_inference_policy(device=env.device) # Use this to load from trained pt file

    # print(f"bc policy0: {policy}")
    # if EXPORT_DAgger_POLICY:
    #     path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'DAgger_policies')
    #     export_DAgger_policy_as_jit(bc_runner.alg.actor_critic, path)
    #     print('Exported policy as jit script to: ', path)
    
    # resume_DAgger_path = "legged_gym/logs/next_exp/exported/DAgger_policies/policy.pt"
    # policy = load_policy(resume_DAgger_path)

    # ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy(device=env.device) 

    # print(f"bc policy2: {policy}")

    # load ik policy
    ik_policy = G1_29_ArmIK()

    

    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    if RECORD_FRAMES:
        from isaacgym import gymapi
        from isaacgym import gymutil
        # Create a directory to save frames
        frames_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        # Create a camera handle
        cam_props = gymapi.CameraProperties()
        cam_props.width = 640
        cam_props.height = 480
        cam_props.enable_tensors = True

        env.camera_handle = env.gym.create_camera_sensor(env.envs[0], cam_props)
        # env.gym.set_camera_location(env.camera_handle, env.envs[0], gymapi.Vec3(0.9, 0.0, 1.2), gymapi.Vec3(0.0, 0.0, 0.7))
        env.gym.set_camera_location(env.camera_handle, env.envs[0], gymapi.Vec3(0.12, 0.0, 0.68), gymapi.Vec3(0.25, 0.0, 0.5))


    env.reset_idx(torch.arange(env.num_envs).to("cuda:0"))
    # env.it = 750
    env.it = 0
    mpe=[]
    mqe=[]
    for j in range(10*int(env.max_episode_length)):
        print(f"step: {j}")
        env.action_curriculum_ratio = 0.0

        current_step = env.episode_length_buf.long() 
        left_gts1 = env.left_gts[torch.arange(env.num_envs), current_step, :]
        left_grs = env.left_grs[torch.arange(env.num_envs), current_step, :]
        right_gts1 = env.right_gts[torch.arange(env.num_envs), current_step, :]
        right_grs = env.right_grs[torch.arange(env.num_envs), current_step, :]

        # left_gts = obs[:,env.num_one_step_obs*5+37+27+27+14+1:env.num_one_step_obs*5+37+27+27+14+1+3]
        # left_grs = obs[:,env.num_one_step_obs*5+37+27+27+14+1+3:env.num_one_step_obs*5+37+27+27+14+1+3+4]
        # right_gts = obs[:,env.num_one_step_obs*5+37+27+27+14+1+3+4:env.num_one_step_obs*5+37+27+27+14+1+3+4+3]
        # right_grs = obs[:,env.num_one_step_obs*5+37+27+27+14+1+3+4+3:env.num_one_step_obs*5+37+27+27+14+1+3+4+3+4]

        torso_pos = env._get_torso_pose()
        torso_height = torso_pos[:, 2]

        torso_pose1 = copy.deepcopy(torso_pos)
        torso_pose1[:,2] = 0.0

        left_gts = left_gts1 - torso_pose1
        right_gts = right_gts1 - torso_pose1

        
        # 构造批量目标位姿
        L_tf_targets = [
            pin.SE3(
                pin.Quaternion(
                    float(left_grs[j][3].cpu().detach().numpy()),
                    float(left_grs[j][0].cpu().detach().numpy()),
                    float(left_grs[j][1].cpu().detach().numpy()),
                    float(left_grs[j][2].cpu().detach().numpy()),
                ),
                np.array([
                    float(left_gts[j][0].cpu().detach().numpy()),
                    float(left_gts[j][1].cpu().detach().numpy()),
                    float(left_gts[j][2].cpu().detach().numpy()) -torso_height[j].cpu().detach().numpy() + 0.05
                ])
            )
            for j in range(obs.shape[0])
        ]
        # print(f"L_tf_targets: {L_tf_targets}")

        R_tf_targets = [
            pin.SE3(
                pin.Quaternion(
                    float(right_grs[j][3].cpu().detach().numpy()),
                    float(right_grs[j][0].cpu().detach().numpy()),
                    float(right_grs[j][1].cpu().detach().numpy()),
                    float(right_grs[j][2].cpu().detach().numpy()),
                ),
                np.array([
                    float(right_gts[j][0].cpu().detach().numpy()),
                    float(right_gts[j][1].cpu().detach().numpy()),
                    float(right_gts[j][2].cpu().detach().numpy())-torso_height[j].cpu().detach().numpy() + 0.05
                ])
            )
            for j in range(obs.shape[0])
        ]

        # 获取当前的关节角度和速度
        current_lr_arm_motor_q = obs[:, env.num_one_step_obs*5+10+13:env.num_one_step_obs*5+10+27].view(-1, 14).cpu().detach().numpy()
        current_lr_arm_dq = obs[:, env.num_one_step_obs*5+37+13:env.num_one_step_obs*5+37+27].view(-1, 14).cpu().detach().numpy() *20

        sol_qs = np.zeros((obs.shape[0], 14))

        # sol_qs = self.solve_batch_ik_parallel(
        #     self.ik_policy,
        #     L_tf_targets,
        #     R_tf_targets,
        #     current_lr_arm_motor_q,
        #     current_lr_arm_dq
        # )

        for k in range(obs.shape[0]):
            try:
                # Get single element from each batch
                left_target = L_tf_targets[k].homogeneous
                right_target = R_tf_targets[k].homogeneous
                current_q = np.array(current_lr_arm_motor_q[k])  # Keep as 2D array
                current_dq = np.array(current_lr_arm_dq[k])       # Keep as 2D array
                
                # Solve IK for this single element
                q, tau = ik_policy.solve_ik(left_target, right_target, current_q, current_dq)
                
                # Store successful solution
                sol_qs[k] = q
                
            except Exception as e:
                print(f"Failed to solve IK for element {k}:", e)
                sol_qs[k] = np.zeros(14)  # If failed, return zero solution

        up_actions = sol_qs * 4



        # 检查哪些环境的 start_flag 为 1
        mask = env.start_flag == 1
        # 将满足条件的环境的 up_actions 设置为零
        up_actions[mask.cpu().detach().numpy()] = np.zeros((mask.sum().item(), 14))
        
        up_actions = torch.tensor(up_actions, device=env.device, dtype=torch.float32)
        upper_body_actions = up_actions # upper body actions
        

        ######## ============================== manager_actions ============================== #########
        # print(f"obs.shape: {obs.shape}")
        manager_actions = policy(obs.detach())

        # print(f"manager_actions: {manager_actions}")
        env.commands[:, 0] = torch.clamp(manager_actions[:,-3], lin_vel_x[0], lin_vel_x[1]) # x_vel
        env.commands[:, 1] = torch.clamp(manager_actions[:,-2], lin_vel_y[0], lin_vel_y[1]) # y_vel
        env.commands[:, 2] = torch.clamp(manager_actions[:,-1]*4, -2.5, 2.5) # yaw_vel
        # env.commands[:, 3] = 0.0 # stand still 
        # print(f"manager_actions[:,-3]: {manager_actions[:,-3]}")
        height = manager_actions[:, -4] # height
        env.commands[:, 4] = torch.clamp(height , min=0.28, max=0.74) # height
        # env.commands[:, 4] = torch.clamp(height , min=0.74, max=0.74) # height


        #### ====================================lower_body_actions================================= ####
        num_segments = obs.shape[1] // env.num_one_step_obs 
        indices_to_keep = torch.cat([torch.arange(i * env.num_one_step_obs, (i + 1) * env.num_one_step_obs - 14*5) for i in range(num_segments)], dim=0)
        lower_policy_obs = obs[:, indices_to_keep]

        lower_body_actions = lower_body_policy(lower_policy_obs)
        lower_body_actions = torch.cat((lower_body_actions, torch.zeros((lower_body_actions.size(0), 1), device=lower_body_actions.device)), dim=1) # yaw_vel = 0.0
        
        ### ============================== upper_body_actions padding ================================== ###
        upper_body_actions = torch.tensor(upper_body_actions, device=lower_body_actions.device, dtype=lower_body_actions.dtype)
        

        #### =================================concat actions ================================== ####
        # print(f"upper_body_actions.shape: {upper_body_actions.shape}")
        # print(f"lower_body_actions.shape: {lower_body_actions.shape}")
        actions = torch.cat((lower_body_actions, upper_body_actions), dim=1)
        # print(f"actions: {actions}")
        obs, _, _, _, _, _, _ = env.step(actions.detach())
        # eval
        # mean square error of pos
        left_hand_pos, left_hand_quat, right_hand_pos, right_hand_quat = env._get_hands_pose()
        # mse
        pos_mean = (torch.mean(torch.norm(left_gts1 - left_hand_pos, dim=1)) + torch.mean(torch.norm(right_gts1 - right_hand_pos, dim=1)))/2.0
        print(f"left_grs: {left_grs}")
        print(f"right_grs: {right_grs}")
        quat_mean = (torch.mean(quaternion_angle_error(left_hand_quat, left_grs)) + torch.mean(quaternion_angle_error(right_hand_quat,right_grs)))/2.0
        quat_mean = quat_mean*57.29578
        print(f"pos_mean: {pos_mean}, quat_mean: {quat_mean}")
        # eval
        mpe.append(pos_mean.item())
        mqe.append(quat_mean.item())

        if RECORD_FRAMES:
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)

            img = env.gym.get_camera_image(env.sim, env.envs[0], env.camera_handle, gymapi.IMAGE_COLOR)

            if img is not None:
                img = img.reshape(480, 640, 4)        # BGRA
                img = img[:, :, :3][..., ::-1]         # Convert to RGB
                from PIL import Image
                frame_path = os.path.join(frames_dir, f'frame_{j:05d}.png')
                Image.fromarray(img).save(frame_path)

        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)
    
    # eval
    mpe = np.mean(mpe)
    mqe = np.mean(mqe)
    print(f"mpe: {mpe}")
    print(f"mqe: {mqe}")


@torch.jit.script
def quaternion_angle_error(q1, q2):
    """
    计算两个四元数之间的旋转角度误差
    :param q1: 当前四元数 (n, 4)
    :param q2: 目标四元数 (n, 4)
    :return: 旋转角度误差 (n,)
    """
    # 计算四元数的点积
    dot_product = torch.sum(q1 * q2, dim=1).abs()  # 取绝对值，确保对称性
    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # 防止数值误差导致超出 [-1, 1]
    
    # 计算旋转角度
    angle_error = 2 * torch.acos(dot_product)  # 旋转角度（弧度）
    return angle_error

if __name__ == '__main__':
    EXPORT_POLICY = True
    EXPORT_DAgger_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0., height=0.73)