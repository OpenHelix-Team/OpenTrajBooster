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

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import torch
import onnxruntime as ort

import rsl_rl
# from rsl_rl.algorithms import ReachPPO
# from rsl_rl.algorithms import ReachBC
# from rsl_rl.algorithms import ReachPPOBC
from rsl_rl.algorithms import ReachDAgger
from typing import Union
from rsl_rl.modules import ReachActorCritic
from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state

import pinocchio as pin
from rsl_rl.ik_modules.robot_arm_ik import G1_29_ArmIK
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
import random


class ReachOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu',
                 resume_path=None):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
            num_one_step_critic_obs = self.env.num_one_step_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
            num_one_step_critic_obs = self.env.num_one_step_obs
        self.num_actor_obs = self.env.num_obs
        self.num_critic_obs = num_critic_obs
        self.actor_history_length = self.env.actor_history_length
        self.critic_history_length = self.env.critic_history_length
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ReachActorCritic
        actor_critic: ReachActorCritic = actor_critic_class( 
                                                        self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_one_step_obs,
                                                        num_one_step_critic_obs,
                                                        self.env.actor_history_length,
                                                        self.env.critic_history_length,
                                                        self.env.num_upper_dof,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # ReachBC or ReachPPO or ReachPPOBC
        # self.alg: Union[ReachBC, ReachPPO, ReachPPOBC] = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.alg: ReachDAgger = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [4])
        self.alg.init_dagger_dataset()

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        # Load lower_body_policy
        if resume_path is not None:
            # self.lower_body_policy = self.load_onnx_policy()
            self.lower_body_policy = self.load_policy(resume_path)
            self.lower_body_policy.eval()
        else:  # error
            raise ValueError("No resume path provided")
        
        _, _ = self.env.reset()
        # load ik policy
        self.ik_policy = G1_29_ArmIK(Unit_Test = False, Visualization = False)

        self.height_history = torch.zeros(self.env.num_envs, 6, device=self.device)
        self.v_x_history = torch.zeros(self.env.num_envs, 6, device=self.device)
        self.v_y_history = torch.zeros(self.env.num_envs, 6, device=self.device)

    def load_onnx_policy(self):
        worker_path = os.path.join(LEGGED_GYM_ROOT_DIR, "worker_model/deploy0929.onnx")
        model = ort.InferenceSession(worker_path)
        def run_inference(input_tensor):
            ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
            ort_outs = model.run(None, ort_inputs)
            return torch.tensor(ort_outs[0], device="cuda:0")
        return run_inference

    def load_policy(self, resume_path):
        # load policy
        body = torch.jit.load(resume_path, map_location="cuda:0")
        def policy(obs):
            action = body.forward(obs)
            return action
        return policy
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "wandb")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")
            
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        # -------------------  For debugging  ------------------- #
        # x_vel=0.5
        # y_vel=0.0
        self.env.commands[:, 0] = 0.0 # init x_vel
        self.env.commands[:, 1] = 0.0 # init y_vel
        # lin_vel_x = [-0.0, 0.0] # min max [m/s]
        # lin_vel_y = [-0.0, 0.0]   # min max [m/s]
        lin_vel_x = [-0.8, 0.8] # min max [m/s]
        lin_vel_y = [-0.5, 0.5]   # min max [m/s]
        yaw_vel=0.
        height=0.73
        self.env.commands[:, 2] = yaw_vel
        self.env.commands[:, 4] = height
        height =self.env.commands[:, 4]
        # -------------------  For debugging  ------------------- #
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for step_i in range(self.num_steps_per_env):
                    # -------------------  For debugging  ------------------- #
                    self.env.action_curriculum_ratio = 0.0

                    #### ------------------------ upper_body_actions ------------------ ####


                    # left_gts = obs[:,self.env.num_one_step_obs*5+37+27+27+14+1:self.env.num_one_step_obs*5+37+27+27+14+1+3]
                    # left_grs = obs[:,self.env.num_one_step_obs*5+37+27+27+14+1+3:self.env.num_one_step_obs*5+37+27+27+14+1+3+4]
                    # right_gts = obs[:,self.env.num_one_step_obs*5+37+27+27+14+1+3+4:self.env.num_one_step_obs*5+37+27+27+14+1+3+4+3]
                    # right_grs = obs[:,self.env.num_one_step_obs*5+37+27+27+14+1+3+4+3:self.env.num_one_step_obs*5+37+27+27+14+1+3+4+3+4]

                    left_gts = obs[:,self.env.num_one_step_obs*5+10+27*2+12:self.env.num_one_step_obs*5+10+27*2+12+3]
                    left_grs = obs[:,self.env.num_one_step_obs*5+10+27*2+12+3:self.env.num_one_step_obs*5+10+27*2+12+3+4]
                    right_gts = obs[:,self.env.num_one_step_obs*5+10+27*2+12+7:self.env.num_one_step_obs*5+10+27*2+12+7+3]
                    right_grs = obs[:,self.env.num_one_step_obs*5+10+27*2+12+7+3:self.env.num_one_step_obs*5+10+27*2+12+7+3+4]

                    torso_pos = self.env._get_torso_pose()
                    torso_height = torso_pos[:, 2]
                    
                    # 构造批量目标位姿
                    L_tf_targets = [
                        pin.SE3(
                            pin.Quaternion(
                                float(left_grs[j][3].cpu().numpy()),
                                float(left_grs[j][0].cpu().numpy()),
                                float(left_grs[j][1].cpu().numpy()),
                                float(left_grs[j][2].cpu().numpy()),
                            ),
                            np.array([
                                float(left_gts[j][0].cpu().numpy()),
                                float(left_gts[j][1].cpu().numpy()),
                                float(left_gts[j][2].cpu().numpy()) -torso_height[j].cpu().numpy() +0.05
                            ])
                        )
                        for j in range(obs.shape[0])
                    ]
                    # print(f"L_tf_targets: {L_tf_targets}")

                    R_tf_targets = [
                        pin.SE3(
                            pin.Quaternion(
                                float(right_grs[j][3].cpu().numpy()),
                                float(right_grs[j][0].cpu().numpy()),
                                float(right_grs[j][1].cpu().numpy()),
                                float(right_grs[j][2].cpu().numpy()),
                            ),
                            np.array([
                                float(right_gts[j][0].cpu().numpy()),
                                float(right_gts[j][1].cpu().numpy()),
                                float(right_gts[j][2].cpu().numpy())-torso_height[j].cpu().numpy() + 0.05
                            ])
                        )
                        for j in range(obs.shape[0])
                    ]

                    # 获取当前的关节角度和速度
                    current_lr_arm_motor_q = obs[:, self.env.num_one_step_obs*5+10+13:self.env.num_one_step_obs*5+10+27].view(-1, 14).cpu().numpy()
                    current_lr_arm_dq = obs[:, self.env.num_one_step_obs*5+37+13:self.env.num_one_step_obs*5+37+27].view(-1, 14).cpu().numpy() * 20

                    sol_qs = np.zeros((obs.shape[0], 14))

                    for j in range(obs.shape[0]):
                        try:
                            # Get single element from each batch
                            left_target = L_tf_targets[j].homogeneous
                            right_target = R_tf_targets[j].homogeneous
                            current_q = np.array(current_lr_arm_motor_q[j])  # Keep as 2D array
                            current_dq = np.array(current_lr_arm_dq[j])       # Keep as 2D array
                            
                            # Solve IK for this single element
                            q, tau = self.ik_policy.solve_ik(left_target, right_target, current_q, current_dq)
                            
                            # Store successful solution
                            sol_qs[j] = q
                            
                        except Exception as e:
                            print(f"Failed to solve IK for element {j}:", e)
                            sol_qs[j] = np.zeros(14)  # If failed, return zero solution

                    up_actions = sol_qs * 4


                    # 检查哪些环境的 start_flag 为 1
                    mask = self.env.start_flag == 1
                    # 将满足条件的环境的 up_actions 设置为零
                    up_actions[mask.cpu().numpy()] = np.zeros((mask.sum().item(), 14))
                    
                    up_actions = torch.tensor(up_actions, device=self.device, dtype=torch.float32)
                    upper_body_actions = up_actions # upper body actions
                    

                    ######## ============================== manager_actions ============================== #########

                    manager_actions  = self.alg.act(obs, critic_obs)
                    
                    
                    # print(f"manager_actions: {manager_actions}")
                    vx = manager_actions[:,-3].detach()
                    vy = manager_actions[:,-2].detach()
                    vyaw = manager_actions[:,-1].detach()
                    self.env.commands[:, 0] = torch.clamp(vx, lin_vel_x[0], lin_vel_x[1]) # x_vel
                    self.env.commands[:, 1] = torch.clamp(vy, lin_vel_y[0], lin_vel_y[1]) # y_vel
                    self.env.commands[:, 2] = torch.clamp(vyaw, -1.5, 1.5)   # yaw_vel
                    # self.env.commands[:, 2] = 0 
                    height = manager_actions[:, -4].detach() # height
                    # print(f"height: {height}")
                    # print(f"manager_actions:{manager_actions}")
                    self.env.commands[:, 4] = torch.clamp(height , min=0.28, max=0.74) # height
                    # self.env.commands[:, 4] = torch.clamp(height , min=0.73, max=0.743)

                    #### =------------------------ lower_body_actions ------------------ ####
        
                    # remove upper body obs
                    num_segments = obs.shape[1] // self.env.num_one_step_obs 
                    indices_to_keep = torch.cat([torch.arange(i * self.env.num_one_step_obs, (i + 1) * self.env.num_one_step_obs - 14*5) for i in range(num_segments)], dim=0)
                    lower_policy_obs = obs[:, indices_to_keep]
                    # print(f"lower_policy_obs.shape: {lower_policy_obs.shape}")
                    # print(f"lower_policy_obs: {lower_policy_obs[7]}")

                    
                    
                    lower_body_actions = self.lower_body_policy(lower_policy_obs)
                    lower_body_actions = torch.cat((lower_body_actions, torch.zeros((lower_body_actions.size(0), 1), device=lower_body_actions.device)), dim=1) # yaw_vel = 0.0
                    actions = torch.cat((lower_body_actions, upper_body_actions), dim=1)
                    actions = actions.to(torch.float32)

                    # Step 5 
                    for _ in range(5):
                        obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = self.env.step(actions)                
                        # remove upper body obs
                        num_segments = obs.shape[1] // self.env.num_one_step_obs 
                        indices_to_keep = torch.cat([torch.arange(i * self.env.num_one_step_obs, (i + 1) * self.env.num_one_step_obs - 14*5) for i in range(num_segments)], dim=0)
                        lower_policy_obs = obs[:, indices_to_keep]
                        
                        lower_body_actions = self.lower_body_policy(lower_policy_obs)
                        lower_body_actions = torch.cat((lower_body_actions, torch.zeros((lower_body_actions.size(0), 1), device=lower_body_actions.device)), dim=1) # yaw_vel = 0.0
                        actions = torch.cat((lower_body_actions, upper_body_actions), dim=1)
                        actions = actions.to(torch.float32)

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = termination_privileged_obs.to(self.device)

                    next_critic_obs = critic_obs.clone().detach()
                    next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()

                    self.alg.process_env_step(rewards, dones, infos, next_critic_obs)
                
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
                
            # mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss, mean_actor_sym_loss, mean_critic_sym_loss = self.alg.update()
            if self.cfg["algorithm_class_name"] == "ReachPPO":
                mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss, mean_actor_sym_loss, mean_critic_sym_loss = self.alg.update()
                mean_bc_loss = 0
                self.env.it = max(it-200,0) ## for classical PPO after BC
            elif self.cfg["algorithm_class_name"] == "ReachDAgger":
                mean_bc_loss, mean_estimation_loss, mean_swap_loss, mean_actor_sym_loss, mean_critic_sym_loss =self.alg.update()
                # mean_bc_loss, mean_estimation_loss, mean_swap_loss, mean_actor_sym_loss, mean_critic_sym_loss = 0,0,0,0,0
                
                mean_value_loss = 0
                mean_surrogate_loss = 0
                # mean_DAgger_loss = 0
                self.env.it = it
                # self.alg.add_dagger_data()

                # self.alg.storage.clear()
                
                if it % 10 == 0:
                    self.alg.add_dagger_data()
                    mean_DAgger_loss = self.alg.update_DAgger()
                # mean_DAgger_loss = self.alg.update_DAgger()
            elif self.cfg["algorithm_class_name"] == "ReachPPOBC":
                # self.env.it = max(it-800,0)  ## for PPO+BC after BC
                self.env.it = max(it-100,0)
                mean_value_loss, mean_surrogate_loss, mean_bc_loss, mean_estimation_loss, mean_swap_loss, mean_actor_sym_loss, mean_critic_sym_loss = self.alg.update()
               
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
            self.current_learning_iteration = it
        
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std[0:10].mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/bc_loss', locs['mean_bc_loss'], locs['it'])
        self.writer.add_scalar('Loss/DAgger_loss', locs['mean_DAgger_loss'], locs['it'])
        self.writer.add_scalar('Loss/Estimation Loss', locs['mean_estimation_loss'], locs['it'])
        self.writer.add_scalar('Loss/Swap Loss', locs['mean_swap_loss'], locs['it'])
        self.writer.add_scalar('Loss/Actor Sym Loss', locs['mean_actor_sym_loss'], locs['it'])
        self.writer.add_scalar('Loss/Critic Sym Loss', locs['mean_critic_sym_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
                self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'BC loss:':>{pad}} {locs['mean_bc_loss']:.4f}\n"""
                            f"""{'DAgger loss:':>{pad}} {locs['mean_DAgger_loss']:.4f}\n"""
                          f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                          f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                          f"""{'Mean actor sym loss:':>{pad}} {locs['mean_actor_sym_loss']:.4f}\n"""
                          f"""{'Mean critic sym loss:':>{pad}} {locs['mean_critic_sym_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'BC loss:':>{pad}} {locs['mean_bc_loss']:.4f}\n"""
                          f"""{'DAgger loss:':>{pad}} {locs['mean_DAgger_loss']:.4f}\n"""
                          f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                          f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                          f"""{'Mean actor sym loss:':>{pad}} {locs['mean_actor_sym_loss']:.4f}\n"""
                          f"""{'Mean critic sym loss:':>{pad}} {locs['mean_critic_sym_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'estimator_optimizer_state_dict': self.alg.actor_critic.estimator.optimizer.state_dict(),
            'iter': self.current_learning_iteration + 1,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=False):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.actor_critic.estimator.optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def train_mode(self):
        self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)