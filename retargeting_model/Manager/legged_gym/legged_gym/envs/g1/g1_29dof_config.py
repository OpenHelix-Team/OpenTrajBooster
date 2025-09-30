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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.75] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,         
            "waist_yaw_joint":0.,
            "waist_roll_joint": 0.,
            "waist_pitch_joint": 0.,
            "left_shoulder_pitch_joint": 0.,
            "left_shoulder_roll_joint": 0.,
            "left_shoulder_yaw_joint": 0.,
            "left_elbow_joint": 0.,
            "left_wrist_roll_joint": 0.,
            "left_wrist_pitch_joint": 0.,
            "left_wrist_yaw_joint": 0.,
            "left_hand_index_0_joint": 0.,
            "left_hand_index_1_joint": 0.,
            "left_hand_middle_0_joint": 0.,
            "left_hand_middle_1_joint": 0.,
            "left_hand_thumb_0_joint": 0.,
            "left_hand_thumb_1_joint": 0.,
            "left_hand_thumb_2_joint": 0.,
            "right_shoulder_pitch_joint": 0.,
            "right_shoulder_roll_joint": -0.,#-0.3
            "right_shoulder_yaw_joint": 0.,
            "right_elbow_joint": 0.,#0.8
            "right_wrist_roll_joint": 0.,
            "right_wrist_pitch_joint": 0.,
            "right_wrist_yaw_joint": 0.,
            "right_hand_index_0_joint": 0.,
            "right_hand_index_1_joint": 0.,
            "right_hand_middle_0_joint": 0.,
            "right_hand_middle_1_joint": 0.,
            "right_hand_thumb_0_joint": 0.,
            "right_hand_thumb_1_joint": 0.,
            "right_hand_thumb_2_joint": 0.,
           
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'M'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 150,
                     'hip_roll': 150,
                     'hip_pitch': 150,
                     'knee': 300,
                     'ankle': 40,
                     
                     "waist": 300,
                     "shoulder": 80,
                     "wrist": 40,
                     "elbow": 80,
                     "hand": 10
                     }  # [N*m/rad]
        
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,

                     "waist": 5,
                     "shoulder": 3,
                     "wrist": 1.5,
                     "elbow": 3,
                     "hand": 2
                     }   # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0

    class commands( LeggedRobotCfg.commands ):
        curriculum = False # NOTE set True later
        max_curriculum = 1.4
        num_commands = 5 # lin_vel_x, lin_vel_y, ang_vel_yaw, heading, height, orientation
        resampling_time = 4. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        heading_to_ang_vel = False
        class ranges( LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.8, 1.2] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.8, 0.8]    # min max [rad/s]
            heading = [-3.14, 3.14]
            height = [-0.5, 0.0]

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        left_foot_name = "left_foot"
        right_foot_name = "right_foot"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ['torso']
        curriculum_joints = []
        left_leg_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_pitch_joint']
        right_leg_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_pitch_joint']
        left_hip_joints = ['left_hip_roll_joint', "left_hip_pitch_joint", "left_hip_yaw_joint"]
        right_hip_joints = ['right_hip_roll_joint', "right_hip_pitch_joint", "right_hip_yaw_joint"]
        hip_pitch_joints = ['right_hip_pitch_joint', 'left_hip_pitch_joint']
        knee_joints = ['left_knee_joint', 'right_knee_joint']
        ankle_joints = ["left_ankle_roll_joint", "right_ankle_roll_joint"]
        upper_body_link = "torso_link"
        imu_link = "imu_in_pelvis"
        knee_names = ["left_knee_link", "left_hip_yaw_link", "right_knee_link", "right_hip_yaw_link"]
        self_collision = 1
        flip_visual_attachments = False
        ankle_sole_distance = 0.02

        
    class domain_rand(LeggedRobotCfg.domain_rand):
        
        use_random = False
        
        randomize_joint_injection = use_random
        joint_injection_range = [-0.05, 0.05]
        
        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.05, 0.05]

        randomize_payload_mass = use_random
        payload_mass_range = [-5, 10]
        
        hand_payload_mass_range = [-0.1, 0.3]

        randomize_com_displacement = False
        com_displacement_range = [-0.1, 0.1]
        
        randomize_body_displacement = use_random
        body_displacement_range = [-0.1, 0.1]

        randomize_link_mass = use_random
        link_mass_range = [0.8, 1.2]
        
        randomize_friction = use_random
        friction_range = [0.1, 3.0]
        
        randomize_restitution = use_random
        restitution_range = [0.0, 1.0]
        
        randomize_kp = use_random
        kp_range = [0.9, 1.1]
        
        randomize_kd = use_random
        kd_range = [0.9, 1.1]
        
        randomize_initial_joint_pos = use_random
        initial_joint_pos_scale = [0.8, 1.2]
        initial_joint_pos_offset = [-0.1, 0.1]
        
        push_robots = use_random
        push_interval_s = 4
        upper_interval_s = 1
        max_push_vel_xy = 0.5
        
        init_upper_ratio = 0.
        delay = use_random

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            tracking_x_vel = 1.5
            tracking_y_vel = 1.
            tracking_ang_vel = 2.
            lin_vel_z = -0.5
            ang_vel_xy = -0.025
            orientation = -1.5
            action_rate = -0.01
            tracking_base_height = 2.
            deviation_hip_joint = -0.2
            deviation_ankle_joint = -0.5
            deviation_knee_joint = -0.75
            dof_acc = -2.5e-7
            dof_pos_limits = -2.
            feet_air_time = 0.05
            feet_clearance = -0.25
            feet_distance_lateral = 0.5
            knee_distance_lateral = 1.0
            feet_ground_parallel = -2.0
            feet_parallel = -3.0
            smoothness = -0.05
            joint_power = -2e-5
            feet_stumble = -1.5
            torques = -2.5e-6
            dof_vel = -1e-4
            dof_vel_limits = -2e-3
            torque_limits = -0.1
            no_fly = 0.75
            joint_tracking_error = -0.1
            feet_slip = -0.25
            feet_contact_forces = -0.00025
            contact_momentum = 2.5e-4
            action_vanish = -1.0
            stand_still = -0.15    
        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.975
        soft_dof_vel_limit = 0.80
        soft_torque_limit = 0.95
        base_height_target = 0.74
        max_contact_force = 400.
        least_feet_distance = 0.2
        least_feet_distance_lateral = 0.2
        most_feet_distance_lateral = 0.35
        most_knee_distance_lateral = 0.35
        least_knee_distance_lateral = 0.2
        clearance_height_target = 0.14
        
    class env( LeggedRobotCfg.rewards ):
        num_envs = 4096
        num_actions = 12
        num_dofs = 27
        num_one_step_observations = 2 * num_dofs + 10 + num_actions # 54 + 10 + 12 = 22 + 54 = 76
        num_one_step_privileged_obs = num_one_step_observations + 3
        num_actor_history = 6
        num_critic_history = 1
        num_observations = num_actor_history * num_one_step_observations
        num_privileged_obs = num_critic_history * num_one_step_privileged_obs
        action_curriculum = True
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20
        
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'

    class noise( LeggedRobotCfg.terrain ):
        add_noise = True
        noise_level = 1.0
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            dof_pos = 0.02
            dof_vel = 2.0
            lin_vel = 0.1
            ang_vel = 0.5
            gravity = 0.05
            height_measurement = 0.1

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        use_flip = False
        entropy_coef = 0.01
        symmetry_scale = 1.0
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        save_interval = 200
        num_steps_per_env = 50
        max_iterations = 100000
        run_name = ''
        experiment_name = ''
        wandb_project = ""
        logger = "wandb"        
        # logger = "tensorboard"        
        wandb_user = "" # enter your own wandb user name here