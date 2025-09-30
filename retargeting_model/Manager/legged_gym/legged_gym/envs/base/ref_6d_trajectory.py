import numpy as np
import torch
import random
import joblib
import random
# from legged_gym.envs.utils.flags import flags

from scipy.spatial.transform import Rotation as sRot
# from legged_gym.envs.poselib.poselib.skeleton.skeleton3d import SkeletonMotion
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R
import os


class TrajGenerator6D():

    def __init__(self, num_envs, episode_dur, num_verts, device, starting_still_dt=1.5):

        self._device = device
        self._dt = episode_dur / (num_verts - 1)
        self._sharp_turn_prob = 0.1
        self._dtheta_max = 0.1
        self._speed_min =  0.0

        

        self.left_gts_flat = torch.zeros((num_envs * num_verts, 3), dtype=torch.float32, device=self._device)
        self.right_gts_flat = torch.zeros((num_envs * num_verts, 3), dtype=torch.float32, device=self._device)
        self.left_grs_flat = torch.zeros((num_envs * num_verts, 4), dtype=torch.float32, device=self._device)
        self.right_grs_flat = torch.zeros((num_envs * num_verts, 4), dtype=torch.float32, device=self._device)
        self.left_gavs_flat = torch.zeros((num_envs * num_verts, 3), dtype=torch.float32, device=self._device)
        self.right_gavs_flat = torch.zeros((num_envs * num_verts, 3), dtype=torch.float32, device=self._device)
        self.left_gvs_flat = torch.zeros((num_envs * num_verts, 3), dtype=torch.float32, device=self._device)
        self.right_gvs_flat = torch.zeros((num_envs * num_verts, 3), dtype=torch.float32, device=self._device)
        self.traj_lengths = torch.zeros(num_envs, dtype=torch.int32, device=self._device)
        
        self.left_gts = self.left_gts_flat.view((num_envs, num_verts, 3))
        self.right_gts = self.right_gts_flat.view((num_envs, num_verts, 3))
        self.left_grs = self.left_grs_flat.view((num_envs, num_verts, 4))
        self.right_grs = self.right_grs_flat.view((num_envs, num_verts, 4))
        self.left_gavs = self.left_gavs_flat.view((num_envs, num_verts, 3))
        self.right_gavs = self.right_gavs_flat.view((num_envs, num_verts, 3))
        self.left_gvs = self.left_gvs_flat.view((num_envs, num_verts, 3))
        self.right_gvs = self.right_gvs_flat.view((num_envs, num_verts, 3))

        self.target_command_heights = torch.zeros(num_envs, num_verts, 1, dtype=torch.float32, device=self._device)
        self.target_command_vel_x = torch.zeros(num_envs, num_verts, 1, dtype=torch.float32, device=self._device)
        self.target_command_vel_y = torch.zeros(num_envs, num_verts, 1, dtype=torch.float32, device=self._device)

        self.ref_target_delta_command_heights = torch.zeros(num_envs, num_verts, 1, dtype=torch.float32, device=self._device)
        self.target_delta_command_heights = torch.zeros(num_envs, num_verts, 1, dtype=torch.float32, device=self._device)
        self.target_delta_command_x = torch.zeros(num_envs, num_verts, 1, dtype=torch.float32, device=self._device)
        self.target_delta_command_y = torch.zeros(num_envs, num_verts, 1, dtype=torch.float32, device=self._device)

        self.eval = False
        self.dir = {} 

        # self.gts = self.gts_flat.view((num_envs, num_verts, 3))
        # self.grs = self.grs_flat.view((num_envs, num_verts, 4))
        # self.gavs = self.gavs_flat.view((num_envs, num_verts, 3))
        # self.gvs = self.gvs_flat.view((num_envs, num_verts, 3))

        
        # env_ids = torch.arange(self.get_num_envs(), dtype=int)

        # self.heading = torch.zeros(num_envs, 1)
        # self.raise_time = 0.25 # Time for going stright up. 
        real_traj = True
        if real_traj:
            data_path = 'data/all2_episodes_hand_trajs_converted.pkl'
            
            self.data = joblib.load(data_path)
            self.total_data_size = len(self.data) 
            print(f"Total trajectory data size: {self.total_data_size}")
            self.indices = list(range(self.total_data_size))  # 全部数据的索引
            random.shuffle(self.indices)  # 随机打乱索引
            self.current_index = 0  # 当前索引位置

            char_path = 'data/trajectory_Hello.pkl'
            self.char_data = joblib.load(char_path)

        return
        
    def sample_dtheat_dspeed(self, n, it):
        self._speed_max = min((it//200)*0.002, 0.02)
        self._accel_max = min((it//200)*0.002, 0.08)

        # random sample speed_max[n] range:[0,self._speed_max]
        speed_max2 = torch.rand([n], device=self._device) * self._speed_max

        
        num_verts = self.get_num_verts()
        dtheta = 2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0  # Sample the angles at each waypoint
        dtheta *= self._dtheta_max * self._dt

        dtheta_sharp = np.pi * (2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0)  # Sharp Angles Angle
        sharp_probs = self._sharp_turn_prob * torch.ones_like(dtheta)
        sharp_mask = torch.bernoulli(sharp_probs) == 1.0
        dtheta[sharp_mask] = dtheta_sharp[sharp_mask]

        dtheta[:, 0] = np.pi * (2 * torch.rand([n], device=self._device) - 1.0)  # Heading

        dspeed = 2 * torch.rand([n, num_verts - 1], device=self._device) - 1.0
        dspeed *= self._accel_max * self._dt
        dspeed[:, 0] = (speed_max2 - self._speed_min) * torch.rand([n], device=self._device) + self._speed_min  # Speed

        speed = torch.zeros_like(dspeed)
        speed[:, 0] = dspeed[:, 0]
        for i in range(1, dspeed.shape[-1]):
            speed[:, i] = torch.clip(speed[:, i - 1] + dspeed[:, i], self._speed_min, 0.01)

        dtheta = torch.cumsum(dtheta, dim=-1)
        return speed, dspeed, dtheta

    def sample_constrained_z(self, batch, num_verts, max_h = 0.25, hard_level =1.0):
        # divider = 10
        divider =int(num_verts//5 // hard_level) 
        num_sampling_points = batch * num_verts
        x = np.arange(0, num_sampling_points//divider, 1)
        z = np.random.uniform(-max_h, max_h, num_sampling_points//divider)
        xs = np.arange(0, num_sampling_points//divider, 1/divider)
        interpolator = PchipInterpolator(x, z)
        zs = interpolator(xs)
        return zs
    
    def reset(self, env_ids, init_left_pos, init_right_pos, init_left_quat, init_right_quat):
        starting_still_frames = 10
        n = len(env_ids)
        num_verts = self.get_num_verts()
        if (n > 0):
            speed_left, dspeed_left, dtheta_left = self.sample_dtheat_dspeed(n)
            speed_right, dspeed_right, dtheta_right = self.sample_dtheat_dspeed(n)
            seg_len_left = speed_left * self._dt
            seg_len_right = speed_right * self._dt

            dpos_left = torch.stack([torch.cos(dtheta_left), -torch.sin(dtheta_left), torch.zeros_like(dtheta_left)], dim=-1) # Z direction
            dpos_left *= seg_len_left.unsqueeze(-1)
            vert_pos_left = torch.cumsum(dpos_left, dim=-2)

            x_min, x_max = 0.1, 0.4  # 设置 x 的范围
            y_min, y_max = 0.0, 0.3  # 设置 y 的范围
            vert_pos_left[..., 0] = torch.clamp(vert_pos_left[..., 0], x_min, x_max)  # 限制 x 范围
            vert_pos_left[..., 1] = torch.clamp(vert_pos_left[..., 1], y_min, y_max)  # 限制 y 范围
            # vert_pos_right[..., 0] = torch.clamp(vert_pos_right[..., 0], x_min, x_max)  # 限制 x 范围
            # vert_pos_right[..., 1] = torch.clamp(vert_pos_right[..., 1], -y_max, y_min)  # 限制 y 范围

            dpos_right = torch.stack([torch.cos(dtheta_right), -torch.sin(dtheta_right), torch.zeros_like(dtheta_right)], dim=-1) # Z direction
            dpos_right *= seg_len_right.unsqueeze(-1)
            vert_pos_right = torch.cumsum(dpos_right, dim=-2)

            all_zs_left = self.sample_constrained_z(n, num_verts)
            all_zs_left = all_zs_left[:(n * (num_verts - 1))].reshape(n, num_verts - 1)
            vert_pos_left[..., 2] = torch.from_numpy(all_zs_left).float()

            all_zs_right = self.sample_constrained_z(n, num_verts)
            all_zs_right = all_zs_right[:(n * (num_verts - 1))].reshape(n, num_verts - 1)
            vert_pos_right[..., 2] = torch.from_numpy(all_zs_right).float()

            ending_rot_left = torch.from_numpy(sRot.random(n).as_quat()).to(init_left_pos)
            ending_rot_right = torch.from_numpy(sRot.random(n).as_quat()).to(init_right_pos)

            vert_pos_left = vert_pos_left - (vert_pos_left[:, 0:1, :] - init_left_pos[:, None, :])
            vert_pos_right = vert_pos_right - (vert_pos_right[:, 0:1, :] - init_right_pos[:, None, :])

            vert_pos_left[..., 2] = torch.clamp(vert_pos_left[..., 2], 0.7, 1.4)
            vert_pos_right[..., 2] = torch.clamp(vert_pos_right[..., 2], 0.7, 1.4)

            # 确保轨迹点之间的距离在 [0.15m, 1m] 之间
            distances = torch.norm(vert_pos_left - vert_pos_right, dim=-1, keepdim=True)

            # 超过 1m 的部分：缩短距离，使其在最大范围内
            too_far = distances > 0.9
            scaling_factor_far = 0.9 / distances.clamp(min=0.9)
            vert_pos_right = torch.where(too_far, vert_pos_left + (vert_pos_right - vert_pos_left) * scaling_factor_far, vert_pos_right)

            # 小于 0.15m 的部分：拉远距离，使其在最小范围内
            too_close = distances < 0.15
            scaling_factor_close = 0.15 / distances.clamp(max=0.15)
            vert_pos_right = torch.where(too_close, vert_pos_left + (vert_pos_right - vert_pos_left) * scaling_factor_close, vert_pos_right)

            # ----------------------------------- attention ----------------------------------- #
            # Ensure the distance constraint between the two trajectories, maybe remove soon!
            # ----------------------------------- attention ----------------------------------- #
            # for i in range(vert_pos_left.shape[1]):
            #     for j in range(vert_pos_left.shape[0]):  # 遍历每个样本
            #         dist = torch.norm(vert_pos_left[j, i, :] - vert_pos_right[j, i, :], dim=-1)
            #         while dist < 0.15 or dist > 1.0:
            #             adjustment = vert_pos_left[j, i, :] - vert_pos_left[j, i-1, :]
            #             vert_pos_right[j, i, :] = vert_pos_right[j, i-1, :] + adjustment  # right hand follows left hand
            #             dist = torch.norm(vert_pos_left[j, i, :] - vert_pos_right[j, i, :], dim=-1)

            self.left_gts[env_ids, :starting_still_frames] = init_left_pos[..., None, :] # first starting_still_frames Frames should be still.
            self.right_gts[env_ids, :starting_still_frames] = init_right_pos[..., None, :] # first starting_still_frames Frames should be still.

            self.left_gts[env_ids, (starting_still_frames):] = vert_pos_left[:, :-(starting_still_frames - 1)]
            self.right_gts[env_ids, (starting_still_frames):] = vert_pos_right[:, :-(starting_still_frames - 1)]

            num_verts = self.get_num_verts()
            slerp_weights = (torch.arange(num_verts - starting_still_frames) / (num_verts - starting_still_frames)).repeat(n, 1).to(init_left_pos)
            slerp_weights = torch.cat([torch.zeros([n, starting_still_frames]).to(init_left_pos), slerp_weights], dim=-1)

            self.left_grs[env_ids] = slerp(init_left_quat[:, None].repeat(1, num_verts, 1).view(-1, 4), ending_rot_left[:, None].repeat(1, num_verts, 1).view(-1, 4), slerp_weights.view(-1, 1)).view(n, self.get_num_verts(), -1)
            self.right_grs[env_ids] = slerp(init_right_quat[:, None].repeat(1, num_verts, 1).view(-1, 4), ending_rot_right[:, None].repeat(1, num_verts, 1).view(-1, 4), slerp_weights.view(-1, 1)).view(n, self.get_num_verts(), -1)

            # self.left_gavs[env_ids] = SkeletonMotion._compute_velocity(p=self.left_gts[env_ids, :, None, :].cpu(), time_delta=self._dt, guassian_filter=False)[:, :, 0].to(self._device)
            # self.right_gavs[env_ids] = SkeletonMotion._compute_velocity(p=self.right_gts[env_ids, :, None, :].cpu(), time_delta=self._dt, guassian_filter=False)[:, :, 0].to(self._device)
            # self.left_gvs[env_ids] = SkeletonMotion._compute_angular_velocity(r=self.left_grs[env_ids, :, None, :].cpu(), time_delta=self._dt, guassian_filter=False)[:, :, 0].to(self._device)
            # self.right_gvs[env_ids] = SkeletonMotion._compute_angular_velocity(r=self.right_grs[env_ids, :, None, :].cpu(), time_delta=self._dt, guassian_filter=False)[:, :, 0].to(self._device)
        return

    def load_retarget_traj(self, env_ids):
        n = len(env_ids)
        # create a directory for each env_id
        for i in env_ids:
            env_id = int(i)  # 将 tensor 转为 int
            source_path = self.data[env_id]['source']
            sub_dir = os.path.dirname(source_path)
            full_dir = os.path.join('your retarget data', sub_dir)
            os.makedirs(full_dir, exist_ok=True)
            self.dir[env_id] = full_dir 
        
        left_hand_pos = [
            torch.tensor(self.data[i]["left_hand_pos"], dtype=torch.float32).to(self._device)
            for i in env_ids
        ]
        left_hand_quat = [
            torch.tensor(self.data[i]["left_hand_quat"], dtype=torch.float32).to(self._device)
            for i in env_ids
        ]
        right_hand_pos = [
            torch.tensor(self.data[i]["right_hand_pos"], dtype=torch.float32).to(self._device)
            for i in env_ids
        ]
        right_hand_quat = [
            torch.tensor(self.data[i]["right_hand_quat"], dtype=torch.float32).to(self._device)
            for i in env_ids
        ]

        return left_hand_pos, left_hand_quat, right_hand_pos, right_hand_quat

    def load_real_traj(self, env_ids):
        n = len(env_ids)
        if self.current_index + n > self.total_data_size:
            # 如果剩余数据不足以满足需求，重新打乱并从头开始
            random.shuffle(self.indices)
            self.current_index = 0

        # 从未使用过的数据中选择
        selected_indices = self.indices[self.current_index:self.current_index + n]
        self.current_index += n

        # 提取对应的数据
        selected_trajs = [self.data[idx] for idx in selected_indices]

        # 随机生成起始索引
        self.current_index = random.randint(0, self.total_data_size - 1)

        # 如果当前索引加上需要的数量超过总数据大小，则从头开始循环
        selected_indices = []
        for _ in range(n):
            selected_indices.append(self.indices[self.current_index])
            self.current_index = (self.current_index + 1) % self.total_data_size  # 循环队列

        # 提取对应的数据
        selected_trajs = [self.data[idx] for idx in selected_indices]

        # 将数据转换为 PyTorch 张量列表
        left_hand_pos = [
            torch.tensor(traj["left_hand_pos"], dtype=torch.float32).to(self._device)
            for traj in selected_trajs
        ]

        left_hand_quat = [
            torch.tensor(traj["left_hand_quat"], dtype=torch.float32).to(self._device)
            for traj in selected_trajs
        ]

        right_hand_pos = [
            torch.tensor(traj["right_hand_pos"], dtype=torch.float32).to(self._device)
            for traj in selected_trajs
        ]

        right_hand_quat = [
            torch.tensor(traj["right_hand_quat"], dtype=torch.float32).to(self._device)
            for traj in selected_trajs
        ]
            
        return left_hand_pos, left_hand_quat, right_hand_pos, right_hand_quat
    

    def load_char_traj(self, env_ids):
        x, y, z  = self.char_data
        left_hand_pos = [x,y,z+0.3]
        left_hand_pos =torch.tensor(left_hand_pos, dtype=torch.float32).to(self._device)
        right_hand_pos = [x,y-0.3,z+0.3]
        right_hand_pos =torch.tensor(right_hand_pos, dtype=torch.float32).to(self._device)


        return left_hand_pos, right_hand_pos
    
    def retarget_reset(self, env_ids, init_left_pos, init_right_pos, init_left_quat, init_right_quat, init_root_pos, it):
        n = len(env_ids)
        starting_still_frames = 50
        transition_frames = 50

        left_hand_pos, left_hand_quat, right_hand_pos, right_hand_quat = self.load_retarget_traj(env_ids)

        for i in range(len(left_hand_pos)):
            left_hand_pos[i] = left_hand_pos[i] + init_root_pos[i:i+1, :]
            right_hand_pos[i] = right_hand_pos[i] + init_root_pos[i:i+1, :]

        # Get the number of points in the real trajectory
        num_real_points_list = [traj.shape[0] for traj in left_hand_pos]
        # 30 Hz -> 50 Hz
        real_traj_frames_list = [int(num_real_points / 3 * 5) for num_real_points in num_real_points_list]
        # Generate the first part: starting_still_frames
        self.left_gts[env_ids, :starting_still_frames] = init_left_pos[..., None, :]  # Keep initial position
        self.right_gts[env_ids, :starting_still_frames] = init_right_pos[..., None, :]  # Keep initial position
        self.left_grs[env_ids, :starting_still_frames] = init_left_quat[..., None, :]  # Keep initial rotation
        self.right_grs[env_ids, :starting_still_frames] = init_right_quat[..., None, :]  # Keep initial rotation

        # Generate the second part: transition_frames
        # Interpolate positions
        for i, env_id in enumerate(env_ids):
            
            # Generate interpolation weights
            weights = torch.linspace(0, 1, transition_frames, device=self._device).unsqueeze(-1)

            # Interpolate left hand position
            start_left_pos = init_left_pos[i:i+1, :]
            end_left_pos = left_hand_pos[i][0:1, :]
            interpolated_left_pos = (1 - weights) * start_left_pos + weights * end_left_pos

            # Interpolate right hand position
            start_right_pos = init_right_pos[i:i+1, :]
            end_right_pos = right_hand_pos[i][0:1, :]
            interpolated_right_pos = (1 - weights) * start_right_pos + weights * end_right_pos

            # Fill interpolated positions into trajectory
            self.left_gts[env_id, starting_still_frames:starting_still_frames + transition_frames] = interpolated_left_pos
            self.right_gts[env_id, starting_still_frames:starting_still_frames + transition_frames] = interpolated_right_pos

            # Interpolate rotations using SLERP
            start_left_quat = init_left_quat[i:i+1, :]
            end_left_quat = left_hand_quat[i][0:1, :]
            interpolated_left_quat = slerp(start_left_quat, end_left_quat, weights)

            start_right_quat = init_right_quat[i:i+1, :]
            end_right_quat = right_hand_quat[i][0:1, :]
            interpolated_right_quat = slerp(start_right_quat, end_right_quat, weights)

            # Fill interpolated rotations into trajectory
            self.left_grs[env_id, starting_still_frames:starting_still_frames + transition_frames] = interpolated_left_quat
            self.right_grs[env_id, starting_still_frames:starting_still_frames + transition_frames] = interpolated_right_quat
        
        # Generate the third part: real trajectory
        left_hand_pos_resampled = [
            torch.nn.functional.interpolate(
                traj.T.unsqueeze(0),  # Shape: (1, 3, num_real_points)
                size=real_traj_frames_list[i],  # 每条轨迹的目标帧数
                mode='linear',
                align_corners=True
            ).squeeze(0).T  # Shape: (real_traj_frames, 3)
            for i, traj in enumerate(left_hand_pos)
        ]

        right_hand_pos_resampled = [
            torch.nn.functional.interpolate(
                traj.T.unsqueeze(0),  # Shape: (1, 3, num_real_points)
                size=real_traj_frames_list[i],  # 每条轨迹的目标帧数
                mode='linear',
                align_corners=True
            ).squeeze(0).T  # Shape: (real_traj_frames, 3)
            for i, traj in enumerate(right_hand_pos)
        ]

        left_hand_quat_resampled = [
            self.interpolate_quaternions_via_axis_angle(traj.unsqueeze(0), real_traj_frames_list[i])[0]
            for i, traj in enumerate(left_hand_quat)
        ]

        right_hand_quat_resampled = [
            self.interpolate_quaternions_via_axis_angle(traj.unsqueeze(0), real_traj_frames_list[i])[0]
            for i, traj in enumerate(right_hand_quat)
        ]

        start_idx = starting_still_frames + transition_frames  # 计算起始索引

        for i, env_id in enumerate(env_ids):
            end_idx = start_idx + real_traj_frames_list[i] -2  # 根据每条轨迹的长度动态计算结束索引

            # 插入位置轨迹
            self.left_gts[env_id, start_idx:end_idx] = left_hand_pos_resampled[i][2:]
            self.right_gts[env_id, start_idx:end_idx] = right_hand_pos_resampled[i][2:]

            # 插入旋转轨迹
            self.left_grs[env_id, start_idx:end_idx] = left_hand_quat_resampled[i][2:]
            self.right_grs[env_id, start_idx:end_idx] = right_hand_quat_resampled[i][2:]

            # clamp z
            self.left_gts[env_id, start_idx:end_idx, 2] = self.left_gts[env_id, start_idx:end_idx, 2]
            self.right_gts[env_id, start_idx:end_idx, 2] = self.right_gts[env_id, start_idx:end_idx, 2]

            self.traj_lengths[env_id] = end_idx  # 记录每个环境的轨迹长度
        
        return




        



    def char_reset(self, env_ids, init_left_pos, init_right_pos, init_left_quat, init_right_quat, init_root_pos, it):
        left_hand_pos, right_hand_pos = self.load_char_traj(env_ids)
        print(f"left_hand_pos.shape: {left_hand_pos.shape}")
        # shape： ([3, traj_len]) -> ([batch, traj_len, 3])
        left_hand_pos = left_hand_pos.T.unsqueeze(0).repeat(len(env_ids), 1, 1)
        right_hand_pos = right_hand_pos.T.unsqueeze(0).repeat(len(env_ids), 1, 1)

        # left_hand_quat = init_left_quat
        left_hand_quat = torch.zeros((1,4540,4), dtype=torch.float32, device=self._device)+ init_left_quat
        right_hand_quat = torch.zeros((1,4540,4), dtype=torch.float32, device=self._device)+ init_right_quat

        print(f"left_hand_pos.shape: {left_hand_pos.shape}")
        print(f"left_hand_quat.shape: {left_hand_quat.shape}")

        num_real_points_list = [traj.shape[0] for traj in left_hand_pos]
        # 30 Hz -> 50 Hz
        # real_traj_frames_list = [int(num_real_points / 3 * 5) for num_real_points in num_real_points_list]
        real_traj_frames_list = [int(num_real_points / 2) for num_real_points in num_real_points_list]

        left_hand_pos_resampled = [
            torch.nn.functional.interpolate(
                traj.T.unsqueeze(0),  # Shape: (1, 3, num_real_points)
                size=real_traj_frames_list[i],  # 每条轨迹的目标帧数
                mode='linear',
                align_corners=True
            ).squeeze(0).T  # Shape: (real_traj_frames, 3)
            for i, traj in enumerate(left_hand_pos)
        ]

        right_hand_pos_resampled = [
            torch.nn.functional.interpolate(
                traj.T.unsqueeze(0),  # Shape: (1, 3, num_real_points)
                size=real_traj_frames_list[i],  # 每条轨迹的目标帧数
                mode='linear',
                align_corners=True
            ).squeeze(0).T  # Shape: (real_traj_frames, 3)
            for i, traj in enumerate(right_hand_pos)
        ]

        left_hand_quat_resampled = [
            self.interpolate_quaternions_via_axis_angle(traj.unsqueeze(0), real_traj_frames_list[i])[0]
            for i, traj in enumerate(left_hand_quat)
        ]

        right_hand_quat_resampled = [
            self.interpolate_quaternions_via_axis_angle(traj.unsqueeze(0), real_traj_frames_list[i])[0]
            for i, traj in enumerate(right_hand_quat)
        ]

        left_hand_pos_resampled = torch.stack(left_hand_pos_resampled, dim=0)  # 将列表堆叠成张量
        right_hand_pos_resampled = torch.stack(right_hand_pos_resampled, dim=0)
        left_hand_quat_resampled = torch.stack(left_hand_quat_resampled, dim=0)
        right_hand_quat_resampled = torch.stack(right_hand_quat_resampled, dim=0)

        self.left_gts[env_ids, :real_traj_frames_list[0]] = left_hand_pos_resampled
        self.right_gts[env_ids, :real_traj_frames_list[0]] = right_hand_pos_resampled
        self.left_grs[env_ids, :real_traj_frames_list[0]] = left_hand_quat_resampled
        self.right_grs[env_ids, :real_traj_frames_list[0]] = right_hand_quat_resampled

        for env_id in env_ids:
            # 计算每个环境的轨迹长度
            self.traj_lengths[env_id] = real_traj_frames_list[0]



    def real_reset(self, env_ids, init_left_pos, init_right_pos, init_left_quat, init_right_quat, init_root_pos, it):
        starting_still_frames = 50
        transition_frames = 50
        n = len(env_ids)
        init_root_pos[:,2] = 0.0
        # num_verts = self.get_num_verts()

        if n > 0:
            # Load real trajectory data
            left_hand_pos, left_hand_quat, right_hand_pos, right_hand_quat = self.load_real_traj(env_ids)

            for i in range(len(left_hand_pos)):
                left_hand_pos[i] = left_hand_pos[i] + init_root_pos[i:i+1, :]
                right_hand_pos[i] = right_hand_pos[i] + init_root_pos[i:i+1, :]
            
            # Get the number of points in the real trajectory
            num_real_points_list = [traj.shape[0] for traj in left_hand_pos]
            # 30 Hz -> 50 Hz
            real_traj_frames_list = [int(num_real_points / 3 * 5) for num_real_points in num_real_points_list]

            
            # Generate the first part: starting_still_frames
            self.left_gts[env_ids, :starting_still_frames] = init_left_pos[..., None, :]  # Keep initial position
            self.right_gts[env_ids, :starting_still_frames] = init_right_pos[..., None, :]  # Keep initial position
            self.left_grs[env_ids, :starting_still_frames] = init_left_quat[..., None, :]  # Keep initial rotation
            self.right_grs[env_ids, :starting_still_frames] = init_right_quat[..., None, :]  # Keep initial rotation
      
            # Generate the second part: transition_frames
            # Interpolate positions
            for i, env_id in enumerate(env_ids):
                
                # Generate interpolation weights
                weights = torch.linspace(0, 1, transition_frames, device=self._device).unsqueeze(-1)

                # Interpolate left hand position
                start_left_pos = init_left_pos[i:i+1, :]
                end_left_pos = left_hand_pos[i][0:1, :]
                # clamp z
                end_left_pos[:, 2] = end_left_pos[:, 2] - 0.06
                interpolated_left_pos = (1 - weights) * start_left_pos + weights * end_left_pos

                # Interpolate right hand position
                start_right_pos = init_right_pos[i:i+1, :]
                end_right_pos = right_hand_pos[i][0:1, :]
                # clamp z
                end_right_pos[:, 2] = end_right_pos[:, 2] - 0.06
                interpolated_right_pos = (1 - weights) * start_right_pos + weights * end_right_pos

                # Fill interpolated positions into trajectory
                self.left_gts[env_id, starting_still_frames:starting_still_frames + transition_frames] = interpolated_left_pos
                self.right_gts[env_id, starting_still_frames:starting_still_frames + transition_frames] = interpolated_right_pos

                # Interpolate rotations using SLERP
                start_left_quat = init_left_quat[i:i+1, :]
                end_left_quat = left_hand_quat[i][0:1, :]
                interpolated_left_quat = slerp(start_left_quat, end_left_quat, weights)

                start_right_quat = init_right_quat[i:i+1, :]
                end_right_quat = right_hand_quat[i][0:1, :]
                interpolated_right_quat = slerp(start_right_quat, end_right_quat, weights)

                # Fill interpolated rotations into trajectory
                self.left_grs[env_id, starting_still_frames:starting_still_frames + transition_frames] = interpolated_left_quat
                self.right_grs[env_id, starting_still_frames:starting_still_frames + transition_frames] = interpolated_right_quat


            # Generate the third part: real trajectory
            left_hand_pos_resampled = [
                torch.nn.functional.interpolate(
                    traj.T.unsqueeze(0),  # Shape: (1, 3, num_real_points)
                    size=real_traj_frames_list[i],  # 每条轨迹的目标帧数
                    mode='linear',
                    align_corners=True
                ).squeeze(0).T  # Shape: (real_traj_frames, 3)
                for i, traj in enumerate(left_hand_pos)
            ]

            right_hand_pos_resampled = [
                torch.nn.functional.interpolate(
                    traj.T.unsqueeze(0),  # Shape: (1, 3, num_real_points)
                    size=real_traj_frames_list[i],  # 每条轨迹的目标帧数
                    mode='linear',
                    align_corners=True
                ).squeeze(0).T  # Shape: (real_traj_frames, 3)
                for i, traj in enumerate(right_hand_pos)
            ]

            left_hand_quat_resampled = [
                self.interpolate_quaternions_via_axis_angle(traj.unsqueeze(0), real_traj_frames_list[i])[0]
                for i, traj in enumerate(left_hand_quat)
            ]

            right_hand_quat_resampled = [
                self.interpolate_quaternions_via_axis_angle(traj.unsqueeze(0), real_traj_frames_list[i])[0]
                for i, traj in enumerate(right_hand_quat)
            ]

            start_idx = starting_still_frames + transition_frames  # 计算起始索引

            # 逐条处理每个环境的轨迹
            for i, env_id in enumerate(env_ids):
                end_idx = start_idx + real_traj_frames_list[i] - 2   # 根据每条轨迹的长度动态计算结束索引

                # 插入位置轨迹
                self.left_gts[env_id, start_idx:end_idx] = left_hand_pos_resampled[i][2:]
                self.right_gts[env_id, start_idx:end_idx] = right_hand_pos_resampled[i][2:]

                # 插入旋转轨迹
                self.left_grs[env_id, start_idx:end_idx] = left_hand_quat_resampled[i][2:]
                self.right_grs[env_id, start_idx:end_idx] = right_hand_quat_resampled[i][2:]

                # clamp z
                self.left_gts[env_id, start_idx:end_idx, 2] = self.left_gts[env_id, start_idx:end_idx, 2] - 0.06
                self.right_gts[env_id, start_idx:end_idx, 2] = self.right_gts[env_id, start_idx:end_idx, 2] - 0.06

                self.traj_lengths[env_id] = end_idx  # 记录每个环境的轨迹长度
            if not self.eval:
                self.delta_traj(env_ids, it)
                
            if self.eval:
                self.target_delta_command_heights[env_ids, 1:] = 0.0
                self.target_delta_command_x[env_ids, 1:] = 0.0
                self.target_delta_command_y[env_ids, 1:] = 0.0
                self.target_command_heights[env_ids] = 0.74 

            # # if self.target_delta_command_heights[env_ids,:,0] < -0.20, self.target_delta_command_y[env_ids,:,0] =0
            # if self.target_delta_command_heights[env_ids,:,0] < -0.20:
            #     self.target_delta_command_y[env_ids,:,0] = 
            #     self.target_delta_command_x[env_ids,:,0] = 

            # add x
            self.left_gts[env_ids,:,0] += self.target_delta_command_x[env_ids,:,0] 
            self.right_gts[env_ids,:,0] += self.target_delta_command_x[env_ids,:,0]
            # add y
            self.left_gts[env_ids,:,1] += self.target_delta_command_y[env_ids,:,0]
            self.right_gts[env_ids,:,1] += self.target_delta_command_y[env_ids,:,0]
            # add z
            self.left_gts[env_ids,:,2] += self.ref_target_delta_command_heights[env_ids,:,0]
            self.right_gts[env_ids,:,2] += self.ref_target_delta_command_heights[env_ids,:,0]

        return
    
    def delta_traj(self, env_ids, it):
        n = len(env_ids)
        num_verts = self.get_num_verts()
        if (n > 0):
            # ============================ z ========================= #
            all_zs_left = self.sample_constrained_z(n, num_verts, max_h=0.42, hard_level=min(2.0,(it//100)+1.0))
            all_zs_left = all_zs_left[:(n * (num_verts - 1))].reshape(n, num_verts - 1)
            self.target_delta_command_heights[env_ids,1:] = torch.from_numpy(all_zs_left).float().unsqueeze(-1).to(env_ids.device)
            self.ref_target_delta_command_heights[env_ids,1:] = torch.clamp(self.target_delta_command_heights[env_ids,1:], min=-0.26, max=0.36) -0.16 # range[-0.42, 0.20] for hand
            self.target_delta_command_heights[env_ids,1:] = torch.clamp(self.target_delta_command_heights[env_ids,1:], min=-0.26, max=0.16) -0.16 # range[-0.42, 0.0] for command
            
            
            self.target_command_heights[env_ids] = 0.74 + self.target_delta_command_heights[env_ids]

            # ============================ x, y ========================= #
            speed_torso, dspeed_torso, dtheta_torso = self.sample_dtheat_dspeed(n, it)
            seg_len_torso = speed_torso 
            dpos_torso = torch.stack([torch.cos(dtheta_torso), -torch.sin(dtheta_torso), torch.zeros_like(dtheta_torso)], dim=-1)
            dpos_torso *= seg_len_torso.unsqueeze(-1)
            mask = self.target_delta_command_heights[env_ids, 1:, 0] > -0.16 # Create a mask based on the height
            dpos_torso = dpos_torso * mask.unsqueeze(-1)  # Apply the mask to dpos_torso
            vert_pos_torso = torch.cumsum(dpos_torso, dim=-2)

            # # dummy vert_pos_torso
            vert_pos_torso =torch.zeros_like(vert_pos_torso)
            

            vert_pos_torso[..., 0] = torch.clamp(vert_pos_torso[..., 0], -2.0, 2.0)  # 限制 x 范围
            vert_pos_torso[..., 1] = torch.clamp(vert_pos_torso[..., 1], -2.0, 2.0)  # 限制 y 范围
            
            # print(f"vert_pos_torso.shape: {vert_pos_torso.shape}")

            self.target_delta_command_x[env_ids,1:] = vert_pos_torso[..., 0].unsqueeze(-1)
            self.target_delta_command_y[env_ids,1:] = vert_pos_torso[..., 1].unsqueeze(-1)

            # 计算每 5 步的速度
            step_interval = 5
            dt_interval = step_interval * self._dt

            # 计算差分
            vert_pos_torso_diff = vert_pos_torso[..., step_interval:, :] - vert_pos_torso[..., :-step_interval, :]

            # 计算速度
            vel_torso = vert_pos_torso_diff / dt_interval

            # 将速度插入到 target_command_vel_x 和 target_command_vel_y 中
            self.target_command_vel_x[env_ids, 1:-step_interval] = vel_torso[..., 0:1]
            self.target_command_vel_y[env_ids, 1:-step_interval] = vel_torso[..., 1:2]

            # 对后 step_interval 帧的速度填充为 0（因为无法计算差分）
            self.target_command_vel_x[env_ids, -step_interval:] = 0
            self.target_command_vel_y[env_ids, -step_interval:] = 0

        return
        
    def traj_lengths(self):
        return self.traj_lengths


    def input_new_trajs(self, env_ids):
        import json
        import requests
        from scipy.interpolate import interp1d
        x = requests.get(f'http://{SERVER}:{PORT}/path?num_envs={len(env_ids)}')

        data_lists = [value for idx, value in x.json().items()]
        coord = np.array(data_lists)
        x = np.linspace(0, coord.shape[1] - 1, num=coord.shape[1])
        fx = interp1d(x, coord[..., 0], kind='linear')
        fy = interp1d(x, coord[..., 1], kind='linear')
        x4 = np.linspace(0, coord.shape[1] - 1, num=coord.shape[1] * 10)
        coord_dense = np.stack([fx(x4), fy(x4), np.zeros([len(env_ids), x4.shape[0]])], axis=-1)
        coord_dense = np.concatenate([coord_dense, coord_dense[..., -1:, :]], axis=-2)
        self.gts[env_ids] = torch.from_numpy(coord_dense).float().to(env_ids.device)
        return self.gts[env_ids]

    def get_num_verts(self):
        return self.left_gts.shape[1]

    def get_num_segs(self):
        return self.get_num_verts() - 1

    def get_num_envs(self):
        return self.left_gts.shape[0]

    def get_traj_duration(self):
        num_verts = self.get_num_verts()
        dur = num_verts * self._dt 
        return dur
    
    # def get_dir(self, env_ids=None):
    #     if env_ids is None:
    #         return self.dir
    #     else:
    #         return self.dir[env_ids]
        
    def get_dir(self, env_ids=None):
        if env_ids is None:
            return self.dir
        else:
            if env_ids in self.dir:
                return self.dir[env_ids]
            else:
                raise KeyError(f"Environment ID {env_ids} not found in directory mapping.")
    
    def get_retarget_traj_verts(self, env_ids=None):
        if env_ids is None:
            return self.left_gts, self.left_grs, self.right_gts, self.right_grs
        else:
            return self.left_gts[env_ids], self.left_grs[env_ids], self.right_gts[env_ids], self.right_grs[env_ids]

    def get_traj_verts(self, env_ids=None):
        if env_ids is None:
            return self.left_gts, self.left_grs, self.right_gts, self.right_grs, self.target_command_heights, self.target_delta_command_x, self.target_delta_command_y
        else:
            return self.left_gts[env_ids], self.left_grs[env_ids], self.right_gts[env_ids], self.right_grs[env_ids], self.target_command_heights[env_ids], self.target_delta_command_x[env_ids], self.target_delta_command_y[env_ids]
        
    def get_motion_state(self, traj_ids, times):
        traj_dur = self.get_traj_duration()
        num_verts = self.get_num_verts()
        num_segs = self.get_num_segs()

        traj_phase = torch.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs
        seg_id0 = torch.floor(seg_idx).long()
        seg_id1 = torch.ceil(seg_idx).long()
        lerp = seg_idx - seg_id0

        pos0 = self.gts_flat[traj_ids * num_verts + seg_id0]
        pos1 = self.gts_flat[traj_ids * num_verts + seg_id1]

        lerp = lerp.unsqueeze(-1)
        o_rb_pos = (1.0 - lerp) * pos0 + lerp * pos1


        rb_rot0 = self.grs_flat[traj_ids * num_verts + seg_id0]
        rb_rot1 = self.grs_flat[traj_ids * num_verts + seg_id1]
        
        o_rb_rot = slerp(rb_rot0, rb_rot1, lerp)

        ang_vel0, ang_vel1 = self.gavs_flat[traj_ids * num_verts + seg_id0], self.gavs_flat[traj_ids * num_verts + seg_id1]
        lin_vel0, lin_vel1 = self.gvs_flat[traj_ids * num_verts + seg_id0], self.gvs_flat[traj_ids * num_verts + seg_id1]

        o_lin_vel = (1.0 - lerp) * lin_vel0 + lerp * lin_vel1
        o_ang_vel = (1.0 - lerp) * ang_vel0 + lerp * ang_vel1
        

        return {
            "o_ang_vel": o_ang_vel[:, None].clone(),
            "o_rb_rot": o_rb_rot[:, None].clone(),
            "o_rb_pos": o_rb_pos[:, None].clone(),
            "o_lin_vel": o_lin_vel[:, None].clone(),
        }

    def mock_calc_pos(self, env_ids, traj_ids, times, query_value_gradient):
        traj_dur = self.get_traj_duration()
        num_verts = self.get_num_verts()
        num_segs = self.get_num_segs()

        traj_phase = torch.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs
        seg_id0 = torch.floor(seg_idx).long()
        seg_id1 = torch.ceil(seg_idx).long()
        lerp = seg_idx - seg_id0

        pos0 = self.gts_flat[traj_ids * num_verts + seg_id0]
        pos1 = self.gts_flat[traj_ids * num_verts + seg_id1]

        lerp = lerp.unsqueeze(-1)
        pos = (1.0 - lerp) * pos0 + lerp * pos1

        new_obs, func = query_value_gradient(env_ids, pos)
        if not new_obs is None:
            # ZL: computes grad
            with torch.enable_grad():
                new_obs.requires_grad_(True)
                new_val = func(new_obs)
                disc_grad = torch.autograd.grad(new_val, new_obs, grad_outputs=torch.ones_like(new_val), create_graph=False, retain_graph=True, only_inputs=True)

        return pos
    
    def quat_to_axis_angle(self, quat):
        """将四元数转换为轴角表示"""
        rotation = R.from_quat(quat.cpu().numpy())
        return torch.tensor(rotation.as_rotvec(), dtype=torch.float32, device=quat.device)

    def axis_angle_to_quat(self, axis_angle):
        """将轴角表示转换为四元数"""
        rotation = R.from_rotvec(axis_angle.cpu().numpy())
        return torch.tensor(rotation.as_quat(), dtype=torch.float32, device=axis_angle.device)

    def interpolate_quaternions_via_axis_angle(self, quat, target_frames):
        """
        使用轴角表示对四元数进行插值
        :param quat: 输入的四元数 (n, num_real_points, 4)
        :param target_frames: 插值后的帧数
        :return: 插值后的四元数 (n, target_frames, 4)
        """
        # 将四元数转换为轴角表示
        axis_angle = torch.stack([self.quat_to_axis_angle(q) for q in quat], dim=0)  # Shape: (n, num_real_points, 3)

        # 在轴角空间中插值
        axis_angle = axis_angle.permute(0, 2, 1)  # Shape: (n, 3, num_real_points)
        axis_angle_resampled = torch.nn.functional.interpolate(
            axis_angle, size=target_frames, mode='linear', align_corners=True
        ).permute(0, 2, 1)  # Shape: (n, target_frames, 3)

        # 将插值后的轴角表示转换回四元数
        quat_resampled = torch.stack([self.axis_angle_to_quat(aa) for aa in axis_angle_resampled], dim=0)  # Shape: (n, target_frames, 4)
        return quat_resampled
    

@torch.jit.script
def slerp(q0, q1, t):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta

    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q