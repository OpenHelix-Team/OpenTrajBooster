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

import torch
import numpy as np

class DaggerRolloutStorage:
    def __init__(self):
        self.actor_obs_list = []
        self.gt_actions_list = []

    def add_data(self, data):
        observations, gt_actions = data
        # 强制在 CPU 上存储
        self.actor_obs_list.append(observations.cpu())
        self.gt_actions_list.append(gt_actions.cpu())

    def get_dagger_generator(self, num_mini_batches, num_learning_epochs, device='cuda'):
        # 组合成大 tensor，在 CPU 上
        actor_obs = torch.cat(self.actor_obs_list, dim=0)
        gt_actions = torch.cat(self.gt_actions_list, dim=0)

        total_size = actor_obs.shape[0]
        batch_size = 256*50

        for _ in range(num_learning_epochs):
            indices = torch.randperm(total_size)
            for start in range(0, total_size, batch_size):
                end = start + batch_size
                batch_inds = indices[start:end]

                # 注意：这里只在这一批上传到 GPU
                yield (
                    actor_obs[batch_inds].to(device),
                    gt_actions[batch_inds].to(device)
                )

    def clear(self):
        self.actor_obs_list.clear()
        self.gt_actions_list.clear()
