# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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

import torch
import torch.nn as nn

import model.amp_network_builder as amp_network_builder
from model.gnn_utils import GATModel


class ParkourBuilder(amp_network_builder.AMPBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(amp_network_builder.AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)  # PARAMS = CFG.NETWORK
            # [ ] cnn的输入列表要扩大
            # 如何选取obs的channel？一方面是对task的高效表征所以不应该太差，另一方面又要考虑压缩和性能，和action的隐状态相比尺度如何
            self.graph_obs_net = GATModel(**params["graph_obs_net"])
            return

        def load(self, params):
            super().load(params)

            self._disc_units = params["disc"]["units"]
            self._disc_activation = params["disc"]["activation"]
            self._disc_initializer = params["disc"]["initializer"]
            return

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            states = obs_dict.get("rnn_states", None)
            task_obs = self.graph_obs_net(obs_dict["graph_obs"])
            next_task_obs = self.graph_obs_net(obs_dict["graph_obs_next"])

            obs = torch.cat([obs, task_obs, next_task_obs], dim=-1)
            actor_outputs = self.eval_actor(obs)
            value = self.eval_critic(obs)

            output = actor_outputs + (value, states)

            return output

    def build(self, name, **kwargs):
        net = ParkourBuilder.Network(self.params, **kwargs)
        return net
