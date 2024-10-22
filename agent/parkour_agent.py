from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common

from isaacgym.torch_utils import *

import time
import numpy as np
import torch

import agent.amp_agent as amp_agent


# TODO 图的normalize
class ParkourAgent(amp_agent.AMPAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        self.experience_buffer.tensor_dict["graph_obs"] = [None] * self.experience_buffer.horizon_length
        self.experience_buffer.tensor_dict["graph_obs_next"] = [None] * self.experience_buffer.horizon_length

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict["graph_obs"] = batch_dict["graph_obs"]
        self.dataset.values_dict["graph_obs_next"] = batch_dict["graph_obs_next"]

