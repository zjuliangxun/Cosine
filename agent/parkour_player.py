import torch

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd

import agent.amp_player as amp_player


class ParkourPlayerContinuous(amp_player.AMPPlayerContinuous):
    def __init__(self, config):
        self._normalize_amp_input = config.get("normalize_amp_input", True)
        self._normalize_graph_input = config.get("normalize_graph_input", True)
        self._disc_reward_scale = config["disc_reward_scale"]
        super().__init__(config)
        return

    def restore(self, fn):
        if fn != "Base":
            super().restore(fn)
            checkpoint = torch_ext.load_checkpoint(fn)
            if self._normalize_amp_input:
                self._amp_input_mean_std.load_state_dict(checkpoint["amp_input_mean_std"])
            if self._normalize_graph_input:
                self._graph_input_mean_std.load_state_dict(checkpoint["graph_input_mean_std"])
        return

    def _build_net(self, config):
        super()._build_net(config)

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(
                (self._amp_observation_space[0] // self.env.task._num_amp_obs_steps,)
            ).to(self.device)
            self._amp_input_mean_std.eval()
        if self._normalize_graph_input:
            self._graph_input_mean_std = RunningMeanStd((config["graph_input_mean_std"],)).to(self.device)
            self._graph_input_mean_std.eval()
        return

    def _post_step(self, info):
        super()._post_step(info)
        if self.env.task.viewer:
            self._amp_debug(info)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        length = self.config.get("varlen_input_shape", 0)
        if length > 0:
            input_shape = list(config["input_shape"])
            input_shape[0] += length
            config["input_shape"] = tuple(input_shape)
        return config

    # functions for _amp_debug
    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info["amp_obs"]
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs)
            disc_reward = amp_rewards["disc_rewards"]

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward)

        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            shape = amp_obs.shape
            amp_obs = amp_obs.view(-1, self.env.amp_observation_space.shape[0] // self.env.task._num_amp_obs_steps)
            amp_obs = self._amp_input_mean_std(amp_obs)
            amp_obs = amp_obs.view(shape)
        return amp_obs

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {"disc_rewards": disc_r}
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale
        return disc_r

    # functions for amp_debug end
