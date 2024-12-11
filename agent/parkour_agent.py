from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common

from isaacgym.torch_utils import *

import time
import numpy as np
import torch

import agent.amp_agent as amp_agent
import data.replay_buffer as replay_buffer


class ParkourAgent(amp_agent.AMPAgent):
    """Use vanilla AMP backend, no conditional GAN"""

    def __init__(self, base_name, config):
        super().__init__(base_name, config)

        # [ ] cond disc 需要额外调用_preproc_task_obs
        self._normalize_graph_input = config.get("normalize_graph_input", True)
        if self._normalize_graph_input:
            self._graph_input_mean_std = RunningMeanStd((config["graph_input_mean_std"],)).to(self.ppo_device)

        return

    def set_eval(self):
        super().set_eval()
        if self._normalize_graph_input:
            self._graph_input_mean_std.eval()
        return

    def set_train(self):
        super().set_train()
        if self._normalize_graph_input:
            self._graph_input_mean_std.train()
        return

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_graph_input:
            state["graph_input_mean_std"] = self._graph_input_mean_std.state_dict()

        return state

    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self._normalize_graph_input:
            self._graph_input_mean_std.load_state_dict(weights["graph_input_mean_std"])
        return

    def _build_amp_buffers(self):
        super()._build_amp_buffers()
        self.experience_buffer.tensor_dict["graph_obs"] = [None] * self.experience_buffer.horizon_length
        self.experience_buffer.tensor_dict["graph_obs_next"] = [None] * self.experience_buffer.horizon_length

    def _preproc_task_obs(self, graph_obs):
        if self._normalize_graph_input:
            graph_obs.x = self._graph_input_mean_std(graph_obs.x)
        return graph_obs

    def calc_gradients(self, input_dict):
        batch_dict_toadd = {
            "graph_obs": self._preproc_task_obs(input_dict["graph_obs"]),
            "graph_obs_next": self._preproc_task_obs(input_dict["graph_obs_next"]),
        }
        return super().calc_gradients(input_dict, batch_dict_toadd=batch_dict_toadd)

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict["graph_obs"] = batch_dict["graph_obs"]
        self.dataset.values_dict["graph_obs_next"] = batch_dict["graph_obs_next"]

    def play_steps(self):
        self.set_eval()
        ## change
        if self.dones is None:
            done_indices = []
        else:
            done_indices = self.dones.nonzero(as_tuple=False)[:: self.num_agents][:, 0]
        # end
        update_list = self.update_list

        for n in range(self.horizon_length):

            ## change
            self.obs, infos = self.env_reset(done_indices)
            # end
            self.experience_buffer.update_data("obses", n, self.obs["obs"])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs, self._rand_action_probs, infos=infos)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data("states", n, self.obs["states"])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data("rewards", n, shaped_rewards)
            self.experience_buffer.update_data("next_obses", n, self.obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones)
            self.experience_buffer.update_data("amp_obs", n, infos["amp_obs"])
            self.experience_buffer.update_data("rand_action_mask", n, res_dict["rand_action_mask"])
            # only add these 2 line
            self.experience_buffer.update_data("graph_obs", n, infos["graph_obs"])
            self.experience_buffer.update_data("graph_obs_next", n, infos["graph_obs_next"])
            ## end
            terminated = infos["terminate"].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= 1.0 - terminated
            self.experience_buffer.update_data("next_values", n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[:: self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            if self.vec_env.env.task.viewer:
                self._amp_debug(infos)

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_next_values = self.experience_buffer.tensor_dict["next_values"]

        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_amp_obs = self.experience_buffer.tensor_dict["amp_obs"]
        amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict["returns"] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size

        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        return batch_dict

    def get_action_values(self, obs_dict, rand_action_probs, **kwargs):
        processed_obs = self._preproc_obs(obs_dict["obs"])

        self.model.eval()
        # change
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": processed_obs,
            "rnn_states": self.rnn_states,
            "graph_obs": kwargs["infos"]["graph_obs"],
            "graph_obs_next": kwargs["infos"]["graph_obs_next"],
        }
        # done

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs_dict["states"]
                input_dict = {
                    "is_train": False,
                    "states": states,
                }
                value = self.get_central_value(input_dict)
                res_dict["values"] = value

        if self.normalize_value:
            res_dict["values"] = self.value_mean_std(res_dict["values"], True)

        rand_action_mask = torch.bernoulli(rand_action_probs)
        det_action_mask = rand_action_mask == 0.0
        res_dict["actions"][det_action_mask] = res_dict["mus"][det_action_mask]
        res_dict["rand_action_mask"] = rand_action_mask

        return res_dict

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
        # [ ] 从env获得当前的图像并且用下列api写出来
        # from isaacgym import gymapi
        # if self._log_image_flag and self.rank == 0:
        #     img = self.gym.get_camera_image(sim, envs[i], camera_handles[i][j], gymapi.IMAGE_COLOR)
        #     self.writer.add_figure("林麻子matplotlib figure林祖泉", fig, global_step=None, close=False, walltime=None)
        # return
    ########### [x] reward and grad
    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards["disc_rewards"]
        # contact reward for self._task_reward_w
        combined_rewards = self._task_reward_w * task_rewards + +self._disc_reward_w * disc_r
        return combined_rewards

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_advs(self, batch_dict):
        returns = batch_dict["returns"]
        values = batch_dict["values"]
        rand_action_mask = batch_dict["rand_action_mask"]

        advantages = returns - values
        advantages = torch.sum(advantages, axis=1)
        if self.normalize_advantage:
            advantages = torch_ext.normalization_with_masks(advantages, rand_action_mask)

        return advantages

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {"disc_rewards": disc_r}
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
            disc_r *= self._disc_reward_scale

        return disc_r

    ########### [x] help funcs
    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info["disc_rewards"] = batch_dict["disc_rewards"]
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)

        if self._disc_reward_w > 0:
            self.writer.add_scalar(
                "losses/disc_loss",
                torch_ext.mean_list(train_info["disc_loss"]).item(),
                frame,
            )

            self.writer.add_scalar(
                "info/disc_agent_acc",
                torch_ext.mean_list(train_info["disc_agent_acc"]).item(),
                frame,
            )
            self.writer.add_scalar(
                "info/disc_demo_acc",
                torch_ext.mean_list(train_info["disc_demo_acc"]).item(),
                frame,
            )
            self.writer.add_scalar(
                "info/disc_agent_logit",
                torch_ext.mean_list(train_info["disc_agent_logit"]).item(),
                frame,
            )
            self.writer.add_scalar(
                "info/disc_demo_logit",
                torch_ext.mean_list(train_info["disc_demo_logit"]).item(),
                frame,
            )
            self.writer.add_scalar(
                "info/disc_grad_penalty",
                torch_ext.mean_list(train_info["disc_grad_penalty"]).item(),
                frame,
            )
            self.writer.add_scalar(
                "info/disc_logit_loss",
                torch_ext.mean_list(train_info["disc_logit_loss"]).item(),
                frame,
            )

            disc_reward_std, disc_reward_mean = torch.std_mean(train_info["disc_rewards"])
            self.writer.add_scalar("info/disc_reward_mean", disc_reward_mean.item(), frame)
            self.writer.add_scalar("info/disc_reward_std", disc_reward_std.item(), frame)
        return

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
