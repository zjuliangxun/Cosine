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

from enum import Enum
import numpy as np
import time
from isaacgym.torch_utils import *
import torch


from utils import torch_utils
from data.motion_lib import MotionLib
from sim.humanoid import Humanoid, dof_to_obs
from sim.strategy.reset import ResetStrategy
from sim.strategy.early_term import TerminateByHeight


class HumanoidAMPBase(Humanoid):
    def __init__(
        self, cfg, sim_params, physics_engine, device_type, device_id, headless
    ):
        self.reset_strategy = ResetStrategy(
            self,
            state_init=cfg["env"]["stateInit"],
            hybrid_init_prob=cfg["env"]["hybridInitProb"],
        )
        self.terminate_strategy = TerminateByHeight()

        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            device_type=device_type,
            device_id=device_id,
            headless=headless,
        )

        ############### moiton lib ################
        self._equal_motion_weights = cfg["env"].get("equal_motion_weights", False)
        motion_file = cfg["env"]["motion_file"]
        self._load_motion(motion_file)

        ############### amp realted buffers ################
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert self._num_amp_obs_steps >= 2
        self._num_amp_obs_enc_steps = cfg["env"].get(
            "numAMPEncObsSteps", self._num_amp_obs_steps
        )

        self._amp_obs_buf = torch.zeros(
            (self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float,
        )
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        return

    def post_physics_step(self):
        super().post_physics_step()

        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def get_num_enc_amp_obs(self):
        return self._num_amp_obs_enc_steps * self._num_amp_obs_per_step

    def fetch_amp_obs_demo(self, num_samples):
        motion_ids = self._motion_lib.sample_motions(num_samples)

        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(
            motion_ids, truncate_time=truncate_time
        )
        motion_times0 += truncate_time

        amp_obs_demo_flat = (
            self.build_amp_obs_demo(motion_ids, motion_times0, self._num_amp_obs_steps)
            .to(self.device)
            .view(-1, self.get_num_amp_obs())
        )

        return amp_obs_demo_flat

    def fetch_amp_obs_demo_per_id(self, num_samples, motion_id):
        motion_ids = torch.tensor(
            [motion_id for _ in range(num_samples)], dtype=torch.long
        ).view(-1)

        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        enc_window_size = self.dt * (self._num_amp_obs_enc_steps - 1)

        enc_motion_times = self._motion_lib.sample_time(
            motion_ids, truncate_time=enc_window_size
        )
        # make sure not to add more than motion clip length, negative amp_obs will show zero index amp_obs instead
        enc_motion_times += torch.clip(
            self._motion_lib._motion_lengths[motion_ids], max=enc_window_size
        )

        enc_amp_obs_demo = self.build_amp_obs_demo(
            motion_ids, enc_motion_times, self._num_amp_obs_enc_steps
        ).view(-1, self._num_amp_obs_enc_steps, self._num_amp_obs_per_step)

        enc_amp_obs_demo_flat = enc_amp_obs_demo.to(self.device).view(
            -1, self.get_num_enc_amp_obs()
        )

        return motion_ids, enc_motion_times, enc_amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0, num_steps):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, num_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, num_steps, device=self.device)
        motion_times = torch.clip(motion_times + time_steps, min=0)

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
            self._motion_lib.get_motion_state(motion_ids, motion_times)
        )
        amp_obs_demo = build_amp_observations(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_pos,
            self._local_root_obs,
            self._root_height_obs,
            self._dof_obs_size,
            self._dof_offsets,
        )
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros(
            (num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float32,
        )
        self._enc_amp_obs_demo_buf = torch.zeros(
            (num_samples, self._num_amp_obs_enc_steps, self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float32,
        )
        return

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if asset_file == "mjcf/amp_humanoid.xml":
            self._num_amp_obs_per_step = (
                13 + self._dof_obs_size + 28 + 3 * num_key_bodies
            )  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif asset_file == "mjcf/amp_humanoid_sword_shield.xml":
            self._num_amp_obs_per_step = (
                13 + self._dof_obs_size + 31 + 3 * num_key_bodies
            )  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert False

        return

    def _load_motion(self, motion_file):
        assert self._dof_offsets[-1] == self.num_dof
        self._motion_lib = MotionLib(
            motion_file=motion_file,
            dof_body_ids=self._dof_body_ids,
            dof_offsets=self._dof_offsets,
            key_body_ids=self._key_body_ids.cpu().numpy(),
            equal_motion_weights=self._equal_motion_weights,
            device=self.device,
        )
        return

    def _reset_envs(self, env_ids):
        self.reset_strategy.reset_env(env_ids)

    def _compute_reset(self):
        self.terminate_strategy.compute_reset()

    def get_task_obs_size(self):
        return 0

    def _set_env_state(
        self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel
    ):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel

        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if env_ids is None:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return

    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if env_ids is None:
            self._curr_amp_obs_buf[:] = build_amp_observations(
                self._rigid_body_pos[:, 0, :],
                self._rigid_body_rot[:, 0, :],
                self._rigid_body_vel[:, 0, :],
                self._rigid_body_ang_vel[:, 0, :],
                self._dof_pos,
                self._dof_vel,
                key_body_pos,
                self._local_root_obs,
                self._root_height_obs,
                self._dof_obs_size,
                self._dof_offsets,
            )
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(
                self._rigid_body_pos[env_ids][:, 0, :],
                self._rigid_body_rot[env_ids][:, 0, :],
                self._rigid_body_vel[env_ids][:, 0, :],
                self._rigid_body_ang_vel[env_ids][:, 0, :],
                self._dof_pos[env_ids],
                self._dof_vel[env_ids],
                key_body_pos[env_ids],
                self._local_root_obs,
                self._root_height_obs,
                self._dof_obs_size,
                self._dof_offsets,
            )
        return


class HumanoidAMPTask(HumanoidAMPBase):
    def __init__(
        self, cfg, sim_params, physics_engine, device_type, device_id, headless
    ):
        self._enable_task_obs = cfg["env"]["enableTaskObs"]

        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=physics_engine,
            device_type=device_type,
            device_id=device_id,
            headless=headless,
        )
        return

    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if self._enable_task_obs:
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size

    def get_task_obs_size(self):
        return 0

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_task()
        return

    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return

    def _update_task(self):
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        self._reset_task(env_ids)
        return

    def _reset_task(self, env_ids):
        return

    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)

        if self._enable_task_obs:
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([humanoid_obs, task_obs], dim=-1)
        else:
            obs = humanoid_obs

        if env_ids is None:
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return

    def _compute_task_obs(self, env_ids=None):
        return NotImplemented

    def _compute_reward(self, actions):
        return NotImplemented

    def _draw_task(self):
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def build_amp_observations(
    root_pos,
    root_rot,
    root_vel,
    root_ang_vel,
    dof_pos,
    dof_vel,
    key_body_pos,
    local_root_obs,
    root_height_obs,
    dof_obs_size,
    dof_offsets,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if local_root_obs:
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)

    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2],
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    )

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat(
        (
            root_h_obs,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            flat_local_key_pos,
        ),
        dim=-1,
    )
    return obs
