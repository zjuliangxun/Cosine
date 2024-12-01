from typing import Any, TYPE_CHECKING
from enum import Enum
import numpy as np
import torch

if TYPE_CHECKING:
    from ..humanoid import Humanoid
    from ..humanoid_amp import HumanoidAMPBase
    from ..parkour_single import ParkourSingle


class TerminateByHeight:
    def __init__(self, ctx):
        self.ctx: Humanoid = ctx
        # self._termination_height = termination_height
        # NOTE this para can be optimized later

    def __call__(self):
        self.compute_reset()
        return

    def compute_reset(self):
        ctx = self.ctx
        ctx.reset_buf[:], ctx._terminate_buf[:] = compute_humanoid_reset(
            ctx.reset_buf,
            ctx.progress_buf,
            ctx._contact_forces,
            ctx._contact_body_ids,
            ctx._rigid_body_pos,
            ctx.max_episode_length,
            ctx._enable_early_termination,
            ctx._termination_heights,
        )
        return


@torch.jit.script
def compute_humanoid_reset(
    reset_buf,
    progress_buf,
    contact_buf,
    contact_body_ids,
    rigid_body_pos,
    max_episode_length,
    enable_early_termination,
    termination_heights,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= progress_buf > 1
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated


class TerminateByContact(TerminateByHeight):
    def __init__(self, ctx, time_out_thresh: int = 120):
        super().__init__(ctx)
        if TYPE_CHECKING:
            self.ctx: ParkourSingle
        self.time_out_thresh = time_out_thresh  # int(4 / ctx.dt)

    def compute_reset(self):
        ctx = self.ctx
        ctx.reset_buf[:], ctx._terminate_buf[:] = compute_humanoid_reset_cg(
            ctx.time_out_buf,
            ctx.terrain_cg_nums[ctx.env_terrain_id],
            ctx.cg_progress_buf,
            ctx.progress_buf,
            ctx._rigid_body_pos,
            ctx.max_episode_length,
            ctx._enable_early_termination,
            ctx._termination_heights,
            self.time_out_thresh,
        )
        return

    # def check_termination(self):
    #     """Check if environments need to be reset"""
    #     self.reset_buf = torch.zeros(
    #         (self.num_envs,), dtype=torch.bool, device=self.device
    #     )
    #     roll_cutoff = torch.abs(self.roll) > 1.5
    #     pitch_cutoff = torch.abs(self.pitch) > 1.5
    #     reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals
    #     height_cutoff = self.root_states[:, 2] < -0.25

    #     self.time_out_buf = (
    #         self.episode_length_buf > self.max_episode_length
    #     )  # no terminal reward for time-outs
    #     self.time_out_buf |= reach_goal_cutoff

    #     self.reset_buf |= self.time_out_buf
    #     self.reset_buf |= roll_cutoff
    #     self.reset_buf |= pitch_cutoff
    #     self.reset_buf |= height_cutoff


@torch.jit.script
def compute_humanoid_reset_cg(
    time_out_buf,
    max_cg_num_buf,
    cg_progress_buf,
    progress_buf,
    rigid_body_pos,
    max_episode_length,
    enable_early_termination,
    termination_heights,
    time_out_thresh,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(progress_buf)
    # terminate if all body under the thresh/many frames without goal reached
    if enable_early_termination:
        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        # fall_height[:, contact_body_ids] = False
        fall_height = torch.all(fall_height, dim=-1)

        has_time_out = time_out_buf > time_out_thresh

        has_fallen = torch.logical_and(has_time_out, fall_height)
        terminated = torch.where(has_fallen, torch.ones_like(progress_buf), terminated)

    has_done = torch.logical_and(
        progress_buf >= max_episode_length - 1,
        # all the cgs in a grid has been traversed
        cg_progress_buf >= max_cg_num_buf,
    )
    reset = torch.where(has_done, torch.ones_like(progress_buf), terminated)

    return reset, terminated
