from typing import Any, TYPE_CHECKING
from enum import Enum
import numpy as np
import torch
from isaacgym.torch_utils import to_torch

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

    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated
    )

    return reset, terminated


class TerminateByContact(TerminateByHeight):
    def __init__(self, ctx: ParkourSingle):
        super().__init__(ctx)

    def compute_reset(self):
        ctx = self.ctx
        # TODO 遍历每个接触图，如果长时间contact reward 太低，就认为失败了
        return
