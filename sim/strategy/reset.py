from typing import Any, TYPE_CHECKING
from enum import Enum
import numpy as np
import torch
import isaacgym.torch_utils as torch_utils
from sim.jit_functions import build_amp_observations

if TYPE_CHECKING:
    from ..humanoid_amp import HumanoidAMPBase
    from ..parkour_single import ParkourSingle


class AMPResetStrategy:
    """Basic reset strategy for humanoid amp tasks"""

    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, ctx, state_init: str, hybrid_init_prob: float):
        self.ctx: HumanoidAMPBase = ctx
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._state_init = self.StateInit[state_init]
        self._hybrid_init_prob = hybrid_init_prob
        self._reset_ref_motion_ids = None
        self._reset_ref_motion_times = None

    def __call__(self, env_ids) -> Any:
        self.reset_env(env_ids)

    def reset_env(self, env_ids):
        ctx = self.ctx
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        if len(env_ids) > 0:
            self._reset_actors(env_ids)
            ctx._reset_env_tensors(env_ids)
            ctx._refresh_sim_tensors()
            ## must be called before _compute_observations, which uses cg_progress_buf
            self._refresh_cg_buf(env_ids)
            ## done
            ctx._compute_observations(env_ids)

        self._init_amp_obs(env_ids)
        return

    def _refresh_cg_buf(env_ids):
        pass

    def _reset_actors(self, env_ids):
        if self._state_init == self.StateInit.Default:
            self._reset_default(env_ids)
        elif self._state_init == self.StateInit.Start or self._state_init == self.StateInit.Random:
            self._reset_ref_state_init(env_ids)
        elif self._state_init == self.StateInit.Hybrid:
            self._reset_hybrid_state_init(env_ids)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return

    def _reset_default(self, env_ids):
        ctx = self.ctx
        ctx._humanoid_root_states[env_ids] = ctx._initial_humanoid_root_states[env_ids]
        ctx._dof_pos[env_ids] = ctx._initial_dof_pos[env_ids]
        ctx._dof_vel[env_ids] = ctx._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        ctx = self.ctx
        num_envs = env_ids.shape[0]

        cur_cg_ids = ctx.get_env_cur_cg_id(env_ids)
        motion_cats = ctx.cg_skill_id[cur_cg_ids]
        motion_ids = ctx._motion_lib.sample_motions(num_envs, motion_cats)

        if self._state_init == self.StateInit.Random or self._state_init == self.StateInit.Hybrid:
            motion_times = ctx._motion_lib.sample_time(motion_ids)
        elif self._state_init == self.StateInit.Start:
            motion_times = torch.zeros(num_envs, device=ctx.device)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = ctx._motion_lib.get_motion_state(
            motion_ids, motion_times
        )

        ctx._set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
        )

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _reset_hybrid_state_init(self, env_ids):
        ctx = self.ctx
        num_envs = env_ids.shape[0]
        ref_probs = torch_utils.to_torch(np.array([self._hybrid_init_prob] * num_envs), device=ctx.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if len(ref_reset_ids) > 0:
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if len(default_reset_ids) > 0:
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        ctx = self.ctx
        ctx._compute_amp_observations(env_ids)

        if len(self._reset_default_env_ids) > 0:
            self._init_amp_obs_default(self._reset_default_env_ids)

        if len(self._reset_ref_env_ids) > 0:
            self._init_amp_obs_ref(
                self._reset_ref_env_ids,
                self._reset_ref_motion_ids,
                self._reset_ref_motion_times,
            )

        return

    def _init_amp_obs_default(self, env_ids):
        ctx = self.ctx
        curr_amp_obs = ctx._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        ctx._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        ctx = self.ctx
        dt = ctx.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, ctx._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, ctx._num_amp_obs_steps - 1, device=ctx.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = ctx._motion_lib.get_motion_state(
            motion_ids, motion_times
        )
        amp_obs_demo = build_amp_observations(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_pos,
            ctx._local_root_obs,
            ctx._root_height_obs,
            ctx._dof_obs_size,
            ctx._dof_offsets,
        )
        ctx._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(ctx._hist_amp_obs_buf[env_ids].shape)
        return


"""
在训练时先只用一个技能不加任何deform（甚至可以用mimic reward），支持RSI
然后训练中使用deform/组合技能，不支持RSI
---------------------------------------
随机挑一个cg/全局第一个cg
default: 在origin放到默认态度
start和hybrid: 其实都一样关键在于如何处理地形和速度变化后对于动作的形变
              可以学习https://zhuanlan.zhihu.com/p/465476711，
              把cg全部接触的序列添加到某个grid的回放set中去，然后再更新它。

"""


class CgRSIResetStrategy(AMPResetStrategy):
    """Supports RSI when training single skill without deformation"""

    def __init__(self, ctx, state_init: str, hybrid_init_prob: float):
        super().__init__(ctx, state_init, hybrid_init_prob)
        if TYPE_CHECKING:
            self.ctx: ParkourSingle = ctx

    def _refresh_cg_buf(self, env_ids):
        self.ctx.time_out_buf[env_ids] = 0
        self.ctx.goal_reach_time_buf[env_ids] = 0
        self.ctx.goal_reached_buf[env_ids] = 0

        self.ctx.node_progress_buf[self._reset_default_env_ids] = 0
        self.ctx.cg_progress_buf[self._reset_default_env_ids] = 0
        self.ctx.cg_progress_buf[self._reset_ref_env_ids] = 0
        if self._state_init == self.StateInit.Start:
            self.ctx.node_progress_buf[self._reset_ref_env_ids] = 0
        else:
            self.ctx.node_progress_buf[self._reset_ref_env_ids] = self._reset_ref_node_order

    def _transform(self, root_pos, root_rot, root_vel, root_ang_vel):
        ctx = self.ctx
        root_pos = torch_utils.tf_apply(ctx.root_trans[:, 0:4], ctx.root_trans[:, 4:], root_pos)
        root_rot = torch_utils.tf_apply(q=ctx.root_trans[:, 0:4], v=root_rot)
        root_vel = torch_utils.tf_apply(q=ctx.root_trans[:, 0:4], v=root_vel)
        root_ang_vel = torch_utils.tf_apply(q=ctx.root_trans[:, 0:4], v=root_ang_vel)

        return root_pos, root_rot, root_vel, root_ang_vel

    def _reset_ref_state_init(self, env_ids):
        ctx = self.ctx
        num_envs = env_ids.shape[0]
        cur_cg_ids = ctx.get_env_cur_cg_id(env_ids)
        motion_cats = ctx.cg_skill_id[cur_cg_ids]
        motion_ids = ctx._motion_lib.sample_motions(num_envs, motion_cats)

        if self._state_init == self.StateInit.Random or self._state_init == self.StateInit.Hybrid:
            # randomly sample the cg's node rather than the time.
            rand_order = torch.rand_like(env_ids) * self.ctx.grid_cg_ord_num[cur_cg_ids]
            rand_order = torch.floor(rand_order).long()
            motion_times = ctx._motion_lib.sample_time_byorder(motion_ids, rand_order)
            self._reset_ref_node_order = rand_order

        elif self._state_init == self.StateInit.Start:
            motion_times = torch.zeros(num_envs, device=ctx.device)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = ctx._motion_lib.get_motion_state(
            motion_ids, motion_times
        )

        root_pos, root_rot, root_vel, root_ang_vel = self._transform(root_pos, root_rot, root_vel, root_ang_vel)

        ctx._set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
        )

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        ctx = self.ctx
        dt = ctx.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, ctx._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, ctx._num_amp_obs_steps - 1, device=ctx.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = ctx._motion_lib.get_motion_state(
            motion_ids, motion_times
        )

        root_pos, root_rot, root_vel, root_ang_vel = self._transform(root_pos, root_rot, root_vel, root_ang_vel)

        amp_obs_demo = build_amp_observations(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_pos,
            ctx._local_root_obs,
            ctx._root_height_obs,
            ctx._dof_obs_size,
            ctx._dof_offsets,
        )
        ctx._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(ctx._hist_amp_obs_buf[env_ids].shape)
        return


class CgOriginResetStrategy(AMPResetStrategy):
    """
    Do not support RSI when training combination skill with deformation
    Init the pose using the default one at cg's origin.
    """

    def __init__(self, ctx, state_init: str, hybrid_init_prob: float):
        super().__init__(ctx, state_init, hybrid_init_prob)

    def _reset_actors(self, env_ids):
        if self._state_init == self.StateInit.Default:
            self._reset_default(env_ids)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return

    def _refresh_cg_buf(self, env_ids):
        self.ctx.time_out_buf[env_ids] = 0
        self.ctx.goal_reach_time_buf[env_ids] = 0
        self.ctx.goal_reached_buf[env_ids] = 0
        self.ctx.node_progress_buf[self._reset_default_env_ids] = 0
        self.ctx.cg_progress_buf[self._reset_default_env_ids] = 0


# class SingleSkillRSIResetStrategy(AMPResetStrategy):
#     def __init__(self, ctx: ParkourSingle, state_init: str, hybrid_init_prob: float):
#         super().__init__(ctx, state_init, hybrid_init_prob)
#         ctx.env_terrain_id
#         if TYPE_CHECKING:
#             self.ctx: ParkourSingle = ctx
#     def reset_env(self, env_ids):
#         super().reset_env(env_ids)
#         self.ctx.time_out_buf[env_ids] = 0
#         self.ctx.goal_reach_time_buf[env_ids] = 0
#         self.ctx.goal_reached_buf[env_ids] = 0

#         self.ctx.node_progress_buf[self._reset_default_env_ids] = 0
#         self.ctx.cg_progress_buf[self._reset_default_env_ids] = 0
#         self.ctx.node_progress_buf[self._reset_ref_env_ids] = 0
#         if self._state_init == self.StateInit.Start:
#             self.ctx.cg_progress_buf[self._reset_ref_env_ids] = 0
#         else:
#             self.ctx.cg_progress_buf[self._reset_ref_env_ids] = ...

#     def _reset_default(self, env_ids):
#         ctx = self.ctx
#         ctx._humanoid_root_states[env_ids] = ctx._initial_humanoid_root_states[env_ids]
#         ctx._dof_pos[env_ids] = ctx._initial_dof_pos[env_ids]
#         ctx._dof_vel[env_ids] = ctx._initial_dof_vel[env_ids]
#         ctx._reset_default_env_ids = env_ids
#         return

#     def _reset_ref_state_init(self, env_ids):
#         ctx = self.ctx
#         num_envs = env_ids.shape[0]
#         motion_cats = ctx.cg_skill_id[env_ids]
#         motion_ids = ctx._motion_lib.sample_motions(num_envs, motion_cats)

#         if (
#             self._state_init == self.StateInit.Random
#             or self._state_init == self.StateInit.Hybrid
#         ):
#             motion_times = ctx._motion_lib.sample_time(motion_ids)
#         elif self._state_init == self.StateInit.Start:
#             motion_times = torch.zeros(num_envs, device=ctx.device)
#         else:
#             assert False, "Unsupported state initialization strategy: {:s}".format(
#                 str(self._state_init)
#             )

#         root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
#             ctx._motion_lib.get_motion_state(motion_ids, motion_times)
#         )

#         root_pos, root_rot, dof_pos, root_vel, root_ang_vel = self._transform(
#             root_pos, root_rot, dof_pos, root_vel, root_ang_vel
#         )

#         ctx._set_env_state(
#             env_ids=env_ids,
#             root_pos=root_pos,
#             root_rot=root_rot,
#             dof_pos=dof_pos,
#             root_vel=root_vel,
#             root_ang_vel=root_ang_vel,
#             dof_vel=dof_vel,
#         )

#         self._reset_ref_env_ids = env_ids
#         self._reset_ref_motion_ids = motion_ids
#         self._reset_ref_motion_times = motion_times
#         return


#     def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
#         ctx = self.ctx
#         dt = ctx.dt
#         motion_ids = torch.tile(
#             motion_ids.unsqueeze(-1), [1, ctx._num_amp_obs_steps - 1]
#         )
#         motion_times = motion_times.unsqueeze(-1)
#         time_steps = -dt * (
#             torch.arange(0, ctx._num_amp_obs_steps - 1, device=ctx.device) + 1
#         )
#         motion_times = motion_times + time_steps

#         motion_ids = motion_ids.view(-1)
#         motion_times = motion_times.view(-1)
#         root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
#             ctx._motion_lib.get_motion_state(motion_ids, motion_times)
#         )
#         amp_obs_demo = build_amp_observations(
#             root_pos,
#             root_rot,
#             root_vel,
#             root_ang_vel,
#             dof_pos,
#             dof_vel,
#             key_pos,
#             ctx._local_root_obs,
#             ctx._root_height_obs,
#             ctx._dof_obs_size,
#             ctx._dof_offsets,
#         )
#         ctx._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(
#             ctx._hist_amp_obs_buf[env_ids].shape
#         )
#         return
