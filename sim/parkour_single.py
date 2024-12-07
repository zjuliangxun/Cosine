from typing import List
from time import time
from omegaconf import OmegaConf
from isaacgym import torch_utils, gymapi, gymutil
from torch_geometric.data import Data, Batch
import torch
import numpy as np

from data.skill_lib import SkillLib
from data.contact_graph_base import CNode
from data.contact_graph import ContactGraph

from sim.humanoid_amp import HumanoidAMPTask
from sim.terrian.base_terrian import TerrainParkour


# TODO legged robots有很多trick
# TODO 定期刷新env系统，有些变量应该需要处理归入一个函数中去
class ParkourSingle(HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.terrain: TerrainParkour = None
        self._cg_nums = 0
        # the environment.yaml
        cfg = OmegaConf.structured(cfg)

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        assert self._dof_offsets[-1] == self.num_dof

    def create_terrain_related_buffers(self):
        #################### init contact graph buffers #####################
        if isinstance(self.terrain, TerrainParkour):  # [ ] 没有处理地面的case
            graph_list = self.terrain.get_graph_list()
            # 每个环境属于哪个grid
            self.env_terrain_id = torch.randint(
                0, self.cfg.terrain.num_sub_terrains, (self.num_envs,), device=self.device
            )
            # 每个grid的所有cg数目和node数目
            self.terrain_cg_nums = torch.tensor([len(x) for x in graph_list], device=self.device, dtype=torch.long)
            self.grid_cg_ord_num = torch.tensor(
                [x.order for cgs in graph_list for x in cgs],
                device=self.device,
                dtype=torch.long,
            )
            self.cg_sustain_times = torch.tensor(
                [x.sustain_time for cgs in graph_list for x in cgs],
                device=self.device,
                dtype=torch.long,
            )
            self.cg_skill_id = torch.tensor(
                [self._motion_lib.get_skill_id(cg.skill_type) for cgs in graph_list for cg in cgs],
                device=self.device,
                dtype=torch.long,
            )
            # 初始化每个cg在大的图里的i位置
            grid_cg_offset, cnt = [], 0
            self._cg_rot_inv, self._cg_trans_inv = [], []
            for cgs in graph_list:
                grid_cg_offset.append(cnt)
                cnt += len(cgs)
                for cg in cgs:
                    self._cg_rot_inv.append(cg._root_rotation)
                    self._cg_trans_inv.append(cg._root_translation)
            self.grid_cg_offset = torch.tensor(grid_cg_offset, device=self.device, dtype=torch.long)  # cumsum
            self._cg_rot_inv, self._cg_trans_inv = torch_utils.tf_inverse(
                torch.cat(self._cg_rot_inv).view(-1, 4),
                torch.cat(self._cg_trans_inv).view(-1, 3),
            )
            # put all the cgs into one PyG Batch object
            self.grid_cgs = Batch.from_data_list(
                [
                    Data(
                        x=cg.get_feat_tensor(),
                        edge_index=cg.edge_index_tensor,
                    )
                    for grid_cgs in graph_list
                    for cg in grid_cgs
                ]
            )

            self.node_progress_buf = torch.zeros_like(self.progress_buf)
            self.cg_progress_buf = torch.zeros_like(self.progress_buf)
            # records the frames spent on waiting reaching the current goal
            self.time_out_buf = torch.zeros_like(self.progress_buf)
            # num of frames contacting with the current goal
            self.goal_reach_time_buf = torch.zeros_like(self.progress_buf)
            self.goal_reached_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self.root_trans = [torch.cat(list(cgs[0].get_coord()), dim=-1) for cgs in graph_list]
            self.root_trans = torch.cat(self.root_trans, dim=0)  # TODO need 刷新当环境改变，

            return

    ############### parkour env creation methods ################

    def _create_envs(self, num_envs, spacing, num_per_row):
        return super()._create_envs(num_envs, spacing, num_per_row)
        # TODO 添加penalised_contact_indices、termination_contact_indices
        # 添加触点的sensor和camera
        # get env origins

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        char_h = 0.89
        env_origin = self.terrain.get_env_origin(self.env_terrain_id[env_id])
        env_origin[2] += char_h
        start_pose.p = gymapi.Vec3(*env_origin)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(
            env_ptr,
            humanoid_asset,
            start_pose,
            "humanoid",
            col_group,
            col_filter,
            segmentation_id,
        )

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(
                env_ptr,
                humanoid_handle,
                j,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.54, 0.85, 0.2),
            )

        if self._pd_control:
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        self.humanoid_handles.append(humanoid_handle)

        return

    def _create_ground_plane(self):
        """create terrain after sim initialization/before env creation"""
        # [ ] if self.cfg.depth.use_camera:
        #     self.graphics_device_id = self.sim_device_id  # required in headless mode

        start = time()
        print("Start creating ground...")
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type == "plane":
            super()._create_ground_plane()
        elif mesh_type in ["heightfield", "trimesh"]:
            self.create_terrain_related_buffers()
            if mesh_type == "heightfield":
                self._create_heightfield()
            elif mesh_type == "trimesh":
                self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))

    def _create_heightfield(self):
        """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.cfg.terrain.horizontal_scale
        hf_params.row_scale = self.cfg.terrain.horizontal_scale
        hf_params.vertical_scale = self.cfg.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.border
        hf_params.transform.p.y = -self.terrain.border
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.flatten(order="C"), hf_params)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        )

    def _create_trimesh(self):
        """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        Very slow when horizontal_scale is small
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        print("Adding trimesh to simulation...")
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order="C"),
            self.terrain.triangles.flatten(order="C"),
            tm_params,
        )
        print("Trimesh added")
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        )

    def _load_motion(self, motion_file):
        assert self._dof_offsets[-1] == self.num_dof
        self._motion_lib = SkillLib(
            motion_file=motion_file,
            dof_body_ids=self._dof_body_ids,
            dof_offsets=self._dof_offsets,
            key_body_ids=self._key_body_ids.cpu().numpy(),
            equal_motion_weights=self._equal_motion_weights,
            device=self.device,
        )
        return

    ############### Compute obs/rewards ################

    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        self._compute_task_obs(env_ids)

        if env_ids is None:
            self.obs_buf[:] = humanoid_obs
        else:
            self.obs_buf[env_ids] = humanoid_obs
        return

    def _compute_task_obs(self, env_ids=None):
        """the graph obs"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        env_cg_id = self.get_env_cur_cg_id(env_ids)
        q, t = torch_utils.tf_combine(
            self._cg_rot_inv[env_cg_id],
            self._cg_trans_inv[env_cg_id],
            self._humanoid_root_states[env_ids, 3:7],
            self._humanoid_root_states[env_ids, 0:3],
        )
        delta_tf = torch.cat([q, t], dim=1)

        for ids, key in zip(self._get_env_cur_next_cg_id(env_ids), ("graph_obs", "graph_obs_next")):
            batch = Batch.from_data_list(self.grid_cgs.index_select(ids))
            node_counts = torch.bincount(batch.batch)
            expand_tf = torch.cat(
                [delta_tf[i].unsqueeze(0).expand(node_counts[i], -1) for i in range(delta_tf.size(0))]
            )
            batch.x = CNode.tf_apply_onfeat(expand_tf[..., 0:4], expand_tf[..., 4:], batch.x)
            self.extras[key] = batch

        return

    def _compute_reward(self, actions):
        """Since it is not deepmimic-type reward, we only return contact reward"""
        # NOTE maybe multiple reward works
        # NOTE 用不用-exp函数
        batch = self.extras["graph_obs"]
        mask = batch.x[:, -1] == self.node_progress_buf[batch.batch]
        filtered_x = batch.x[mask]
        skeleid = get_skeleton_id(filtered_x)  # BUG input data all left foots!
        col_indices = (
            (skeleid * 15 + 1).unsqueeze(1)
            + torch.arange(3, device=skeleid.device).unsqueeze(0).repeat(skeleid.size(0), 1)
        ).to(torch.long)
        row_indices = torch.arange(skeleid.shape[0], device=skeleid.device)
        filtered_batch_id = batch.batch[mask]
        ske_state = self.obs_buf[filtered_batch_id][row_indices[:, None], col_indices]
        self.goal_reached_buf.fill_(False)

        # reward distance
        d = torch.norm(ske_state[..., :3] - filtered_x[..., :3], dim=-1)
        rd = torch.exp(-d + 1e-7)

        self.goal_reached_buf.scatter_reduce_(0, filtered_batch_id, d < self.cfg.contact.reached_thresh, reduce="sum")
        self.goal_reach_time_buf[self.goal_reached_buf] += 1

        # BUG Reproducibility issue
        self.rew_buf[:] = self.rew_buf.scatter_reduce(0, filtered_batch_id, rd, reduce="sum")

        self._update_reward_goals()
        return

        # TODO body的位置是哪里？怎么表达接触力和法向量
        # TODO 加上wococo的通用人形稳定奖励

    ############### steps ################
    def _update_reward_goals(self):
        self.time_out_buf += 1
        # for a node group, if contact frames exceeds required num, then move the goal to the next node(e.g. for idle)
        cur_cg_id = self.get_env_cur_cg_id()
        mask_update = self.goal_reach_time_buf >= self.cg_sustain_times[cur_cg_id]
        self.node_progress_buf[mask_update] += 1
        self.time_out_buf[mask_update] = 0
        # if all nodes in a cg are finished
        mask_node_done = self.node_progress_buf > self.grid_cg_ord_num[cur_cg_id]
        self.node_progress_buf[mask_node_done] = 0
        self.cg_progress_buf[mask_node_done] += 1
        return

    ############### utils ################
    def get_env_cur_cg_id(self, env_ids=None):
        if env_ids is None:
            return self.grid_cg_offset[self.env_terrain_id] + self.cg_progress_buf
        env_grid_id = self.env_terrain_id[env_ids]
        return self.grid_cg_offset[env_grid_id] + self.cg_progress_buf[env_ids]

    def _get_env_cur_next_cg_id(self, env_ids):
        env_grid_id = self.env_terrain_id[env_ids]
        env_cg_id = self.cg_progress_buf[env_ids]
        env_cur_cg_id = self.grid_cg_offset[env_grid_id] + env_cg_id
        env_next_cg_id = torch.where(
            env_cg_id == self.terrain_cg_nums[env_grid_id] - 1,  # env_cg_id starts from 0!
            env_cur_cg_id,
            env_cur_cg_id + 1,
        )
        return env_cur_cg_id, env_next_cg_id

    ############### draw task ################
    def _draw_task(self):
        if self.cfg.visualize.vis_contacts:
            self.gym.clear_lines(self.viewer)

            # draw contact goals
            lookat_id = 0  # the id of rendering env
            unreach_goal = gymutil.WireframeBoxGeometry(0.1, 0.1, 0.15, None, color=(1, 0, 0))
            reached_goal = gymutil.WireframeBoxGeometry(0.1, 0.1, 0.15, None, color=(0, 1, 0))
            current_goal = gymutil.WireframeBoxGeometry(0.1, 0.1, 0.15, None, color=(1, 0.67, 0))

            gridid = self.env_terrain_id[lookat_id].cpu().item()
            cg_offset = self.grid_cg_offset[gridid].cpu().item()
            cur_cg_id = self.cg_progress_buf[lookat_id].cpu().item()  # [ ] 到底是全局还是每个grid的cg号？
            cur_node_id = self.node_progress_buf[lookat_id].cpu().item()

            cg_list: List[ContactGraph] = self.terrain.get_graph_list()[gridid]
            for i, cg in enumerate(cg_list):
                goal_positions = cg.get_feat_tensor(device="cpu")[:, :3]
                for j in range(len(goal_positions)):
                    goal = goal_positions[j]
                    goal_z = goal[2]
                    pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)

                    i += cg_offset
                    if i == cur_cg_id and j == cur_node_id:
                        flag = current_goal
                    elif i < cur_cg_id or (i == cur_cg_id and j < cur_node_id):
                        flag = reached_goal
                    else:
                        flag = unreach_goal
                    gymutil.draw_lines(flag, self.gym, self.viewer, self.envs[lookat_id], pose)

        super()._draw_task()
        return

    def fetch_amp_obs_demo(self, num_samples):
        motion_ids = self._motion_lib.sample_motions(num_samples)

        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo_flat = (
            self.build_amp_obs_demo(motion_ids, motion_times0, self._num_amp_obs_steps)
            .to(self.device)
            .view(-1, self.get_num_amp_obs())
        )

        return amp_obs_demo_flat

    def fetch_amp_obs_demo_per_id(self, num_samples, motion_id):
        motion_ids = torch.tensor([motion_id for _ in range(num_samples)], dtype=torch.long).view(-1)

        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        enc_window_size = self.dt * (self._num_amp_obs_enc_steps - 1)

        enc_motion_times = self._motion_lib.sample_time(motion_ids, truncate_time=enc_window_size)
        # make sure not to add more than motion clip length, negative amp_obs will show zero index amp_obs instead
        enc_motion_times += torch.clip(self._motion_lib._motion_lengths[motion_ids], max=enc_window_size)

        enc_amp_obs_demo = self.build_amp_obs_demo(motion_ids, enc_motion_times, self._num_amp_obs_enc_steps).view(
            -1, self._num_amp_obs_enc_steps, self._num_amp_obs_per_step
        )

        enc_amp_obs_demo_flat = enc_amp_obs_demo.to(self.device).view(-1, self.get_num_enc_amp_obs())

        return motion_ids, enc_motion_times, enc_amp_obs_demo_flat

    def fetch_amp_obs_demo_enc_pair(self, num_samples):
        motion_ids = self._motion_lib.sample_motions(num_samples)

        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        enc_window_size = self.dt * (self._num_amp_obs_enc_steps - 1)

        enc_motion_times = self._motion_lib.sample_time(motion_ids, truncate_time=enc_window_size)
        # make sure not to add more than motion clip length, negative amp_obs will show zero index amp_obs instead
        enc_motion_times += torch.clip(self._motion_lib._motion_lengths[motion_ids], max=enc_window_size)

        # sub-window-size is for the amp_obs contained within the enc-amp-obs. make sure we sample only within the valid portion of the motion
        sub_window_size = (
            torch.clip(self._motion_lib._motion_lengths[motion_ids], max=enc_window_size)
            - self.dt * self._num_amp_obs_steps
        )
        motion_times = enc_motion_times - torch.rand(enc_motion_times.shape, device=self.device) * sub_window_size

        enc_amp_obs_demo = self.build_amp_obs_demo(motion_ids, enc_motion_times, self._num_amp_obs_enc_steps).view(
            -1, self._num_amp_obs_enc_steps, self._num_amp_obs_per_step
        )
        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times, self._num_amp_obs_steps).view(
            -1, self._num_amp_obs_steps, self._num_amp_obs_per_step
        )

        enc_amp_obs_demo_flat = enc_amp_obs_demo.to(self.device).view(-1, self.get_num_enc_amp_obs())
        amp_obs_demo_flat = amp_obs_demo.to(self.device).view(-1, self.get_num_amp_obs())

        return motion_ids, enc_motion_times, enc_amp_obs_demo_flat, motion_times, amp_obs_demo_flat

    def fetch_amp_obs_demo_pair(self, num_samples):
        motion_ids = self._motion_lib.sample_motions(num_samples)
        cat_motion_ids = torch.cat((motion_ids, motion_ids), dim=0)

        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        enc_window_size = self.dt * (self._num_amp_obs_enc_steps - 1)

        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=enc_window_size)
        motion_times0 += torch.clip(self._motion_lib._motion_lengths[motion_ids], max=enc_window_size)

        motion_times1 = motion_times0 + torch.rand(motion_times0.shape, device=self._motion_lib._device) * 0.5
        motion_times1 = torch.min(motion_times1, self._motion_lib._motion_lengths[motion_ids])

        motion_times = torch.cat((motion_times0, motion_times1), dim=0)

        amp_obs_demo = self.build_amp_obs_demo(cat_motion_ids, motion_times, self._num_amp_obs_enc_steps).view(
            -1, self._num_amp_obs_enc_steps, self._num_amp_obs_per_step
        )
        amp_obs_demo0, amp_obs_demo1 = torch.split(amp_obs_demo, num_samples)

        amp_obs_demo0_flat = amp_obs_demo0.to(self.device).view(-1, self.get_num_enc_amp_obs())

        amp_obs_demo1_flat = amp_obs_demo1.to(self.device).view(-1, self.get_num_enc_amp_obs())

        return motion_ids, motion_times0, amp_obs_demo0_flat, motion_times1, amp_obs_demo1_flat
