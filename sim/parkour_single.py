from typing import List
from isaacgym import gymtorch, torch_utils, gymapi
from torch_geometric.data import Data, Batch
import torch
import time
from data.contact_graph_base import CNode
from sim.humanoid_amp import HumanoidAMPTask
from sim.terrian.base_terrian import TerrainParkour


class ParkourSingle(HumanoidAMPTask):
    def __init__(
        self, cfg, sim_params, physics_engine, device_type, device_id, headless
    ):
        self.terrain = None
        self._cg_nums = 0

        super().__init__(
            cfg, sim_params, physics_engine, device_type, device_id, headless
        )
        # ---------------init contact graph buffers----------------
        if isinstance(self.terrain, TerrainParkour):
            graph_list = self.terrain.get_graph_list()
            # 每个环境属于哪个grid
            self.env_terrain_id = torch.randint(
                0, self.cfg.num_sub_terrains, (self.num_envs), device=self.device
            )
            # 每个grid的所有cg数目和node数目
            self.terrain_cg_nums = torch.tensor(
                [len(x) for x in graph_list], device=self.device, dtype=torch.long
            )
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
            # 初始化每个cg在大的图里的i位置
            grid_cg_offset, x = [], 0
            self._cg_rot_inv, self._cg_trans_inv = [], []
            for cgs in graph_list:
                grid_cg_offset.append(x)
                x += len(cgs)
                for cg in cgs:
                    self._cg_rot_inv.append(cg._root_rotation)
                    self._cg_trans_inv.append(cg._root_translation)
            self.grid_cg_offset = torch.tensor(
                grid_cg_offset, device=self.device, dtype=torch.long
            )  # cumsum
            self._cg_rot_inv, self._cg_trans_inv = torch_utils.tf_inverse(
                torch.cat(self._cg_rot_inv),
                torch.cat(self._cg_trans_inv),
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
            )  # TODO 给最后一个加上idle图

            self.node_progress_buf = torch.zeros_like(self.progress_buf)
            self.cg_progress_buf = torch.zeros_like(self.progress_buf)
            # records the frames spent on waiting reaching the current goal
            self.time_out_buf = torch.zeros_like(self.progress_buf)
            # num of frames contacting with the current goal
            self.goal_reach_time_buf = torch.zeros_like(self.progress_buf)
            self.goal_reach_buf = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
            return

    ############### parkour env creation methods ################

    def _create_envs(self, num_envs, spacing, num_per_row):
        return super()._create_envs(num_envs, spacing, num_per_row)
        # TODO 添加penalised_contact_indices、termination_contact_indices
        # 添加触点的sensor和camera
        # get env origins

    def _create_ground_plane(self):
        """create terrain after sim initialization/before env creation"""
        # if self.cfg.depth.use_camera: TODO
        #     self.graphics_device_id = self.sim_device_id  # required in headless mode

        start = time()
        print("Start creating ground...")
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type == "plane":
            super()._create_ground_plane()
        elif mesh_type in ["heightfield", "trimesh"]:
            self.terrain = TerrainParkour(self.cfg.terrain, self.num_envs)
            if mesh_type == "heightfield":
                self._create_heightfield()
            elif mesh_type == "trimesh":
                self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]"
            )
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

        self.gym.add_heightfield(
            self.sim, self.terrain.heightsamples.flatten(order="C"), hf_params
        )
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
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
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

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

        env_grid_id = self.env_terrain_id[env_ids]
        q, t = torch_utils.tf_combine(
            self._cg_rot_inv[env_grid_id],
            self._cg_trans_inv[env_grid_id],
            self._rigid_body_pos[env_ids],
            self._rigid_body_rot[env_ids],  # quat
        )
        delta_tf = torch.cat([q, t], dim=1)

        for ids, key in zip(
            self._get_env_cur_next_cg_id(env_ids), ("graph_obs", "graph_obs_next")
        ):
            batch = Batch.from_data_list(self.grid_cgs.index_select(ids))
            node_counts = torch.bincount(batch.batch)
            expand_tf = torch.cat(
                [
                    delta_tf[i].unsqueeze(0).expand(node_counts[i], -1)
                    for i in range(delta_tf.size(0))
                ]
            )
            batch.x = CNode.tf_apply_onfeat(expand_tf[0:4], expand_tf[4:], batch.x)
            self.extras[key] = batch

        return

    def _compute_reward(self, actions):
        """Since it is not deepmimic-type reward, we only return contact reward"""
        # NOTE maybe multiple reward works
        # NOTE 用不用-exp函数
        batch = self.extras["graph_obs"]
        mask = batch.x[:, -1] == self.node_progress_buf
        filtered_x = batch.x[mask]
        filtered_batch_id = batch.batch[mask]
        skeleid = filtered_x[:, -2]
        indices = (skeleid * 15 + 1).unsqueeze(1) + torch.arange(3).unsqueeze(0)
        indices = indices.to(torch.long)
        ske_state = self.obs_buf[filtered_batch_id][indices]
        self.goal_reach_time_buf.set_(False)

        # reward distance
        d = torch.norm(ske_state[..., :3] - filtered_x[..., :3])
        rd = torch.exp(-d + 1e-7)
        self.goal_reach_buf.scatter_reduce_(
            0, filtered_batch_id, d > self.cfg.reward.rd_thresh, reduce="sum"
        )

        # BUG Reproducibility issue
        self.rew_buf[:] = self.rew_buf.scatter_reduce(
            0, filtered_batch_id, rd, reduce="sum"
        )

        self.goal_reach_time_buf[self.goal_reach_buf] += 1
        self._update_reward_goals()
        return

        # TODO body的位置是哪里？怎么表达接触力和法向量
        # TODO 加上wococo的通用人形稳定奖励

    ############### steps ################
    def _update_reward_goals(self):
        self.time_out_buf += 1
        # for a node group, if contact frames exceeds required num, then move the goal to the next node(e.g. for idle)
        cur_cg_id = self._get_env_cur_cg_id()
        mask_update = self.goal_reach_time_buf >= self.cg_sustain_times[cur_cg_id]
        self.node_progress_buf[mask_update] += 1
        self.time_out_buf[mask_update] = 0
        # if all nodes in a cg are finished
        mask_node_done = self.node_progress_buf > self.grid_cg_ord_num[cur_cg_id]
        self.node_progress_buf[mask_node_done] = 0
        self.cg_progress_buf[mask_node_done] += 1
        return

    ############### utils ################
    def _get_env_cur_cg_id(self, env_ids=None):
        if env_ids is None:
            return self.grid_cg_offset[self.env_terrain_id] + self.cg_progress_buf
        env_grid_id = self.env_terrain_id[env_ids]
        return self.grid_cg_offset[env_grid_id] + self.cg_progress_buf[env_ids]

    def _get_env_cur_next_cg_id(self, env_ids):
        env_grid_id = self.env_terrain_id[env_ids]
        env_cg_id = self.cg_progress_buf[env_ids]
        env_cur_cg_id = self.grid_cg_offset[env_grid_id] + env_cg_id
        env_next_cg_id = torch.where(
            env_cg_id == self.terrain_cg_nums[env_grid_id],
            self._cg_nums,
            env_cur_cg_id + 1,
        )
        return env_cur_cg_id, env_next_cg_id


# def _compute_reward_nobatch(self, actions):
#     for i in range(self.num_envs):
#         cg: Data = self.extras["graph_obs"][i]
#         if self.node_progress_buf[i] == cg.num_nodes:
#             self.node_progress_buf[i] = 0
#             self.cg_progress_buf[i] += 1
#         else:
#             nodeid = self.node_progress_buf[i]

#             goals = cg.x[:, cg.x[:, -1] == nodeid]
#             skeleid = goals[:, -2]
#             indices = (skeleid * 15 + 1).unsqueeze(1) + torch.arange(3).unsqueeze(0)
#             ske_pos = (ske_state := self.obs_buf[i][indices])[:3]

#     return
