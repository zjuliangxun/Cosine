from isaacgym import gymtorch, torch_utils, gymapi
from enum import Enum
import torch
import time
from data.motion_lib import MotionLib
from sim.humanoid_amp import HumanoidAMPTask
from sim.humanoid import Humanoid, dof_to_obs
from sim.terrian.base_terrian import TerrainParkour


class ParkourSingle(HumanoidAMPTask):
    def __init__(
        self, cfg, sim_params, physics_engine, device_type, device_id, headless
    ):
        self.terrain: TerrainParkour = None
        super().__init__(
            cfg, sim_params, physics_engine, device_type, device_id, headless
        )

        return

    ############### parkour env creation methods ################
    def _create_envs(self, num_envs, spacing, num_per_row):
        return super()._create_envs(num_envs, spacing, num_per_row)
        # TODO 添加penalised_contact_indices、termination_contact_indices
        # 添加触点的sensor和camera

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

    ############### parkour env creation methods ################
