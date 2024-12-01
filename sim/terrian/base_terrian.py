from typing import List
from isaacgym import terrain_utils, torch_utils
import random
import numpy as np
import torch

from sim.terrian.terrain_leggedgym import Terrain
from sim.terrian.geometric import draw_rectangle_, FrameMap
from data.skill_lib import SkillLib
from data.contact_graph import ContactGraph


class TerrainParkour(Terrain):
    def __init__(self, cfg, num_envs, motion_lib: SkillLib):
        self._skill_combinations = cfg.skill_combinations
        self._combination_weights = [i.weight for i in self._skill_combinations]
        self._motion_lib = motion_lib

        self._ego2pixel_scale = torch.tensor([cfg.horizontal_scale, cfg.horizontal_scale, cfg.vertical_scale])

        self.graph_list: List[List[ContactGraph]] = []
        # _yaw_distribution = torch.distributions.MultivariateNormal(
        #     torch.zeros(3), torch.diag_embed(torch.tensor([0, 0, 1]))
        # )
        self._yaw_distribution = torch.distributions.Normal(0, 0.37)
        super().__init__(cfg, num_envs)

    def get_graph_list(self):
        return self.graph_list

    def ego2pixel(self, q, t, ego_position: torch.Tensor) -> np.ndarray:
        ego_position = torch_utils.tf_apply(q, t, ego_position)
        pixel_coord = torch.round(ego_position / self._ego2pixel_scale).to(torch.int32)
        return (pixel_coord[..., 0:2]).cpu().numpy()

    def curiculum(self, use_random=False, max_difficulty=False):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                if len(self._combination_weights) > 1:
                    combine_item = np.random.choice(self._skill_combinations, p=self._combination_weights)
                else:
                    combine_item = self._skill_combinations[0]
                if not combine_item.fix_order:
                    random.shuffle(combine_item.skills)

                skills = combine_item.skills + ["idle"] if self.cfg.add_idle_to_last else combine_item.skills
                if max_difficulty:
                    terrain = self.make_terrain(skills, i, j, np.random.uniform(0.7, 1))
                else:
                    terrain = self.make_terrain(skills, i, j, combine_item.get("diffculty", None))

                self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, row, col, skills, difficulty=0.1):
        terrain = FrameMap(
            width=self.length_per_env_pixels,
            length=self.width_per_env_pixels,
            offset=np.array([0, self.env_width // 2]),
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )

        graph_list: List[ContactGraph] = []
        lines, rotations = [], []
        for skill_name in skills:
            assert type(skill_name) == str
            cg = self._motion_lib.get_cg_by_skill(skill_name)
            device = cg.device
            if self.cfg.enable_deform:
                cg.deform()
            if self.cfg.enable_scale:
                cg.transform(scale=torch.tensor([1.0, 1.0, 1.0], device=device))

            yaw = self._yaw_distribution.sample()
            line = cg.get_main_line()
            line = torch_utils.quat_apply(
                torch_utils.quat_from_euler_xyz(roll=torch.tensor(0), pitch=torch.tensor(0), yaw=yaw).to(device),
                line,
            ).view(-1, 3)

            lines.append(line)
            rotations.append(yaw)
            graph_list.append(cg)

        ################## connect graphs ##################
        device = lines[0].device
        env_ori_x, env_ori_y = self.env_origin_xy(row, col)
        global2env_trans = torch.tensor([env_ori_x, env_ori_y, 0], device=device)
        translations = [torch.tensor([0, 0, 0], device=device)]
        translated_lines = [lines[0] + translations[0]]
        for i in range(1, len(lines)):
            translation = translated_lines[i - 1][1] - lines[i][0]
            translated_line = lines[i] + translation
            translated_lines.append(translated_line)
            translations.append(translation)
            graph_list[i].set_coord(*(translation + global2env_trans).tolist(), 0, 0, rotations[i])

        graph_list[0].set_coord(*(global2env_trans).tolist(), 0.0, 0.0, rotations[0])

        ################## generate skill terrains ##################
        for cg, translation, rotation in zip(graph_list, translations, rotations):
            sub_pixel_length, sub_pixel_height = (
                1 + int(cg.len_x / self.cfg.horizontal_scale),
                1 + int(cg.len_y / self.cfg.horizontal_scale),
            )
            subterrain = FrameMap(
                width=sub_pixel_length,
                length=sub_pixel_height,
                offset=np.array([0, sub_pixel_height // 2]),
                vertical_scale=self.cfg.vertical_scale,
                horizontal_scale=self.cfg.horizontal_scale,
            )
            subterrain.set_coord(translation, rotation)

            if cg.skill_type == "vault" or cg.skill_type == "jump":
                self.generate_vault_terrain(subterrain, cg, difficulty)
            elif (
                cg.skill_type == "run" or cg.skill_type == "walk" or cg.skill_type == "roll" or cg.skill_type == "idle"
            ):
                self.generate_walk_terrain(subterrain, cg, difficulty)
            else:
                return NotImplementedError

            terrain.draw_on_self(subterrain)

        self.graph_list.append(graph_list)
        return terrain

    def generate_walk_terrain(self, subterrain: FrameMap, cg: ContactGraph, difficulty):
        discrete_obstacles_height = 0.03 + difficulty * 0.15
        num_rectangles = 20
        rectangle_min_size = 0.5
        rectangle_max_size = 1.0
        terrain_utils.discrete_obstacles_terrain(
            subterrain,
            discrete_obstacles_height,
            rectangle_min_size,
            rectangle_max_size,
            num_rectangles,
            platform_size=3.0,
        )

        # enforce the contact point is on the surface
        for node in cg.nodes:
            px, py = subterrain.to_uv(node.position[:2])
            height = subterrain.height_field_raw[px, py].item() * self.cfg.vertical_scale
            node.position[2] = height

    def generate_vault_terrain(self, subterrain, cg: ContactGraph, difficulty):
        lastx = 0
        for node in cg.nodes:
            x, _, height = node.position.tolist()
            height = int(height / self.cfg.vertical_scale)
            delta = int(1 / self.cfg.horizontal_scale)
            x = int(x / self.cfg.horizontal_scale)
            if x - delta < (lastx + x) * 0.75:
                sc = x - delta
            else:
                sc = (lastx + x) // 2
            subterrain.height_field_raw[..., sc:] = height  # BUG EGO2PIXEL
            lastx = x
        return

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x, env_origin_y = self.env_origin_xy(i, j)
        x1 = int((self.env_length / 2.0 - 0.5) / terrain.horizontal_scale)  # within 1 meter square range
        x2 = int((self.env_length / 2.0 + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 0.5) / terrain.horizontal_scale)
        if self.cfg.origin_zero_z:
            env_origin_z = 0
        else:
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        # self.env_slope_vec[i, j] = terrain.slope_vector

    def env_origin_xy(self, row, col):
        return row * self.env_length + 1.0, (col + 0.5) * self.env_width
