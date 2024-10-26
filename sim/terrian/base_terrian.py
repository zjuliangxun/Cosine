from typing import List
import random
import numpy as np
import torch
from isaacgym import terrain_utils
from isaacgym import torch_utils

from sim.terrian.terrain_leggedgym import Terrain
from sim.terrian.geometric import draw_rectangle_
from data.skill_lib import SkillLib
from data.contact_graph import ContactGraph


class TerrainParkour(Terrain):
    def __init__(self, cfg, num_envs, motion_lib: SkillLib):
        self._skill_combinations = cfg.skill_combinations
        self._combination_weights = [i.weight for i in self._skill_combinations]
        self._motion_lib = motion_lib

        self._offset = torch.tensor([5, cfg.terrrain_width // 2, 0])
        self._ego2pixel_scale = torch.tensor(
            [cfg.horizontal_scale, cfg.horizontal_scale, cfg.vertical_scale]
        )

        self.graph_list: List[List[ContactGraph]] = []
        # _roll_distribution = torch.distributions.MultivariateNormal(
        #     torch.zeros(3), torch.diag_embed(torch.tensor([0, 0, 1]))
        # )
        self._roll_distribution = torch.distributions.Normal(0, 0.37)
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
                combine_item = np.random.choice(
                    self._skill_combinations, p=self._combination_weights
                )
                if not combine_item.fix_order:
                    random.shuffle(combine_item.skills)
                if max_difficulty and combine_item.diffculty is not None:  # TODO CHECK
                    terrain = self.make_terrain(
                        combine_item.skills, np.random.uniform(0.7, 1)
                    )
                else:
                    terrain = self.make_terrain(
                        combine_item.skills, combine_item.diffculty
                    )

                self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, skills, difficulty=0.1):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.length_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )

        graph_list: List[ContactGraph] = []
        lines, rotations = [], []
        for skill in skills:
            assert type(skill) == str
            cg = self._motion_lib.get_cg_by_skill(skill)
            if self.cfg.enable_deform:
                cg.deform()
            if self.cfg.enable_scale:
                cg.transform(scale=torch.tensor([1.0, 1.0, 1.0], device=cg.device))

            roll = self._roll_distribution.sample()
            line = cg.get_main_line()
            line = torch_utils.quat_apply(
                torch_utils.quat_from_euler_xyz(roll=roll, pitch=0, yaw=0),
                line,
            )

            lines.append(line)
            rotations.append(roll)
            graph_list.append(cg)
            self._cg_nums += 1
            # self.add_roughness(terrain)

        # connect graphs------------------------------------------------
        translations = [self._offset.to(lines[0].device)]
        translated_lines = [lines[0] + translations[0]]
        for i in range(1, len(lines)):
            translation = translated_lines[i - 1][1] - lines[i][0]
            translated_line = lines[i] + translation
            translated_lines.append(translated_line)
            translations.append(translation)
            graph_list[i].set_coord(*translation.tolist(), 0, 0, rotations[i])

        # NOTE 感觉是从左下角起的env，所以要把cg的坐标系转换到env坐标系
        graph_list[0].set_coord(*self._offset.tolist(), 0.0, 0, rotations[0])

        # generate skill terrains------------------------------------------------
        for cg, translation, rotation in zip(graph_list, translations, rotations):
            sub_pixel_length, sub_pixel_height = (
                int(cg.len_x / self.cfg.horizontal_scale),
                int(cg.len_y / self.cfg.horizontal_scale),
            )
            subterrain = np.zeros((sub_pixel_length, sub_pixel_height), dtype=np.int16)

            if cg.skill_type == "vault" or cg.skill_type == "jump":
                self.generate_vault_terrain(subterrain, cg, difficulty)
            elif (
                cg.skill_type == "run"
                or cg.skill_type == "walk"
                or cg.skill_type == "roll"
                or cg.skill_type == "idle"
            ):
                self.generate_walk_terrain(subterrain, cg, difficulty)
            else:
                return NotImplementedError

            draw_rectangle_(
                subterrain,
                terrain.height_field_raw,
                rotation,
                translation.cpu().numpy(),
            )

        self.graph_list.append(graph_list)
        return terrain

    def generate_walk_terrain(self, subterrain, cg: ContactGraph, difficulty):
        discrete_obstacles_height = 0.03 + difficulty * 0.15
        num_rectangles = 20
        rectangle_min_size = 0.5
        rectangle_max_size = 2.0
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
            x, y, height = node.position.tolist()
            y = int(y / self.cfg.horizontal_scale)
            x = int(x / self.cfg.horizontal_scale)
            height = subterrain.height_field_raw[x, y].item() * self.cfg.vertical_scale
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
            subterrain[sc:, ...] = height
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

        # env_origin_x = (i + 0.5) * self.env_length
        env_origin_x = i * self.env_length + 1.0
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int(
            (self.env_length / 2.0 - 0.5) / terrain.horizontal_scale
        )  # within 1 meter square range
        x2 = int((self.env_length / 2.0 + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 0.5) / terrain.horizontal_scale)
        if self.cfg.origin_zero_z:
            env_origin_z = 0
        else:
            env_origin_z = (
                np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
            )
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        # self.env_slope_vec[i, j] = terrain.slope_vector
