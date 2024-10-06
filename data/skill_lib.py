import os
import yaml

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion
from poselib.poselib.core.rotation3d import *
from isaacgym.torch_utils import *

from data.motion_lib import *
from torch import Tensor
from typing import Tuple

import torch

from data.contact_graph import ContactGraph
import random, copy


# TODO 统一skill id和pkl导入的id
class LoadedMotionsCG(LoadedMotions):
    motion_cgs: Tuple[ContactGraph]
    motion_weights_byskill: Tensor

    def __init__(
        self,
        motions: Tuple[SkeletonMotion],
        motion_lengths: Tensor,
        motion_weights: Tensor,
        motion_fps: Tensor,
        motion_dt: Tensor,
        motion_num_frames: Tensor,
        motion_files: Tuple[str],
        motion_cgs: Tuple[ContactGraph],
        motion_weights_byskill: Tensor,
    ):
        super().__init__(
            motions,
            motion_lengths,
            motion_weights,
            motion_fps,
            motion_dt,
            motion_num_frames,
            motion_files,
            motion_weights_byskill,
        )
        self.motion_cgs = motion_cgs
        self.register_buffer("motion_weights", motion_weights_byskill, persistent=False)


class SkillLib(MotionLib):
    def __init__(
        self,
        motion_file,
        dof_body_ids,
        dof_offsets,
        key_body_ids,
        equal_motion_weights,
        device="cpu",
    ):
        # motion weights are used for random sampling
        self._skill_categories = {}
        self._skill_nums = 0
        super().__init__(
            motion_file,
            dof_body_ids,
            dof_offsets,
            key_body_ids,
            equal_motion_weights,
            device,
        )

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if ext == ".yaml":
            dir_name = os.path.dirname(motion_file)
            motion_files = []
            motion_weights = []
            graph_files = []

            with open(os.path.join(os.getcwd(), motion_file), "r") as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_list = motion_config["motions"]
            for motion_entry in motion_list:
                curr_file = motion_entry["file"]
                curr_weight = motion_entry["weight"]
                curr_graph = motion_entry["graph"]
                assert curr_weight >= 0

                curr_file = os.path.join(dir_name, curr_file)
                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
                graph_files.append(curr_graph)
        else:
            raise ValueError("Unsupported motion file format. Use YAML.")

        return motion_files, graph_files, motion_weights

    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []
        self._motion_cgs = []

        total_len = 0.0

        motion_files, graph_files, motion_weights = self._fetch_motion_files(
            motion_file
        )
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print(
                "Loading {:d}/{:d} motion files: {:s}".format(
                    f + 1, num_motion_files, curr_file
                )
            )
            curr_motion: SkeletonMotion = SkeletonMotion.from_file(curr_file)
            curr_cg: ContactGraph = ContactGraph.from_file(graph_files[f])

            self._skill_categories.setdefault(curr_cg.skill_type, set()).add(f)

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)

            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels

            # Moving motion tensors to the GPU
            if USE_CACHE:
                curr_cg = curr_cg.to(self._device)
                curr_motion = DeviceCache(curr_motion, self._device)
            else:
                curr_motion.tensor = curr_motion.tensor.to(self._device)
                curr_cg = curr_cg.to(self._device)
                curr_motion._skeleton_tree._parent_indices = (
                    curr_motion._skeleton_tree._parent_indices.to(self._device)
                )
                curr_motion._skeleton_tree._local_translation = (
                    curr_motion._skeleton_tree._local_translation.to(self._device)
                )
                curr_motion._rotation = curr_motion._rotation.to(self._device)

            self._motions.append(curr_motion)
            self._motion_cgs.append(curr_cg)
            self._motion_lengths.append(curr_len)
            print("Loaded motion with length {:.3f}s".format(curr_len))

            if self._equal_motion_weights:
                curr_weight = 1.0
            else:
                curr_weight = motion_weights[f]
            self._motion_weights.append(curr_weight)
            self._motion_files.append(curr_file)

        self._skill_nums = len(self._skill_categories)
        self._max_skill_len = max(len(lst) for lst in self._skill_categories.values)

        self._motion_lengths = torch.tensor(
            self._motion_lengths, device=self._device, dtype=torch.float32
        )

        self._motion_weights = torch.tensor(
            self._motion_weights, dtype=torch.float32, device=self._device
        )

        self._motion_fps = torch.tensor(
            self._motion_fps, device=self._device, dtype=torch.float32
        )
        self._motion_dt = torch.tensor(
            self._motion_dt, device=self._device, dtype=torch.float32
        )
        self._motion_num_frames = torch.tensor(
            self._motion_num_frames, device=self._device
        )

        indices = torch.full(
            (len(self._skill_categories), self._max_skill_len),
            -1,
            dtype=torch.long,
            device=self._device,
        )
        for i, se in enumerate(self._skill_categories):
            indices[i, : len(se)] = torch.tensor(list(se), dtype=torch.long)

        self._motion_weights_byskill = torch.where(
            indices >= 0,
            self._motion_weights[indices],
            torch.zeros_like(indices, dtype=torch.float32),
        )
        self._motion_weights /= self._motion_weights.sum()
        self._motion_weights_byskill /= self._motion_weights_byskill.sum(dim=1)

        self.state = LoadedMotionsCG(
            motions=tuple(self._motions),
            motion_cgs=tuple(self._motion_cgs),
            motion_lengths=self._motion_lengths,
            motion_weights=self._motion_weights,
            motion_fps=self._motion_fps,
            motion_dt=self._motion_dt,
            motion_num_frames=self._motion_num_frames,
            motion_files=tuple(motion_files),
            motion_weights_byskill=self._motion_weights_byskill,
        )

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print(
            "Loaded {:d} motions with a total length of {:.3f}s.".format(
                num_motions, total_len
            )
        )

        return motion_files

    def sample_motions(self, n, motion_cats):
        motion_ids = torch.multinomial(
            self.state.motion_weights_byskill[motion_cats],
            num_samples=n,
            replacement=True,
        )
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None, phase=None):
        if phase is None:
            phase = torch.rand(motion_ids.shape, device=self._device)

        motion_len = self.state.motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time

        # don't allow negative phase
        motion_time = phase * torch.clip(motion_len, min=0)
        return motion_time

    def get_cg_by_skill(self, skill_name) -> ContactGraph:
        if skill_name in self._skill_categories:
            motion_id = random.sample(self._skill_categories[skill_name], 1)[0]
            return copy.deepcopy(self.state.motion_cgs[motion_id])
        else:
            raise ValueError("Skill name not found in the skill library")

    def get_skill_id(self, skill_name):
        if skill_name in self._skill_categories:
            return self._skill_categories[skill_name]
        else:
            raise ValueError("Skill name not found in the skill library")
