import isaacgym.torch_utils as tf
import torch
import pickle
from enum import Enum
from typing import List, Union, Tuple


class SkeletonID(Enum):
    ARM = 1
    LEG = 2
    HEAD = 3
    TORSO = 4


class DeviceMixin:
    def __init__(self, device="cpu") -> None:
        # self._dtype = torch.get_default_dtype()
        self._device = torch.device(device)

    # @property
    # def dtype(self):
    #     return self._dtype

    @property
    def device(self):
        device = self._device
        if isinstance(device, str):
            return device
        if device.type == "cuda" and device.index is None:
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return device

    def to(self, device=None, dtype=None):
        self._device = device
        self._dtype = dtype
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(device=device))
            elif isinstance(value, List):
                for v in value:
                    if isinstance(v, DeviceMixin):
                        v.to(device=device)


class CNode(DeviceMixin):
    def __init__(
        self,
        pos: List[float],
        normal,
        skeleton_id: SkeletonID,
        order: int,
        start_time,
        end_time,
        sustain_time: int = 1,
        device="cpu",
    ):

        self._position = pos
        self._normal = normal
        self._velocity = None
        self._order = order
        self._sustain_time = sustain_time
        self.start_time = start_time
        self.end_time = end_time
        self.skeleton_id = skeleton_id.value if isinstance(skeleton_id, SkeletonID) else int(skeleton_id)
        for attr, value in self.__dict__.items():
            if isinstance(value, List):
                setattr(self, attr, tf.to_torch(value, device=device))

    @property
    def order(self):
        return self._order

    @property
    def sustain_time(self):
        return self._sustain_time

    @property
    def position(self):
        return self._position

    @property
    def normal(self):
        return self._normal

    def tf_apply(self, q=None, t=None, scale=None):
        for name in ["position", "normal"]:
            v = getattr(self, name)
            if scale is not None:
                v = v * scale
            if q is not None and t is not None:
                v = tf.tf_apply(q, t, v)
            setattr(self, name, v)

    @classmethod
    def tf_apply_onfeat(cls, q, t, feat):
        if q.device != feat.device:
            q = q.to(feat.device)
            t = t.to(feat.device)
        return torch.cat(
            [
                tf.tf_apply(q, t, feat[..., 0:3]),
                tf.quat_apply(q, feat[..., 3:6]),
                feat[..., 6:],
            ],
            dim=1,
        )

    def node_feature(self):
        return torch.cat(
            [
                self._position,
                self.normal,
                torch.tensor([self.skeleton_id, self._order], device=self._position.device),
            ]
        )


class CEdge:
    def __init__(self, start_node, end_node, order, start_frame, end_frame):
        self.start_node = int(start_node)
        self.end_node = int(end_node)
        self._order = int(order)
        self._base_order = 0
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)

    @property
    def order(self):
        return self._base_order + self._order

    def base_order_offset(self, offset):
        self._base_order += offset
        return self._base_order


class GraphBase(DeviceMixin):
    def __init__(self, directional: bool = True, device="cpu") -> None:
        super().__init__()
        self.nodes: List[CNode] = []
        self.edges: List[CEdge] = []
        self.node_nums = 0
        self.edge_nums = 0
        self._node_feat_tensor = None
        self._edge_feat_tensor = None
        self.adj_matrix = None
        self.directional = directional

    def add_node(self, nodes: List[CNode]):
        self.nodes += nodes
        self.node_nums += len(nodes)

    def add_edge(self, edges: List[CEdge]):
        self.edges += edges
        self.edge_nums += len(edges)

    def build_adj_matrix(self):
        return NotImplementedError

    def get_feat_tensor(self):
        # if not built yet, build and store the tensor(in init coord)
        # apply the coord transform and return it.
        return NotImplementedError

    def is_homogeneous(self, graph):
        # decide whether two graphs are matched(are the same skill)
        return NotImplementedError

    def get_subgraph(self, nodeid: List[int]) -> "GraphBase":
        return NotImplementedError

    def merge(self, graph):
        return NotImplementedError

    def deform(self):
        return NotImplementedError

    def segment(self, graph_srcs, graph_tars):
        return NotImplementedError

    def serialize(self):
        return NotImplementedError


class ContactGraphBase(GraphBase):
    def __init__(self, directional, device="cpu") -> None:
        super().__init__(directional)
        self._root_rotation = None
        self._root_translation = None
        self._main_direction = None

    def get_coord(self):
        return self._root_rotation, self._root_translation

    def set_coord(self, q, t):
        return NotImplementedError

    @property
    def main_direction(self):
        return NotImplementedError

    @property
    def skill_type(self):
        return NotImplementedError

    @property
    def anchors(self):
        return NotImplementedError

    def build_anchor_pts(self):
        # generate the points(foot here)which can be linked with other graphs
        return NotImplementedError

    def to_file(self, path):
        assert path.endswith(".pkl"), "path should end with .pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, path):
        assert path.endswith(".pkl"), "Only support .pkl file"
        with open(path, "rb") as f:
            return pickle.load(f)
