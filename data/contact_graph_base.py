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
    def __init__(self, pos: List[float], normal, skeleton_id: SkeletonID, device="cpu"):
        super().__init__(device=device)
        self._position = pos
        self._normal = normal
        self._velocity = None
        self.skeleton_id = (
            skeleton_id.value
            if isinstance(skeleton_id, SkeletonID)
            else int(skeleton_id)
        )
        for attr, value in self.__dict__.items():
            if isinstance(value, List):
                setattr(self, attr, tf.to_torch(value, device=device))

    @property
    def position(self):
        return self._position

    @property
    def normal(self):
        return self._normal

    def tf_apply(self, q, t):
        for v in [self.position, self.normal]:
            v = tf.tf_apply(q, t, v)

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
            ]
        )

    def node_feature(self):
        return torch.cat(
            [
                self.position,
                self.normal,
                torch.tensor([self.skeleton_id.value], device=self.position.device),
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

    def __getstate__(self):
        return {
            "start_node": self.start_node,
            "end_node": self.end_node,
            "order": self._order,
            "base_order": self._base_order,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }

    def __setstate__(self, state):
        self.start_node = state["start_node"]
        self.end_node = state["end_node"]
        self._order = state["order"]
        self._base_order = state["base_order"]
        self.start_frame = state["start_frame"]
        self.end_frame = state["end_frame"]


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

    def deform(self, nodeid: List[int], t, q):
        # TODO select some nodes by id and apply a transformation to them
        # note that under the init graph coord! along x...
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
        # stores the original heading direction of the skill TODO: main dir all x in the beg??
        return NotImplementedError

    @property
    def skill_type(self):  # 动作标签
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
