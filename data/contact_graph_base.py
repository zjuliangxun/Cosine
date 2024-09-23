import torch
from enum import Enum
from typing import List, Union, Tuple
import isaacgym.torch_utils as tf


class SkeletonID(Enum):
    ARM = 1
    LEG = 2
    HEAD = 3
    TORSO = 4


class CNode:
    def __init__(self, pos: Union[torch.Tensor | List[float]], normal, skeleton_id):
        self._position = pos
        self._normal = normal
        self._velocity = None
        self.skeleton_id = skeleton_id
        for attr, value in self.__dict__.items():
            if isinstance(value, List[float]):
                setattr(self, attr, tf.to_torch(value))

    @property
    def position(self):
        return self._position

    @property
    def normal(self):
        return self._normal

    def to(self, device):
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(device))

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
        self.start_node = start_node
        self.end_node = end_node
        self._order = order
        self._base_order = 0
        self.start_frame = start_frame
        self.end_frame = end_frame

    @property
    def order(self):
        return self._base_order + self._order

    def base_order_offset(self, offset):
        self._base_order += offset
        return self._base_order


class GraphBaseInterface:
    def __init__(self, directional: bool = True) -> None:
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
        self.node_nums += 1

    def add_edge(self, edges: List[CEdge]):
        self.edges += edges
        self.edge_nums += 1

    def build_adj_matrix(self):
        return NotImplementedError

    def get_feat_tensor(self):
        # if not built yet, build and store the tensor(in init coord)
        # apply the coord transform and return it.
        return NotImplementedError

    def is_homogeneous(self, graph):
        # decide whether two graphs are matched(are the same skill)
        return NotImplementedError

    def get_subgraph(self, nodeid: List[int]):
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


class ContactGraphInterface(GraphBaseInterface):
    def __init__(self, directional) -> None:
        super().__init__(directional)
        self._coordinate = None

    @property
    def coordinate(self):
        # coord(node)=world coords
        return NotImplementedError

    @property
    def main_direction(self):
        # stores the original heading direction of the skill TODO: main dir all x in the beg??
        return NotImplementedError

    @property
    def skill_type(self):  # 动作标签
        return NotImplementedError

    def anchor_pts(self):
        # return the points(foot here)which can be linked with other graphs
        return NotImplementedError
