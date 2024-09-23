import isaacgym.torch_utils as tf
import torch
import json, pickle
from typing import List, Union, Tuple

from contact_graph_base import *

# from scipy.spatial.transform import Rotation as sRot


class ContactGraph(ContactGraphInterface):
    def __init__(self, nodes, edges, use_edge_feat=False, directional=True):
        super().__init__(directional)
        self._has_deformed = False  # flag to rebuild feature
        self.use_edge_feat = use_edge_feat
        self._max_edge_order = 0
        self._order2edgemap = {}
        self._coordinate = self.set_coord(0, 0, 0, 0, 0, 0)
        self.add_node(nodes)
        self.add_edge(edges)
        self.build_adj_matrix()

    @property
    def coordinate(self):
        return self._coordinate

    def set_coord(self, x, y, z, yaw, pitch, roll):
        self._coordinate = (
            tf.quat_from_euler_xyz(
                torch.tensor(yaw), torch.tensor(pitch), torch.tensor(roll)
            ),
            torch.tensor([x, y, z]),
        )

    @property
    def main_direction(self):
        if self._main_direction is None:
            # x-axis by default, up z TODO
            self._main_direction = torch.tensor([1, 0, 0])
            # if len(self.nodes) >= 2: self._main_direction = self.nodes[-1].position - self.nodes[0].position # Assuming main direction is the vector from the first node to the last
        return self._main_direction

    def build_adj_matrix(self):
        self._order2edgemap = {}
        self.adj_matrix = torch.zeros(1 + self.node_nums, 1 + self.node_nums)
        for edge in self.edges:
            self.adj_matrix[edge.start_node, edge.end_node] = 1
            self.adj_matrix[edge.end_node, edge.start_node] = 1
            self._max_edge_order = max(edge.order, self._max_edge_order)
        for i in range(self._max_edge_order, self._max_edge_order + 1):
            self._order2edgemap[i] = []
        for i, e in enumerate(self.edges):
            self._order2edgemap[e.order].append(i)

    def get_feat_tensor(self, device="cpu"):
        if self._has_deformed:
            self._has_deformed = not self._has_deformed
            self._node_feat_tensor = torch.stack(
                [node.node_feature() for node in self.nodes], device=device
            )
            if self.use_edge_feat:
                edge_list = [(edge.start_node, edge.end_node) for edge in self.edges]
                self._edge_feat_tensor = torch.tensor(
                    edge_list, dtype=torch.long, device=device
                )
        # TODO check if need reverse
        ret = tuple(CNode.tf_apply_onfeat(*self.coordinate, self._node_feat_tensor))
        if self.use_edge_feat:
            ret += tuple(self._edge_feat_tensor)
        return ret

    def get_edges_same_order(self, order):
        if order > self._max_edge_order or order < 0:
            raise ValueError
        return [self.edges[idx] for idx in self._order2edgemap[order]]

    def deform(self, nodeid: List[int], *, q, t, scale, scale_direc):
        self._has_deformed = True
        for i in nodeid:
            self.nodes[i].tf_apply(q, t)

    def get_subgraph(self, nodeid: List[int]):
        nodes = [self.nodes[i] for i in nodeid]
        edges = [
            CEdge(
                e.start_node,
                e.end_node,
                e.order,
                e.start_frame,
                e.end_frame,
            )  # TODO this keeps the original order and frame number
            for e in self.edges
            if e.start_node in nodeid and e.end_node in nodeid
        ]
        return ContactGraph(nodes, edges, self.use_edge_feat, self.directional)

    def merge(self, graph, q, t):
        self._has_deformed = True
        # choose the anchor_points based on head/tail
        # calculate the possible range the input CG anchor may locate
        # rotate then translate the input graph so that the anchor points are the same
        # merge nodes and edges (increment nums; edge base offset)
        return NotImplementedError

    def __getstate__(self):
        state = {
            "use_edge_feat": self.use_edge_feat,
            "_max_edge_order": self._max_edge_order,
            "_order2edgemap": self._order2edgemap,
            "_coordinate": self._coordinate,
            "directional": self.directional,
            "node_nums": self.node_nums,
            "edge_nums": self.edge_nums,
            "_node_feat_tensor": self._node_feat_tensor,
            "_edge_feat_tensor": self._edge_feat_tensor,
            "adj_matrix": self.adj_matrix,
            "directional": self.directional,
            "nodes": self.nodes,
            "edges": self.edges,
        }
        return state

    def __setstate__(self, state):
        self.use_edge_feat = state["use_edge_feat"]
        self._max_edge_order = state["_max_edge_order"]
        self._order2edgemap = state["_order2edgemap"]
        self._coordinate = state["_coordinate"]
        self.directional = state["directional"]
        self.nodes = state["nodes"]
        self.edges = state["edges"]
        self.node_nums = state["node_nums"]
        self.edge_nums = state["edge_nums"]
        self._node_feat_tensor = state["_node_feat_tensor"]
        self._edge_feat_tensor = state["_edge_feat_tensor"]
        self.adj_matrix = state["adj_matrix"]
        self.directional = state["directional"]

    @property
    def skill_type(self):
        return self._skill_name

    def anchor_pts(self):
        # the nodes of input graphs can fall in circles of head/tail anchor pts
        tail, head = set(), set()
        self._max_radius = 2.5
        for e in self.edges:
            if e.order == 0:
                head.add(e.start_node)
            if e.order == self._max_edge_order:
                tail.add(e.end_node)
