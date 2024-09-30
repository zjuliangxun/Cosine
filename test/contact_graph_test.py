import unittest
import torch
from data.contact_graph import ContactGraph


class TestContactGraph(unittest.TestCase):
    def setUp(self):
        # Create some mock nodes and edges
        self.nodes = [
            MockNode(
                position=torch.tensor([0, 0, 0]),
                normal=torch.tensor([1, 0, 0]),
                skeleton_id=MockSkeletonID(0),
            ),
            MockNode(
                position=torch.tensor([1, 0, 0]),
                normal=torch.tensor([0, 1, 0]),
                skeleton_id=MockSkeletonID(1),
            ),
        ]
        self.edges = [
            MockEdge(start_node=0, end_node=1, order=0, start_frame=0, end_frame=1)
        ]
        self.graph = ContactGraph(self.nodes, self.edges)

    def test_initialization(self):
        self.assertEqual(len(self.graph.nodes), 2)
        self.assertEqual(len(self.graph.edges), 1)
        self.assertFalse(self.graph._has_deformed)

    def test_set_coord(self):
        self.graph.set_coord(1, 2, 3, 0.1, 0.2, 0.3)
        self.assertTrue(torch.equal(self.graph.coordinate[1], torch.tensor([1, 2, 3])))

    def test_main_direction(self):
        direction = self.graph.main_direction
        self.assertTrue(torch.equal(direction, torch.tensor([1, 0, 0])))

    def test_build_adj_matrix(self):
        self.graph.build_adj_matrix()
        self.assertEqual(self.graph.adj_matrix[0, 1], 1)
        self.assertEqual(self.graph.adj_matrix[1, 0], 1)

    def test_get_feat_tensor(self):
        feat_tensor = self.graph.get_feat_tensor()
        self.assertIsInstance(feat_tensor, tuple)

    def test_get_edges_same_order(self):
        edges = self.graph.get_edges_same_order(0)
        self.assertEqual(len(edges), 1)

    def test_deform(self):
        self.graph.deform(
            [0],
            q=torch.tensor([1, 0, 0, 0]),
            t=torch.tensor([1, 1, 1]),
            scale=1,
            scale_direc=1,
        )
        self.assertTrue(self.graph._has_deformed)

    def test_get_subgraph(self):
        subgraph = self.graph.get_subgraph([0])
        self.assertEqual(len(subgraph.nodes), 1)

    def test_serialize(self):
        serialized = self.graph.serialize()
        self.assertIsInstance(serialized, str)

    def test_anchor_pts(self):
        anchor_points = self.graph.anchor_pts()
        self.assertIsInstance(anchor_points, tuple)


class MockNode:
    def __init__(self, position, normal, skeleton_id):
        self.position = position
        self.normal = normal
        self.skeleton_id = skeleton_id

    def node_feature(self):
        return self.position

    def tf_apply(self, q, t):
        pass


class MockEdge:
    def __init__(self, start_node, end_node, order, start_frame, end_frame):
        self.start_node = start_node
        self.end_node = end_node
        self.order = order
        self.start_frame = start_frame
        self.end_frame = end_frame


class MockSkeletonID:
    def __init__(self, value):
        self.value = value


if __name__ == "__main__":
    unittest.main()


# if __name__ == "__main__":
#     from data.contact_graph import ContactGraph
#     from data.contact_graph_base import CNode, CEdge
#     import pickle

#     c = [CNode([0.0, 1, 2], [2.0, 3, 4], 7), CNode([0.0, 1, 2], [2.0, 3, 4], 7)]
#     e = [CEdge(0.0, 1.0, 0.0, 0.0, 1.0)]
#     cg = ContactGraph(c, e)
#     c[0].to(device="cuda")
#     cg.to(device="cuda")
#     with open("contact_graph.pkl", "wb") as f:
#         pickle.dump(cg, f)
