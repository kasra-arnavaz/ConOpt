import unittest
import torch
from pathlib import Path
import sys
sys.path.append("src")

from mesh.mesh_factory import MeshFactoryFromStl

class TestMeshFactoryFromStl(unittest.TestCase):

    def setUp(self):
        file = Path("tests/data/caterpillar.stl")
        self.mesh = MeshFactoryFromStl(file, device="cpu").create()

    def tests_if_node_position_has_correct_shape(self):
        actual = self.mesh.nodes.position.shape
        expected = (840, 3)
        self.assertEqual(expected, actual)

    def tests_if_first_node_position_is_correct(self):
        expected = torch.tensor([-14.1578, 10.2862, 7.5], dtype=torch.float32)
        actual = self.mesh.nodes.position[0]
        self.assertTrue(torch.equal(expected, actual))

    def tests_if_last_node_position_is_correct(self):
        expected = torch.tensor([-4.89074, -1.03956, 300], dtype=torch.float32)
        actual = self.mesh.nodes.position[-1]
        self.assertTrue(torch.equal(expected, actual))

    def tests_if_element_triangles_has_correct_shape(self):
        actual = self.mesh.elements.triangles.shape
        expected = (1676, 3)
        self.assertEqual(expected, actual)

    def tests_if_first_element_triangles_is_correct(self):
        expected = torch.tensor([0, 1, 2], dtype=torch.int32)
        actual = self.mesh.elements.triangles[0]
        self.assertTrue(torch.equal(expected, actual))

    def tests_if_last_element_triangles_is_correct(self):
        expected = torch.tensor([747, 738, 753], dtype=torch.int32)
        actual = self.mesh.elements.triangles[-1]
        self.assertTrue(torch.equal(expected, actual))
        


if __name__ == "__main__":
    unittest.main()
