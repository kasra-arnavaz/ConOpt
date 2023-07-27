import unittest
import torch
from pathlib import Path
import sys
sys.path.append("src")

from mesh.mesh_factory import MeshFactoryFromObj

class TestMeshFactoryFromObj(unittest.TestCase):

    def setUp(self):
        file = Path("tests/data/caterpillar.obj")
        self.mesh = MeshFactoryFromObj(file).create()

    def tests_if_node_position_has_correct_shape(self):
        actual = self.mesh.nodes.position.shape
        expected = (1472, 3)
        self.assertEqual(expected, actual)

    def tests_if_first_node_position_is_correct(self):
        expected = torch.tensor([0.0, -1.616098, 4.25], dtype=torch.float32)
        actual = self.mesh.nodes.position[0]
        self.assertTrue(torch.equal(expected, actual))

    def tests_if_last_node_position_is_correct(self):
        expected = torch.tensor([0.4166103, 0.5363491, 22.618374600000003], dtype=torch.float32)
        actual = self.mesh.nodes.position[-1]
        self.assertTrue(torch.equal(expected, actual))

    def tests_if_element_triangles_has_correct_shape(self):
        actual = self.mesh.elements.triangles.shape
        expected = (2940, 3)
        self.assertEqual(expected, actual)

    def tests_if_first_element_triangles_is_correct(self):
        expected = torch.tensor([787, 1381, 224], dtype=torch.int64)
        actual = self.mesh.elements.triangles[0]
        self.assertTrue(torch.equal(expected, actual))

    def tests_if_last_element_triangles_is_correct(self):
        expected = torch.tensor([1464, 586, 540], dtype=torch.int64)
        actual = self.mesh.elements.triangles[-1]
        self.assertTrue(torch.equal(expected, actual))
        


if __name__ == "__main__":
    unittest.main()
