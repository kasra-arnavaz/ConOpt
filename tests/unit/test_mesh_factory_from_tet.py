import unittest
import torch
from pathlib import Path
import sys

sys.path.append("src")

from mesh.mesh_factory import MeshFactoryFromTet


class TestMeshFactoryFromTet(unittest.TestCase):
    def setUp(self):
        file = Path("tests/data/caterpillar.tet")
        self.mesh = MeshFactoryFromTet(file, device="cpu").create()

    def tests_if_node_position_has_correct_shape(self):
        actual = self.mesh.nodes.position.shape
        expected = (2166, 3)
        self.assertEqual(expected, actual)

    def tests_if_first_node_position_is_correct(self):
        expected = torch.tensor([0.0, -1.616098, 4.25], dtype=torch.float32)
        actual = self.mesh.nodes.position[0]
        self.assertTrue(torch.equal(expected, actual))

    def tests_if_last_node_position_is_correct(self):
        expected = torch.tensor([-0.7189126209995993, -0.1738021780468882, -0.7364678651660144], dtype=torch.float32)
        actual = self.mesh.nodes.position[-1]
        self.assertTrue(torch.equal(expected, actual))

    def tests_if_element_tetrahedra_has_correct_shape(self):
        actual = self.mesh.elements.tetrahedra.shape
        expected = (7873, 4)
        self.assertEqual(expected, actual)

    def tests_if_first_element_tetrahedra_is_correct(self):
        expected = torch.tensor([1716, 1473, 2081, 1940], dtype=torch.int32)
        actual = self.mesh.elements.tetrahedra[0]
        self.assertTrue(torch.equal(expected, actual))

    def tests_if_last_element_tetrahedra_is_correct(self):
        expected = torch.tensor([786, 105, 60, 2080], dtype=torch.int32)
        actual = self.mesh.elements.tetrahedra[-1]
        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
