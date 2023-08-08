import unittest
import torch
from pathlib import Path
import sys
sys.path.append("src")

from mesh.mesh_factory import MeshFactoryFromMsh

class TestMeshFactoryFromMsh(unittest.TestCase):

    def setUp(self):
        file = Path("tests/data/caterpillar.msh")
        self.mesh = MeshFactoryFromMsh(file, device="cpu").create()

    def tests_if_node_position_has_correct_shape(self):
        actual = self.mesh.nodes.position.shape
        expected = (2802, 3)
        self.assertEqual(expected, actual)

    def tests_if_first_node_position_is_correct(self):
        expected = torch.tensor([-17.386711165489424, 0.023640618123745545, 7.433517472403718], dtype=torch.float32)
        actual = self.mesh.nodes.position[0]
        self.assertTrue(torch.equal(expected, actual))

    def tests_if_last_node_position_is_correct(self):
        expected = torch.tensor([-9.545216567553763, -0.15252530224620196, 16.91802172177038], dtype=torch.float32)
        actual = self.mesh.nodes.position[-1]
        self.assertTrue(torch.equal(expected, actual))

    def tests_if_element_tetrahedra_has_correct_shape(self):
        actual = self.mesh.elements.tetrahedra.shape
        expected = (10642, 4)
        self.assertEqual(expected, actual)

    def tests_if_first_element_tetrahedra_is_correct(self):
        expected = torch.tensor([2771, 2779, 734, 1135], dtype=torch.int32)
        actual = self.mesh.elements.tetrahedra[0]
        self.assertTrue(torch.equal(expected, actual))

    def tests_if_last_element_tetrahedra_is_correct(self):
        expected = torch.tensor([2067, 820, 101, 1160], dtype=torch.int32)
        actual = self.mesh.elements.tetrahedra[-1]
        self.assertTrue(torch.equal(expected, actual))
        


if __name__ == "__main__":
    unittest.main()
