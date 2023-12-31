import unittest
from pathlib import Path
import sys

sys.path.append("src")
from mesh.mesh_factory import MeshFactoryFromScad
from mesh.scad import Scad


class TestMeshFactoryFromScad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        cls.mesh = MeshFactoryFromScad(scad, ideal_edge_length=0.02, device="cpu").create()

    def tests_if_node_position_is_not_none(self):
        self.assertIsNotNone(self.mesh.nodes.position)

    def tests_if_element_tetrahedra_is_not_none(self):
        self.assertIsNotNone(self.mesh.elements.tetrahedra)


if __name__ == "__main__":
    unittest.main()
