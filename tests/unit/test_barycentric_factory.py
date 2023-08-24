import unittest
import sys
from pathlib import Path
import torch

sys.path.append("src")

from cable.barycentric_factory import BarycentricListFactory
from mesh.mesh_factory import MeshFactoryFromScad
from cable.holes_factory import HolesListFactory
from cable.holes_initial_position import HolesInitialPosition
from mesh.scad import Scad


class TestListBarycentricFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        holes_positions = HolesInitialPosition(scad).get()
        cls.mesh = MeshFactoryFromScad(scad).create()
        cls.holes = HolesListFactory(holes_positions).create()

    def tests_if_error_is_raised_creating_barycentric(self):
        try:
            BarycentricListFactory(mesh=self.mesh, holes=self.holes).create()
        except:
            self.fail()

    def tests_if_HxN_has_the_correct_shape(self):
        barycentrics = BarycentricListFactory(mesh=self.mesh, holes=self.holes).create()
        for barycentric, holes in zip(barycentrics, self.holes):
            self.assertEqual(list(barycentric.HxN.shape), [len(holes), len(self.mesh.nodes)])

    def tests_if_NxH_has_the_correct_shape(self):
        barycentrics = BarycentricListFactory(mesh=self.mesh, holes=self.holes).create()
        for barycentric, holes in zip(barycentrics, self.holes):
            self.assertEqual(list(barycentric.NxH.shape), [len(self.mesh.nodes), len(holes)])


if __name__ == "__main__":
    unittest.main()
