import unittest
from pathlib import Path
import sys
import torch

sys.path.append("src")

from cable.holes_force import HolesForce
from cable.cable import Cable
from mesh.scad import Scad
from mesh.mesh_factory import MeshFactoryFromScad
from cable.holes_initial_position import HolesInitialPosition
from cable.holes_factory import HolesFactoryFromListOfPositions
from mesh.nodes_force import NodesForce
from cable.barycentric_factory import BarycentricFactory


class TestNodesForce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        holes_position = HolesInitialPosition(scad).get()
        cls.holes = HolesFactoryFromListOfPositions(holes_position, device="cuda").create()[0]
        pull_ratio = torch.tensor(0.5, requires_grad=True, device="cuda")
        cable = Cable(stiffness=100, damping=0.01, pull_ratio=pull_ratio, holes=cls.holes)
        HolesForce(cable=cable, device="cuda").update()
        cls.mesh = MeshFactoryFromScad(scad).create()
        cls.barycentric = BarycentricFactory(mesh=cls.mesh, holes=cls.holes, device="cuda").create()

    def tests_if_nodes_force_is_changed(self):
        old_nodes_force = self.mesh.nodes.force
        NodesForce(nodes=self.mesh.nodes, holes=self.holes, barycentric=self.barycentric).update()
        new_nodes_force = self.mesh.nodes.force
        self.assertFalse(torch.equal(old_nodes_force, new_nodes_force))


if __name__ == "__main__":
    unittest.main()
