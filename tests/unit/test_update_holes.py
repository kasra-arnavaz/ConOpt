import unittest
import sys
from pathlib import Path
import torch
sys.path.append("src")

from cable.barycentric_factory import BarycentricFactory
from mesh.mesh_factory import MeshFactoryFromScad
from cable.holes_factory import HolesFactoryFromListOfPositions
from cable.holes_initial_position import HolesInitialPosition
from mesh.scad import Scad
from simulation.update_holes import HolesPositionAndVelocity, HolesForce
from cable.cable import Cable
from mesh.scad import Scad
from cable.holes_initial_position import HolesInitialPosition
from cable.holes_factory import HolesFactoryFromListOfPositions


class TestHolesForce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        holes_position = HolesInitialPosition(scad).get()
        holes = HolesFactoryFromListOfPositions(holes_position, device="cuda").create()[0]
        pull_ratio = torch.tensor(0.5, requires_grad=True, device="cuda")
        cls.cable = Cable(stiffness=100, damping=0.01, pull_ratio=pull_ratio, holes=holes)

    def test_if_holes_force_is_changed(self):
        old_holes_force = self.cable.holes.force
        HolesForce(cable=self.cable, device="cuda").update()
        new_holes_force = self.cable.holes.force
        self.assertFalse(torch.equal(old_holes_force, new_holes_force))


class TestHolesPositionAndVelocity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        holes_positions = HolesInitialPosition(scad).get()
        cls.mesh = MeshFactoryFromScad(scad).create()
        cls.holes = HolesFactoryFromListOfPositions(holes_positions).create()

    def tests_if_the_first_update_of_holes_position_and_velocity_does_nothing(self):
        for holes in self.holes:
            holes_position_before = holes.position.clone()
            holes_velocity_before = holes.velocity.clone()
            barycentric = BarycentricFactory(mesh=self.mesh, holes=holes).create()
            HolesPositionAndVelocity(holes=holes, nodes=self.mesh.nodes, barycentric=barycentric).update()
            self.assertTrue(torch.allclose(holes.position, holes_position_before, atol=1e-5))
            self.assertTrue(torch.allclose(holes.velocity, holes_velocity_before, atol=1e-5))





            
if __name__ == "__main__":
    unittest.main()