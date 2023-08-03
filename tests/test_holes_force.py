import unittest
from pathlib import Path
import sys
import torch

sys.path.append("src")

from cable.holes_force import HolesForce
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


if __name__ == "__main__":
    unittest.main()
