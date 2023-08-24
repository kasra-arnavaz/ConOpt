import torch
import unittest
import sys

sys.path.append("src")
from pathlib import Path
from mesh.scad import Scad
from cable.cable_factory import CableListFactory
from cable.holes_factory import HolesListFactory
from cable.holes_initial_position import HolesInitialPosition
from cable.cable import Cable


class TestCableFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        holes_position = HolesInitialPosition(scad).get()
        cls.holes = HolesListFactory(holes_position).create()
        cls.pull_ratio = [
            torch.tensor(0.5, device="cuda"),
            torch.tensor(0.0, device="cuda"),
            torch.tensor(0.0, device="cuda"),
        ]

    def tests_if_cables_are_created_from_cable_factory(self):
        cables = CableListFactory(stiffness=100, damping=0.1, pull_ratio=self.pull_ratio, holes=self.holes).create()
        for cable in cables:
            self.assertIsInstance(cable, Cable)


if __name__ == "__main__":
    unittest.main()
