import unittest
import sys
import torch
from pathlib import Path

sys.path.append("src")

from cable.holes_initial_position import CaterpillarHolesInitialPosition, StarfishHolesInitialPosition
from mesh.scad import Scad


class TestCaterpillarHolesInitialPosition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        cls.holes_position = CaterpillarHolesInitialPosition(scad).get()

    def tests_if_holes_are_sorted_ascendingly_in_height(self):
        for hole in self.holes_position:
            height = hole[:, -1]
            self.assertTrue(torch.equal(height, height.sort(descending=False).values))

    def tests_holes_are_of_correct_type(self):
        self.assertIsInstance(self.holes_position, list)
        for hole in self.holes_position:
            self.assertIsInstance(hole, torch.Tensor)

class TestStarfishHolesInitialPosition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/starfish.scad")
        parameters = Path("tests/data/starfish_scad_params.json")
        scad = Scad(file, parameters)
        cls.holes_position = StarfishHolesInitialPosition(scad).get()

    def tests_holes_are_of_correct_type(self):
        self.assertIsInstance(self.holes_position, list)
        for hole in self.holes_position:
            self.assertIsInstance(hole, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
