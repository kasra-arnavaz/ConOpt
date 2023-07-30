import unittest
import sys
import torch
from pathlib import Path

sys.path.append("src")

from mesh.holes_position import HolesPositionWhenUnloaded
from mesh.scad import Scad


class TestHolesPositionWhenUnloaded(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        cls.holes_position = HolesPositionWhenUnloaded(scad).get()

    def tests_if_holes_are_sorted_ascendingly_in_height(self):
        for hole in self.holes_position:
            height = hole[:, -1]
            self.assertTrue(torch.equal(height, height.sort(descending=False).values))


if __name__ == "__main__":
    unittest.main()
