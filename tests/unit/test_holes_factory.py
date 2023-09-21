import unittest
import sys
from pathlib import Path

sys.path.append("src")
from cable.holes import Holes
from cable.holes_factory import HolesListFactory
from cable.holes_initial_position import CaterpillarHolesInitialPosition
from mesh.scad import Scad


class TestHolesFactoryFromListOfPositions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        cls.holes_position = CaterpillarHolesInitialPosition(scad).get()

    def tests_if_a_list_of_holes_is_created(self):
        holes_list = HolesListFactory(list_of_positions=self.holes_position).create()
        self.assertTrue(isinstance(holes_list, list))
        self.assertTrue(all(isinstance(holes, Holes) for holes in holes_list))


if __name__ == "__main__":
    unittest.main()
