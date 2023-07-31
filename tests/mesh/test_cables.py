import unittest
import torch
import sys
import numpy as np
from pathlib import Path

sys.path.append("src")
from mesh.cables import Cables
from mesh.scad import Scad
from mesh.holes_position import HolesPositionWhenUnloaded


class TestCable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        cls.holes_position_when_unloaded = HolesPositionWhenUnloaded(scad).get()

    def tests_if_holes_position_per_cable_is_none_by_default(self):
        self.assertIsNone(Cables().holes_position_per_cable)

    def tests_if_value_error_is_raised_when_holes_position_per_cable_is_a_tuple_of_torch_tensors(self):
        cables = Cables()
        tensor = torch.arange(15).reshape(5, 3).to(dtype=torch.float32)
        holes_position_per_cable = (tensor, tensor, tensor)
        with self.assertRaises(TypeError):
            cables.holes_position_per_cable = holes_position_per_cable

    def tests_if_value_error_is_raised_when_holes_position_per_cable_is_a_list_of_numpy_arrays(self):
        cables = Cables()
        array = np.arange(15).reshape(5, 3).astype(np.float32)
        holes_position_per_cable = (array, array, array)
        with self.assertRaises(TypeError):
            cables.holes_position_per_cable = holes_position_per_cable

    def tests_if_holes_position_per_cable_can_be_set_in_unloaded_state(self):
        cables = Cables()
        try:
            cables.holes_position_per_cable = self.holes_position_when_unloaded
        except:
            self.fail("Failed to assign <holes_position_per_cables> to cable object.")

    def tests_if_len_is_set_correctly_in_unloaded_state(self):
        cables = Cables()
        cables.holes_position_per_cable = self.holes_position_when_unloaded
        self.assertEqual(len(cables), 3)

    def tests_if_num_holes_per_cable_is_set_correctly_in_unloaded_state(self):
        cables = Cables()
        cables.holes_position_per_cable = self.holes_position_when_unloaded
        self.assertEqual(cables.num_holes_per_cable, [18, 18, 18])
if __name__ == "__main__":
    unittest.main()
