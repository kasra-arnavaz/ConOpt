import unittest
import torch
import sys
import numpy as np
from pathlib import Path

sys.path.append("src")
from mesh.cable import Cable
from mesh.scad import Scad
from mesh.holes_position import HolesPositionWhenUnloaded


class TestCable(unittest.TestCase):
    def tests_if_holes_position_is_none_by_default(self):
        self.assertIsNone(Cable().holes_position)

    def tests_if_value_error_is_raised_when_holes_position_is_a_tuple_of_torch_tensors(self):
        cable = Cable()
        tensor = torch.arange(15).reshape(5, 3).to(dtype=torch.float32)
        holes_position = (tensor, tensor, tensor)
        with self.assertRaises(TypeError):
            cable.holes_position = holes_position

    def tests_if_value_error_is_raised_when_holes_position_is_a_list_of_numpy_arrays(self):
        cable = Cable()
        array = np.arange(15).reshape(5, 3).astype(np.float32)
        holes_position = (array, array, array)
        with self.assertRaises(TypeError):
            cable.holes_position = holes_position

    def tests_if_holes_position_can_be_set_in_unloaded_state(self):
        cable = Cable()
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        holes_position = HolesPositionWhenUnloaded(scad).get()
        try:
            cable.holes_position = holes_position
        except:
            self.fail("Failed to assign holes_positions to cable object.")


if __name__ == "__main__":
    unittest.main()
