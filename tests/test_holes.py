import unittest
import torch
import sys
import numpy as np
from pathlib import Path

sys.path.append("src")
from cable.holes import Holes

class TestHoles(unittest.TestCase):

    def tests_if_type_error_is_raised_when_position_is_initialized_to_list(self):
        with self.assertRaises(TypeError):
            Holes(position=[1, 2, 3])

    def tests_if_type_error_is_raised_when_position_is_initialized_to_numpy_array(self):
        p = np.arange(15).reshape(5, 3).astype(np.float32)
        with self.assertRaises(TypeError):
            Holes(position=p)

    def tests_if_value_error_is_raised_when_position_is_initialized_to_1d_tensor(self):
        p = torch.arange(12).to(dtype=torch.float32)
        with self.assertRaises(ValueError):
            Holes(position=p)

    def tests_if_value_error_is_raised_when_positions_last_dimension_is_initialized_to_2(self):
        p = torch.arange(12).reshape(6, 2).to(dtype=torch.float32)
        with self.assertRaises(ValueError):
            Holes(position=p)

    def tests_if_value_error_is_raised_when_position_dtype_is_initialzied_to_torch_int64(self):
        p = torch.arange(12).reshape(4, 3).to(dtype=torch.int64)
        with self.assertRaises(ValueError):
            Holes(position=p)

    def tests_the_len_of_holes(self):
        position = torch.arange(24).reshape(8,3).to(dtype=torch.float32)
        holes = Holes(position=position)
        self.assertEqual(len(holes), 8)

if __name__ == "__main__":
    unittest.main()
