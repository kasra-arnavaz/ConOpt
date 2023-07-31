import unittest
import torch
import sys
import numpy as np

sys.path.append("src")
from mesh.cable import Cable
from mesh.holes import Holes

class TestCable(unittest.TestCase):

    def setUp(self) -> None:
        p = torch.arange(12).reshape(-1,3).to(dtype=torch.float32)
        self.holes = Holes(position=p)

    def tests_if_pull_ratio_is_a_zero_tensor_by_default(self):
        cable = Cable(holes=self.holes)
        zero_tensor = torch.tensor(0., dtype=torch.float32)
        self.assertTrue(torch.equal(cable.pull_ratio, zero_tensor))

    def tests_if_a_type_error_is_raised_if_pull_ratio_is_numpy_array(self):
        pull_ratio = np.array(0.)
        with self.assertRaises(TypeError):
            Cable(holes=self.holes, pull_ratio=pull_ratio)

    def tests_if_a_value_error_is_raised_if_pull_ratio_1D_torch_tensor(self):
        pull_ratio = torch.tensor([0.])
        with self.assertRaises(ValueError):
            Cable(holes=self.holes, pull_ratio=pull_ratio)

    def tests_if_a_value_error_is_raised_if_pull_ratios_dtype_is_int64(self):
        pull_ratio = torch.tensor(0, dtype=torch.int64)
        with self.assertRaises(ValueError):
            Cable(holes=self.holes, pull_ratio=pull_ratio)

    def tests_if_pull_ratio_is_clipped_to_zero_if_negative(self):
        cable = Cable(holes=self.holes)
        cable.pull_ratio = torch.tensor(-1., dtype=torch.float32)
        expected = torch.tensor(0., dtype=torch.float32)
        self.assertTrue(torch.equal(cable.pull_ratio, expected))

if __name__ == "__main__":
    unittest.main()
