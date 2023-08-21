import unittest
import sys
import torch

sys.path.append("src")

from cable.barycentric import Barycentric


class TestBarycentric(unittest.TestCase):
    def tests_if_value_error_is_raised_when_HxN_does_not_sum_to_one(self):
        HxN = torch.tensor([[0.1, 0.5, 0.2, 0.2], [0.8, 0.1, 0.1, 0.1]])
        NxH = torch.zeros([4, 2])
        with self.assertRaises(ValueError):
            Barycentric(HxN, NxH)

    def tests_if_value_error_is_not_raised_when_HxN_does_sum_to_one(self):
        HxN = torch.tensor([[0.1, 0.5, 0.2, 0.2], [0.8, 0.0, 0.1, 0.1]])
        NxH = torch.zeros([4, 2])
        try:
            Barycentric(HxN, NxH)
        except ValueError:
            self.fail()


if __name__ == "__main__":
    unittest.main()
