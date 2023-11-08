import unittest
import torch
import sys

sys.path.append("src")

from utils.interpolate import linear_interpolate_between_two_points, linear_interpolate


class TestLinearInterpolate(unittest.TestCase):
    def tests_values_of_linear_interpolation_between_two_points(self):
        inp = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        x0 = torch.tensor(0.0)
        x1 = torch.tensor(0.5)
        y0 = torch.tensor(0.0)
        y1 = torch.tensor(0.2)
        actual = linear_interpolate_between_two_points(x0, x1, y0, y1, inp)
        expected = torch.tensor([0.0, 0.04, 0.08, 0.12, 0.16, 0.2])
        self.assertTrue(torch.allclose(actual, expected))

    def tests_values_of_linear_interpolation_between_three_points(self):
        inp = torch.arange(0, 1.1, 0.1)
        xs = torch.tensor([0.0, 0.5, 1.0])
        ys = torch.tensor([0.0, 0.2, 0.0])
        actual = linear_interpolate(xs, ys, inp)
        expected = [torch.tensor(e) for e in [0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.16, 0.12, 0.08, 0.04]]
        for a, e in zip(actual, expected):
            self.assertTrue(torch.allclose(a, e))


if __name__ == "__main__":
    unittest.main()
