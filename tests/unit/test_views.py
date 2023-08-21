import unittest
import sys
import torch

sys.path.append("src")
from rendering.views import ThreeExteriorViews, SixExteriorViews, ThreeInteriorViews, SixInteriorViews


class TestViews(unittest.TestCase):
    def tests_if_three_exterior_views_are_a_list_of_tensors(self):
        views = ThreeExteriorViews().get()
        self.assertIsInstance(views, list)
        for R, T in zip(views[0], views[1]):
            self.assertIsInstance(R, torch.Tensor)
            self.assertIsInstance(T, torch.Tensor)

    def tests_if_six_exterior_views_are_a_list_of_tensors(self):
        views = SixExteriorViews().get()
        self.assertIsInstance(views, list)
        for R, T in zip(views[0], views[1]):
            self.assertIsInstance(R, torch.Tensor)
            self.assertIsInstance(T, torch.Tensor)

    def tests_if_three_interior_views_are_a_list_of_tensors(self):
        views = ThreeInteriorViews(center=torch.tensor([0.0, 0.0, 0.0])).get()
        self.assertIsInstance(views, list)
        for R, T in zip(views[0], views[1]):
            self.assertIsInstance(R, torch.Tensor)
            self.assertIsInstance(T, torch.Tensor)

    def tests_if_six_interior_views_are_a_list_of_tensors(self):
        views = SixInteriorViews(center=torch.tensor([0.0, 0.0, 0.0])).get()
        self.assertIsInstance(views, list)
        for R, T in zip(views[0], views[1]):
            self.assertIsInstance(R, torch.Tensor)
            self.assertIsInstance(T, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
