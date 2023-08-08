import unittest
import sys
import torch

sys.path.append("src")

from point.transform import Transform, get_quaternion
from point.points import Points


class TestTransform(unittest.TestCase):
    def tests_get_quaternion_when_angle_is_zero(self):
        q = get_quaternion(vector=[1, 0, 0], angle_in_degrees=0.0)
        self.assertEqual(q, [1.0, 0.0, 0.0, 0.0])

    def tests_get_quaternion_when_rotating_90_degrees_around_x_axis(self):
        q = get_quaternion(vector=[1, 0, 0], angle_in_degrees=90.0)
        actual = torch.tensor(q)
        expected = torch.tensor([0.7071067811865475, 0.7071067811865475, 0.0, 0.0])
        self.assertTrue(torch.equal(expected, actual))

    def test_translation(self):
        points = Points(position=torch.tensor([[1.0, -2.0, -7.0]]))
        Transform(translation=[-2.0, 5.0, 0], device="cpu").apply(points)
        expected_output = torch.tensor([-1.0, 3.0, -7.0])
        self.assertTrue(torch.allclose(points.position, expected_output))

    def test_rotation(self):
        points = Points(position=torch.tensor([[0.0, 1.0, 0.0]]))
        rotation = get_quaternion(vector=[1, 0, 0], angle_in_degrees=90.0)
        Transform(rotation=rotation, device="cpu").apply(points)
        expected_output = torch.tensor([0.0, 0.0, 1.0])
        self.assertTrue(torch.allclose(points.position, expected_output))

    def test_scale(self):
        points = Points(position=torch.tensor([[1.0, 0.0, 3.0]], device="cuda"))
        Transform(scale=[1.0, 4.0, 2.0], device="cuda").apply(points)
        expected_output = torch.tensor([1.0, 0.0, 6.0], device="cuda")
        self.assertTrue(torch.allclose(points.position, expected_output))

    def test_translation_rotation_scale(self):
        points = Points(position=torch.tensor([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]]))
        translation = [1.0, 1.0, 0.0]
        rotation = get_quaternion(vector=[1, 0, 0], angle_in_degrees=90.0)
        scale = [2.0, 2.0, 2.0]
        Transform(translation, rotation, scale, device="cpu").apply(points)
        expected_output = torch.tensor([[4.0, 0.0, 4.0], [4.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(points.position, expected_output))


if __name__ == "__main__":
    unittest.main()
