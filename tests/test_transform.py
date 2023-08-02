import unittest
import sys
import torch
sys.path.append("src")

from utils.transform import Transform, get_quaternion

class TestTransform(unittest.TestCase):

    def tests_get_quaternion_when_angle_is_zero(self):
        q = get_quaternion(vector=[1, 0, 0], angle_in_degrees=0.)
        self.assertEqual(q, [1.0 , 0.0, 0.0, 0.0])

    def tests_get_quaternion_when_rotating_90_degrees_around_x_axis(self):
        q = get_quaternion(vector=[1, 0, 0], angle_in_degrees=90.)
        actual = torch.tensor(q)
        expected = torch.tensor([0.7071067811865475, 0.7071067811865475, 0.0, 0.0])
        self.assertTrue(torch.equal(expected, actual))
    
    def test_translation(self):
        point = torch.tensor([1.,-2.,-7.])
        actual_output = Transform(translation=[-2.,5.,0], device="cpu").apply(point)
        expected_output = torch.tensor([-1., 3., -7.])
        self.assertTrue(torch.allclose(actual_output, expected_output))

    def test_rotation(self):
        point = torch.tensor([0., 1., 0.])
        rotation = get_quaternion(vector=[1,0,0], angle_in_degrees=90.)
        actual_output = Transform(rotation=rotation, device="cpu").apply(point)
        expected_output = torch.tensor([0., 0., 1.])
        self.assertTrue(torch.allclose(actual_output, expected_output))

    def test_scale(self):
        point = torch.tensor([1., 0., 3.])
        actual_output = Transform(scale=[1., 4., 2.], device="cuda").apply(point)
        expected_output = torch.tensor([1., 0., 6.], device="cuda")
        self.assertTrue(torch.allclose(actual_output, expected_output))

    def test_translation_rotation_scale(self):
        point = torch.tensor([[1.,1.,0.],
                              [1.,-1.,0.],
                              [-1.,1.,0.],
                              [-1.,-1.,0.]])
        translation = [1.,1.,0.]
        rotation = get_quaternion(vector=[1,0,0], angle_in_degrees=90.)
        scale = [2., 2., 2.]
        actual_output = Transform(translation, rotation, scale, device="cpu").apply(point)
        expected_output = torch.tensor([[4.,0.,4.],
                                        [4.,0.,0.],
                                        [0.,0.,4.],
                                        [0.,0.,0.]])
        self.assertTrue(torch.allclose(actual_output, expected_output))

        
if __name__ == "__main__":
    unittest.main()