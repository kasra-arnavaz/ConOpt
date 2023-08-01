import unittest
import torch
import sys

sys.path.append("src")
from point.points import Points


class TestPoints(unittest.TestCase):
    def tests_if_type_error_is_raised_when_position_is_initialized_to_list(self):
        with self.assertRaises(TypeError):
            Points(position=[1, 2, 3])

    def tests_if_type_error_is_raised_when_position_is_set_to_list(self):
        points = Points(position=torch.zeros([10, 3], dtype=torch.float32))
        with self.assertRaises(TypeError):
            points.position = [1, 2, 3]

    def tests_if_value_error_is_raised_when_position_is_initialized_to_1d_tensor(self):
        with self.assertRaises(ValueError):
            p = torch.arange(12).to(dtype=torch.float32)
            Points(position=p)

    def tests_if_value_error_is_raised_when_position_is_set_to_1d_tensor(self):
        p = torch.arange(12).reshape(4, 3).to(dtype=torch.float32)
        points = Points(position=p)
        with self.assertRaises(ValueError):
            points.position = torch.arange(12).to(dtype=torch.float32)

    def tests_if_value_error_is_raised_when_position_is_initialized_to_3d_tensor(self):
        p = torch.arange(12).reshape(2, 2, 3).to(dtype=torch.float32)
        with self.assertRaises(ValueError):
            Points(position=p)

    def tests_if_value_error_is_raised_when_position_is_set_to_3d_tensor(self):
        p = torch.arange(12).reshape(4, 3).to(dtype=torch.float32)
        points = Points(position=p)
        with self.assertRaises(ValueError):
            points.position = p.reshape(2, 2, 3).to(dtype=torch.float32)

    def tests_if_value_error_is_raised_when_positions_last_dimension_is_initialized_to_2(self):
        p = torch.arange(12).reshape(6, 2).to(dtype=torch.float32)
        with self.assertRaises(ValueError):
            Points(position=p)

    def tests_if_value_error_is_raised_when_positions_last_dimension_is_set_to_2(self):
        p = torch.arange(12).reshape(4, 3).to(dtype=torch.float32)
        points = Points(position=p)
        with self.assertRaises(ValueError):
            points.position = torch.arange(12).reshape(6, 2).to(dtype=torch.float32)

    def tests_if_value_error_is_raised_when_position_dtype_is_initialzied_to_torch_int64(self):
        p = torch.arange(12).reshape(4, 3).to(dtype=torch.int64)
        with self.assertRaises(ValueError):
            Points(position=p)

    def tests_if_value_error_is_raised_when_position_dtype_is_set_to_torch_int64(self):
        p = torch.arange(12).reshape(4, 3).to(dtype=torch.float32)
        points = Points(position=p)
        with self.assertRaises(ValueError):
            points.position = torch.arange(12).reshape(4, 3).to(dtype=torch.int64)

    def tests_if_velocity_is_zeros_like_position_by_default(self):
        p = torch.arange(12).reshape(4, 3).to(dtype=torch.float32)
        points = Points(position=p)
        self.assertTrue(torch.equal(points.velocity, torch.zeros_like(points.position)))

    def tests_if_value_error_is_raised_when_velocity_is_set_to_tensor_of_a_different_shape_to_position(self):
        p = torch.arange(12).reshape(4, 3).to(dtype=torch.float32)
        points = Points(position=p)
        with self.assertRaises(ValueError):
            points.velocity = torch.arange(24).reshape(8, 3).to(dtype=torch.float32)

    def tests_if_force_is_zeros_like_position_by_default(self):
        p = torch.arange(12).reshape(4, 3).to(dtype=torch.float32)
        points = Points(position=p)
        self.assertTrue(torch.equal(points.force, torch.zeros_like(points.position)))

    def tests_if_value_error_is_raised_when_force_is_set_to_tensor_of_a_different_shape_to_position(self):
        p = torch.arange(12).reshape(4, 3).to(dtype=torch.float32)
        points = Points(position=p)
        with self.assertRaises(ValueError):
            points.force = torch.arange(24).reshape(8, 3).to(dtype=torch.float32)

    def tests_the_length_of_points(self):
        p = torch.arange(21).reshape(7, 3).to(dtype=torch.float32)
        points = Points(position=p)
        self.assertEqual(len(points), 7)


if __name__ == "__main__":
    unittest.main()
