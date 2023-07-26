import unittest
import torch
import sys

sys.path.append("src")
from mesh.nodes import Nodes


class TestNodes(unittest.TestCase):
    def tests_if_type_error_is_raised_when_position_is_initialized_to_list(self):
        with self.assertRaises(TypeError):
            Nodes(position=[1, 2, 3])

    def tests_if_type_error_is_raised_when_position_is_set_to_list(self):
        with self.assertRaises(TypeError):
            nodes = Nodes(position=torch.zeros([10, 3], dtype=torch.float32))
            nodes.position = [1, 2, 3]

    def tests_if_value_error_is_raised_when_position_is_initialized_to_1d_tensor(self):
        with self.assertRaises(ValueError):
            p = torch.arange(12).to(dtype=torch.float32)
            Nodes(position=p)

    def tests_if_value_error_is_raised_when_position_is_set_to_1d_tensor(self):
        with self.assertRaises(ValueError):
            p = torch.arange(12).reshape(4, 3).to(dtype=torch.float32)
            nodes = Nodes(position=p)
            nodes.position = torch.arange(12).to(dtype=torch.float32)

    def tests_if_value_error_is_raised_when_position_is_initialized_to_3d_tensor(self):
        with self.assertRaises(ValueError):
            p = torch.arange(12).reshape(2, 2, 3).to(dtype=torch.float32)
            Nodes(position=p)

    def tests_if_value_error_is_raised_when_position_is_set_to_3d_tensor(self):
        with self.assertRaises(ValueError):
            p = torch.arange(12).reshape(4, 3).to(dtype=torch.float32)
            nodes = Nodes(position=p)
            nodes.position = p.reshape(2, 2, 3).to(dtype=torch.float32)

    def tests_if_value_error_is_raised_when_positions_last_dimension_is_initialized_to_2(self):
        with self.assertRaises(ValueError):
            p = torch.arange(12).reshape(6, 2).to(dtype=torch.float32)
            Nodes(position=p)

    def tests_if_value_error_is_raised_when_positions_last_dimension_is_set_to_2(self):
        with self.assertRaises(ValueError):
            p = torch.arange(12).reshape(4, 3).to(dtype=torch.float32)
            nodes = Nodes(position=p)
            nodes.position = torch.arange(12).reshape(6, 2).to(dtype=torch.float32)

    def tests_if_value_error_is_raised_when_position_dtype_is_initialzied_to_torch_int64(self):
        with self.assertRaises(ValueError):
            p = torch.arange(12).reshape(4,3).to(dtype=torch.int64)
            nodes = Nodes(position=p)

    def tests_if_value_error_is_raised_when_position_dtype_is_set_to_torch_int64(self):
        with self.assertRaises(ValueError):
            p = torch.arange(12).reshape(4,3).to(dtype=torch.float32)
            nodes = Nodes(position=p)
            nodes.position = torch.arange(12).reshape(4,3).to(dtype=torch.int64)

    def tests_if_velocity_is_none_when_initialized(self):
        p = torch.arange(12).reshape(4, 3).to(dtype=torch.float32)
        nodes = Nodes(position=p)
        self.assertIsNone(nodes.velocity)

    def tests_if_value_error_is_raised_when_velocity_is_set_to_tensor_of_a_different_shape_to_position(self):
        with self.assertRaises(ValueError):
            p = torch.arange(12).reshape(4, 3).to(dtype=torch.float32)
            nodes = Nodes(position=p)
            nodes.velocity = torch.arange(24).reshape(8, 3).to(dtype=torch.float32)



    def tests_the_length_of_nodes(self):
        p = torch.arange(21).reshape(7,3).to(dtype=torch.float32)
        nodes = Nodes(position=p)
        self.assertEqual(len(nodes), 7)


if __name__ == "__main__":
    unittest.main()
