import unittest
import torch
import sys

sys.path.append("src")
from mesh.elements import Elements


class TestElements(unittest.TestCase):
    def tests_if_type_error_is_raised_when_triangles_is_initialized_to_list(self):
        with self.assertRaises(TypeError):
            Elements(triangles=[1, 2, 3])

    def tests_if_type_error_is_raised_when_triangles_is_set_to_list(self):
        tri = torch.arange(12).reshape(4, 3).to(dtype=torch.int64)
        elements = Elements(triangles=tri)
        with self.assertRaises(TypeError):
            elements.triangles = [1, 2, 3]

    def tests_if_value_error_is_raised_when_triangles_is_initialized_to_1d_tensor(self):
        with self.assertRaises(ValueError):
            tri = torch.arange(12).to(dtype=torch.int64)
            Elements(triangles=tri)

    def tests_if_value_error_is_raised_when_triangles_is_set_to_1d_tensor(self):
        tri = torch.arange(12).reshape(4, 3).to(dtype=torch.int64)
        elements = Elements(triangles=tri)
        with self.assertRaises(ValueError):
            elements.triangles = torch.arange(12).to(dtype=torch.int64)

    def tests_if_value_error_is_raised_when_triangles_is_initialized_to_3d_tensor(self):
        with self.assertRaises(ValueError):
            tri = torch.arange(12).reshape(2, 2, 3).to(dtype=torch.int64)
            Elements(triangles=tri)

    def tests_if_value_error_is_raised_when_triangles_is_set_to_3d_tensor(self):
        tri = torch.arange(12).reshape(4, 3).to(dtype=torch.int64)
        elements = Elements(triangles=tri)
        with self.assertRaises(ValueError):
            elements.triangles = tri.reshape(2, 2, 3).to(dtype=torch.int64)

    def tests_if_value_error_is_raised_when_triangles_last_dimension_is_initialized_to_2(self):
        with self.assertRaises(ValueError):
            tri = torch.arange(12).reshape(6, 2).to(dtype=torch.int64)
            Elements(triangles=tri)

    def tests_if_value_error_is_raised_when_triangles_last_dimension_is_set_to_2(self):
        tri = torch.arange(12).reshape(4, 3).to(dtype=torch.int64)
        elements = Elements(triangles=tri)
        with self.assertRaises(ValueError):
            elements.triangles = torch.arange(12).reshape(6, 2).to(dtype=torch.int64)

    def tests_if_value_error_is_raised_when_tetrahedras_last_dimension_is_initialized_to_3(self):
        with self.assertRaises(ValueError):
            tet = torch.arange(12).reshape(4, 3).to(dtype=torch.int64)
            Elements(tetrahedra=tet)

    def tests_if_value_error_is_raised_when_tetrahedras_last_dimension_is_set_to_3(self):
        tri = torch.arange(12).reshape(4, 3).to(dtype=torch.int64)
        elements = Elements(triangles=tri)
        with self.assertRaises(ValueError):
            elements.tetrahedra = torch.arange(12).reshape(4, 3).to(dtype=torch.int64)

    def tests_if_tetrahedra_is_none_when_initialized(self):
        tri = torch.arange(12).reshape(4, 3)
        elements = Elements(triangles=tri)
        self.assertIsNone(elements.tetrahedra)

    def tests_if_value_error_is_raised_when_triangles_dtype_is_initialzied_to_torch_float32(self):
        with self.assertRaises(ValueError):
            tri = torch.arange(12).reshape(4, 3).to(dtype=torch.float32)
            elements = Elements(triangles=tri)

    def tests_if_value_error_is_raised_when_position_dtype_is_set_to_torch_float32(self):
        tri = torch.arange(12).reshape(4, 3).to(dtype=torch.int64)
        elements = Elements(triangles=tri)
        with self.assertRaises(ValueError):
            elements.triangles = torch.arange(12).reshape(4, 3).to(dtype=torch.float32)

    def tests_if_value_error_is_raised_when_tetrahedra_dtype_is_initialzied_to_torch_float32(self):
        tet = torch.arange(12).reshape(3, 4).to(dtype=torch.float32)
        with self.assertRaises(ValueError):
            Elements(tetrahedra=tet)

    def tests_if_value_error_is_raised_when_position_dtype_is_set_to_torch_float32(self):
        tet = torch.arange(12).reshape(3, 4).to(dtype=torch.int64)
        elements = Elements(tetrahedra=tet)
        with self.assertRaises(ValueError):
            elements.triangles = torch.arange(12).reshape(4, 3).to(dtype=torch.float32)

    def tests_the_num_of_triangles(self):
        tri = torch.arange(21).reshape(7, 3).to(dtype=torch.int64)
        elements = Elements(triangles=tri)
        self.assertEqual(elements.num_triangles, 7)

    def tests_the_num_of_tetrahedra(self):
        tet = torch.arange(24).reshape(6, 4).to(dtype=torch.int64)
        elements = Elements(tetrahedra=tet)
        self.assertEqual(elements.num_tetrahedra, 6)


if __name__ == "__main__":
    unittest.main()
