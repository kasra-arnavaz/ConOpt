import unittest
import torch
import sys

sys.path.append("src")

from mesh.mesh import Mesh
from mesh.nodes import Nodes
from mesh.elements import Elements


class TestMesh(unittest.TestCase):
    def tests_if_index_error_is_raised_when_triangles_are_initialized_out_of_bounds(self):
        nodes = Nodes(position=torch.arange(9).reshape(3, 3).to(dtype=torch.float32))
        elements = Elements(triangles=torch.tensor([[0, 1, 3]], dtype=torch.int32))
        with self.assertRaises(IndexError):
            Mesh(nodes, elements)

    def tests_if_index_error_is_raised_when_triangles_are_set_out_of_bounds(self):
        nodes = Nodes(position=torch.arange(9).reshape(3, 3).to(dtype=torch.float32))
        elements = Elements()
        mesh = Mesh(nodes, elements)
        with self.assertRaises(IndexError):
            mesh.elements = Elements(triangles=torch.tensor([[0, 1, 3]], dtype=torch.int32))

    def tests_if_index_error_is_raised_when_tetrahedra_are_initialized_out_of_bounds(self):
        nodes = Nodes(position=torch.arange(15).reshape(5, 3).to(dtype=torch.float32))
        elements = Elements(tetrahedra=torch.tensor([[0, 1, 2, 7]], dtype=torch.int32))
        with self.assertRaises(IndexError):
            Mesh(nodes, elements)

    def tests_if_index_error_is_raised_when_tetrahedra_are_set_out_of_bounds(self):
        nodes = Nodes(position=torch.arange(15).reshape(5, 3).to(dtype=torch.float32))
        elements = Elements()
        mesh = Mesh(nodes, elements)
        with self.assertRaises(IndexError):
            mesh.elements = Elements(tetrahedra=torch.tensor([[0, 1, 2, 7]], dtype=torch.int32))

    def tests_if_index_error_is_not_raised_when_triangles_are_initialized_in_bounds(self):
        nodes = Nodes(position=torch.arange(9).reshape(3, 3).to(dtype=torch.float32))
        elements = Elements(triangles=torch.tensor([[0, 1, 2]], dtype=torch.int32))
        try:
            Mesh(nodes, elements)
        except IndexError:
            self.fail("Raised IndexError unexpectedly.")

    def tests_if_index_error_is_not_raised_when_triangles_are_set_in_bounds(self):
        nodes = Nodes(position=torch.arange(9).reshape(3, 3).to(dtype=torch.float32))
        elements = Elements()
        mesh = Mesh(nodes, elements)
        try:
            mesh.elements = Elements(triangles=torch.tensor([[0, 1, 2]], dtype=torch.int32))
        except IndexError:
            self.fail("Raised IndexError unexpectedly.")

    def tests_if_index_error_is_not_raised_when_tetrahedra_are_initialized_in_bounds(self):
        nodes = Nodes(position=torch.arange(15).reshape(5, 3).to(dtype=torch.float32))
        elements = Elements(tetrahedra=torch.tensor([[0, 1, 2, 3], [4, 3, 1, 2]], dtype=torch.int32))
        try:
            Mesh(nodes, elements)
        except IndexError:
            self.fail("Raised IndexError unexpectedly.")

    def tests_if_index_error_is_not_raised_when_tetrahedra_are_set_in_bounds(self):
        nodes = Nodes(position=torch.arange(15).reshape(5, 3).to(dtype=torch.float32))
        elements = Elements()
        mesh = Mesh(nodes, elements)
        try:
            mesh.elements = Elements(tetrahedra=torch.tensor([[0, 1, 2, 3], [4, 3, 1, 2]], dtype=torch.int32))
        except IndexError:
            self.fail("Raised IndexError unexpectedly.")


if __name__ == "__main__":
    unittest.main()
