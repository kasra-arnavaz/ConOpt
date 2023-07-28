import torch
import meshio
from abc import ABC, abstractmethod
from os import PathLike
from mesh.mesh import Mesh
from mesh.nodes import Nodes
from mesh.elements import Elements

class MeshFactory(ABC):
    def __init__(self, file: PathLike):
        self._file = file

    @abstractmethod
    def create(self) -> Mesh:
        pass


class MeshFactoryFromObj(MeshFactory):
    def create(self):
        nodes = Nodes(position=self._get_position())
        elements = Elements(triangles=self._get_triangles())
        return Mesh(nodes, elements)

    def _get_position(self):
        return torch.from_numpy(self._get_obj_mesh().points).to(dtype=torch.float32)

    def _get_triangles(self):
        return torch.from_numpy(self._get_obj_mesh().cells_dict["triangle"]).to(dtype=torch.int64)
    
    def _get_obj_mesh(self):
        return meshio.read(self._file)
    
class MeshFactoryFromStl(MeshFactory):
    def create(self):
        nodes = Nodes(position=self._get_position())
        elements = Elements(triangles=self._get_triangles())
        return Mesh(nodes, elements)

    def _get_position(self):
        return torch.from_numpy(self._get_stl_mesh().points).to(dtype=torch.float32)

    def _get_triangles(self):
        return torch.from_numpy(self._get_stl_mesh().cells_dict["triangle"]).to(dtype=torch.int64)
    
    def _get_stl_mesh(self):
        return meshio.read(self._file)
    

class MeshFactoryFromMsh(MeshFactory):
    def create(self):
        nodes = Nodes(position=self._get_position())
        elements = Elements(tetrahedra=self._get_tetrahedra())
        return Mesh(nodes, elements)
    
    def _get_position(self):
        return torch.from_numpy(self._get_msh_mesh().points).to(dtype=torch.float32)

    def _get_tetrahedra(self):
        return torch.from_numpy(self._get_msh_mesh().cells_dict["tetra"]).to(dtype=torch.int64)

    def _get_msh_mesh(self):
        return meshio.read(self._file)


class MeshFactoryFromTet(MeshFactory):
    def create(self):
        nodes = Nodes(position=self._get_position())
        elements = Elements(tetrahedra=self._get_tetrahedra())
        return Mesh(nodes, elements)

    def _get_position(self):
        lines = self._get_lines()
        nodes = [list(map(float, line[1:])) for line in lines if line[0] == "v"]
        return torch.tensor(nodes, dtype=torch.float32)

    def _get_tetrahedra(self):
        lines = self._get_lines()
        elements = [list(map(int, line[1:])) for line in lines if line[0] == "t"]
        return torch.tensor(elements, dtype=torch.int64)

    def _get_lines(self) -> list:
        with open(self._file) as f:
            lines = f.readlines()
        return [line.split(" ") for line in lines]
