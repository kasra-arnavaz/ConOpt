import torch
import meshio
import os
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from mesh.mesh import Mesh
from mesh.nodes import Nodes
from mesh.elements import Elements
from mesh.scad import Scad


class MeshFactory(ABC):
    def __init__(self, file: PathLike, device: str = "cuda"):
        self._file = file
        self._device = device

    @abstractmethod
    def create(self) -> Mesh:
        pass


class _MeshFactoryTriangles(MeshFactory):
    def create(self):
        nodes = Nodes(position=self._get_position())
        elements = Elements(triangles=self._get_triangles())
        return Mesh(nodes, elements)

    def _get_position(self):
        return torch.from_numpy(self._get_mesh_data().points).to(dtype=torch.float32, device=self._device)

    def _get_triangles(self):
        return torch.from_numpy(self._get_mesh_data().cells_dict["triangle"]).to(dtype=torch.int32, device=self._device)

    def _get_mesh_data(self):
        return meshio.read(self._file)


class MeshFactoryFromObj(_MeshFactoryTriangles):
    pass


class MeshFactoryFromStl(_MeshFactoryTriangles):
    pass


class MeshFactoryFromMsh(MeshFactory):
    def create(self):
        nodes = Nodes(position=self._get_position())
        elements = Elements(tetrahedra=self._get_tetrahedra())
        return Mesh(nodes, elements)

    def _get_position(self):
        return torch.from_numpy(self._get_mesh_data().points).to(dtype=torch.float32, device=self._device)

    def _get_tetrahedra(self):
        return torch.from_numpy(self._get_mesh_data().cells_dict["tetra"]).to(dtype=torch.int32, device=self._device)

    def _get_mesh_data(self):
        return meshio.read(self._file)


class MeshFactoryFromScad(MeshFactory):
    def __init__(self, scad: Scad, ideal_edge_length: float = 0.02, device: str = "cuda"):
        self.PATH = Path(".tmp")
        self.PATH.mkdir(exist_ok=True)
        self._scad = scad
        self._ideal_edge_length = ideal_edge_length
        self._device = device

    def create(self):
        self._create_files()
        msh_factory = MeshFactoryFromMsh(file=self._get_msh_file(), device=self._device)
        position = msh_factory._get_position()
        tetrahedra = msh_factory._get_tetrahedra()
        nodes = Nodes(position=position)
        elements = Elements(tetrahedra=tetrahedra)
        return Mesh(nodes, elements)

    def _create_files(self):
        self._convert_scad_to_stl()
        self._convert_stl_to_msh_and_obj()

    def _convert_scad_to_stl(self):
        stl = self._get_stl_file()
        os.system(f"openscad -q {self._scad.file} -o {stl} -p {self._scad.parameters} -P firstSet")

    def _convert_stl_to_msh_and_obj(self):
        iel = self._ideal_edge_length
        stl = self._get_stl_file()
        msh = self._get_msh_file()
        os.system(f"fTetWild/build/FloatTetwild_bin -i {stl} -o {msh} -l {iel}")

    def _get_msh_file(self):
        return self.PATH / "mesh.msh"

    def _get_stl_file(self):
        return self.PATH / "mesh.stl"


class MeshFactoryFromTet(MeshFactory):
    def create(self):
        nodes = Nodes(position=self._get_position())
        elements = Elements(tetrahedra=self._get_tetrahedra())
        return Mesh(nodes, elements)

    def _get_position(self):
        lines = self._get_lines()
        nodes = [list(map(float, line[1:])) for line in lines if line[0] == "v"]
        return torch.tensor(nodes, dtype=torch.float32, device=self._device)

    def _get_tetrahedra(self):
        lines = self._get_lines()
        elements = [list(map(int, line[1:])) for line in lines if line[0] == "t"]
        return torch.tensor(elements, dtype=torch.int32, device=self._device)

    def _get_lines(self) -> list:
        with open(self._file) as f:
            lines = f.readlines()
        return [line.split(" ") for line in lines]
