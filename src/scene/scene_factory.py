import sys

sys.path.append("src")

from mesh.mesh_factory import MeshFactoryFromScad, MeshFactoryFromObj
from cable.holes_initial_position import HolesInitialPosition
from mesh.scad import Scad
from mesh.mesh_properties import MeshProperties
from point.transform import Transform
from cable.holes_factory import HolesListFactory
from cable.cable_factory import CableListFactory
from scene.scene import Scene
from os import PathLike
from typing import List
import torch
from attrs import define


@define
class SceneFactory:
    scad_file: PathLike
    scad_parameters: PathLike
    ideal_edge_lenght: float
    gripper_properties: MeshProperties
    gripper_transform: Transform
    cable_pull_ratio: List[torch.Tensor]
    cable_stiffness: float
    cable_damping: float

    object_file: PathLike
    object_properties: MeshProperties
    object_transform: Transform

    device: str

    def create(self) -> Scene:
        return Scene(gripper=self._gripper(), object=self._object(), device=self.device)

    def _gripper(self):
        mesh = MeshFactoryFromScad(self._scad(), self.ideal_edge_lenght, self.device).create()
        self.gripper_transform.apply(mesh.nodes)
        mesh.properties = self.gripper_properties
        mesh.cables = self._cables()
        return mesh

    def _object(self):
        mesh = MeshFactoryFromObj(self.object_file, device=self.device).create()
        mesh.properties = self.object_properties
        self.object_transform.apply(mesh.nodes)
        return mesh

    def _scad(self):
        return Scad(file=self.scad_file, parameters=self.scad_parameters)

    def _cables(self):
        return CableListFactory(self._holes(), self.cable_pull_ratio, self.cable_stiffness, self.cable_damping).create()

    def _holes(self):
        holes_position = HolesInitialPosition(self._scad()).get()
        holes = HolesListFactory(holes_position, device=self.device).create()
        for hole in holes:
            self.gripper_transform.apply(hole)
        return holes
