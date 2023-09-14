import sys

sys.path.append("src")

from mesh.mesh_factory import MeshFactoryFromScad, MeshFactoryFromObj, MeshFactoryFromMsh
from cable.holes_initial_position import HolesInitialPosition
from mesh.scad import Scad
from mesh.mesh_properties import MeshProperties
from point.transform import Transform
from cable.holes_factory import HolesListFactory
from cable.cable_factory import CableListFactory
from scene.scene import GripperScene, TouchScene, Scene
from os import PathLike
from typing import List
import torch
from warp_wrapper.contact_properties import ContactProperties
from attrs import define
from abc import ABC, abstractmethod


@define
class SceneFactory(ABC):
    scad_file: PathLike
    scad_parameters: PathLike
    ideal_edge_length: float
    robot_properties: MeshProperties
    robot_transform: Transform
    cable_pull_ratio: List[torch.Tensor]
    cable_stiffness: float
    cable_damping: float
    device: str = "cuda"
    msh_file: PathLike = None
    make_new_robot: bool = True

    @abstractmethod
    def create(self) -> Scene:
        pass

    def _robot(self):
        if self.make_new_robot:
            mesh = MeshFactoryFromScad(self._scad(), self.ideal_edge_length, self.device).create()
        else:
            mesh = MeshFactoryFromMsh(self.msh_file, self.device).create()
        self.robot_transform.apply(mesh.nodes)
        mesh.properties = self.robot_properties
        mesh.cables = self._cables()
        return mesh

    def _scad(self):
        return Scad(file=self.scad_file, parameters=self.scad_parameters)

    def _cables(self):
        return CableListFactory(self._holes(), self.cable_pull_ratio, self.cable_stiffness, self.cable_damping).create()

    def _holes(self):
        holes_position = HolesInitialPosition(self._scad()).get()
        holes = HolesListFactory(holes_position, device=self.device).create()
        for hole in holes:
            self.robot_transform.apply(hole)
        return holes
    
@define
class GripperSceneFactory(SceneFactory):
    scad_file: PathLike
    scad_parameters: PathLike
    ideal_edge_length: float
    robot_properties: MeshProperties
    robot_transform: Transform
    cable_pull_ratio: List[torch.Tensor]
    cable_stiffness: float
    cable_damping: float
    object_file: PathLike
    object_properties: MeshProperties
    object_transform: Transform
    contact_properties: ContactProperties
    device: str = "cuda"
    msh_file: PathLike = None
    make_new_robot: bool = True

    def create(self) -> GripperScene:
        return GripperScene(
            robot=self._robot(),
            object=self._object(),
            contact_properties=self.contact_properties,
            device=self.device,
        )

    def _object(self):
        if self.object_file is None:
            return None
        mesh = MeshFactoryFromObj(self.object_file, device=self.device).create()
        mesh.properties = self.object_properties
        self.object_transform.apply(mesh.nodes)
        return mesh