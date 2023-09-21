import sys

sys.path.append("src")

from mesh.mesh_factory import MeshFactoryFromScad, MeshFactoryFromObj, MeshFactoryFromMsh
from cable.holes_initial_position import CaterpillarHolesInitialPosition
from mesh.scad import Scad
from mesh.mesh_properties import MeshProperties
from point.transform import Transform
from cable.holes_factory import HolesListFactory
from cable.cable_factory import CableListFactory
from scene.scene import GripperScene, TouchScene, Scene
from os import PathLike
from typing import List
import torch
from mesh.mesh import Mesh
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
        holes_position = CaterpillarHolesInitialPosition(self._scad()).get()
        holes = HolesListFactory(holes_position, device=self.device).create()
        for hole in holes:
            self.robot_transform.apply(hole)
        return holes
    
    @staticmethod
    def _create_obj_mesh(file: PathLike, properties: MeshProperties, transform: Transform, device: str) -> Mesh:
        mesh = MeshFactoryFromObj(file, device=device).create()
        mesh.properties = properties
        transform.apply(mesh.nodes)
        return mesh
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
        return self._create_obj_mesh(self.object_file, self.object_properties, self.object_transform, self.device)
    
@define
class TouchSceneFactory(GripperSceneFactory):
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
    obstacle_files: List[PathLike] = None
    obstacle_properties: List[MeshProperties] = None
    obstacle_transforms: List[Transform] = None
    device: str = "cuda"
    msh_file: PathLike = None
    make_new_robot: bool = True

    def create(self) -> TouchScene:
        return TouchScene(
            robot=self._robot(),
            object=self._object(),
            obstacles=self._obstacles(),
            contact_properties=self.contact_properties,
            device=self.device,
        )
    
    def _obstacles(self) -> List[Mesh]:
        if self.obstacle_files is None:
            return None
        obstacles = []
        for file, properties, transform in zip(self.obstacle_files, self.obstacle_properties, self.obstacle_transforms):
            obstacles.append(self._create_obj_mesh(file, properties, transform, self.device))
        return obstacles
