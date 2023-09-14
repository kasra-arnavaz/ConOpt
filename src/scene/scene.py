from attrs import define, field
import sys
from typing import List

sys.path.append("src")

from mesh.mesh import Mesh
from warp.sim import Model
from warp_wrapper.model_factory import ModelFactory
from warp_wrapper.contact_properties import ContactProperties
import copy
import warp as wp
from abc import ABC, abstractmethod

wp.init()


@define(slots=False)
class Scene(ABC):
    robot: Mesh = field()
    device: str = field(default="cuda")
    model: Model = field(init=False)

    @model.default
    @abstractmethod
    def _create_model(self):
        pass

    def all_meshes(self) -> List[Mesh]:
        return [self.robot]

    def __attrs_post_init__(self):
        self._initial_robot_nodes = copy.deepcopy(self.robot.nodes)
        self._initial_partilce_q = wp.clone(self.model.particle_q)
        self._initial_partilce_qd = wp.clone(self.model.particle_qd)

    def reset(self):
        self.robot.nodes = copy.deepcopy(self._initial_robot_nodes)
        self.model.particle_q = self._initial_partilce_q
        self.model.particle_qd = self._initial_partilce_qd

@define
class GripperScene(Scene):
    robot: Mesh = field()
    object: Mesh = field()
    contact_properties: ContactProperties = field()
    device: str = field(default="cuda")
    model: Model = field(init=False)


    @model.default
    def _create_model(self):
        return ModelFactory(
            soft_mesh=self.robot,
            shape_meshes=[self.object],
            contact_properties=self.contact_properties,
            device=self.device,
        ).create()
    
    def all_meshes(self) -> List[Mesh]:
        return [self.robot, self.object]


@define
class TouchScene(Scene):
    robot: Mesh = field()
    object: Mesh = field()
    obstacles: List[Mesh] = field(default=None)
    contact_properties: ContactProperties = field(default=None)
    device: str = field(default="cuda")
    model: Model = field(init=False)
    robot_end_effector_idx: int = field(init=False)

    def all_meshes(self):
        return self.robot + self._shape_meshes()

    @model.default
    def _create_model(self):
        return ModelFactory(
            soft_mesh=self.robot,
            shape_meshes=self._shape_meshes(),
            contact_properties=self.contact_properties,
            device=self.device,
        ).create()
    
    def _shape_meshes(self):
        shape_meshes = self.obstacles.copy() if self.obstacles is not None else []
        shape_meshes.append(self.object)
        return shape_meshes
    
    @robot_end_effector_idx.default
    def _robot_end_effector_idx(self):
        return self.robot.nodes.position[:, 1].argmin()  # assumes robot is facing down
