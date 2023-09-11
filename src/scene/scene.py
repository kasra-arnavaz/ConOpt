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

wp.init()


@define(slots=False)
class Scene:
    gripper: Mesh = field()
    object: Mesh = field(default=None)
    contact_properties: ContactProperties = field(default=None)
    device: str = field(default="cuda")
    model: Model = field(init=False)
    gripper_end_effector_idx: int = field(init=False)

    @model.default
    def _create_model(self):
        return ModelFactory(
            soft_mesh=self.gripper,
            shape_mesh=self.object,
            contact_properties=self.contact_properties,
            device=self.device,
        ).create()

    def __attrs_post_init__(self):
        self._initial_gripper_nodes = copy.deepcopy(self.gripper.nodes)
        self._initial_partilce_q = wp.clone(self.model.particle_q)
        self._initial_partilce_qd = wp.clone(self.model.particle_qd)

    def reset(self):
        self.gripper.nodes = copy.deepcopy(self._initial_gripper_nodes)
        self.model.particle_q = self._initial_partilce_q
        self.model.particle_qd = self._initial_partilce_qd

    @gripper_end_effector_idx.default
    def _gripper_end_effector_idx(self):
        return self.gripper.nodes.position[:, 1].argmin()  # assumes gripper is facing down
