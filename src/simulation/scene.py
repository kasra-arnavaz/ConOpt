from attrs import define, field
import sys
from typing import List

sys.path.append("src")

from mesh.mesh import Mesh
from cable.cable import Cable
from cable.holes import Holes
from warp.sim import Model
from warp_wrapper.model_factory import ModelFactory
import copy
import warp as wp

wp.init()


@define(slots=False)
class Scene:
    gripper: Mesh = field()
    object: Mesh = field(default=None)
    device: str = field(default="cuda")
    model: Model = field()

    @model.default
    def _create_model(self):
        return ModelFactory(soft_mesh=self.gripper, shape_mesh=self.object, device=self.device).create()

    def __attrs_post_init__(self):
        self._initial_gripper = copy.deepcopy(self.gripper)
        self._initial_partilce_q = wp.clone(self.model.particle_q)
        self._initial_partilce_qd = wp.clone(self.model.particle_qd)

    def reset(self):
        self.gripper = copy.deepcopy(self._initial_gripper)
        self.model.particle_q = self._initial_partilce_q
        self.model.particle_qd = self._initial_partilce_qd
