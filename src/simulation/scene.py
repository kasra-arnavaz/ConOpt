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
        self._initial_object = copy.deepcopy(self.object)
        self._initial_model = copy.deepcopy(self.model)
        self._initial_device = self.device

    def reset(self):
        self.gripper = self._initial_gripper
        self.object = self._initial_object
        self.model = self._initial_model
        self.device = self._initial_device
