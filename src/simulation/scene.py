from attrs import define, field
import sys
from typing import List

sys.path.append("src")

from mesh.mesh import Mesh
from cable.cable import Cable
from cable.holes import Holes
from warp.sim import Model
from warp_wrapper.model_factory import ModelFactory


@define
class Scene:
    gripper: Mesh = field()
    object: Mesh = field(default=None)
    device: str = field(default="cuda")
    model: Model = field()

    @model.default
    def _create_model(self):
        return ModelFactory(soft_mesh=self.gripper, shape_mesh=self.object, device=self.device).create()
