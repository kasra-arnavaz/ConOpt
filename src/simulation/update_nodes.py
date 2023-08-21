from mesh.mesh import Nodes
import sys

sys.path.append("src")

from warp_wrapper.update_state import UpdateState
from warp.sim import Model
from cable.holes import Holes
from cable.barycentric import Barycentric

class NodesForce:
    def __init__(self, nodes: Nodes, holes: Holes, barycentric: Barycentric):
        self._nodes = nodes
        self._holes = holes
        self._barycentric = barycentric

    def update(self):
        self._nodes.force = self._nodes.force + self._barycentric.NxH @ self._holes.force


class NodesPositionAndVelocity:
    def __init__(self, nodes: Nodes, model: Model, dt: float):
        self._nodes = nodes
        self._model = model
        self._dt = dt

    def update(self):
        self._nodes.position, self._nodes.velocity = UpdateState.apply(
            self._nodes.force,
            self._nodes.position,
            self._nodes.velocity,
            self._model,
            self._dt,
        )