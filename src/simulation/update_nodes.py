from mesh.mesh import Nodes
import sys

sys.path.append("src")

from warp_wrapper.update_state import UpdateState
from warp_wrapper.state_iterable import StateIterable
from warp.sim import Model
from cable.holes import Holes
from cable.barycentric import Barycentric
from torch.utils.checkpoint import checkpoint


class NodesForce:
    def __init__(self, nodes: Nodes, holes: Holes, barycentric: Barycentric):
        self._nodes = nodes
        self._holes = holes
        self._barycentric = barycentric

    def update(self):
        self._nodes.force = self._nodes.force + self._barycentric.NxH @ self._holes.force


class NodesPositionAndVelocity:
    def __init__(self, nodes: Nodes, model: Model, dt: float, state_now, state_next):
        self._nodes = nodes
        self._model = model
        self._dt = dt
        self.state_now = state_now
        self.state_next = state_next

    def update(self):
        args = (self._nodes.force,
            self._nodes.position,
            self._nodes.velocity,
            self._model,
            self.state_now,
            self.state_next,
            self._dt,
        )
        self._nodes.position, self._nodes.velocity = UpdateState.apply(*args)
            
