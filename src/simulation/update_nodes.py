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
    def __init__(self, barycentric: Barycentric):
        self._barycentric = barycentric

    def __call__(self, holes_force):
        return self._barycentric.NxH @ holes_force


class NodesPositionAndVelocity:
    def __init__(self, model: Model, dt: float):
        self._model = model
        self._dt = dt

    def __call__(self, nodes_force, nodes_position, nodes_velocity):
        args = (nodes_force,
            nodes_position,
            nodes_velocity,
            self._model,
            self._dt,
        )
        return UpdateState.apply(*args)
            
