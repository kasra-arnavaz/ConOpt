from mesh.mesh import Nodes
import sys

sys.path.append("src")

from warp_wrapper.update_state import UpdateState
from warp.sim import Model

import warp as wp


class NodesPositionAndVelocity:
    def __init__(self, nodes: Nodes, model: Model, dt: float):
        self._nodes = nodes
        self._model = model
        self._dt = dt

    def update(self):
        state_now = self._model.state(requires_grad=True)
        state_next = self._model.state(requires_grad=True)
        self._nodes.position, self._nodes.velocity = UpdateState.apply(
            self._nodes.force,
            self._nodes.position,
            self._nodes.velocity,
            self._model,
            self._dt,
            state_now,
            state_next,
        )
