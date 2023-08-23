import sys
from typing import List
import torch

sys.path.append("src")

from warp_wrapper.update_state import UpdateState
from warp.sim import Model
from cable.barycentric import Barycentric


class NodesForce:
    def __init__(self, barycentrics: List[Barycentric]):
        self._barycentrics = barycentrics

    def __call__(self, holes_forces):
        nodes_forces = []
        for barycentric, holes_force in zip(self._barycentrics, holes_forces):
            nodes_forces.append(barycentric.NxH @ holes_force)
        return torch.stack(nodes_forces).sum(dim=0)


class NodesPositionAndVelocity:
    def __init__(self, model: Model, dt: float):
        self._model = model
        self._dt = dt

    def __call__(self, nodes_force, nodes_position, nodes_velocity):
        args = (
            nodes_force,
            nodes_position,
            nodes_velocity,
            self._model,
            self._dt,
        )
        return UpdateState.apply(*args)
