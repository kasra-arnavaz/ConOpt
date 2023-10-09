import sys
import torch

sys.path.append("src")
from typing import List

from cable.cable import Cable
from cable.barycentric import Barycentric


class HolesForce:
    def __init__(self, cables: List[Cable], device: str = "cuda"):
        self._device = device
        self._cables = cables

    def __call__(self, holes_positions: List[torch.Tensor], holes_velocities: List[torch.Tensor]) -> List[torch.Tensor]:
        holes_forces = []
        for holes_position, holes_velocity, cable in zip(holes_positions, holes_velocities, self._cables):
            force = torch.zeros(holes_position.shape, dtype=torch.float32, device=self._device)
            tangent_vector_pointing_to_the_tip = holes_position[1:] - holes_position[:-1]
            f = -next(cable.pull_ratio.iterator) * cable.stiffness * tangent_vector_pointing_to_the_tip
            g = cable.damping * holes_velocity
            force[0] = -f[0] - g[0]
            force[1:-1] = f[:-1] - f[1:] - g[1:-1]
            force[-1] = f[-1] - g[-1]
            holes_forces.append(force)
        return holes_forces


class HolesPositionAndVelocity:
    def __init__(self, barycentrics: List[Barycentric]):
        self._barycentrics = barycentrics

    def __call__(self, nodes_position: torch.Tensor, nodes_velocity: torch.Tensor):
        holes_positions, holes_velocities = [], []
        for barycentric in self._barycentrics:
            holes_positions.append(barycentric.HxN @ nodes_position)
            holes_velocities.append(barycentric.HxN @ nodes_velocity)
        return holes_positions, holes_velocities
