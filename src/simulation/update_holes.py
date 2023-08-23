import sys
import torch

sys.path.append("src")

from cable.cable import Cable
from mesh.nodes import Nodes
from cable.barycentric import Barycentric
from cable.holes import Holes


class HolesForce:
    def __init__(self, cable: Cable, device: str = "cuda"):
        self._device = device
        self._cable = cable

    def __call__(self, holes_position, holes_velocity) -> torch.Tensor:
        force_ = torch.zeros([len(self._cable.holes), 3], dtype=torch.float32, device=self._device, requires_grad=True)
        tangent_vector_pointing_to_the_tip = holes_position[1:] - holes_position[:-1]
        f = -self._cable.pull_ratio * self._cable.stiffness * tangent_vector_pointing_to_the_tip
        g = self._cable.damping * holes_velocity
        force = force_.clone()
        force[0] = -f[0] - g[0]
        force[1:-1] = f[:-1] - f[1:] - g[1:-1]
        force[-1] = f[-1] - g[-1]
        return force



class HolesPositionAndVelocity:
    def __init__(self, barycentric: Barycentric):
        self._barycentric = barycentric

    def __call__(self, nodes_position, nodes_velocity):
        holes_position = self._barycentric.HxN @ nodes_position
        holes_velocity = self._barycentric.HxN @ nodes_velocity
        return holes_position, holes_velocity
