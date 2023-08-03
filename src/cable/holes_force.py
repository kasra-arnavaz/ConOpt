import sys
import torch

sys.path.append("src")

from cable.cable import Cable


class HolesForce:
    def __init__(self, cable: Cable, device: str = "cuda"):
        self._cable = cable
        self._device = device

    def update(self):
        self._cable.holes.force = self.get_force()

    def get_force(self) -> torch.Tensor:
        force_ = torch.zeros([len(self._cable.holes), 3], dtype=torch.float32, device=self._device, requires_grad=True)
        f = -self._cable.pull_ratio * self._cable.stiffness * self._tangent_vector_pointing_to_the_tip
        g = self._cable.damping * self._cable.holes.velocity
        force = force_.clone()
        force[0] = -f[0] - g[0]
        force[1:-1] = f[:-1] - f[1:] - g[1:-1]
        force[-1] = f[-1] - g[-1]
        return force

    @property
    def _tangent_vector_pointing_to_the_tip(self) -> torch.Tensor:
        return self._cable.holes.position[1:] - self._cable.holes.position[:-1]
