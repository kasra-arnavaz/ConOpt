from typing import List
from mesh.mesh import Mesh
from cable.cable_actuation import CableActuation


class MeshForce:
    def __init__(self, mesh: Mesh, actuations: List[CableActuation]):
        self._mesh = mesh
        self._actuations = actuations

    def update_force(self):
        self._mesh.nodes.force = self._get_force()

    def _get_force(self):
        return self._actuation.force
