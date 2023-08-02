import sys
sys.path.append("src")

from mesh.nodes import Nodes
from cable.barycentric import Barycentric
from cable.holes import Holes

class HolesPositionAndVelocity:

    def __init__(self, holes: Holes, nodes: Nodes, barycentric: Barycentric):
        self._holes = holes
        self._nodes = nodes
        self._barycentric = barycentric

    def update(self):
        self._holes.position = self._barycentric.HxN @ self._nodes.position
        self._holes.velocity = self._barycentric.HxN @ self._nodes.velocity
    