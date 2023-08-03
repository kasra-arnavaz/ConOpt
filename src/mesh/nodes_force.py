from mesh.mesh import Nodes
import sys

sys.path.append("src")

from cable.holes import Holes
from cable.barycentric import Barycentric


class NodesForce:
    def __init__(self, nodes: Nodes, holes: Holes, barycentric: Barycentric):
        self._nodes = nodes
        self._holes = holes
        self._barycentric = barycentric

    def update(self):
        self._nodes.force = self._nodes.force + self._barycentric.NxH @ self._holes.force
