from attrs import define, field
from typing import List
from mesh.nodes import Nodes
from mesh.elements import Elements
from cable.cable import Cable
from mesh.mesh_properties import MeshProperties


@define
class Mesh:
    nodes: Nodes = field()
    elements: Elements = field()
    properties: MeshProperties = field(default=None)
    cables: List[Cable] = field(default=None)

    @elements.validator
    def _check_max_index(self, attribute, value):
        elements = [value.triangles, value.tetrahedra]
        names = ["triangles", "tetrahedra"]
        for element, name in zip(elements, names):
            if element is not None:
                if element.amax() >= len(self.nodes):
                    raise IndexError(f"Some indices of {name} larger than the number of nodes.")
