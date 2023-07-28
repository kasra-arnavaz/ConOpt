from attrs import define, field
from mesh.nodes import Nodes
from mesh.elements import Elements


@define
class Mesh:
    nodes: Nodes = field()
    elements: Elements = field()

    @elements.validator
    def _check_max_index(self, attribute, value):
        for element, name in zip([value.triangles, value.tetrahedra], ["triangles", "tetrahedra"]):
            if element is not None:
                if (element.amax() >= len(self.nodes)):
                    raise IndexError(f"Some indices of {name} larger than the number of nodes.")