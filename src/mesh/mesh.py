from attrs import define, field
from mesh.nodes import Nodes
from mesh.elements import Elements


@define
class Mesh:
    nodes: Nodes = field()
    elements: Elements = field()