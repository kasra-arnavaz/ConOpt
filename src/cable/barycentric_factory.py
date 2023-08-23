import warp as wp
import torch
from typing import List
from mesh.mesh import Mesh
from cable.cable import Cable
from cable.holes import Holes
from warp_wrapper.geometry import point_is_in_tetrahedron, barycentric_coordinates
from cable.barycentric import Barycentric


class MultiBarycentricFactory:
    def __init__(self, mesh: Mesh, cables: List[Cable], device: str = "cuda"):
        self._mesh = mesh
        self._cables = cables
        self._device = device

    def create(self) -> List[Barycentric]:
        barycentrics = []
        for cable in self._cables:
            barycentric = BarycentricFactory(mesh=self._mesh, holes=cable.holes, device=self._device).create()
            barycentrics.append(barycentric)
        return barycentrics


class BarycentricFactory:
    def __init__(self, mesh: Mesh, holes: Holes, device: str = "cuda"):
        self._mesh = mesh
        self._holes = holes
        self._device = device

    def create(self) -> Barycentric:
        HxN = self._get_barycentric()
        NxH = self._get_barycentric_pinv(HxN)
        return Barycentric(HxN=HxN, NxH=NxH)

    @staticmethod
    def _get_barycentric_pinv(barycentric: torch.Tensor) -> torch.Tensor:
        return torch.pinverse(barycentric)

    def _get_barycentric(self) -> torch.Tensor:
        holes = wp.from_torch(self._holes.position.contiguous(), dtype=wp.vec3)
        nodes = wp.from_torch(self._mesh.nodes.position.contiguous(), dtype=wp.vec3)
        tetrahedra = wp.from_torch(self._mesh.elements.tetrahedra, dtype=int)
        barycentric_matrix = wp.zeros((len(self._holes), len(self._mesh.nodes)), dtype=float, device=self._device)
        wp.launch(
            kernel=self._barycentric_kernel,
            dim=[len(self._holes), self._mesh.elements.num_tetrahedra, 4],
            inputs=[holes, nodes, tetrahedra, barycentric_matrix],
            device=self._device,
        )
        return wp.to_torch(barycentric_matrix)

    @wp.kernel
    def _barycentric_kernel(
        holes: wp.array(dtype=wp.vec3),
        nodes: wp.array(dtype=wp.vec3),
        tetrahedra: wp.array2d(dtype=int),
        w: wp.array2d(dtype=float),
    ) -> None:
        i, j, k = wp.tid()
        hole_i = holes[i]
        tet_j = tetrahedra[j]
        n_j_0 = nodes[tet_j[0]]
        n_j_1 = nodes[tet_j[1]]
        n_j_2 = nodes[tet_j[2]]
        n_j_3 = nodes[tet_j[3]]
        if point_is_in_tetrahedron(n_j_0, n_j_1, n_j_2, n_j_3, hole_i):
            w[i, tet_j[k]] = barycentric_coordinates(n_j_0, n_j_1, n_j_2, n_j_3, hole_i, k)
