import torch
from abc import ABC, abstractproperty
from typing import List
import warp as wp
from mesh.mesh import Mesh
from cable.cable import Cable
from utils.geometry import point_is_in_tetrahedron, barycentric_coordinates



class CableActuation:
    def __init__(self, mesh: Mesh, cable: Cable, device: str = "cuda"):
        self._mesh = mesh
        self._cable = cable
        self._device = device

    def get_forces_per_cable(self):
            force_ = torch.zeros([len(self._cable.holes), 3], dtype=torch.float32, device=self._device, requires_grad=True)
            tangent_vector = self._get_tangent_vector()
            f = -self._pull * (self._stiffness * tangent_vector)
            g = self._damping * (self._hole.barycentric @ self._mesh.velocity)
            force = force_.clone()
            force[0] = -f[0] - g[0]
            force[1:-1] = f[:-1] - f[1:] - g[1:-1]
            force[-1] = f[-1] - g[-1]
            return self._hole.barycentric_pinv @ force

    def _get_tangent_vector(self) -> torch.Tensor:
        hole_pos = self._hole.barycentric @ self._mesh.nodes
        return hole_pos[1:] - hole_pos[:-1]  # pointing to the tip

    def _get_barycentric_pinv(self) -> List[torch.Tensor]:
        return [torch.pinverse(barycentric) for barycentric in self._barycentric]

    def _get_barycentric(self) -> List[torch.Tensor]:
        barycentrics = []
        for holes_position, num_holes in zip(self._cables.holes_position_per_cable, self._cables.num_holes_per_cable):
            holes = wp.from_torch(holes_position.contiguous(), dtype=wp.vec3)
            nodes = wp.from_torch(self._mesh.nodes.position.contiguous(), dtype=wp.vec3)
            tetrahedra = wp.from_torch(self._mesh.elements.tetrahedra, dtype=int)
            barycentric_matrix = wp.zeros(num_holes, len(self._mesh.nodes), dtype=float, device=self._device)
            wp.launch(
                kernel=self._barycentric_kernel,
                dim=[num_holes, self._mesh.elements.num_tetrahedra, 4],
                inputs=[holes, nodes, tetrahedra, barycentric_matrix],
                device=self._device)
            barycentrics.append(wp.to_torch(barycentric_matrix))
        return barycentrics
    
    @wp.kernel
    def _barycentric_kernel(
        holes: wp.array(dtype=wp.vec3),
        nodes: wp.array(dtype=wp.vec3),
        tetrahedra: wp.array2d(dtype=int),
        w: wp.array2d(dtype=float)) -> None:
        i, j, k = wp.tid()
        hole_i = holes[i]
        tet_j = tetrahedra[j]
        n_j_0 = nodes[tet_j[0]]
        n_j_1 = nodes[tet_j[1]]
        n_j_2 = nodes[tet_j[2]]
        n_j_3 = nodes[tet_j[3]]
        if point_is_in_tetrahedron(n_j_0, n_j_1, n_j_2, n_j_3, hole_i):
            w[i, tet_j[k]] = barycentric_coordinates(n_j_0, n_j_1, n_j_2, n_j_3, hole_i, k)
