import torch
import numpy as np
import warp as wp
from warp.sim import Model, ModelBuilder
import sys

sys.path.append("src")
from mesh.mesh import Mesh

wp.init()


class ModelFactory:
    def __init__(self, soft_mesh: Mesh = None, shape_mesh: Mesh = None, device: str = "cuda"):
        self._soft_mesh = soft_mesh
        self._shape_mesh = shape_mesh
        self._device = device

    def create(self) -> Model:
        builder: ModelBuilder = wp.sim.ModelBuilder()
        self._add_soft_mesh(builder) if self._soft_mesh is not None else None
        self._add_shape_mesh(builder) if self._shape_mesh is not None else None
        model = builder.finalize(self._device)
        model = self._update_model_attributes(model)
        return model

    def _add_soft_mesh(self, builder: ModelBuilder):
        properties = self._soft_mesh.properties
        k_lambda, k_mu = wp.utils.lame_parameters(properties.youngs_modulus, properties.poissons_ratio)
        builder.add_soft_mesh(
            pos=np.array([0, 0, 0]),
            rot=np.array([0, 0, 0, 1]),
            scale=1,
            vel=[0, 0, 0],
            vertices=self._soft_mesh.nodes.position.detach().cpu().numpy(),
            indices=self._soft_mesh.elements.tetrahedra.cpu().numpy().reshape(-1),
            density=properties.density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=properties.damping_factor,
        )
        self._apply_boundary_conditions(builder)
        self._set_soft_mesh_triangles(builder)

    def _apply_boundary_conditions(self, builder: ModelBuilder):
        builder.particle_mass = torch.tensor(builder.particle_mass, device=self._device)
        mask = self._is_inside_frozen_bounding_box(self._soft_mesh)
        builder.particle_mass[mask] = 0.0
        builder.particle_mass = builder.particle_mass.tolist()

    @staticmethod
    def _is_inside_frozen_bounding_box(soft_mesh: Mesh):
        frozen_bounding_box = soft_mesh.properties.frozen_bounding_box
        points = soft_mesh.nodes.position
        bounding_box_min, bounding_box_max = torch.tensor(frozen_bounding_box, device=points.device).reshape(2, 3)
        return torch.all((bounding_box_min <= points) & (points <= bounding_box_max), dim=1)

    def _set_soft_mesh_triangles(self, builder: ModelBuilder):
        self._soft_mesh.elements.triangles = torch.tensor(builder.tri_indices, device=self._device, dtype=torch.int32)

    def _add_shape_mesh(self, builder: ModelBuilder):
        builder.add_shape_mesh(
            body=-1,
            mesh=wp.sim.Mesh(
                self._shape_mesh.nodes.position.detach().cpu().numpy(),
                self._shape_mesh.elements.triangles.cpu().numpy().reshape(-1),
            ),
            density=self._shape_mesh.properties.density,
        )

    @staticmethod
    def _update_model_attributes(model: Model):
        model.tri_ke, model.tri_ka, model.tri_kd, model.tri_kb = 0.0, 0.0, 0.0, 0.0
        model.ground = False
        model.gravity = (0.0, -9.8, 0.0)
        model.soft_contact_distance = 0.0
        model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_kf = 0.1, 0.1, 0.1
        return model
