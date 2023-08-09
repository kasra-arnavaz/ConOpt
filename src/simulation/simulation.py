import sys
from typing import List
import tqdm
import torch
from torch.utils.checkpoint import checkpoint

sys.path.append("src")
from mesh.mesh import Mesh
from cable.cable import Cable
from cable.holes_position_and_velocity import HolesPositionAndVelocity
from cable.holes_force import HolesForce
from mesh.nodes_force import NodesForce
from mesh.nodes_position_and_velocity import NodesPositionAndVelocity
from cable.barycentric_factory import BarycentricFactory
from warp_wrapper.model_factory import ModelFactory


class Simulation:
    def __init__(self, gripper_mesh: Mesh, cables: List[Cable], duration: float, dt: float, object_mesh: Mesh = None, device: str = "cuda"):
        self._mesh = gripper_mesh
        self._cables = cables
        self._num_steps = int(duration / dt)
        self._dt = dt
        self._device = device
        self._holes = [cable.holes for cable in cables]
        self._barycentric = [BarycentricFactory(self._mesh, holes, self._device).create() for holes in self._holes]
        self._model = ModelFactory(soft_mesh=gripper_mesh, shape_mesh=object_mesh, device="cuda").create()

    def run(self):
        for _ in tqdm.tqdm(range(self._num_steps), desc="Simulation", colour="green", leave=False):
            self._step()

    def _step(self):
        self._zero_forces()
        self._update_holes()
        self._update_nodes_force()
        checkpoint(self._update_nodes_position_and_velocity)

    def _update_holes(self):
        for c, h, b in zip(self._cables, self._holes, self._barycentric):
            HolesPositionAndVelocity(holes=h, nodes=self._mesh.nodes, barycentric=b).update()
            HolesForce(cable=c, device=self._device).update()

    def _update_nodes_force(self):
        for h, b in zip(self._holes, self._barycentric):
            NodesForce(nodes=self._mesh.nodes, holes=h, barycentric=b).update()

    def _update_nodes_position_and_velocity(self):
        NodesPositionAndVelocity(nodes=self._mesh.nodes, model=self._model, dt=self._dt).update()

    def _zero_forces(self):
        self._mesh.nodes.force = torch.zeros_like(self._mesh.nodes.force)
        for hole in self._holes:
            hole.force = torch.zeros_like(hole.force)
