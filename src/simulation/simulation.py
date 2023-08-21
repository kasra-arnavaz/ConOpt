import sys
from typing import List
import tqdm
import torch
from torch.utils.checkpoint import checkpoint

sys.path.append("src")
from mesh.mesh import Mesh
from cable.cable import Cable
from simulation.update_holes import HolesForce, HolesPositionAndVelocity
from simulation.update_nodes import NodesForce, NodesPositionAndVelocity
from cable.barycentric_factory import BarycentricFactory
from warp_wrapper.model_factory import ModelFactory
from warp_wrapper.state_iterable import StateIterable


class Simulation:
    def __init__(
        self,
        gripper_mesh: Mesh,
        cables: List[Cable],
        duration: float,
        dt: float,
        object_mesh: Mesh = None,
        device: str = "cuda",
    ):
        self._NUM_SUBSTEPS = 23040

        self._mesh = gripper_mesh
        self._cables = cables
        total_steps = int(duration / dt)
        self._num_segments = int(total_steps / self._NUM_SUBSTEPS)
        self._dt = dt
        self._device = device
        self._barycentrics = [BarycentricFactory(self._mesh, cable.holes, device).create() for cable in self._cables]
        self._object_mesh = object_mesh
        self._model = ModelFactory(soft_mesh=gripper_mesh, shape_mesh=object_mesh, device=device).create()
        # self._state_iterable = StateIterable(model=self._model)

    def run(self):
        for _ in tqdm.tqdm(range(self._num_segments), desc="Simulation", colour="green", leave=False):
            # with set_checkpoint_early_stop(False):
            checkpoint(self._update_segment, use_reentrant=False)

    def _update_segment(self):
        self._NUM_SUBSTEPS = 1
        for _ in range(self._NUM_SUBSTEPS):
            self._zero_forces()
            self._update_holes()
            self._update_nodes_force()
            self._update_nodes_position_and_velocity()

    def _update_holes(self):
        for c, b in zip(self._cables, self._barycentrics):
            HolesPositionAndVelocity(holes=c.holes, nodes=self._mesh.nodes, barycentric=b).update()
            HolesForce(cable=c, device=self._device).update()

    def _update_nodes_force(self):
        for c, b in zip(self._cables, self._barycentrics):
            NodesForce(nodes=self._mesh.nodes, holes=c.holes, barycentric=b).update()

    def _update_nodes_position_and_velocity(self):
        NodesPositionAndVelocity(nodes=self._mesh.nodes, model=self._model, dt=self._dt).update()

    def _zero_forces(self):
        self._mesh.nodes.force = torch.zeros_like(self._mesh.nodes.force, requires_grad=True)
        for cable in self._cables:
            cable.holes.force = torch.zeros_like(cable.holes.force, requires_grad=True)
