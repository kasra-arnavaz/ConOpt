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
from point.point_iterable import PointIterable

import gc

def count_tensors():
    count = 0
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            count += 1
    return count

class Simulation:
    def __init__(
        self,
        gripper_mesh: Mesh,
        cables: List[Cable],
        duration: float,
        dt: float,
        object_mesh: Mesh = None,
        device: str = "cuda",
        segment_duration: float = 0.5
    ):
        self._mesh = gripper_mesh
        self._cables = cables
        self._num_segments = int(duration / segment_duration)
        self._segment_steps = int(segment_duration / dt)
        self._dt = dt
        self._device = device
        self._barycentrics = [BarycentricFactory(gripper_mesh, cable.holes, device).create() for cable in self._cables]
        self._object_mesh = object_mesh
        self._model = ModelFactory(soft_mesh=gripper_mesh, shape_mesh=object_mesh, device=device).create()
        # self._state_iterable = StateIterable(self._model, num=self._segment_steps)
        # self._nodes_iterable = PointIterable(gripper_mesh.nodes, num=self._segment_steps)
        self.free_memory = []

    def run(self, use_checkpoint: bool = True):
        for _ in tqdm.tqdm(range(self._num_segments), desc="Simulation", colour="green", leave=False):
            self._append_free_memory()
            if use_checkpoint:
                checkpoint(self._update_segment, use_reentrant=False)
            else:
                self._update_segment()
            self._append_free_memory()

    def _append_free_memory(self):
        self.free_memory.append(torch.cuda.mem_get_info()[0] / (1024 * 1024 * 1024))

    def _update_segment(self):
        # self._segment_steps = 2
        for _ in range(self._segment_steps):
            self._update_holes_position_and_velocity()
            self._update_holes_force()
            self._update_nodes_force()
            self._update_nodes_position_and_velocity()
            self._zero_forces()

    def _update_holes_position_and_velocity(self):
        for c, b in zip(self._cables, self._barycentrics):
            HolesPositionAndVelocity(holes=c.holes, nodes=self._mesh.nodes, barycentric=b).update()

    def _update_holes_force(self):
        for c in self._cables:
            HolesForce(cable=c, device=self._device).update()

    def _update_nodes_force(self):
        for c, b in zip(self._cables, self._barycentrics):
            NodesForce(nodes=self._mesh.nodes, holes=c.holes, barycentric=b).update()

    def _update_nodes_position_and_velocity(self):
        NodesPositionAndVelocity(nodes=self._mesh.nodes, model=self._model, dt=self._dt).update()

    def _zero_forces(self):
        self._mesh.nodes.force = torch.zeros_like(self._mesh.nodes.force)
        for cable in self._cables:
            cable.holes.force = torch.zeros_like(cable.holes.force)
