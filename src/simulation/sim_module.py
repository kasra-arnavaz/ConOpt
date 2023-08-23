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
import warp as wp


class Simulation(torch.nn.Module):
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
        super(Simulation, self).__init__()
        self._mesh = gripper_mesh
        self._cables = cables
        self._num_segments = int(duration / segment_duration)
        self._segment_steps = int(segment_duration / dt)
        self._dt = dt
        self._duration = duration
        self._device = device
        self._barycentrics = [BarycentricFactory(gripper_mesh, cable.holes, device).create() for cable in self._cables]
        self._object_mesh = object_mesh
        self._model = ModelFactory(soft_mesh=gripper_mesh, shape_mesh=object_mesh, device=device).create()
        self.free_memory = []
        self._holes_position_and_velocity = HolesPositionAndVelocity(barycentric=self._barycentrics[0])
        self._holes_force = HolesForce(cable=self._cables[0], device=self._device)
        self._nodes_force = NodesForce(barycentric=self._barycentrics[0])
        self._nodes_position_and_velocity = NodesPositionAndVelocity(model=self._model, dt=self._dt)

    def step(self, np, nv):
        hp, hv = self._holes_position_and_velocity(np, nv)
        hf = self._holes_force(hp, hv)
        nf = self._nodes_force(hf)
        return self._nodes_position_and_velocity(nf, np, nv)
    
    def forward(self, np, nv):

        def segment(np, nv, num_steps):
            for _ in range(num_steps):
                np, nv = self.step(np, nv)
            return np, nv
        
        self._append_free_memory()

        for _ in range(self._num_segments):
            np, nv = checkpoint(segment, np, nv, self._segment_steps)

        self._append_free_memory()

        return np, nv

    def _append_free_memory(self):
        self.free_memory.append(torch.cuda.mem_get_info()[0] / (1024 * 1024 * 1024))