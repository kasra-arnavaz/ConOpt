import sys
from typing import List
import tqdm
import torch
from torch.utils.checkpoint import checkpoint

sys.path.append("src")
from mesh.mesh import Mesh
from cable.cable import Cable
from cable.holes import Holes
from simulation.update_holes import HolesForce, HolesPositionAndVelocity
from simulation.update_nodes import NodesForce, NodesPositionAndVelocity
from simulation.simulation_properties import SimulationProperties
from cable.barycentric_factory import BarycentricFactory
from warp_wrapper.model_factory import ModelFactory


class Simulation(torch.nn.Module):
    def __init__(
        self,
        gripper_mesh: Mesh,
        cables: List[Cable],
        holes: List[Holes],
        properties: SimulationProperties,
        object_mesh: Mesh = None,
    ):
        super(Simulation, self).__init__()
        self._mesh = gripper_mesh
        self._cables = cables
        self._properties = properties
        self._num_segments = int(properties.duration / properties.segment_duration)
        self._segment_steps = int(properties.segment_duration / properties.dt)
        self._barycentrics = [BarycentricFactory(gripper_mesh, hole, properties.device).create() for hole in holes]
        self._object_mesh = object_mesh
        self._model = ModelFactory(soft_mesh=gripper_mesh, shape_mesh=object_mesh, device=properties.device).create()
        self.free_memory = []
        self._holes_position_and_velocity = HolesPositionAndVelocity(barycentrics=self._barycentrics)
        self._holes_force = HolesForce(cables=self._cables, device=properties.device)
        self._nodes_force = NodesForce(barycentrics=self._barycentrics)
        self._nodes_position_and_velocity = NodesPositionAndVelocity(model=self._model, dt=properties.dt)

    def forward(self, np, nv):
        def segment(np, nv, num_steps):
            for _ in tqdm.tqdm(range(num_steps), desc="Segment", leave=False):
                np, nv = self.step(np, nv)
            return np, nv

        self._append_free_memory()
        for _ in tqdm.tqdm(range(self._num_segments), "Simulation", colour="green"):
            np, nv = checkpoint(segment, np, nv, self._segment_steps)
            self._append_free_memory()

        return np, nv

    def step(self, np, nv):
        hp, hv = self._holes_position_and_velocity(np, nv)
        hf = self._holes_force(hp, hv)
        nf = self._nodes_force(hf)
        return self._nodes_position_and_velocity(nf, np, nv)

    def _append_free_memory(self):
        self.free_memory.append(torch.cuda.mem_get_info()[0] / (1024 * 1024 * 1024))
