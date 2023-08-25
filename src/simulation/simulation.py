import sys
from typing import List
import tqdm
import torch
from torch.utils.checkpoint import checkpoint

sys.path.append("src")
from simulation.scene import Scene
from simulation.update_holes import HolesForce, HolesPositionAndVelocity
from simulation.update_nodes import NodesForce, NodesPositionAndVelocity
from simulation.simulation_properties import SimulationProperties
from cable.barycentric_factory import BarycentricListFactory


class Simulation(torch.nn.Module):
    def __init__(self, scene: Scene, properties: SimulationProperties, cables):
        super().__init__()
        self.scene = scene
        self.free_memory = []
        self._properties = properties
        self.cables = cables
        holes = [cable.holes for cable in cables]
        self._barycentrics = BarycentricListFactory(scene.gripper, holes, properties.device).create()
        # functions
        self._holes_position_and_velocity = HolesPositionAndVelocity(barycentrics=self._barycentrics)
        self._holes_force = HolesForce(cables=cables, device=properties.device)
        self._nodes_force = NodesForce(barycentrics=self._barycentrics)
        self._nodes_position_and_velocity = NodesPositionAndVelocity(model=scene.model, dt=properties.dt)

    def forward(self, np, nv):
        def segment(np, nv, num_steps):
            for _ in range(num_steps):
                np, nv = self.step(np, nv)
            return np, nv

        self._append_free_memory()
        for _ in tqdm.tqdm(range(self._properties.num_segments), "Simulation", colour="green", leave=False):
            np.requires_grad_()
            nv.requires_grad_()
            np, nv = checkpoint(segment, np, nv, self._properties.num_steps_per_segment)
            self._append_free_memory()

        return np, nv

    def step(self, np, nv):
        hp, hv = self._holes_position_and_velocity(np, nv)
        hf = self._holes_force(hp, hv)
        nf = self._nodes_force(hf)
        return self._nodes_position_and_velocity(nf, np, nv)

    def _append_free_memory(self):
        self.free_memory.append(torch.cuda.mem_get_info()[0] / (1024 * 1024 * 1024))
