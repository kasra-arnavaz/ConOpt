import sys
import tqdm
import torch
from torch.utils.checkpoint import checkpoint

sys.path.append("src")
from scene.scene import Scene
from simulation.update_holes import HolesForce, HolesPositionAndVelocity
from simulation.update_nodes import NodesForce, NodesPositionAndVelocity
from simulation.simulation_properties import SimulationProperties
from cable.barycentric_factory import BarycentricListFactory


class Simulation(torch.nn.Module):
    def __init__(self, scene: Scene, properties: SimulationProperties, use_checkpoint: bool = True):
        super().__init__()
        self.free_memory = []
        self.properties = properties
        holes = [cable.holes for cable in scene.robot.cables]
        self._barycentrics = BarycentricListFactory(scene.robot, holes, properties.device).create()
        self._use_checkpoint = use_checkpoint
        # functions
        self._holes_position_and_velocity = HolesPositionAndVelocity(barycentrics=self._barycentrics)
        self._holes_force = HolesForce(cables=scene.robot.cables, device=properties.device)
        self._nodes_force = NodesForce(barycentrics=self._barycentrics)
        self._nodes_position_and_velocity = NodesPositionAndVelocity(model=scene.model, dt=properties.dt)

    def forward(self, nodes_position, nodes_velocity, p0, p1, p2):
        def segment(nodes_position, nodes_velocity, p0, p1, p2, num_steps):
            for _ in range(num_steps):
                nodes_position, nodes_velocity, p0, p1, p2 = self.step(nodes_position, nodes_velocity, p0, p1, p2)
            return nodes_position, nodes_velocity, p0, p1, p2

        self._append_free_memory()
        nodes_position.requires_grad_()
        nodes_velocity.requires_grad_()
        if self._use_checkpoint:
            nodes_position, nodes_velocity, p0, p1, p2 = checkpoint(
                segment, nodes_position, nodes_velocity, p0, p1, p2, self.properties.num_steps_per_segment
            )
        else:
            nodes_position, nodes_velocity = segment(
                nodes_position, nodes_velocity, self.properties.num_steps_per_segment
            )
        self._append_free_memory()

        return nodes_position, nodes_velocity, p0, p1, p2

    def step(self, nodes_position, nodes_velocity, p0, p1, p2):
        holes_position, holes_velocity = self._holes_position_and_velocity(nodes_position, nodes_velocity)
        holes_force = self._holes_force(holes_position, holes_velocity, p0, p1, p2)
        nodes_force = self._nodes_force(holes_force)
        p0 = next(self._holes_force._cables[0].pull_ratio.iterator)
        p1 = next(self._holes_force._cables[1].pull_ratio.iterator)
        p2 = next(self._holes_force._cables[2].pull_ratio.iterator)
        np, nv = self._nodes_position_and_velocity(nodes_force, nodes_position, nodes_velocity)
        return np, nv, p0, p1, p2

    def _append_free_memory(self):
        self.free_memory.append(torch.cuda.mem_get_info()[0] / (1024 * 1024 * 1024))
