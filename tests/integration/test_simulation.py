import torch
import unittest
import sys

sys.path.append("src")
from pathlib import Path
from simulation.simulation import Simulation
from scene.scene_factory import SceneFactory
from point.transform import Transform, get_quaternion
from mesh.mesh_properties import MeshProperties
from simulation.simulation_properties import SimulationProperties
from simulation.update_scene import update_scene


class TestSimulation(unittest.TestCase):
    """Only checks memory and gradients but final mesh needs to be viewed manually
    to make sure the outcome is accurate."""

    @classmethod
    def setUpClass(cls):
        device = "cuda"
        # gripper
        scad_file = Path("tests/data/caterpillar.scad")
        scad_parameters = Path("tests/data/caterpillar_scad_params.json")
        ideal_edge_length = 0.02
        gripper_properties = MeshProperties(
            name="caterpillar",
            density=1080.0,
            youngs_modulus=149_000,
            poissons_ratio=0.40,
            damping_factor=0.4,
            frozen_bounding_box=[-float("inf"), -0.01, -float("inf"), float("inf"), float("inf"), float("inf")],
        )
        gripper_transform = Transform(
            rotation=get_quaternion(vector=[1, 0, 0], angle_in_degrees=90), scale=[0.001, 0.001, 0.001], device=device
        )
        cable_pull_ratio = [
            torch.tensor(0.5, device=device, requires_grad=True),
            torch.tensor(0.0, device=device, requires_grad=True),
            torch.tensor(0.0, device=device, requires_grad=True),
        ]
        cable_stiffness, cable_damping = 100, 0.01
        # object
        object_file = Path("tests/data/cylinder.obj")
        object_properties = MeshProperties(name="cylinder", density=1080.0)
        object_transform = Transform(translation=[60, -60, -20], scale=[0.0015, 0.0015, 0.01], device=device)

        cls.scene = SceneFactory(
            scad_file=scad_file,
            scad_parameters=scad_parameters,
            ideal_edge_lenght=ideal_edge_length,
            gripper_properties=gripper_properties,
            gripper_transform=gripper_transform,
            cable_pull_ratio=cable_pull_ratio,
            cable_stiffness=cable_stiffness,
            cable_damping=cable_damping,
            object_file=object_file,
            object_properties=object_properties,
            object_transform=object_transform,
            device=device,
        ).create()

        sim_properties = SimulationProperties(
            duration=0.02, segment_duration=0.01, dt=2.1701388888888886e-05, device=device
        )
        cls.simulation = Simulation(scene=cls.scene, properties=sim_properties)
        update_scene(scene=cls.scene, simulation=cls.simulation, viewer=None)

    def tests_if_simulation_releases_memory_after_each_segment(self):
        memory = self.simulation.free_memory
        self.assertTrue(max(memory) - min(memory) <= 0.1)

    def tests_if_a_gradients_of_pull_ratio_are_not_none_given_a_loss(self):
        loss = self.scene.gripper.nodes.position.sum().requires_grad_()
        loss.backward()
        for cable in self.scene.gripper.cables:
            self.assertIsNotNone(cable.pull_ratio.grad)


if __name__ == "__main__":
    unittest.main()
