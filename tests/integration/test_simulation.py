import torch
import unittest
import sys

sys.path.append("src")
from pathlib import Path
from simulation.simulation import Simulation
from mesh.mesh_factory import MeshFactoryFromScad, MeshFactoryFromObj
from mesh.scad import Scad
from cable.cable_factory import CableFactory
from cable.holes_factory import HolesFactoryFromListOfPositions
from cable.holes_initial_position import HolesInitialPosition
from point.transform import Transform, get_quaternion
from mesh.mesh_properties import MeshProperties
from simulation.simulation_properties import SimulationProperties


class TestSimulationWithCheckpoint(unittest.TestCase):
    """Only runs simulation but final mesh needs to be viewed manually
    to make sure the outcome is accurate."""

    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        cls.gripper_mesh = MeshFactoryFromScad(scad, ideal_edge_length=0.02).create()
        object_mesh = MeshFactoryFromObj(Path("tests/data/cylinder.obj"), device="cuda").create()
        cls.gripper_mesh.properties = MeshProperties(
            name="caterpillar",
            density=1080.0,
            youngs_modulus=149_000,
            poissons_ratio=0.40,
            damping_factor=0.4,
            frozen_bounding_box=[-float("inf"), -0.01, -float("inf"), float("inf"), float("inf"), float("inf")],
        )
        object_mesh.properties = MeshProperties(name="cylinder", density=1080.0)
        transform = Transform(
            rotation=get_quaternion(vector=[1, 0, 0], angle_in_degrees=90), scale=[0.001, 0.001, 0.001]
        )
        holes_position = HolesInitialPosition(scad).get()
        holes = HolesFactoryFromListOfPositions(holes_position).create()
        transform.apply(object_mesh.nodes)
        transform.apply(cls.gripper_mesh.nodes)
        for hole in holes:
            transform.apply(hole)
        cls.pull_ratio = [
            torch.tensor(0.2, device="cuda", requires_grad=True),
            torch.tensor(0.0, device="cuda", requires_grad=True),
            torch.tensor(0.0, device="cuda", requires_grad=True),
        ]

        cables = CableFactory(stiffness=100, damping=0.01, pull_ratio=cls.pull_ratio).create()
        sim_properties = SimulationProperties(
            duration=0.02, segment_duration=0.01, dt=2.1701388888888886e-05, device="cuda"
        )
        cls.simulation = Simulation(
            gripper_mesh=cls.gripper_mesh,
            object_mesh=object_mesh,
            cables=cables,
            holes=holes,
            properties=sim_properties,
        )
        cls.gripper_mesh.nodes.position, cls.gripper_mesh.nodes.velocity = cls.simulation(
            cls.gripper_mesh.nodes.position, cls.gripper_mesh.nodes.velocity
        )

    def tests_if_simulation_releases_memory_after_each_segment(self):
        memory = self.simulation.free_memory
        self.assertTrue(max(memory) - min(memory) <= 0.1)

    def tests_if_a_gradients_of_pull_ratio_are_not_none_given_a_loss(self):
        loss = self.gripper_mesh.nodes.position.sum()
        loss.backward()
        for pull in self.pull_ratio:
            self.assertIsNotNone(pull.grad)


if __name__ == "__main__":
    unittest.main()
