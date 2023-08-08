import torch
import unittest
import sys

sys.path.append("src")
from pathlib import Path
from simulation.simulation import Simulation
from mesh.mesh_factory import MeshFactoryFromScad
from mesh.scad import Scad
from cable.cable_factory import CableFactory
from cable.holes_factory import HolesFactoryFromListOfPositions
from cable.holes_initial_position import HolesInitialPosition
from point.transform import Transform, get_quaternion
from mesh.mesh_properties import MeshProperties


class TestSimulation(unittest.TestCase):
    """Only runs simulation but final mesh needs to be viewed manually
    to make sure the outcome is accurate."""

    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        mesh = MeshFactoryFromScad(scad, ideal_edge_length=0.02).create()
        mesh.properties = MeshProperties(
            name="caterpillar",
            density=1080.0,
            youngs_modulus=200_000,
            poissons_ratio=0.49,
            damping_factor=0.4,
            frozen_bounding_box=[-float("inf"), -0.01, -float("inf"), float("inf"), float("inf"), float("inf")],
        )
        transform = Transform(
            rotation=get_quaternion(vector=[1, 0, 0], angle_in_degrees=90), scale=[0.001, 0.001, 0.001]
        )
        holes_position = HolesInitialPosition(scad).get()
        holes = HolesFactoryFromListOfPositions(holes_position).create()
        transform.apply(mesh.nodes)
        for hole in holes:
            transform.apply(hole)
        pull_ratio = [
            torch.tensor(0.8, device="cuda"),
            torch.tensor(0.0, device="cuda"),
            torch.tensor(0.0, device="cuda"),
        ]
        cables = CableFactory(stiffness=100, damping=0.01, pull_ratio=pull_ratio, holes=holes).create()
        cls.simulation = Simulation(mesh=mesh, cables=cables, duration=1.5, dt=2.1701388888888886e-05, device="cuda")

    def tests_if_simulation_runs(self):
        try:
            self.simulation.run()
        except:
            self.fail()


if __name__ == "__main__":
    unittest.main()