import unittest
import sys
from pathlib import Path
import torch

sys.path.append("src")

from cable.barycentric_factory import BarycentricListFactory
from mesh.mesh_factory import MeshFactoryFromMsh
from simulation.update_holes import HolesPositionAndVelocity, HolesForce
from mesh.scad import Scad
from cable.holes_initial_position import CaterpillarHolesInitialPosition
from cable.holes_factory import HolesListFactory
from cable.cable_factory import CableListFactory
from cable.pull_ratio import TimeInvariablePullRatio
from simulation.simulation_properties import SimulationProperties

class TestHolesForce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        holes_position = CaterpillarHolesInitialPosition(scad).get()
        holes = HolesListFactory(holes_position, device="cuda").create()
        sim_properties = SimulationProperties(dt=0.1, duration=1.0, segment_duration=0.1)
        pull_ratio = [
            TimeInvariablePullRatio(pull_ratio=torch.tensor(0.5, device="cuda"), simulation_properties=sim_properties, device="cuda"),
            TimeInvariablePullRatio(pull_ratio=torch.tensor(0.5, device="cuda"), simulation_properties=sim_properties, device="cuda"),
            TimeInvariablePullRatio(pull_ratio=torch.tensor(0.5, device="cuda"), simulation_properties=sim_properties, device="cuda"),
        ]
        cls.cables = CableListFactory(stiffness=100, damping=0.01, pull_ratio=pull_ratio, holes=holes).create()

    def test_if_holes_force_is_changed(self):
        old_holes_force = [cable.holes.force for cable in self.cables]
        holes_positions = [cable.holes.position for cable in self.cables]
        holes_velocities = [cable.holes.velocity for cable in self.cables]
        fn = HolesForce(cables=self.cables, device="cuda")
        new_holes_force = fn(holes_positions, holes_velocities)
        for old_force, new_force in zip(old_holes_force, new_holes_force):
            self.assertFalse(torch.equal(old_force, new_force))


class TestHolesPositionAndVelocity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        msh_file = Path("tests/data/caterpillar.msh")
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        holes_positions = CaterpillarHolesInitialPosition(scad).get()
        cls.mesh = MeshFactoryFromMsh(msh_file).create()
        cls.holes = HolesListFactory(holes_positions).create()

    def tests_if_the_first_update_of_holes_position_and_velocity_does_nothing(self):
        old_holes_positions = [holes.position for holes in self.holes]
        old_holes_velocities = [holes.velocity for holes in self.holes]
        barycentrics = BarycentricListFactory(mesh=self.mesh, holes=self.holes).create()
        fn = HolesPositionAndVelocity(barycentrics=barycentrics)
        new_holes_positions, new_holes_velocities = fn(self.mesh.nodes.position, self.mesh.nodes.velocity)
        for old_positions, new_positions in zip(old_holes_positions, new_holes_positions):
            self.assertTrue(torch.allclose(old_positions, new_positions, atol=1e-5))
        for old_velocities, new_velocities in zip(old_holes_velocities, new_holes_velocities):
            self.assertTrue(torch.allclose(old_velocities, new_velocities, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
