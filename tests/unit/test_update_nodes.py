import unittest
from pathlib import Path
import sys
import torch

sys.path.append("src")

from mesh.scad import Scad
from mesh.mesh_factory import MeshFactoryFromMsh
from simulation.update_holes import HolesForce
from simulation.update_nodes import NodesPositionAndVelocity, NodesForce
from mesh.mesh_properties import MeshProperties
from cable.holes_initial_position import CaterpillarHolesInitialPosition
from cable.holes_factory import HolesListFactory
from cable.cable_factory import CableListFactory
from cable.barycentric_factory import BarycentricListFactory
from warp_wrapper.model_factory import ModelFactory
from warp_wrapper.contact_properties import ContactProperties
from cable.pull_ratio import TimeInvariablePullRatio
from simulation.simulation_properties import SimulationProperties


class TestNodesPositionAndVelocity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        msh_file = Path("tests/data/caterpillar.msh")
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)

        holes_position = CaterpillarHolesInitialPosition(scad).get()
        holes = HolesListFactory(holes_position, device="cuda").create()
        holes_positions = [holes.position for holes in holes]
        holes_velocities = [holes.velocity for holes in holes]
        sim_properties = SimulationProperties(dt=0.1, duration=1.0, segment_duration=0.1)
        pull_ratio = [
            TimeInvariablePullRatio(
                pull_ratio=torch.tensor(0.5, device="cuda"), simulation_properties=sim_properties, device="cuda"
            ),
            TimeInvariablePullRatio(
                pull_ratio=torch.tensor(0.0, device="cuda"), simulation_properties=sim_properties, device="cuda"
            ),
            TimeInvariablePullRatio(
                pull_ratio=torch.tensor(0.0, device="cuda"), simulation_properties=sim_properties, device="cuda"
            ),
        ]
        cables = CableListFactory(stiffness=100, damping=0.01, pull_ratio=pull_ratio, holes=holes).create()

        fn = HolesForce(cables=cables, device="cuda")
        holes_forces = fn(holes_positions, holes_velocities)
        cls.mesh = MeshFactoryFromMsh(msh_file).create()
        cls.mesh.properties = MeshProperties(
            name="caterpillar",
            youngs_modulus=149_000,
            poissons_ratio=0.45,
            damping_factor=0.4,
            density=1080.0,
            frozen_bounding_box=[-float("inf"), -float("inf"), 0, float("inf"), float("inf"), 2],
        )
        cls.mesh.cables = cables
        barycentrics = BarycentricListFactory(mesh=cls.mesh, holes=holes, device="cuda").create()
        fn = NodesForce(barycentrics=barycentrics)
        cls.mesh.nodes.force = fn(holes_forces)
        contact_properties = ContactProperties(distance=0.01, ke=2.0, kd=0.1, kf=0.1, ground=False)
        cls.model = ModelFactory(soft_mesh=cls.mesh, contact_properties=contact_properties, device="cuda").create()

    def tests_if_nodes_position_and_velocity_is_changed(self):
        old_nodes_position, old_nodes_velocity = self.mesh.nodes.position, self.mesh.nodes.velocity
        fn = NodesPositionAndVelocity(model=self.model, dt=2.1701388888888886e-05)
        nodes = self.mesh.nodes
        new_nodes_position, new_nodes_velocity = fn(nodes.force, nodes.position, nodes.velocity)
        self.assertFalse(torch.equal(old_nodes_position, new_nodes_position))
        self.assertFalse(torch.equal(old_nodes_velocity, new_nodes_velocity))


class TestNodesForce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        msh_file = Path("tests/data/caterpillar.msh")
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        holes_position = CaterpillarHolesInitialPosition(scad).get()
        cls.holes = HolesListFactory(holes_position, device="cuda").create()
        holes_positions = [holes.position for holes in cls.holes]
        holes_velocities = [holes.velocity for holes in cls.holes]
        sim_properties = SimulationProperties(dt=0.1, duration=1.0, segment_duration=0.1)
        pull_ratio = [
            TimeInvariablePullRatio(
                pull_ratio=torch.tensor(0.5, device="cuda"), simulation_properties=sim_properties, device="cuda"
            ),
            TimeInvariablePullRatio(
                pull_ratio=torch.tensor(0.0, device="cuda"), simulation_properties=sim_properties, device="cuda"
            ),
            TimeInvariablePullRatio(
                pull_ratio=torch.tensor(0.0, device="cuda"), simulation_properties=sim_properties, device="cuda"
            ),
        ]
        cables = CableListFactory(stiffness=100, damping=0.01, pull_ratio=pull_ratio, holes=cls.holes).create()
        fn = HolesForce(cables=cables, device="cuda")
        cls.holes_forces = fn(holes_positions, holes_velocities)
        cls.mesh = MeshFactoryFromMsh(msh_file).create()
        cls.barycentrics = BarycentricListFactory(mesh=cls.mesh, holes=cls.holes, device="cuda").create()

    def tests_if_nodes_force_is_changed(self):
        old_nodes_force = self.mesh.nodes.force
        fn = NodesForce(barycentrics=self.barycentrics)
        new_nodes_force = fn(self.holes_forces)
        self.assertFalse(torch.equal(old_nodes_force, new_nodes_force))


if __name__ == "__main__":
    unittest.main()
