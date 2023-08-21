import unittest
from pathlib import Path
import sys
import torch

sys.path.append("src")

from mesh.scad import Scad
from mesh.mesh_factory import MeshFactoryFromScad
from simulation.update_holes import HolesForce
from simulation.update_nodes import NodesPositionAndVelocity, NodesForce
from mesh.mesh_properties import MeshProperties
from cable.cable import Cable
from cable.holes_initial_position import HolesInitialPosition
from cable.holes_factory import HolesFactoryFromListOfPositions
from cable.barycentric_factory import BarycentricFactory
from warp_wrapper.model_factory import ModelFactory


class TestNodesPositionAndVelocity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        holes_position = HolesInitialPosition(scad).get()
        cls.holes = HolesFactoryFromListOfPositions(holes_position, device="cuda").create()[0]
        pull_ratio = torch.tensor(0.5, requires_grad=True, device="cuda")
        cable = Cable(stiffness=100, damping=0.01, pull_ratio=pull_ratio, holes=cls.holes)
        HolesForce(cable=cable, device="cuda").update()
        cls.mesh = MeshFactoryFromScad(scad).create()
        cls.mesh.properties = MeshProperties(
            name="caterpillar",
            youngs_modulus=149_000,
            poissons_ratio=0.45,
            damping_factor=0.4,
            density=1080.0,
            frozen_bounding_box=[-float("inf"), -float("inf"), 0, float("inf"), float("inf"), 2],
        )
        cls.barycentric = BarycentricFactory(mesh=cls.mesh, holes=cls.holes, device="cuda").create()
        NodesForce(nodes=cls.mesh.nodes, holes=cls.holes, barycentric=cls.barycentric).update()
        cls.model = ModelFactory(soft_mesh=cls.mesh, device="cuda").create()

    def tests_if_nodes_position_and_velocity_is_changed(self):
        old_nodes_position, old_nodes_velocity = self.mesh.nodes.position, self.mesh.nodes.velocity
        NodesPositionAndVelocity(nodes=self.mesh.nodes, model=self.model, dt=2.1701388888888886e-05).update()
        new_nodes_position, new_nodes_velocity = self.mesh.nodes.position, self.mesh.nodes.velocity
        self.assertFalse(torch.equal(old_nodes_position, new_nodes_position))
        self.assertFalse(torch.equal(old_nodes_velocity, new_nodes_velocity))


class TestNodesForce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        holes_position = HolesInitialPosition(scad).get()
        cls.holes = HolesFactoryFromListOfPositions(holes_position, device="cuda").create()[0]
        pull_ratio = torch.tensor(0.5, requires_grad=True, device="cuda")
        cable = Cable(stiffness=100, damping=0.01, pull_ratio=pull_ratio, holes=cls.holes)
        HolesForce(cable=cable, device="cuda").update()
        cls.mesh = MeshFactoryFromScad(scad).create()
        cls.barycentric = BarycentricFactory(mesh=cls.mesh, holes=cls.holes, device="cuda").create()

    def tests_if_nodes_force_is_changed(self):
        old_nodes_force = self.mesh.nodes.force
        NodesForce(nodes=self.mesh.nodes, holes=self.holes, barycentric=self.barycentric).update()
        new_nodes_force = self.mesh.nodes.force
        self.assertFalse(torch.equal(old_nodes_force, new_nodes_force))


if __name__ == "__main__":
    unittest.main()
