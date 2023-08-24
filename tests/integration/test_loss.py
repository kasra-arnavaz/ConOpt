import torch
import unittest
import sys

sys.path.append("src")
from pathlib import Path
from mesh.mesh_factory import MeshFactoryFromScad, MeshFactoryFromObj
from mesh.scad import Scad
from mesh.mesh_properties import MeshProperties
from simulation.simulation import Simulation
from cable.cable_factory import CableListFactory
from cable.holes_factory import HolesListFactory
from cable.holes_initial_position import HolesInitialPosition
from point.transform import Transform, get_quaternion
from rendering.views import ThreeInteriorViews
from rendering.rendering import ExteriorDepthRendering, InteriorGapRendering, InteriorContactRendering
from objective.loss import MaxGripLoss
from simulation.simulation_properties import SimulationProperties
from simulation.scene import Scene


class TestMaxGripLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        gripper_mesh = MeshFactoryFromScad(scad, ideal_edge_length=0.02).create()
        object_mesh = MeshFactoryFromObj(Path("tests/data/cylinder.obj"), device="cuda").create()
        gripper_mesh.properties = MeshProperties(
            name="caterpillar",
            density=1080.0,
            youngs_modulus=200_000,
            poissons_ratio=0.49,
            damping_factor=0.4,
            frozen_bounding_box=[-float("inf"), -0.01, -float("inf"), float("inf"), float("inf"), float("inf")],
        )
        object_mesh.properties = MeshProperties(name="cylinder", density=1080.0)
        transform = Transform(
            rotation=get_quaternion(vector=[1, 0, 0], angle_in_degrees=90), scale=[0.001, 0.001, 0.001]
        )
        holes_position = HolesInitialPosition(scad).get()
        holes = HolesListFactory(holes_position).create()
        transform.apply(gripper_mesh.nodes)
        for hole in holes:
            transform.apply(hole)
        pull_ratio = [
            torch.tensor(0.5, device="cuda"),
            torch.tensor(0.0, device="cuda"),
            torch.tensor(0.0, device="cuda"),
        ]
        transform.apply(object_mesh.nodes)
        cables = CableListFactory(stiffness=100, damping=0.01, pull_ratio=pull_ratio, holes=holes).create()
        gripper_mesh.cables = cables
        sim_properties = SimulationProperties(
            duration=0.02, segment_duration=0.01, dt=2.1701388888888886e-05, device="cuda"
        )
        scene = Scene(gripper=gripper_mesh, object=object_mesh, device="cuda")
        simulation = Simulation(scene=scene, properties=sim_properties)
        gripper_mesh.nodes.position, gripper_mesh.nodes.velocity = simulation(
            gripper_mesh.nodes.position, gripper_mesh.nodes.position
        )
        views = ThreeInteriorViews(center=object_mesh.nodes.position.mean(dim=0), device="cuda")
        rendering = InteriorGapRendering(gripper_mesh=gripper_mesh, object_mesh=object_mesh, views=views)
        cls.loss = MaxGripLoss(rendering=rendering, device="cuda").get_loss()

    def tests_if_max_grip_loss_is_of_type_torch_tensor(self):
        self.assertIsInstance(self.loss, torch.Tensor)

    def tests_if_max_grip_loss_has_correct_shape(self):
        self.assertEqual(list(self.loss.shape), [1])


if __name__ == "__main__":
    unittest.main()
