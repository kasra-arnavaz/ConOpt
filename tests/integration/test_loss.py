import torch
import unittest
import sys

sys.path.append("src")
from pathlib import Path
from mesh.mesh_properties import MeshProperties
from simulation.simulation import Simulation
from point.transform import Transform, get_quaternion
from rendering.views import ThreeInteriorViews
from rendering.rendering import InteriorGapRendering
from objective.loss import MaxGripLoss, PointTouchLoss
from simulation.simulation_properties import SimulationProperties
from scene.scene_factory import GripperSceneFactory, TouchSceneFactory
from simulation.update_scene import update_scene
from warp_wrapper.contact_properties import ContactProperties


class TestMaxGripLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda"
        # robot
        msh_file = Path("tests/data/caterpillar.msh")
        scad_file = Path("tests/data/caterpillar.scad")
        scad_parameters = Path("tests/data/caterpillar_scad_params.json")
        ideal_edge_length = 0.02
        robot_properties = MeshProperties(
            name="caterpillar",
            density=1080.0,
            youngs_modulus=149_000,
            poissons_ratio=0.45,
            damping_factor=0.4,
            frozen_bounding_box=[-float("inf"), -0.01, -float("inf"), float("inf"), -float("inf"), float("inf")],
        )
        robot_transform = Transform(
            rotation=get_quaternion(vector=[1, 0, 0], angle_in_degrees=90),
            scale=[0.001, 0.001, 0.001],
            device=cls.device,
        )
        cable_pull_ratio = [
            torch.tensor(0.5, device=cls.device),
            torch.tensor(0.0, device=cls.device),
            torch.tensor(0.0, device=cls.device),
        ]
        cable_stiffness, cable_damping = 100, 0.01
        # object
        object_file = Path("tests/data/cylinder.obj")
        object_properties = MeshProperties(name="cylinder", density=1080.0)
        object_transform = Transform(translation=[60, -60, -20], scale=[0.0015, 0.0015, 0.01], device=cls.device)

        contact_properties = ContactProperties(distance=0.001, ke=2.0, kd=0.1, kf=0.1)

        cls.scene = GripperSceneFactory(
            msh_file=msh_file,
            scad_file=scad_file,
            scad_parameters=scad_parameters,
            ideal_edge_length=ideal_edge_length,
            robot_properties=robot_properties,
            robot_transform=robot_transform,
            cable_pull_ratio=cable_pull_ratio,
            cable_stiffness=cable_stiffness,
            cable_damping=cable_damping,
            object_file=object_file,
            object_properties=object_properties,
            object_transform=object_transform,
            contact_properties=contact_properties,
            device=cls.device,
            make_new_robot=False
        ).create()
        sim_properties = SimulationProperties(
            duration=0.02, segment_duration=0.01, dt=2.1701388888888886e-05, device=cls.device
        )
        simulation = Simulation(scene=cls.scene, properties=sim_properties)
        update_scene(scene=cls.scene, simulation=simulation)
        views = ThreeInteriorViews(center=cls.scene.object.nodes.position.mean(dim=0), device=cls.device)
        cls.rendering = InteriorGapRendering(scene=cls.scene, views=views, device=cls.device)

    def tests_if_max_grip_loss_is_of_type_torch_tensor(self):
        loss = MaxGripLoss(rendering=self.rendering, device=self.device).get_loss()
        self.assertIsInstance(loss, torch.Tensor)


class TestPointTouchLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda"
        # robot
        msh_file = Path("tests/data/caterpillar.msh")
        scad_file = Path("tests/data/caterpillar.scad")
        scad_parameters = Path("tests/data/caterpillar_scad_params.json")
        ideal_edge_length = 0.02
        robot_properties = MeshProperties(
            name="caterpillar",
            density=1080.0,
            youngs_modulus=149_000,
            poissons_ratio=0.45,
            damping_factor=0.4,
            frozen_bounding_box=[-float("inf"), -0.01, -float("inf"), float("inf"), -float("inf"), float("inf")],
        )
        robot_transform = Transform(
            rotation=get_quaternion(vector=[1, 0, 0], angle_in_degrees=90),
            scale=[0.001, 0.001, 0.001],
            device=cls.device,
        )
        cable_pull_ratio = [
            torch.tensor(0.5, device=cls.device),
            torch.tensor(0.0, device=cls.device),
            torch.tensor(0.0, device=cls.device),
        ]
        cable_stiffness, cable_damping = 100, 0.01
        # object
        object_file = Path("tests/data/sphere.obj")
        object_properties = MeshProperties(name="sphere", density=1080.0)
        object_transform = Transform(translation=[-20, -20, -20], scale=[0.01, 0.01, 0.01], device=cls.device)

        contact_properties = ContactProperties(distance=0.001, ke=2.0, kd=0.1, kf=0.1)

        cls.scene = TouchSceneFactory(
            msh_file=msh_file,
            scad_file=scad_file,
            scad_parameters=scad_parameters,
            ideal_edge_length=ideal_edge_length,
            robot_properties=robot_properties,
            robot_transform=robot_transform,
            cable_pull_ratio=cable_pull_ratio,
            cable_stiffness=cable_stiffness,
            cable_damping=cable_damping,
            object_file=object_file,
            object_properties=object_properties,
            object_transform=object_transform,
            contact_properties=contact_properties,
            device=cls.device,
            make_new_robot=False
        ).create()
        sim_properties = SimulationProperties(
            duration=0.02, segment_duration=0.01, dt=2.1701388888888886e-05, device=cls.device
        )
        simulation = Simulation(scene=cls.scene, properties=sim_properties)
        update_scene(scene=cls.scene, simulation=simulation)

    def tests_if_point_torch_loss_is_of_type_torch_tensor(self):
        loss = PointTouchLoss(scene=self.scene).get_loss()
        self.assertIsInstance(loss, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
