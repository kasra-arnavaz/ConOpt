import torch
import unittest
import sys

sys.path.append("src")
from pathlib import Path
from mesh.mesh_properties import MeshProperties
from simulation.simulation import Simulation
from point.transform import Transform, get_quaternion
from rendering.views import ThreeInteriorViews, SixInteriorViews
from rendering.rendering import InteriorGapRendering, InteriorDistanceRendering
from objective.loss import MaxGripLoss, PointTouchLoss, ObstacleAvoidanceLoss
from simulation.simulation_properties import SimulationProperties
from scene.scene_factory import GripperSceneFactory, TouchSceneFactory
from simulation.update_scene import UpdateScene
from warp_wrapper.contact_properties import ContactProperties
from rendering.z_buffer import ZBuffer
from cable.pull_ratio import TimeInvariablePullRatio
from objective.variables import Variables


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
            poissons_ratio=0.40,
            damping_factor=0.4,
            frozen_bounding_box=[-float("inf"), -0.01, -float("inf"), float("inf"), -float("inf"), float("inf")],
        )
        robot_transform = Transform(
            rotation=get_quaternion(vector=[1, 0, 0], angle_in_degrees=90),
            scale=[0.001, 0.001, 0.001],
            device=cls.device,
        )
        sim_properties = SimulationProperties(
            duration=0.02, segment_duration=0.01, dt=2.1701388888888886e-05, device=cls.device
        )
        pull_ratio = [
            TimeInvariablePullRatio(
                pull_ratio=torch.tensor(0.0, device="cuda"), simulation_properties=sim_properties, device="cuda"
            ),
            TimeInvariablePullRatio(
                pull_ratio=torch.tensor(0.0, device="cuda"), simulation_properties=sim_properties, device="cuda"
            ),
            TimeInvariablePullRatio(
                pull_ratio=torch.tensor(0.0, device="cuda"), simulation_properties=sim_properties, device="cuda"
            ),
        ]
        cable_stiffness, cable_damping = 100, 0.01
        # object
        object_file = Path("tests/data/cylinder.obj")
        object_properties = MeshProperties(name="cylinder", density=1080.0)
        object_transform = Transform(translation=[60, -60, -20], scale=[0.0015, 0.0015, 0.01], device=cls.device)

        contact_properties = ContactProperties(distance=0.001, ke=2.0, kd=0.1, kf=0.1, ground=False)

        cls.scene = GripperSceneFactory(
            msh_file=msh_file,
            scad_file=scad_file,
            scad_parameters=scad_parameters,
            ideal_edge_length=ideal_edge_length,
            robot_properties=robot_properties,
            robot_transform=robot_transform,
            cable_pull_ratio=pull_ratio,
            cable_stiffness=cable_stiffness,
            cable_damping=cable_damping,
            object_file=object_file,
            object_properties=object_properties,
            object_transform=object_transform,
            contact_properties=contact_properties,
            device=cls.device,
            make_new_robot=False,
        ).create()

        simulation = Simulation(scene=cls.scene, properties=sim_properties)
        UpdateScene(scene=cls.scene, simulation=simulation).update_scene()
        views = ThreeInteriorViews(center=cls.scene.object.nodes.position.mean(dim=0), device=cls.device)
        robot_zbuf = ZBuffer(mesh=cls.scene.robot, views=views, device=cls.device)
        other_zbuf = ZBuffer(mesh=cls.scene.object, views=views, device=cls.device)
        cls.rendering = InteriorGapRendering(robot_zbuf=robot_zbuf, other_zbuf=other_zbuf)

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
            poissons_ratio=0.40,
            damping_factor=0.4,
            frozen_bounding_box=[-float("inf"), -0.01, -float("inf"), float("inf"), -float("inf"), float("inf")],
        )
        robot_transform = Transform(
            rotation=get_quaternion(vector=[1, 0, 0], angle_in_degrees=90),
            scale=[0.001, 0.001, 0.001],
            device=cls.device,
        )
        sim_properties = SimulationProperties(
            duration=0.02, segment_duration=0.01, dt=2.1701388888888886e-05, device=cls.device
        )
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
        cable_stiffness, cable_damping = 100, 0.01
        # object
        object_file = Path("tests/data/sphere.obj")
        object_properties = MeshProperties(name="sphere", density=1080.0)
        object_transform = Transform(translation=[-20, -20, -20], scale=[0.01, 0.01, 0.01], device=cls.device)

        contact_properties = ContactProperties(distance=0.001, ke=2.0, kd=0.1, kf=0.1, ground=False)

        cls.scene = TouchSceneFactory(
            msh_file=msh_file,
            scad_file=scad_file,
            scad_parameters=scad_parameters,
            ideal_edge_length=ideal_edge_length,
            robot_properties=robot_properties,
            robot_transform=robot_transform,
            cable_pull_ratio=pull_ratio,
            cable_stiffness=cable_stiffness,
            cable_damping=cable_damping,
            object_file=object_file,
            object_properties=object_properties,
            object_transform=object_transform,
            contact_properties=contact_properties,
            device=cls.device,
            make_new_robot=False,
        ).create()
        simulation = Simulation(scene=cls.scene, properties=sim_properties)
        UpdateScene(scene=cls.scene, simulation=simulation).update_scene()

    def tests_if_point_torch_loss_is_of_type_torch_tensor(self):
        loss = PointTouchLoss(scene=self.scene).get_loss()
        self.assertIsInstance(loss, torch.Tensor)


class TestObstacleAvoidanceLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        device = "cuda"
        # robot
        msh_file = Path("tests/data/caterpillar.msh")
        scad_file = Path("tests/data/caterpillar.scad")
        scad_parameters = Path("tests/data/caterpillar_unequal_scad_params.json")
        ideal_edge_length = 0.02
        robot_properties = MeshProperties(
            name="caterpillar",
            density=1080.0,
            youngs_modulus=149_000,
            poissons_ratio=0.40,
            damping_factor=0.4,
            frozen_bounding_box=[-float("inf"), -0.01, -float("inf"), float("inf"), -float("inf"), float("inf")],
        )
        robot_transform = Transform(
            rotation=get_quaternion(vector=[1, 0, 0], angle_in_degrees=90),
            scale=[0.001, 0.001, 0.001],
            device=device,
        )
        sim_properties = SimulationProperties(
            duration=0.01, segment_duration=0.01, dt=2.1701388888888886e-05, device=device
        )
        pull_ratio = [
            TimeInvariablePullRatio(
                pull_ratio=torch.tensor(0.5, device="cuda", requires_grad=True),
                simulation_properties=sim_properties,
                device="cuda",
            ),
            TimeInvariablePullRatio(
                pull_ratio=torch.tensor(0.0, device="cuda", requires_grad=True),
                simulation_properties=sim_properties,
                device="cuda",
            ),
            TimeInvariablePullRatio(
                pull_ratio=torch.tensor(0.0, device="cuda", requires_grad=True),
                simulation_properties=sim_properties,
                device="cuda",
            ),
        ]
        cable_stiffness, cable_damping = 100, 0.01
        # object
        object_file = Path("tests/data/sphere.obj")
        object_properties = MeshProperties(name="object", density=1080.0)
        object_transform = Transform(translation=[-8.26, -38.56, -11.46], scale=[0.005, 0.005, 0.005], device=device)

        # obstacle
        obstacle_files = [Path("tests/data/cylinder.obj")] * 2
        obstacle_properties = [MeshProperties(name="obstacle_0", density=1080.0)] * 2
        transform_0 = Transform(
            translation=[60.0, -80.0, -30.0],
            rotation=get_quaternion(vector=[0, 1, 0], angle_in_degrees=120),
            scale=[0.001, 0.001, 0.001],
        )
        transfrom_1 = Transform(
            translation=[60.0, -180.0, -30.0],
            rotation=get_quaternion(vector=[0, 1, 0], angle_in_degrees=120),
            scale=[0.001, 0.001, 0.001],
        )
        obstacle_transforms = [transform_0, transfrom_1]

        contact_properties = ContactProperties(distance=None, ke=None, kd=None, kf=None, ground=None)

        scene = TouchSceneFactory(
            msh_file=msh_file,
            scad_file=scad_file,
            scad_parameters=scad_parameters,
            ideal_edge_length=ideal_edge_length,
            robot_properties=robot_properties,
            robot_transform=robot_transform,
            cable_pull_ratio=pull_ratio,
            cable_stiffness=cable_stiffness,
            cable_damping=cable_damping,
            object_file=object_file,
            object_properties=object_properties,
            object_transform=object_transform,
            obstacle_files=obstacle_files,
            obstacle_properties=obstacle_properties,
            obstacle_transforms=obstacle_transforms,
            contact_properties=contact_properties,
            device=device,
            make_new_robot=False,
        ).create()

        cls.variables = Variables()
        for cable in scene.robot.cables:
            for opt in cable.pull_ratio.optimizable:
                cls.variables.add_parameter(opt)

        simulation = Simulation(scene=scene, properties=sim_properties)
        views_obstacle_0 = SixInteriorViews(center=scene.obstacles[0].nodes.position.mean(dim=0), device=device)
        views_obstacle_1 = SixInteriorViews(center=scene.obstacles[1].nodes.position.mean(dim=0), device=device)
        robot_zbuf_0 = ZBuffer(mesh=scene.robot, views=views_obstacle_0, device=device)
        robot_zbuf_1 = ZBuffer(mesh=scene.robot, views=views_obstacle_1, device=device)
        robot_zbufs = [robot_zbuf_0, robot_zbuf_1]
        obstacle_zbuf_0 = ZBuffer(mesh=scene.obstacles[0], views=views_obstacle_0, device=device)
        obstacle_zbuf_1 = ZBuffer(mesh=scene.obstacles[1], views=views_obstacle_1, device=device)
        obstacle_zbufs = [obstacle_zbuf_0, obstacle_zbuf_1]
        rendering = [
            InteriorDistanceRendering(robot_zbuf=rz, other_zbuf=oz) for rz, oz in zip(robot_zbufs, obstacle_zbufs)
        ]
        cls.loss_obstacle_0 = ObstacleAvoidanceLoss(rendering[0], device=device)
        cls.loss_obstacle_1 = ObstacleAvoidanceLoss(rendering[1], device=device)

        scene.add_observer(cls.loss_obstacle_0)
        scene.add_observer(cls.loss_obstacle_1)
        UpdateScene(scene=scene, simulation=simulation).update_scene()

    def tests_if_max_grip_loss_is_of_type_torch_tensor(self):
        self.assertIsInstance(self.loss_obstacle_0.loss, torch.Tensor)
        self.assertIsInstance(self.loss_obstacle_1.loss, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
