import torch
import unittest
import sys

sys.path.append("src")
from pathlib import Path
from mesh.mesh_properties import MeshProperties
from simulation.simulation import Simulation
from point.transform import Transform, get_quaternion
from rendering.views import SixExteriorViews, SixInteriorViews
from rendering.rendering import ExteriorDepthRendering, InteriorGapRendering, InteriorContactRendering
from simulation.simulation_properties import SimulationProperties
from scene.scene_factory import GripperSceneFactory
from warp_wrapper.contact_properties import ContactProperties
from simulation.update_scene import UpdateScene
from rendering.z_buffer import ZBuffer
from scene.scene_observer import SceneImages
from cable.pull_ratio import TimeInvariablePullRatio


class TestRenderingVisualization(unittest.TestCase):
    """Only runs rendering but final images need to be viewed manually
    to make sure the outcome is accurate."""

    @classmethod
    def setUpClass(cls):
        device = "cuda"
        # robot
        msh_file = Path("data/long_caterpillar.msh")
        scad_file = Path("data/caterpillar.scad")
        scad_parameters = Path("data/long_caterpillar_scad_params.json")
        ideal_edge_length = 0.02
        robot_properties = MeshProperties(
            name="caterpillar",
            density=1080.0,
            youngs_modulus=149_000,
            poissons_ratio=0.45,
            damping_factor=0.4,
            frozen_bounding_box=[-float("inf"), -0.01, -float("inf"), float("inf"), float("inf"), float("inf")],
        )
        robot_transform = Transform(
            rotation=get_quaternion(vector=[1, 0, 0], angle_in_degrees=90),
            scale=[0.001, 0.001, 0.001],
            device=device,
        )
        sim_properties = SimulationProperties(
            duration=2.0, segment_duration=0.1, dt=2.1701388888888886e-05, device=device
        )
        pull_ratio = [
            TimeInvariablePullRatio(
                pull_ratio=torch.tensor(0.5, device=device), simulation_properties=sim_properties, device=device
            ),
            TimeInvariablePullRatio(
                pull_ratio=torch.tensor(0.0, device=device), simulation_properties=sim_properties, device=device
            ),
            TimeInvariablePullRatio(
                pull_ratio=torch.tensor(0.0, device=device), simulation_properties=sim_properties, device=device
            ),
            TimeInvariablePullRatio(
                pull_ratio=torch.tensor(0.5, device=device), simulation_properties=sim_properties, device=device
            ),
            TimeInvariablePullRatio(
                pull_ratio=torch.tensor(0.0, device=device), simulation_properties=sim_properties, device=device
            ),
        ]
        cable_stiffness, cable_damping = 100, 0.01
        # object
        object_file = Path("tests/data/cylinder.obj")
        object_properties = MeshProperties(name="cylinder", density=1080.0)
        object_transform = Transform(translation=[60, -60, -20], scale=[0.0015, 0.0015, 0.01], device=device)

        contact_properties = ContactProperties(distance=0.001, ke=2.0, kd=0.1, kf=0.1, ground=False)

        scene = GripperSceneFactory(
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
            device=device,
            make_new_robot=False,
        ).create()

        simulation = Simulation(scene=scene, properties=sim_properties)
        # exterior
        views = SixExteriorViews(distance=0.5, device=device)
        zbufs = [ZBuffer(mesh=mesh, views=views, device=device) for mesh in scene.all_meshes()]
        rendering = ExteriorDepthRendering(zbufs)
        exterior_images = SceneImages(rendering=rendering, path=".tmp", prefix="exterior")
        # gap
        views = SixInteriorViews(center=scene.object.nodes.position.mean(dim=0), device=device)
        robot_zbuf = ZBuffer(mesh=scene.robot, views=views, device=device)
        other_zbuf = ZBuffer(mesh=scene.object, views=views, device=device)
        rendering = InteriorGapRendering(robot_zbuf=robot_zbuf, other_zbuf=other_zbuf)
        gap_images = SceneImages(rendering=rendering, path=".tmp", prefix="gap")
        # contact
        views = SixInteriorViews(center=scene.object.nodes.position.mean(dim=0), device=device)
        robot_zbuf = ZBuffer(mesh=scene.robot, views=views, device=device)
        other_zbuf = ZBuffer(mesh=scene.object, views=views, device=device)
        rendering = InteriorContactRendering(robot_zbuf=robot_zbuf, other_zbuf=other_zbuf)
        contact_images = SceneImages(rendering=rendering, path=".tmp", prefix="contact")
        # add observers
        scene.add_observer(exterior_images)
        scene.add_observer(gap_images)
        scene.add_observer(contact_images)
        UpdateScene(scene=scene, simulation=simulation).update_scene()

    def tests_if_exterior_depth_rendering_of_robot_runs_given_six_views(self):
        print("CHECK .tmp/exterior_i.png")

    def tests_if_interior_gap_rendering_runs_given_six_views(self):
        print("CHECK .tmp/gap_i.png")

    def tests_if_interior_contact_rendering_runs_given_six_views(self):
        print("CHECK .tmp/contact_i.png")


if __name__ == "__main__":
    unittest.main()
