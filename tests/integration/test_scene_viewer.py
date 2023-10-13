import torch
import unittest
import sys

sys.path.append("src")
from pathlib import Path
from simulation.simulation import Simulation
from scene.scene_factory import TouchSceneFactory, GripperSceneFactory
from warp_wrapper.contact_properties import ContactProperties
from point.transform import Transform, get_quaternion
from mesh.mesh_properties import MeshProperties
from simulation.simulation_properties import SimulationProperties
from simulation.update_scene import update_scene
from scene.scene_viewer import SceneViewer
from cable.pull_ratio import TimeInvariablePullRatio

class TestSceneViewer(unittest.TestCase):
    """Only checks memory and gradients but final mesh needs to be viewed manually
    to make sure the outcome is accurate."""

    @classmethod
    def setUpClass(cls):
        device = "cuda"
        # robot
        msh_file = Path("data/long_caterpillar.msh")
        scad_file = Path("tests/data/caterpillar.scad")
        scad_parameters = Path("tests/data/caterpillar_scad_params.json")
        ideal_edge_length = 0.02
        robot_properties = MeshProperties(
            name="caterpillar",
            density=1080.0,
            youngs_modulus=149_000,
            poissons_ratio=0.40,
            damping_factor=0.4,
            frozen_bounding_box=[-float("inf"), -0.01, -float("inf"), float("inf"), float("inf"), float("inf")],
        )
        robot_transform = Transform(
            rotation=get_quaternion(vector=[1, 0, 0], angle_in_degrees=90), scale=[0.001, 0.001, 0.001], device=device
        )
        sim_properties = SimulationProperties(
            duration=0.5, segment_duration=0.05, dt=2.1701388888888886e-05, device=device
        )
        pull_ratio = [
            TimeInvariablePullRatio(pull_ratio=torch.tensor(0.5, device=device), simulation_properties=sim_properties, device=device),
            TimeInvariablePullRatio(pull_ratio=torch.tensor(0.0, device=device), simulation_properties=sim_properties, device=device),
            TimeInvariablePullRatio(pull_ratio=torch.tensor(0.0, device=device), simulation_properties=sim_properties, device=device),
        ]
        cable_stiffness, cable_damping = 100, 0.01
        # object
        object_file = Path("tests/data/cylinder.obj")
        object_properties = MeshProperties(name="cylinder", density=1080.0)
        object_transform = Transform(translation=[60, -60, -20], scale=[0.0015, 0.0015, 0.01], device=device)

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
            device=device,
            make_new_robot=False
        ).create()


        cls.simulation = Simulation(scene=cls.scene, properties=sim_properties)
        cls.viewer = SceneViewer(scene=cls.scene, path=".tmp")

    def tests_if_simulation_runs_with_viewer(self):
        try:
            update_scene(scene=self.scene, simulation=self.simulation, viewer=self.viewer)
            print("CHECK .tmp/scene.usd")
        except:
            self.fail()


if __name__ == "__main__":
    unittest.main()
