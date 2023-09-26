import unittest
from pathlib import Path
import sys
import torch
sys.path.append("src")
from mesh.mesh_properties import MeshProperties
from point.transform import Transform, get_quaternion
from warp_wrapper.contact_properties import ContactProperties
from scene.scene_factory import GripperSceneFactory
from mesh.mesh import Mesh
from warp.sim import Model

class TestGripperSceneFactory(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        device = "cuda"
        # robot
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
            rotation=get_quaternion(vector=[1, 0, 0], angle_in_degrees=90), scale=[0.001, 0.001, 0.001], device=device
        )
        cable_pull_ratio = [
            torch.tensor(0.5, device=device, requires_grad=True),
            torch.tensor(0.0, device=device, requires_grad=True),
            torch.tensor(0.0, device=device, requires_grad=True),
        ]
        cable_stiffness, cable_damping = 100, 0.01
        # object
        object_file = Path("tests/data/cylinder.obj")
        object_properties = MeshProperties(name="cylinder", density=1080.0)
        object_transform = Transform(translation=[60, -60, -20], scale=[0.0015, 0.0015, 0.01], device=device)

        contact_properties = ContactProperties(distance=0.001, ke=2.0, kd=0.1, kf=0.1, ground=False)
        
        cls.scene = GripperSceneFactory(
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
            device=device,
            make_new_robot=True
        ).create()
    
    def tests_type_of_robot_attribute(self):
        self.assertIsInstance(self.scene.robot, Mesh)

    def tests_type_of_object_attribute(self):
        self.assertIsInstance(self.scene.object, Mesh)

    def tests_type_of_model_attribute(self):
        self.assertIsInstance(self.scene.model, Model)


if __name__ == "__main__":
    unittest.main()