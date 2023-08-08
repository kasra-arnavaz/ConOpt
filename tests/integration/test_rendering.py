import torch
import unittest
import sys

sys.path.append("src")
from pathlib import Path
from mesh.mesh_factory import MeshFactoryFromScad, MeshFactoryFromObj
from mesh.scad import Scad
from mesh.mesh_properties import MeshProperties
from simulation.simulation import Simulation
from cable.cable_factory import CableFactory
from cable.holes_factory import HolesFactoryFromListOfPositions
from cable.holes_initial_position import HolesInitialPosition
from point.transform import Transform, get_quaternion
from rendering.views import SixExteriorViews, SixInteriorViews
from rendering.rendering import ExteriorDepthRendering, InteriorGapRendering, InteriorContactRendering


class TestRenderingVisualization(unittest.TestCase):
    """Only runs rendering but final images need to be viewed manually
    to make sure the outcome is accurate."""

    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        gripper_mesh = MeshFactoryFromScad(scad, ideal_edge_length=0.02).create()
        gripper_mesh.properties = MeshProperties(
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
        transform.apply(gripper_mesh.nodes)
        for hole in holes:
            transform.apply(hole)
        pull_ratio = [
            torch.tensor(0.5, device="cuda"),
            torch.tensor(0.0, device="cuda"),
            torch.tensor(0.0, device="cuda"),
        ]
        cables = CableFactory(stiffness=100, damping=0.01, pull_ratio=pull_ratio, holes=holes).create()
        Simulation(mesh=gripper_mesh, cables=cables, duration=0.5, dt=2.1701388888888886e-05, device="cuda").run()
        cls.gripper_mesh = gripper_mesh
        object_mesh = MeshFactoryFromObj(Path("tests/data/cylinder.obj"), device="cuda").create()
        transform.apply(object_mesh.nodes)
        cls.object_mesh = object_mesh

        
    def tests_if_exterior_depth_rendering_of_gripper_runs_given_six_views(self):
        views = SixExteriorViews(distance=0.5, device="cuda")
        rendering = ExteriorDepthRendering(meshes=[self.gripper_mesh], views=views)
        try:
            rendering.get_images()
        except:
            self.fail()

    def tests_if_interior_gap_rendering_runs_given_six_views(self):
        views = SixInteriorViews(center=self.object_mesh.nodes.position.mean(dim=0), device="cuda")
        rendering = InteriorGapRendering(self.gripper_mesh, self.object_mesh, views)
        try:
            rendering.get_images()
        except:
            self.fail()    
    
    def tests_if_interior_contact_rendering_runs_given_six_views(self):
        views = SixInteriorViews(center=self.object_mesh.nodes.position.mean(dim=0), device="cuda")
        rendering = InteriorContactRendering(self.gripper_mesh, self.object_mesh, views)
        try:
            rendering.get_images()
        except:
            self.fail()

if __name__ == "__main__":
    unittest.main()
