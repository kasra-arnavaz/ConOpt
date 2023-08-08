import torch
import unittest
import sys

sys.path.append("src")
from pathlib import Path
from mesh.mesh_factory import MeshFactoryFromScad, MeshFactoryFromObj
from mesh.scad import Scad
from rendering.views import SixExteriorViews, SixInteriorViews
from rendering.rendering import ExteriorDepthRendering, InteriorGapRendering, InteriorContactRendering


class TestRendering(unittest.TestCase):
    """Only runs rendering but final images need to be viewed manually
    to make sure the outcome is accurate."""

    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        cls.gripper_mesh = MeshFactoryFromScad(scad, ideal_edge_length=0.02, device="cuda").create()
        cls.object_mesh = MeshFactoryFromObj(Path("tests/data/cylinder.obj"), device="cuda").create()
        
    def tests_if_exterior_depth_rendering_of_gripper_runs_given_six_views(self):
        views = SixExteriorViews(device="cuda")
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
