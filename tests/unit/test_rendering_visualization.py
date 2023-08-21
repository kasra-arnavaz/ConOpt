import torch
import unittest
import sys

sys.path.append("src")
from pathlib import Path
from mesh.mesh_factory import MeshFactoryFromObj
from point.transform import Transform, get_quaternion
from rendering.views import SixExteriorViews
from rendering.rendering import ExteriorDepthRendering
from rendering.visualization import Visualization


class TestRenderingVisualization(unittest.TestCase):
    """Only runs rendering but final images need to be viewed manually
    to make sure the outcome is accurate."""

    @classmethod
    def setUpClass(cls):
        transform = Transform(rotation=get_quaternion(vector=[1, 0, 0], angle_in_degrees=30), scale=[0.01, 0.01, 0.01])
        object_mesh = MeshFactoryFromObj(Path("tests/data/cylinder.obj"), device="cuda").create()
        transform.apply(object_mesh.nodes)
        cls.object_mesh = object_mesh

    def tests_if_exterior_depth_rendering_of_gripper_is_visualized_given_six_views(self):
        views = SixExteriorViews(distance=1.0, device="cuda")
        rendering = ExteriorDepthRendering(meshes=[self.object_mesh], views=views)
        try:
            Visualization(rendering).save_images(Path(".tmp/rendering.png"))
        except:
            self.fail()


if __name__ == "__main__":
    unittest.main()
