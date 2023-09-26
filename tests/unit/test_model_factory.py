import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.append("src")

from warp_wrapper.model_factory import ModelFactory
from mesh.scad import Scad
from mesh.mesh_factory import MeshFactoryFromScad
from mesh.mesh_properties import MeshProperties
from warp.sim import Model
from warp_wrapper.contact_properties import ContactProperties


class TestModelFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)
        mesh = MeshFactoryFromScad(scad).create()
        mesh.properties = MeshProperties(
            name="caterpillar",
            density=1080.0,
            youngs_modulus=149_000,
            poissons_ratio=0.45,
            damping_factor=0.4,
            frozen_bounding_box=[-np.inf, -np.inf, 0, np.inf, np.inf, 2],
        )
        contact_properties = ContactProperties(distance=0.01, ke=2.0, kd=0.1, kf=0.1, ground=False)
        cls.model_factory = ModelFactory(soft_mesh=mesh, contact_properties=contact_properties, device="cuda")

    def tests_if_model_factory_creates_a_model_with_a_soft_mesh(self):
        model = self.model_factory.create()
        self.assertTrue(isinstance(model, Model))


if __name__ == "__main__":
    unittest.main()
