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
from warp_wrapper.state_iterable import StateIterable


class TestStateIterable(unittest.TestCase):
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
        model = ModelFactory(soft_mesh=mesh, device="cuda").create()
        cls.state_iterable = StateIterable(model=model, num=3)

    def tests_if_states_are_iterated_recursively(self):
        states = [next(self.state_iterable) for _ in range(6)]
        self.assertTrue(states[1] is states[2])
        self.assertTrue(states[3] is states[4])
        self.assertTrue(states[5] is states[0])


if __name__ == "__main__":
    unittest.main()
