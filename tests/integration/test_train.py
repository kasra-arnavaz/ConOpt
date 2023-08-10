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
from rendering.views import ThreeInteriorViews
from rendering.visualization import Visualization
from rendering.rendering import ExteriorDepthRendering, InteriorGapRendering, InteriorContactRendering
from objective.loss import MaxGripLoss
from objective.optimizer import GradientDescent, Adam
from objective.train import Train
from objective.variables import Variables

class TestMaxGripLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        file = Path("tests/data/caterpillar.scad")
        parameters = Path("tests/data/caterpillar_scad_params.json")
        scad = Scad(file, parameters)

        gripper_mesh = MeshFactoryFromScad(scad, ideal_edge_length=0.02, device="cuda").create()
        object_mesh = MeshFactoryFromObj(Path("tests/data/cylinder.obj"), device="cuda").create()

        gripper_mesh.properties = MeshProperties(
            name="caterpillar",
            density=1080.0,
            youngs_modulus=149_000,
            poissons_ratio=0.40,
            damping_factor=0.4,
            frozen_bounding_box=[-float("inf"), -0.01, -float("inf"), float("inf"), float("inf"), float("inf")],
        )
        object_mesh.properties = MeshProperties(name="cylinder", density=1080.)

        transform_gripper = Transform(
            rotation=get_quaternion(vector=[1, 0, 0], angle_in_degrees=90), scale=[0.001, 0.001, 0.001], device="cuda"
        )
        transform_object = Transform(translation=[60, -60, -20], scale=[0.0015, 0.0015, 0.01], device="cuda")

        holes_position = HolesInitialPosition(scad).get()
        holes = HolesFactoryFromListOfPositions(holes_position, device="cuda").create()

        transform_gripper.apply(gripper_mesh.nodes)
        transform_object.apply(object_mesh.nodes)
        for hole in holes:
            transform_gripper.apply(hole)
        pull_ratio = [
            torch.tensor(0.5, device="cuda", requires_grad=True),
            torch.tensor(0.0, device="cuda", requires_grad=True),
            torch.tensor(0.0, device="cuda", requires_grad=True),
        ]
        variables = Variables()
        for p in pull_ratio:
            variables.add_parameter(p)
        cables = CableFactory(stiffness=100, damping=0.01, pull_ratio=pull_ratio, holes=holes).create()
        simulation = Simulation(gripper_mesh=gripper_mesh, object_mesh=object_mesh, cables=cables, duration=0.5, dt=2.1701388888888886e-05, device="cuda")
        views = ThreeInteriorViews(center=object_mesh.nodes.position.mean(dim=0), device="cuda")
        rendering = InteriorGapRendering(gripper_mesh=gripper_mesh, object_mesh=object_mesh, views=views, device="cuda")
        loss = MaxGripLoss(rendering=rendering, device="cuda")
        optimizer = Adam(loss, variables, learning_rate=1e-4)
        cls.train = Train(simulation, loss, optimizer, num_iters=1)

        
    def tests_if_train_runs(self):
        try:
            self.train.run()
        except:
            self.fail()

if __name__ == "__main__":
    unittest.main()
