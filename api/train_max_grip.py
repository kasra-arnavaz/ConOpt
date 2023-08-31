import torch
import unittest
import sys

sys.path.append("src")
from pathlib import Path
from simulation.scene_factory import SceneFactory
from mesh.mesh_properties import MeshProperties
from simulation.simulation import Simulation
from point.transform import Transform, get_quaternion
from rendering.views import ThreeInteriorViews, ThreeExteriorViews
from rendering.visual import Visual
from rendering.rendering import (
    ExteriorDepthRendering,
    InteriorGapRendering,
    InteriorContactRendering,
)
from objective.loss import MaxGripLoss
from objective.optimizer import GradientDescent, Adam
from objective.train import Train
from objective.variables import Variables
from simulation.simulation_properties import SimulationProperties
from objective.log import Log
from simulation.scene_viewer import SceneViewer


class TestMaxGripLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        device = "cuda"
        # gripper
        scad_file = Path("tests/data/caterpillar.scad")
        scad_parameters = Path("tests/data/caterpillar_scad_params.json")
        ideal_edge_length = 0.02
        gripper_properties = MeshProperties(
            name="caterpillar",
            density=1080.0,
            youngs_modulus=149_000,
            poissons_ratio=0.40,
            damping_factor=0.4,
            frozen_bounding_box=[-float("inf"), -0.01, -float("inf"), float("inf"), float("inf"), float("inf")],
        )
        gripper_transform = Transform(
            rotation=get_quaternion(vector=[1, 0, 0], angle_in_degrees=90), scale=[0.001, 0.001, 0.001], device=device
        )
        cable_pull_ratio = [
            torch.tensor(0.0, device=device, requires_grad=True),
            torch.tensor(0.0, device=device, requires_grad=True),
            torch.tensor(0.0, device=device, requires_grad=True),
        ]
        cable_stiffness, cable_damping = 100, 0.01
        # object
        object_file = Path("tests/data/cylinder.obj")
        object_properties = MeshProperties(name="cylinder", density=1080.0)
        object_transform = Transform(translation=[60, -60, -20], scale=[0.0015, 0.0015, 0.01], device=device)

        cls.scene = SceneFactory(
            scad_file=scad_file,
            scad_parameters=scad_parameters,
            ideal_edge_lenght=ideal_edge_length,
            gripper_properties=gripper_properties,
            gripper_transform=gripper_transform,
            cable_pull_ratio=cable_pull_ratio,
            cable_stiffness=cable_stiffness,
            cable_damping=cable_damping,
            object_file=object_file,
            object_properties=object_properties,
            object_transform=object_transform,
            device=device,
        ).create()

        variables = Variables()
        for cable in cls.scene.gripper.cables:
            variables.add_parameter(cable.pull_ratio)
        sim_properties = SimulationProperties(
            duration=1.0, segment_duration=0.1, dt=2.1701388888888886e-05, device=device
        )
        simulation = Simulation(scene=cls.scene, properties=sim_properties)
        views = ThreeInteriorViews(center=cls.scene.object.nodes.position.mean(dim=0), device=device)
        rendering = InteriorGapRendering(
            scene=cls.scene,
            views=views,
            device=device,
        )
        loss = MaxGripLoss(rendering=rendering, device=device)
        optimizer = GradientDescent(loss, variables, learning_rate=1e-3)
        exterior_view = ThreeExteriorViews(distance=0.5, device=device)
        PATH = ".tmp"
        visual = Visual(ExteriorDepthRendering(scene=cls.scene, views=exterior_view, device=device), path=PATH)
        log = Log(loss=loss, variables=variables, path=PATH)
        viewer = SceneViewer(scene=cls.scene, path=PATH)
        cls.train = Train(
            simulation, cls.scene, loss, optimizer, num_iters=100, log=log, visual=visual, scene_viewer=viewer
        )

    def tests_if_train_runs(self):
        try:
            self.train.run(verbose=True)
        except:
            self.fail()


if __name__ == "__main__":
    unittest.main()
