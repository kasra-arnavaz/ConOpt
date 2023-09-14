import torch
import sys

sys.path.append("src")
from pathlib import Path
from scene.scene_factory import SceneFactoryFromMsh, SceneFactoryFromScad
from mesh.mesh_properties import MeshProperties
from simulation.simulation import Simulation
from point.transform import Transform, get_quaternion
from rendering.views import ThreeInteriorViews, ThreeExteriorViews, SixInteriorViews
from rendering.visual import Visual
from rendering.rendering import (
    ExteriorDepthRendering,
    InteriorGapRendering,
    InteriorContactRendering,
)
from objective.loss import PointTouchLoss
from objective.optimizer import GradientDescent, Adam
from objective.train import Train
from objective.variables import Variables
from simulation.simulation_properties import SimulationProperties
from objective.log import Log
from warp_wrapper.contact_properties import ContactProperties
from config.config import Config
import argparse
from utils.path import get_next_numbered_path


def main(args):
    config = Config.from_yaml(args.config)
    DEVICE = config.device
    PATH = get_next_numbered_path(config.out_path)
    config.to_yaml(path=PATH)

    # gripper
    msh_file = Path(config.msh_file)
    scad_file = Path(config.scad_file)
    scad_parameters = Path(config.scad_parameters)
    ideal_edge_length = config.ideal_edge_length
    gripper_properties = MeshProperties(
        name="gripper",
        density=config.gripper_density,
        youngs_modulus=config.gripper_youngs_modulus,
        poissons_ratio=config.gripper_poissons_ratio,
        damping_factor=config.gripper_damping_factor,
        frozen_bounding_box=config.gripper_frozen_bounding_box,
    )
    gripper_transform = Transform(
        translation=config.gripper_translation,
        rotation=get_quaternion(
            vector=config.gripper_rotation_vector, angle_in_degrees=config.gripper_rotation_degrees
        ),
        scale=config.gripper_scale,
        device=DEVICE,
    )
    cable_pull_ratio = [torch.tensor(pull, device=DEVICE) for pull in config.cable_pull_ratio]
    cable_stiffness, cable_damping = config.cable_stiffness, config.cable_damping

    # object
    object_file = Path(config.object_file)
    object_properties = MeshProperties(name="object", density=config.object_density)
    object_transform = Transform(
        translation=config.object_translation,
        rotation=get_quaternion(vector=config.object_rotation_vector, angle_in_degrees=config.object_rotation_degrees),
        scale=config.object_scale,
        device=DEVICE,
    )

    contact_properties = ContactProperties(
        distance=config.contact_distance, ke=config.contact_ke, kd=config.contact_kd, kf=config.contact_kf
    )

    scene = SceneFactoryFromMsh(
        msh_file=msh_file,
        scad_file=scad_file,
        scad_parameters=scad_parameters,
        ideal_edge_length=ideal_edge_length,
        gripper_properties=gripper_properties,
        gripper_transform=gripper_transform,
        cable_pull_ratio=cable_pull_ratio,
        cable_stiffness=cable_stiffness,
        cable_damping=cable_damping,
        object_file=object_file,
        object_properties=object_properties,
        object_transform=object_transform,
        contact_properties=contact_properties,
        device=DEVICE,
    ).create()

    variables = Variables()
    for cable in scene.gripper.cables:
        variables.add_parameter(cable.pull_ratio)
    sim_properties = SimulationProperties(
        duration=config.sim_duration, segment_duration=config.sim_segment_duration, dt=config.sim_dt, device=DEVICE
    )
    simulation = Simulation(scene=scene, properties=sim_properties, use_checkpoint=config.use_checkpoint)
    views = SixInteriorViews(center=scene.object.nodes.position.mean(dim=0), device=DEVICE)
    rendering = InteriorGapRendering(
        scene=scene,
        views=views,
        device=DEVICE,
    )
    loss = PointTouchLoss(scene=scene)
    optimizer = GradientDescent(loss, variables, learning_rate=config.learning_rate)
    exterior_view = ThreeExteriorViews(distance=0.5, device=DEVICE)
    visual_ext = Visual(
        ExteriorDepthRendering(scene=scene, views=exterior_view, device=DEVICE), path=PATH, prefix="ext"
    )
    visual_gap = Visual(rendering, path=PATH, prefix="gap")
    log = Log(loss=loss, variables=variables, path=PATH)
    Train(
        simulation,
        scene,
        loss,
        optimizer,
        num_iters=config.num_training_iterations,
        log=log,
        visuals=[visual_ext, visual_gap],
    ).run(verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path like with yaml format")
    args = parser.parse_args()
    main(args)
