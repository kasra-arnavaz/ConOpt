import sys

sys.path.append("src")
from pathlib import Path
from scene.scene_factory import GripperSceneFactory
from mesh.mesh_properties import MeshProperties
from simulation.simulation import Simulation
from point.transform import Transform, get_quaternion
from rendering.views import SixInteriorViews
from rendering.rendering import InteriorGapRendering
from objective.loss import MaxGripLoss
from objective.optimizer import GradientDescent
from objective.train import Train
from objective.variables import Variables
from simulation.simulation_properties import SimulationProperties
from objective.log import Log
from warp_wrapper.contact_properties import ContactProperties
from config.config import Config
import argparse
from utils.path import get_next_numbered_path
from rendering.z_buffer import ZBuffer
from cable.pull_ratio import TimeInvariablePullRatio
from simulation.update_scene import UpdateScene


def main(args):
    config = Config.from_yaml(args.config)
    DEVICE = config.device
    PATH = get_next_numbered_path(config.out_path)
    config.to_yaml(path=PATH)

    # robot
    msh_file = Path(config.msh_file)
    scad_file = Path(config.scad_file)
    scad_parameters = Path(config.scad_parameters)
    ideal_edge_length = config.ideal_edge_length
    robot_properties = MeshProperties(
        name="robot",
        density=config.robot_density,
        youngs_modulus=config.robot_youngs_modulus,
        poissons_ratio=config.robot_poissons_ratio,
        damping_factor=config.robot_damping_factor,
        frozen_bounding_box=config.robot_frozen_bounding_box,
    )
    robot_transform = Transform(
        translation=config.robot_translation,
        rotation=get_quaternion(
            vector=config.robot_rotation_vector,
            angle_in_degrees=config.robot_rotation_degrees,
        ),
        scale=config.robot_scale,
        device=DEVICE,
    )
    sim_properties = SimulationProperties(
        duration=config.sim_duration,
        segment_duration=config.sim_segment_duration,
        dt=config.sim_dt,
        device=DEVICE,
    )
    cable_pull_ratio = [
        TimeInvariablePullRatio(simulation_properties=sim_properties, device=DEVICE) for _ in config.cable_pull_ratio
    ]
    cable_stiffness, cable_damping = config.cable_stiffness, config.cable_damping

    # object
    object_file = Path(config.object_file)
    object_properties = MeshProperties(name="object", density=config.object_density)
    object_transform = Transform(
        translation=config.object_translation,
        rotation=get_quaternion(
            vector=config.object_rotation_vector,
            angle_in_degrees=config.object_rotation_degrees,
        ),
        scale=config.object_scale,
        device=DEVICE,
    )

    contact_properties = ContactProperties(
        distance=config.contact_distance,
        ke=config.contact_ke,
        kd=config.contact_kd,
        kf=config.contact_kf,
        ground=config.ground,
    )

    scene = GripperSceneFactory(
        msh_file=msh_file,
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
        device=DEVICE,
        make_new_robot=False,
    ).create()

    variables = Variables()
    for cable in scene.robot.cables:
        for opt in cable.pull_ratio.optimizable:
            variables.add_parameter(opt)

    simulation = Simulation(scene=scene, properties=sim_properties, use_checkpoint=config.use_checkpoint)
    views = SixInteriorViews(center=scene.object.nodes.position.mean(dim=0), device=DEVICE)
    robot_zbuf = ZBuffer(mesh=scene.robot, views=views, device=DEVICE)
    other_zbuf = ZBuffer(mesh=scene.object, views=views, device=DEVICE)
    rendering = InteriorGapRendering(robot_zbuf=robot_zbuf, other_zbuf=other_zbuf)
    loss = MaxGripLoss(rendering=rendering, device=DEVICE)
    optimizer = GradientDescent(loss, variables, learning_rate=config.learning_rate)
    log = Log(loss=loss, variables=variables, path=PATH)
    update_scene = UpdateScene(scene=scene, simulation=simulation)
    Train(
        scene,
        update_scene,
        loss,
        optimizer,
        num_iters=config.num_training_iterations,
        log=log,
    ).run(verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path like with yaml format")
    args = parser.parse_args()
    main(args)
