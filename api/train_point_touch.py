import sys

sys.path.append("src")
from pathlib import Path
from scene.scene_factory import TouchSceneFactory
from mesh.mesh_properties import MeshProperties
from simulation.simulation import Simulation
from point.transform import Transform, get_quaternion
from rendering.views import SixInteriorViews
from rendering.rendering import InteriorGapRendering, InteriorDistanceRendering
from objective.loss import ObstacleAvoidanceLoss, PointTouchWithObstacleAvoidanceLoss
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
        rotation=get_quaternion(vector=config.robot_rotation_vector, angle_in_degrees=config.robot_rotation_degrees),
        scale=config.robot_scale,
        device=DEVICE,
    )
    sim_properties = SimulationProperties(
        duration=config.sim_duration,
        segment_duration=config.sim_segment_duration,
        dt=config.sim_dt,
        key_timepoints_interval=config.key_timepoints_interval,
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
        rotation=get_quaternion(vector=config.object_rotation_vector, angle_in_degrees=config.object_rotation_degrees),
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
    # obstacle
    try:
        obstacle_files = [Path(file) for file in config.obstacle_file]
        obstacle_properties = [
            MeshProperties(name=f"obstacle_{i}", density=density) for i, density in enumerate(config.obstacle_density)
        ]
        obstacle_transforms = []
        for t, rv, rd, s in zip(
            config.obstacle_translation,
            config.obstacle_rotation_vector,
            config.obstacle_rotation_degrees,
            config.obstacle_scale,
        ):
            obstacle_transforms.append(
                Transform(translation=t, rotation=get_quaternion(vector=rv, angle_in_degrees=rd), scale=s)
            )
    except:
        obstacle_files, obstacle_properties, obstacle_transforms = None, None, None

    contact_properties = ContactProperties(
        distance=config.contact_distance,
        ke=config.contact_ke,
        kd=config.contact_kd,
        kf=config.contact_kf,
        ground=config.ground,
    )

    scene = TouchSceneFactory(
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
        obstacle_files=obstacle_files,
        obstacle_properties=obstacle_properties,
        obstacle_transforms=obstacle_transforms,
        contact_properties=contact_properties,
        device=DEVICE,
        make_new_robot=False,
    ).create()

    variables = Variables()
    for cable in scene.robot.cables:
        for opt in cable.pull_ratio.optimizable:
            variables.add_parameter(opt)

    simulation = Simulation(scene=scene, properties=sim_properties, use_checkpoint=config.use_checkpoint)
    # point touch with obstacle avoidance loss
    views_obstacle_0 = SixInteriorViews(center=scene.obstacles[0].nodes.position.mean(dim=0), device=DEVICE)
    views_obstacle_1 = SixInteriorViews(center=scene.obstacles[1].nodes.position.mean(dim=0), device=DEVICE)
    robot_zbuf_0 = ZBuffer(mesh=scene.robot, views=views_obstacle_0, device=DEVICE)
    robot_zbuf_1 = ZBuffer(mesh=scene.robot, views=views_obstacle_1, device=DEVICE)
    robot_zbufs = [robot_zbuf_0, robot_zbuf_1]
    obstacle_zbuf_0 = ZBuffer(mesh=scene.obstacles[0], views=views_obstacle_0, device=DEVICE)
    obstacle_zbuf_1 = ZBuffer(mesh=scene.obstacles[1], views=views_obstacle_1, device=DEVICE)
    obstacle_zbufs = [obstacle_zbuf_0, obstacle_zbuf_1]
    renderings = [
        InteriorGapRendering(robot_zbuf=rz, other_zbuf=oz) for rz, oz in zip(robot_zbufs, obstacle_zbufs)
    ]  # Consider using InteriorDistanceRendering
    obs_loss = [ObstacleAvoidanceLoss(rendering=r, device=DEVICE) for r in renderings]
    for ol in obs_loss:
        scene.add_observer(ol)
    loss = PointTouchWithObstacleAvoidanceLoss(scene=scene, obstacle_avoidance_losses=obs_loss)
    optimizer = GradientDescent(loss, variables, learning_rate=config.learning_rate)
    log = Log(loss=loss, variables=variables, path=PATH)
    update_scene = UpdateScene(scene=scene, simulation=simulation)
    Train(scene, update_scene, loss, optimizer, num_iters=config.num_training_iterations, log=log).run(verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path like with yaml format")
    args = parser.parse_args()
    main(args)
