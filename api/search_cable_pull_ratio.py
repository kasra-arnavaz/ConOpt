import torch
import sys

sys.path.append("src")
from pathlib import Path
from scene.scene_factory import TouchSceneFactory
from mesh.mesh_properties import MeshProperties
from simulation.simulation import Simulation
from point.transform import Transform, get_quaternion
from simulation.simulation_properties import SimulationProperties
from scene.scene_viewer import SceneViewer
from config.config import Config
import argparse
from utils.path import get_next_numbered_path
from simulation.update_scene import update_scene
from warp_wrapper.contact_properties import ContactProperties
from cable.pull_ratio import TimeInvariablePullRatio, TimeVariablePullRatio

def main(args):
    config = Config.from_yaml(args.config)
    DEVICE = config.device
    PATH = get_next_numbered_path(args.path)
    

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
            vector=config.robot_rotation_vector, angle_in_degrees=config.robot_rotation_degrees
        ),
        scale=config.robot_scale,
        device=DEVICE,
    )
    sim_properties = SimulationProperties(
        duration=config.sim_duration, segment_duration=config.sim_segment_duration, dt=config.sim_dt, device=DEVICE
    )
    config.cable_pull_ratio = args.alpha
    cable_pull_ratio = [TimeInvariablePullRatio(pull_ratio=torch.tensor(pull, device=DEVICE), simulation_properties=sim_properties, device=DEVICE) for pull in config.cable_pull_ratio]
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
    # obstacle
    try:
        obstacle_files = [Path(file) for file in config.obstacle_file]
        obstacle_properties = [MeshProperties(name=f"obstacle_{i}", density=density) for i, density in enumerate(config.obstacle_density)]
        obstacle_transforms = []
        for t, rv, rd, s in zip(config.obstacle_translation, config.obstacle_rotation_vector, config.obstacle_rotation_degrees, config.obstacle_scale):
            obstacle_transforms.append(Transform(translation=t, rotation=get_quaternion(vector=rv, angle_in_degrees=rd), scale=s))
    except:
        obstacle_files, obstacle_properties, obstacle_transforms = None, None, None
    contact_properties = ContactProperties(
        distance=config.contact_distance, ke=config.contact_ke, kd=config.contact_kd, kf=config.contact_kf, ground=config.ground
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
        make_new_robot=False
    ).create()

    
    simulation = Simulation(scene=scene, properties=sim_properties)
    viewer = SceneViewer(scene=scene, path=PATH)
    try:
        update_scene(scene=scene, simulation=simulation, viewer=viewer)
    finally:
        config.to_yaml(path=PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path like with yaml format")
    parser.add_argument("-a", "--alpha", type=float, nargs="+")
    parser.add_argument("-p", "--path", type=str, help="path to where the save the files", default=".dev/scene_search")
    args = parser.parse_args()
    main(args)
