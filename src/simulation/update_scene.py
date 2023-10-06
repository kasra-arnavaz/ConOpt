import sys

sys.path.append("src")

from simulation.simulation import Simulation
from scene.scene import Scene
from scene.scene_viewer import SceneViewer
import tqdm


def update_scene_one_segment(scene: Scene, simulation: Simulation) -> None:
    scene.robot.nodes.position, scene.robot.nodes.velocity, scene.robot.cables[0].alpha, scene.robot.cables[1].alpha, scene.robot.cables[2].alpha = simulation(
        scene.robot.nodes.position, scene.robot.nodes.velocity, scene.robot.cables[0].alpha, scene.robot.cables[1].alpha, scene.robot.cables[2].alpha
    )


def update_scene(scene: Scene, simulation: Simulation, viewer: SceneViewer = None) -> None:
    for i in tqdm.tqdm(range(simulation.properties.num_segments), "Simulation", colour="green", leave=False):
        viewer.save(time=i * simulation.properties.segment_duration) if viewer is not None else None
        update_scene_one_segment(scene=scene, simulation=simulation)
