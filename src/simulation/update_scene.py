import sys

sys.path.append("src")

from simulation.simulation import Simulation
from simulation.scene import Scene
from simulation.scene_viewer import SceneViewer
import tqdm


def update_scene_one_segment(scene: Scene, simulation: Simulation) -> None:
    scene.gripper.nodes.position, scene.gripper.nodes.velocity = simulation(
        scene.gripper.nodes.position, scene.gripper.nodes.velocity
    )


def update_scene(scene: Scene, simulation: Simulation, viewer: SceneViewer = None) -> None:
    for i in tqdm.tqdm(range(simulation.properties.num_segments), "Simulation", colour="green", leave=False):
        viewer.save(time=i * simulation.properties.segment_duration) if viewer is not None else None
        update_scene_one_segment(scene=scene, simulation=simulation)
