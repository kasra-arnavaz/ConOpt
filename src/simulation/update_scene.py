import sys

sys.path.append("src")

from simulation.simulation import Simulation
from scene.scene import Scene
from scene.scene_viewer import SceneViewer
import tqdm
from objective.loss import ObstacleAvoidanceLoss
from typing import List

def update_scene_one_segment(scene: Scene, simulation: Simulation) -> None:
    scene.robot.nodes.position, scene.robot.nodes.velocity = simulation(
        scene.robot.nodes.position, scene.robot.nodes.velocity
    )


def update_scene(scene: Scene, simulation: Simulation, viewer: SceneViewer = None, obstacle_loss: List[ObstacleAvoidanceLoss] = None) -> None:
    for i in tqdm.tqdm(range(simulation.properties.num_segments), "Simulation", colour="green", leave=False):
        viewer.save(time=i * simulation.properties.segment_duration) if viewer is not None else None
        if obstacle_loss is not None:
            for o_loss in obstacle_loss:
                o_loss.stack_loss()
        update_scene_one_segment(scene=scene, simulation=simulation)
