import sys

sys.path.append("src")

from simulation.simulation import Simulation
from simulation.scene import Scene


def update_scene(scene: Scene, simulation: Simulation) -> Scene:
    scene.gripper.nodes.position, scene.gripper.nodes.velocity = simulation(
        scene.gripper.nodes.position, scene.gripper.nodes.velocity
    )
    return scene
