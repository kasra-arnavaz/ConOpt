import sys

sys.path.append("src")

from simulation.simulation import Simulation


def update_scene(simulation: Simulation) -> None:
    simulation.scene.gripper.nodes.position, simulation.scene.gripper.nodes.velocity = simulation(
        simulation.scene.gripper.nodes.position, simulation.scene.gripper.nodes.velocity
    )