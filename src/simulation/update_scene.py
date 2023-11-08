import sys

sys.path.append("src")

from simulation.simulation import Simulation
from scene.scene import Scene
import tqdm


class UpdateScene:
    def __init__(self, scene: Scene, simulation: Simulation):
        self._scene = scene
        self._simulation = simulation

    def update_scene(self):
        for self.i in tqdm.tqdm(
            range(self._simulation.properties.num_segments),
            "Simulation",
            colour="green",
            leave=False,
        ):
            self._notify_observers()
            self._update_one_segment()
            self._notify_observers() if self._is_last_segment() else None

    def _update_one_segment(self) -> None:
        self._scene.robot.nodes.position, self._scene.robot.nodes.velocity = self._simulation(
            self._scene.robot.nodes.position, self._scene.robot.nodes.velocity
        )

    def _notify_observers(self) -> None:
        for obs in self._scene.observers:
            obs.update()

    def _is_last_segment(self) -> bool:
        return self.i == self._simulation.properties.num_segments - 1
