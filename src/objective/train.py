import tqdm
import sys

sys.path.append("src")

from simulation.simulation import Simulation
from objective.loss import Loss
from objective.optimizer import Optimizer
from simulation.update_scene import update_scene
from scene.scene import Scene
from objective.log import Log
from rendering.visual import Visual
from scene.scene_viewer import SceneViewer


class Train:
    def __init__(
        self,
        simulation: Simulation,
        scene: Scene,
        loss: Loss,
        optimizer: Optimizer,
        num_iters: int,
        log: Log = None,
        visual: Visual = None,
        scene_viewer: SceneViewer = None,
    ):
        self._simulation = simulation
        self._scene = scene
        self._loss = loss
        self._optimizer = optimizer
        self._num_iters = num_iters
        self._log = log
        self._visual = visual
        self._scene_viewer = scene_viewer

    def run(self, verbose: bool = True):
        for self.i in tqdm.tqdm(
            range(self._num_iters),
            desc="Training",
            colour="blue",
            disable=verbose,
        ):
            self._visual.save_images(str(self.i)) if self._save_first_visuals() else None
            update_scene(scene=self._scene, simulation=self._simulation, viewer=self._scene_viewer)
            self._visual.save_images(str(self.i + 1)) if self._save_visuals() else None
            self._loss.backward()
            self._optimizer.step()
            self.print() if verbose else None
            self._log.save() if self._log is not None else None
            self._optimizer.zero_grad()
            self._scene.reset()

    def _save_visuals(self):
        return self._visual is not None

    def _save_first_visuals(self):
        return self._save_visuals() and self.i == 0

    def print(self):
        print(f"Iter: {self.i}")
        print(f"Loss: {self._loss.get_loss()}")
        print(f"Grad: {self._optimizer._variables.gradients}")
        print(f"Alpha: {self._optimizer._variables.parameters}")
