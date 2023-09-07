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
from typing import List


class Train:
    def __init__(
        self,
        simulation: Simulation,
        scene: Scene,
        loss: Loss,
        optimizer: Optimizer,
        num_iters: int,
        log: Log = None,
        visuals: List[Visual] = None,
    ):
        self._simulation = simulation
        self._scene = scene
        self._loss = loss
        self._optimizer = optimizer
        self._num_iters = num_iters
        self._log = log
        self._visuals = visuals

    def run(self, verbose: bool = True):
        for self.i in tqdm.tqdm(
            range(self._num_iters),
            desc="Training",
            colour="blue",
            disable=verbose,
        ):
            for visual in self._visuals:
                visual.save_images(str(self.i)) if visual is not None and self.i == 0 else None
            update_scene(scene=self._scene, simulation=self._simulation)
            for visual in self._visuals:
                visual.save_images(str(self.i + 1)) if visual is not None else None
            self._loss.backward()
            self._optimizer.step()
            self.print() if verbose else None
            self._log.save() if self._log is not None else None
            self._optimizer.zero_grad()
            self._scene.reset()

    def print(self):
        print(f"Iter: {self.i+1}")
        print(f"Loss: {self._loss.get_loss()}")
        print(f"Grad: {self._optimizer._variables.gradients}")
        print(f"Alpha: {self._optimizer._variables.parameters}")
