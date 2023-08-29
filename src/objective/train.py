import tqdm
import sys

sys.path.append("src")

from simulation.simulation import Simulation
from objective.loss import Loss
from objective.optimizer import Optimizer
from simulation.update_scene import update_scene
from simulation.scene import Scene
from rendering.visualization import Visualization
from rendering.views import ThreeExteriorViews
from rendering.rendering import ExteriorDepthRendering
from pathlib import Path


class Train:
    def __init__(self, simulation: Simulation, scene: Scene, loss: Loss, optimizer: Optimizer, num_iters: int):
        self._simulation = simulation
        self._scene = scene
        self._loss = loss
        self._optimizer = optimizer
        self._num_iters = num_iters
        # views = ThreeExteriorViews(distance=0.5, device="cuda")
        # self._vis_rendering = ExteriorDepthRendering(scene=simulation.scene, views=views)

    def run(self, verbose: bool = True):
        for i in tqdm.tqdm(
            range(self._num_iters),
            desc="Training",
            colour="blue",
            disable=verbose,
        ):
            # Visualization(self._vis_rendering).save_images(Path(f"log/before_{i}.png"))
            scene = update_scene(scene=self._scene, simulation=self._simulation)
            self._loss.backward()
            self._optimizer.step()
            self.print(i) if verbose else None
            self._optimizer.zero_grad()
            # Visualization(self._vis_rendering).save_images(Path(f"log/after_{i}.png"))
            scene.reset()

    def print(self, iteration):
        print(f"Iter: {iteration}")
        print(f"Loss: {self._loss.get_loss()}")
        print(f"Grad: {self._optimizer._variables.gradients}")
        print(f"Alpha: {self._optimizer._variables.parameters}")
