import tqdm
import sys

sys.path.append("src")

from simulation.simulation import Simulation
from objective.loss import Loss
from objective.optimizer import Optimizer
from rendering.visualization import Visualization
from rendering.views import ThreeExteriorViews
from rendering.rendering import ExteriorDepthRendering
from simulation.update_scene import update_scene


class Train:
    def __init__(self, simulation: Simulation, loss: Loss, optimizer: Optimizer, num_iters: int):
        self._simulation = simulation
        self._loss = loss
        self._optimizer = optimizer
        self._num_iters = num_iters
        views = ThreeExteriorViews(distance=0.5, device="cuda")
        self._vis_rendering = ExteriorDepthRendering(
            meshes=[simulation.scene.gripper, simulation.scene.object], views=views
        )

    def run(self, verbose: bool = True):
        for i in tqdm.tqdm(
            range(self._num_iters),
            desc="Training",
            colour="blue",
            leave=False,
            disable=~verbose,
        ):
            # Visualization(self._vis_rendering).show_images()
            update_scene(self._simulation)
            self._loss.backward()
            self._optimizer.step()
            self.print(i)
            # self._optimizer.zero_grad()
            # Visualization(self._vis_rendering).show_images()
            self._simulation.scene.reset()

    def print(self, iteration):
        print(f"Iter: {iteration}")
        print(f"Loss: {self._loss.get_loss()}")
        print(f"Grad: {self._optimizer._variables.gradients}")
        print(f"Alpha: {self._optimizer._variables.parameters}")
