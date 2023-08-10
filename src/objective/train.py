import tqdm
import sys
sys.path.append("src")

from simulation.simulation import Simulation
from objective.loss import Loss
from objective.optimizer import Optimizer
from rendering.visualization import Visualization
from rendering.views import ThreeExteriorViews
from rendering.rendering import ExteriorDepthRendering

class Train:
    def __init__(self, simulation: Simulation, loss: Loss, optimizer: Optimizer, num_iters: int):
        self._simulation = simulation
        self._loss = loss
        self._optimizer = optimizer
        self._num_iters = num_iters

    def run(self, verbose: bool = True):
        for i in tqdm.tqdm(range(self._num_iters), desc="Training", colour="blue", leave=False, disable=~verbose):
            views = ThreeExteriorViews(distance=0.5, device="cuda")
            rendering = ExteriorDepthRendering(meshes=[self._simulation._mesh, self._simulation._object_mesh], views=views, device="cuda")
            Visualization(rendering).show_images()
            self._simulation.run()
            self._loss.backward()
            Visualization(rendering).show_images()
            self._optimizer.step()
            self.print(i)
            # self._simulation.reset_states()

    def print(self, iteration):
        print(f"Iter: {iteration}")
        print(f"Loss: {self._loss.get_loss()}")
        print(f"Grad: {self._optimizer._variables.gradients}")
        print(f"Alpha: {self._optimizer._variables.parameters}")