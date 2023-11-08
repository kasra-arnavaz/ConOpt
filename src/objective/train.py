import tqdm
import sys

sys.path.append("src")

from objective.loss import Loss
from objective.optimizer import Optimizer
from simulation.update_scene import UpdateScene
from objective.log import Log
from scene.scene import Scene
import torch


class Train:
    def __init__(
        self,
        scene: Scene,
        update_scene: UpdateScene,
        loss: Loss,
        optimizer: Optimizer,
        num_iters: int,
        log: Log = None,
    ):
        self._scene = scene
        self._update_scene = update_scene
        self._loss = loss
        self._optimizer = optimizer
        self._num_iters = num_iters
        self._log = log

    def run(self, verbose: bool = True):
        for self.i in tqdm.tqdm(
            range(self._num_iters),
            desc="Training",
            colour="blue",
            disable=verbose,
        ):
            self._update_scene.update_scene()
            self._loss.backward()
            self._optimizer.step()
            self.print() if verbose else None
            self._log.save() if self._log is not None else None
            self._optimizer.zero_grad()
            try:
                for loss in self._loss.obstacle_avoidance_losses:
                    loss.loss = torch.zeros(1, requires_grad=True, device=loss.loss.device)
            except:
                pass
            self._scene.reset()

    def print(self):
        print(f"Iter: {self.i+1}")
        print(f"Loss: {self._loss.get_loss()}")
        print(f"Grad: {[round(g.item(), 3) for g in self._optimizer._variables.gradients]}")
        print(f"Alpha: {[round(p.item(), 3) for p in self._optimizer._variables.parameters]}")
