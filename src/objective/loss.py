from abc import ABC, abstractmethod
import torch
import sys

sys.path.append("src")

from rendering.rendering import InteriorGapRendering
from scene.scene import Scene


class Loss(ABC):
    @abstractmethod
    def get_loss(self) -> torch.Tensor:
        pass

    def backward(self):
        self.get_loss().backward()


class ToyLoss(Loss):
    def __init__(self, scene: Scene):
        self._scene = scene

    def get_loss(self):
        return self._scene.gripper.nodes.position.sum()


class MaxGripLoss(Loss):
    def __init__(self, rendering: InteriorGapRendering, device: str = "cuda"):
        self._rendering = rendering
        self._device = device

    def get_loss(self):
        images = self._rendering.get_images()
        loss = torch.zeros(1, requires_grad=True, device=self._device)
        for image in images:
            loss = loss + 0.5 * (image**2).sum()
        return loss
