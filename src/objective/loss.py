from abc import ABC, abstractmethod
import torch
import sys

sys.path.append("src")

from rendering.rendering import InteriorGapRendering, InteriorContactRendering
from rendering.views import InteriorViews
from scene.scene import Scene, TouchScene
from typing import List


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
        return self._scene.robot.nodes.position.sum()


class MaxGripLoss(Loss):
    def __init__(self, rendering: InteriorGapRendering, device: str = "cuda"):
        #todo: change init to make rendering based on scene and views
        self._rendering = rendering
        self._device = device

    def get_loss(self):
        images = self._rendering.get_images()
        loss = torch.zeros(1, requires_grad=True, device=self._device)
        for image in images:
            loss = loss + 0.5 * (image**2).sum()
        return loss / len(images)


class PointTouchLoss(Loss):
    def __init__(self, scene: TouchScene):
        self._scene = scene

    def get_loss(self):
        output_position = self._scene.robot.nodes.position[self._scene.robot_end_effector_idx]
        target_position = self._scene.object.nodes.position.mean(dim=0)
        return torch.sum((output_position - target_position) ** 2)
    
class ObstacleAvoidanceLoss(Loss):
    def __init__(self, scene: TouchScene, views: InteriorViews, device: str = "cuda"):
        self._rendering = InteriorContactRendering(scene=scene, views=views, device=device)

    def get_loss(self):
        pass
