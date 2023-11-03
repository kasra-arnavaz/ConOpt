from abc import ABC, abstractmethod
import torch
import sys

sys.path.append("src")

from rendering.rendering import InteriorGapRendering, InteriorContactRendering, InteriorDistanceRendering
from scene.scene import Scene, TouchScene
from typing import List
from objective.variables import Variables


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
    def __init__(self, rendering: InteriorDistanceRendering, device: str = "cuda"):
        self._rendering = rendering
        self._device = device
        self.loss = torch.zeros(1, requires_grad=True, device=self._device)

    @property
    def rendering(self):
        return self._rendering
       
    def get_loss(self):
        loss = torch.zeros(1, requires_grad=True, device=self._device)
        images = self._rendering.get_images()
        for image in images:
            loss = loss + torch.mean(image)
        return loss/len(images)
    
    def stack_loss(self):
        self.loss = self.loss + self.get_loss() # try in place

    
class PointTouchWithObstacleAvoidanceLoss(Loss):
    def __init__(self, scene: TouchScene, obstacle_avoidance_losses: List[ObstacleAvoidanceLoss]):
        self._point_touch_loss = PointTouchLoss(scene)
        self.obstacle_avoidance_losses = obstacle_avoidance_losses

    def get_loss(self):
        loss = self._point_touch_loss.get_loss()
        print(f"tracking loss: {loss}")
        for i, obstacle_avoidance_loss in enumerate(self.obstacle_avoidance_losses):
            loss = loss - obstacle_avoidance_loss.loss
            print(f"obstacle {i} loss: {obstacle_avoidance_loss.loss}")
        return loss
    
class LocomotionLoss(Loss):
    def __init__(self, scene: Scene, target_position: torch.Tensor, variables: Variables):
        self._scene = scene
        self._target_position = target_position
        self._variables = variables

    def get_loss(self):
        tracking_loss = self.get_tracking_loss()
        l1_loss = self.get_l1_loss()
        print(f"track loss: {tracking_loss}")
        print(f"L1 loss: {l1_loss}")
        return 10*tracking_loss + 0.01*l1_loss

    def get_tracking_loss(self):
        output_position = self._scene.robot.nodes.position.mean(dim=0)
        return torch.sum((output_position - self._target_position) ** 2)

    def get_l1_loss(self):
        parameters = self._variables.parameters
        loss = torch.zeros(1, requires_grad=True, device=parameters[0].device)
        for p in parameters:
            loss = loss + torch.abs(p)
        return loss/len(parameters)