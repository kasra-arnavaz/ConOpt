import torch
from typing import List
from abc import ABC, abstractmethod
from cable.holes import Holes


class HolesFactory(ABC):
    @abstractmethod
    def create(self) -> List[Holes]:
        pass


class HolesFactoryFromListOfPositions(HolesFactory):
    def __init__(self, list_of_positions: List[torch.Tensor], device: str = "cuda"):
        self._list_of_positions = list_of_positions
        self._device = device

    def create(self):
        return [Holes(position.to(device=self._device).requires_grad_()) for position in self._list_of_positions]
