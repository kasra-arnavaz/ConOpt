import torch
from typing import List
from cable.holes import Holes


class MultiHolesFactory:
    def __init__(self, list_of_positions: List[torch.Tensor], device: str = "cuda"):
        self._list_of_positions = list_of_positions
        self._device = device

    def create(self) -> List[Holes]:
        return [Holes(position.to(device=self._device)) for position in self._list_of_positions]
