import torch
from typing import List
from cable.cable import Cable
from cable.holes import Holes


class CableFactory:
    def __init__(
        self, stiffness: float, damping: float, pull_ratio: List[torch.Tensor], holes: List[Holes], device: str = "cuda"
    ):
        self._stiffness = stiffness
        self._damping = damping
        self._pull_ratio = pull_ratio
        self._holes = holes
        self._device = device

    def create(self) -> List[Cable]:
        return [
            Cable(holes, pull_ratio, self._stiffness, self._damping)
            for holes, pull_ratio in zip(self._holes, self._pull_ratio)
        ]
