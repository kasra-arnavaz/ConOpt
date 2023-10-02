import torch
from typing import List
from cable.cable import Cable
from cable.holes import Holes
from cable.pull_ratio import PullRatio

class CableListFactory:
    def __init__(self, holes: List[Holes], pull_ratio: List[PullRatio], stiffness: float, damping: float):
        self._holes = holes
        self._pull_ratio = pull_ratio
        self._stiffness = stiffness
        self._damping = damping

    def create(self) -> List[Cable]:
        return [
            Cable(holes=h, pull_ratio=pr, stiffness=self._stiffness, damping=self._damping)
            for pr, h in zip(self._pull_ratio, self._holes)
        ]
