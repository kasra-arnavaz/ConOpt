import torch
from typing import List
from abc import ABC, abstractmethod
from mesh.holes import Holes


class HolesFactory(ABC):

    @abstractmethod
    def create(self) -> List[Holes]:
        pass

class HolesFactoryFromListOfPositions(HolesFactory):

    def __init__(self, list_of_positions: List[torch.Tensor]):
        self._list_of_positions = list_of_positions

    def create(self):
        return [Holes(position) for position in self._list_of_positions]

