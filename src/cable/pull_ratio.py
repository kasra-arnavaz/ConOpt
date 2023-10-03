from abc import ABC, abstractmethod
import torch
from typing import List
import sys
sys.path.append("src")
from simulation.simulation_properties import SimulationProperties


class PullRatio(ABC):

    @abstractmethod
    def get(self) -> List[torch.Tensor]:
        pass

    def __init__(self):
        self.index = 0
        self.direction = 1
        self.data = self.get()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.data[self.index]
        if self.index == len(self.data) - 1:
            self.direction = -1
        elif self.index == 0 and self.direction == -1:
            self.direction = 1
        self.index += self.direction
        return item


class TimeInvariablePullRatio(PullRatio):

    def __init__(self, simulation_properties: SimulationProperties, pull_ratio: torch.Tensor = None, device: str = "cuda"):
        self._pull_ratio = pull_ratio if pull_ratio is not None else torch.tensor(0.0, device=device)
        self._num_steps = simulation_properties.num_steps
        super().__init__()

    @property
    def optimizable(self):
        return [self._pull_ratio]

    def get(self):
        return [self._pull_ratio] * self._num_steps
    
class TimeVariablePullRatio(PullRatio):

    def __init__(self, simulation_properties: SimulationProperties, pull_ratio: List[torch.Tensor] = None, device: str = "cuda"):
        self._time = simulation_properties.key_timepoints
        self._pull_ratio = pull_ratio if pull_ratio is not None else [torch.tensor(0.0, device=device) for _ in range(len(self._time))]
        self._dt = simulation_properties.dt
        self._device = device
        super().__init__()


    @property
    def optimizable(self):
        return self._pull_ratio

    def get(self):
        tensor = torch.empty(0, device=self._device)
        for i in range(len(self._pull_ratio)-1):
            steps = int((self._time[i+1] - self._time[i]) / self._dt)+1
            weight = torch.linspace(0.,1.,steps).to(self._device)
            linear = torch.lerp(input=self._pull_ratio[i], end=self._pull_ratio[i+1], weight=weight)
            if i > 0: linear = linear[1:]
            tensor = torch.cat((tensor, linear))
        index = (torch.tensor(self._time).reshape(-1)/self._dt).to(int)
        out = []
        for i, t in enumerate(tensor):
            if i in index:
                idx = torch.where(index == i)[0]
                out.append(self.optimizable[idx])
            else:
                out.append(t)
        return out
