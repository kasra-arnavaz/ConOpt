from abc import ABC, abstractmethod
import torch
from typing import List
import sys
sys.path.append("src")
from simulation.simulation_properties import SimulationProperties
from functools import cached_property

class PullRatio(ABC):

    @abstractmethod
    def update(self) -> List[torch.Tensor]:
        pass

    @cached_property
    def pull_ratio(self):
        return self.update()

    def __init__(self):
        self.index = 0
        self.ascending = True

    def __iter__(self):
        return self
    
    def __next__(self):
        item = self.pull_ratio[self.index]
        if self.ascending:
            self.index += 1
            if self.index == len(self.pull_ratio):
                self.ascending = False
                self.index -= 1
        else:
            self.index -= 1
            if self.index < 0:
                self.ascending = True
                self.index = 0
        return item



class TimeInvariablePullRatio(PullRatio):

    def __init__(self, simulation_properties: SimulationProperties, pull_ratio: torch.Tensor = None, device: str = "cuda"):
        self._pull_ratio = pull_ratio if pull_ratio is not None else torch.tensor(0.0, device=device, requires_grad=True)
        self._num_steps = simulation_properties.num_steps
        super().__init__()

    @property
    def optimizable(self):
        return [self._pull_ratio]

    def update(self):
        return [self._pull_ratio] * self._num_steps
    

class TimeVariablePullRatio(PullRatio):

    def __init__(self, simulation_properties: SimulationProperties, pull_ratio: List[torch.Tensor] = None, device: str = "cuda"):
        self._time = simulation_properties.key_timepoints
        self._pull_ratio = pull_ratio if pull_ratio is not None else [torch.tensor(0.0, device=device, requires_grad=True) for _ in range(len(self._time))]
        self._dt = simulation_properties.dt
        self._device = device
        super().__init__()

    @property
    def optimizable(self):
        return self._pull_ratio

    def update(self):
        tensor = torch.empty(0, device=self._device)
        optimizable = self.optimizable
        for i in range(len(self._pull_ratio)-1):
            steps = int((self._time[i+1] - self._time[i]) / self._dt) + 1
            weight = torch.linspace(0.,1.,steps).to(self._device)
            linear = torch.lerp(input=self._pull_ratio[i], end=self._pull_ratio[i+1], weight=weight)[:-1]
            tensor = torch.cat((tensor, linear))
        # index = (self._time/self._dt).to(int)
        index = torch.tensor([0, 200, 399])
        out = []
        for i, t in enumerate(tensor):
            if i in index:
                idx = torch.where(index == i)[0]
                out.append(optimizable[idx])
            else:
                out.append(t)
        return out
