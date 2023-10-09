from abc import ABC, abstractmethod
import torch
from typing import List
import sys
sys.path.append("src")
from simulation.simulation_properties import SimulationProperties
from simulation.segment_iterator import SegmentIterator
from functools import cached_property

class PullRatio(ABC):

    def __init__(self, sim_properties: SimulationProperties):
        self._sim_properties = sim_properties
        self.update_pull_ratio()
        self.update_iterator()

    @abstractmethod
    def update_pull_ratio(self) -> None:
        pass

    def update_iterator(self) -> None:
        self.iterator = SegmentIterator(lst=self.pull_ratio, num_segments=self._sim_properties.num_segments)

class TimeInvariablePullRatio(PullRatio):

    def __init__(self, simulation_properties: SimulationProperties, pull_ratio: torch.Tensor = None, device: str = "cuda"):
        self._pull_ratio = pull_ratio if pull_ratio is not None else torch.tensor(0.0, device=device, requires_grad=True)
        self._num_steps = simulation_properties.num_steps
        super().__init__(simulation_properties)

    @property
    def optimizable(self):
        return [self._pull_ratio]

    def update_pull_ratio(self):
        self.pull_ratio = [self._pull_ratio] * self._num_steps
    

class TimeVariablePullRatio(PullRatio):

    def __init__(self, simulation_properties: SimulationProperties, pull_ratio: List[torch.Tensor] = None, device: str = "cuda"):
        self._time = simulation_properties.key_timepoints
        self._pull_ratio = pull_ratio if pull_ratio is not None else [torch.tensor(0.0, device=device, requires_grad=True) for _ in range(len(self._time))]
        self._dt = simulation_properties.dt
        self._device = device
        super().__init__(simulation_properties)

    @property
    def optimizable(self):
        return self._pull_ratio

    def update_pull_ratio(self):
        tensor = torch.empty(0, device=self._device)
        optimizable = self.optimizable
        for i in range(len(self._pull_ratio)-1):
            steps = int((self._time[i+1] - self._time[i]) / self._dt) + 1
            weight = torch.linspace(0.,1.,steps).to(self._device)
            linear = torch.lerp(input=self._pull_ratio[i], end=self._pull_ratio[i+1], weight=weight)[:-1]
            linear[0] = self._pull_ratio[i]
            linear[-1] = self._pull_ratio[i+1]
            tensor = torch.cat((tensor, linear))
        self.pull_ratio = [t for t in tensor]
