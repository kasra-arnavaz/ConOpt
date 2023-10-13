from abc import ABC, abstractmethod
import torch
from typing import List
import sys
sys.path.append("src")
from simulation.simulation_properties import SimulationProperties
from simulation.segment_iterator import SegmentIterator
from functools import cached_property
from utils.interpolate import linear_interpolate

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
        if len(self._time) != len(self._pull_ratio):
            raise ValueError(f"Expected <pull_ratio> to be of length {len(self._time)}, got {len(self._pull_ratio)}.")
        self._dt = simulation_properties.dt
        self._device = device
        super().__init__(simulation_properties)

    @property
    def optimizable(self):
        return self._pull_ratio

    def update_pull_ratio(self):
        t = torch.arange(0, self._sim_properties.duration, self._dt, device=self._device)
        self.pull_ratio = linear_interpolate(self._time, self._pull_ratio, t)

