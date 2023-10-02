from abc import ABC, abstractmethod
import torch
from typing import List


class PullRatio(ABC):

    @abstractmethod
    def get(self) -> List[torch.Tensor]:
        pass

    def __init__(self):
        self.index = 0
        self.direction = 1

    def __iter__(self):
        return self

    def __next__(self):
        data = self.get()
        item = data[self.index]
        if self.index == len(data) - 1:
            self.direction = -1
        elif self.index == 0 and self.direction == -1:
            self.direction = 1
        self.index += self.direction
        return item


class TimeInvariablePullRatio(PullRatio):

    def __init__(self, num_steps: int, pull_ratio: torch.Tensor = None, device: str = "cuda"):
        super().__init__()
        self._pull_ratio = pull_ratio or torch.zeros(1, device=device)
        self._num_steps = num_steps
    @property
    def optimizable(self):
        return [self._pull_ratio]

    def get(self):
        return [self._pull_ratio] * self._num_steps
    
class TimeVariablePullRatio(PullRatio):

    def __init__(self, time: List[torch.Tensor], dt: float, pull_ratio: List[torch.Tensor] = None, device: str = "cuda"):
        super().__init__()
        self._pull_ratio = pull_ratio or [torch.zeros(1, device=device) for _ in range(len(time))]
        self._time = time
        self._dt = dt
        self._device = device

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
