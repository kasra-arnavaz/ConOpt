from abc import ABC, abstractmethod
import torch
from typing import List


class PullRatio(ABC):

    @abstractmethod
    def get(self) -> List[torch.Tensor]:
        pass

    def __init__(self):
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        input_list = self.get()
        if self.index < len(input_list):
            result = input_list[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration


class TimeInvariablePullRatio(PullRatio):

    def __init__(self, pull_ratio: torch.Tensor, num_steps: int):
        super().__init__()
        self._pull_ratio = pull_ratio
        self._num_steps = num_steps

    @property
    def optimizable(self):
        return [self._pull_ratio]

    def get(self):
        return [self._pull_ratio] * self._num_steps
    
class TimeVariablePullRatio(PullRatio):

    def __init__(self, pull_ratio: List[torch.Tensor], time: List[torch.Tensor], dt: float):
        super().__init__()
        self._pull_ratio = pull_ratio
        self._time = time
        self._dt = dt

    @property
    def optimizable(self):
        return self._pull_ratio

    def get(self):
        tensor = torch.empty(0, device=self._pull_ratio[0].device)
        for i in range(len(self._pull_ratio)-1):
            steps = int((self._time[i+1] - self._time[i]) / self._dt)+1
            weight = torch.linspace(0.,1.,steps).to(self._pull_ratio[i].device)
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
