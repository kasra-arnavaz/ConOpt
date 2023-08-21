import torch
from abc import ABC, abstractmethod
from objective.loss import Loss
from objective.variables import Variables


class Optimizer(ABC):
    def __init__(self, loss: Loss, variables: Variables):
        self._loss = loss
        self._variables = variables

    @abstractmethod
    def step(self):
        pass


class GradientDescent(Optimizer):
    def __init__(self, loss: Loss, variables: Variables, learning_rate: float):
        super().__init__(loss, variables)
        self._learning_rate = learning_rate

    @torch.no_grad()
    def step(self):
        self._variables.set_gradients()
        for i in range(len(self._variables)):
            self._variables.parameters[i] = (
                self._variables.parameters[i] - self._variables.gradients[i] * self._learning_rate
            )


class Adam(Optimizer):
    def __init__(self, loss: Loss, variables: Variables, learning_rate: float):
        super().__init__(loss, variables)
        self._learning_rate = learning_rate
        self._betas = [0.9, 0.99]
        self._eps = 1e-07
        self._t = 0

    @torch.no_grad()
    def step(self):
        self._variables.set_gradients()
        for i in range(len(self._variables)):
            self._m = torch.zeros_like(self._variables.parameters[i])
            self._v = torch.zeros_like(self._variables.parameters[i])
            self._m = self._betas[0] * self._m + (1.0 - self._betas[0]) * self._variables.gradients[i]
            self._v = (
                self._betas[1] * self._v
                + (1.0 - self._betas[1]) * self._variables.gradients[i] * self._variables.gradients[i]
            )
            mhat = self._m / (1.0 - (self._betas[0] ** (self._t + 1.0)))
            vhat = self._v / (1.0 - (self._betas[1] ** (self._t + 1.0)))
            self._variables.parameters[i] = self._variables.parameters[i] - self._learning_rate * mhat / (
                torch.sqrt(vhat) + self._eps
            )
        self._t += 1
