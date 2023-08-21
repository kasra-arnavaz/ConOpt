from typing import List
import torch


class Variables:
    def __init__(self):
        self.parameters = []

    def add_parameter(self, p: torch.Tensor):
        if not isinstance(p, torch.Tensor):
            raise ValueError("Only torch.Tensor objects can be added to <parameters>.")
        self.parameters.append(p.requires_grad_())

    def set_gradients(self):
        self.gradients = [parameter.grad for parameter in self.parameters]

    def __len__(self):
        return len(self.parameters)
