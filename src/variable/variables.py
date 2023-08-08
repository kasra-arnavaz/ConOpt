from typing import List
import torch


class Variables:

    def __init__(self):
        self.parameters = []

    def add_parameter(self, p: torch.Tensor):
        if not isinstance(p, torch.Tensor):
            raise ValueError("Only torch.Tensor objects can be added to <parameters>.")
        self.parameters.append(p.requires_grad_())
    
    @property
    def gradients(self):
        return [parameter.grad for parameter in self.parameters]
