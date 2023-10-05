import torch


class Variables:
    def __init__(self):
        self.parameters = []

    def add_parameter(self, p: torch.Tensor):
        if not isinstance(p, torch.Tensor):
            raise ValueError("Only torch.Tensor objects can be added to <parameters>.")
        if not p.requires_grad:
            raise ValueError("Only tensors with requires_grad=True can be added as <parameters>.")
        self.parameters.append(p)

    def set_gradients(self):
        self.gradients = [parameter.grad for parameter in self.parameters]

    def __len__(self):
        return len(self.parameters)
