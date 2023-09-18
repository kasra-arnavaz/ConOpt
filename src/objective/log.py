import os
import torch
import sys

sys.path.append("src")
import copy
from objective.loss import Loss
from objective.variables import Variables
import matplotlib.pyplot as plt


class Log:
    def __init__(self, loss: Loss, variables: Variables, path: str):
        self._loss = loss
        self._variables = variables
        self._path = path
        self.loss_log, self.gradient_log, self.parameters_log = [], [], []
        os.makedirs(path, exist_ok=True)

    def _append_log(self):
        self.loss_log.append(self._loss.get_loss().detach().cpu().tolist())
        self.gradient_log.append(copy.deepcopy(self._variables.gradients))
        self.parameters_log.append(copy.deepcopy(self._variables.parameters))

    def save(self):
        self._append_log()
        torch.save(torch.tensor(self.loss_log), f"{self._path}/loss.pt")
        torch.save(torch.tensor(self.gradient_log), f"{self._path}/gradient.pt")
        torch.save(torch.tensor(self.parameters_log), f"{self._path}/parameter.pt")

    @staticmethod
    def plot(path):
        loss = torch.load(f"{path}/loss.pt")
        gradient = torch.load(f"{path}/gradient.pt")
        parameter = torch.load(f"{path}/parameter.pt")
        for tensor, label in zip([loss, gradient, parameter], ["Loss", "Gradient", "Parameter"]):
            plt.figure()
            plt.plot(tensor)
            plt.xlabel("# Iterations")
            plt.ylabel(label)
        plt.show()

    @staticmethod
    def print(path):
        loss = torch.load(f"{path}/loss.pt")
        gradient = torch.load(f"{path}/gradient.pt")
        parameter = torch.load(f"{path}/parameter.pt")
        print(f"Loss\n{loss}")
        print(f"Gradinets\n{gradient}")
        print(f"Parameters\n{parameter}")

    @staticmethod
    def print_iteration(path, i: int):
        loss = torch.load(f"{path}/loss.pt")
        gradient = torch.load(f"{path}/gradient.pt")
        parameter = torch.load(f"{path}/parameter.pt")
        print(f"Iteration {i}")
        print(f"Loss\n{loss[i]}")
        print(f"Gradinets\n{gradient[i]}")
        print(f"Parameters\n{parameter[i]}")
