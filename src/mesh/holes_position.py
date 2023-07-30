import os
import torch
import json
import numpy as np
from pathlib import Path
from typing import List
from mesh.scad import Scad


class HolesPositionWhenUnloaded:
    def __init__(self, scad: Scad):
        self._scad = scad
        self._PATH = Path(".tmp")
        self._PATH.mkdir(exist_ok=True)

    def get(self) -> List[torch.Tensor]:
        self._create_echo_file()
        return self._sort_holes_position_ascending_by_height()

    def _create_echo_file(self):
        os.system(f"openscad {self._scad.file} -o {self._get_echo_file()} -p {self._scad.parameters} -P firstSet")

    def _sort_holes_position_ascending_by_height(self):
        cables_holes = self._separate_holes_for_each_cable()
        holes_sorted = []
        for cable_holes in cables_holes:
            idx = torch.sort(cable_holes[:, 2], descending=False).indices
            holes_sorted.append(cable_holes[idx])
        return holes_sorted

    def _get_echo_file(self):
        return self._PATH / "hole.echo"

    def _separate_holes_for_each_cable(self):
        holes = self._sort_holes_position_by_cable()
        num_holes_per_cable = self._get_num_holes_per_cable()
        return torch.split(holes, num_holes_per_cable)

    def _sort_holes_position_by_cable(self):
        holes_position = self._get_unsorted_holes_position()
        idx = torch.sort(holes_position[:, 1], descending=False).indices
        return holes_position[idx]

    def _get_unsorted_holes_position(self):
        file = np.loadtxt(self._get_echo_file(), dtype=str, delimiter=",").tolist()
        for line in file:
            line[0] = line[0].replace("ECHO:", "")
        return 0.1 * torch.from_numpy(np.array(file, dtype=np.float32))  # 0.1 for mm to cm

    def _get_num_holes_per_cable(self):
        return (2 * self._get_cable_length()).tolist()

    def _get_cable_length(self):
        num_holes_per_block = self._get_number_of_holes_per_block()
        cable_length = torch.zeros((len(num_holes_per_block), num_holes_per_block[0]))
        for i, nc in enumerate(num_holes_per_block):
            cable_length[i, :nc] = torch.ones(nc)
        return cable_length.sum(dim=0).to(dtype=int)

    def _get_number_of_holes_per_block(self):
        with open(self._scad.parameters, "r") as f:
            num_holes = json.load(f)["parameterSets"]["firstSet"]["Num_holes"]
        return list(map(int, num_holes[1:-1].split(",")))
