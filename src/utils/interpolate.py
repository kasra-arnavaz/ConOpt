import torch
from typing import List

def linear_interpolate_between_two_points(x0, x1, y0, y1, x, device: str = "cuda") -> torch.Tensor:
    x0, x1, y0, y1 = x0.to(device), x1.to(device), y0.to(device), y1.to(device)
    x = x.to(device)
    m = (y1-y0)/(x1-x0)
    return m*(x-x0) + y0

def linear_interpolate(xs, ys, x, device: str = "cuda") -> List[torch.Tensor]:
    out = []
    for i in range(len(xs) - 1):
        mask = (x >=xs[i]) & (x<xs[i+1])
        for xi in x[mask]:
            linear = linear_interpolate_between_two_points(xs[i], xs[i+1], ys[i], ys[i+1], xi, device)
            out.append(linear)
    return out