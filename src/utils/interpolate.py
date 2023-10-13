import torch
from typing import List

def linear_interpolate_between_two_points(x0, x1, y0, y1, x) -> torch.Tensor:
    weight = ((x - x0)/(x1 - x0)).to(dtype=torch.float32)
    return torch.lerp(y0, y1, weight=weight)

def linear_interpolate(xs, ys, x) -> List[torch.Tensor]:
    out = []
    for i in range(len(xs) - 1):
        mask = (x >=xs[i]) & (x<xs[i+1])
        for xi in x[mask]:
            linear = linear_interpolate_between_two_points(xs[i], xs[i+1], ys[i], ys[i+1], xi)
            out.append(linear)
    return out