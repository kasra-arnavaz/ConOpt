import torch
from attrs import define, field

@define(frozen=True)
class Barycentric:
    HxN: torch.Tensor = field()
    NxH: torch.Tensor = field()

    @HxN.validator
    def _check_sum_to_one(self, attribute, value):
        ones = torch.ones(value.shape[0], dtype=torch.float32, device=value.device)
        HxN_sums_to_one = torch.isclose(value.sum(dim=-1), ones, atol=1e-8).all()
        if not HxN_sums_to_one:
            raise ValueError("Expected HxN to sum to one across dim 1.")