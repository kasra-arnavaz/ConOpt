import torch
from attrs import define, field


@define
class Points:
    position: torch.Tensor = field(on_setattr=lambda self, attribute, value: value.requires_grad_())
    velocity: torch.Tensor = field(init=False, on_setattr=lambda self, attribute, value: value.requires_grad_())
    force: torch.Tensor = field(init=False, on_setattr=lambda self, attribute, value: value.requires_grad_())

    def __len__(self):
        return self.position.shape[0]

    @velocity.default
    @force.default
    def _set_zero_by_default(self):
        return torch.zeros_like(self.position, requires_grad=True)

    @position.validator
    def _validate_position(self, attribute, value):
        self._validate_common_traits_of_attributes(name=attribute.name, value=value)

    @velocity.validator
    @force.validator
    def _validate_velocity(self, attribute, value):
        self._validate_common_traits_of_attributes(name=attribute.name, value=value)
        self._validate_equal_shape_with_position(name=attribute.name, value=value)

    def _validate_common_traits_of_attributes(self, name, value):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected <{name}> to be of type torch.Tensor, got {type(value)}.")
        if len(value.shape) != 2:
            raise ValueError(f"Expected <{name}> to be a 2D tensor, got {len(value.shape)}D tensor.")
        if value.shape[-1] != 3:
            raise ValueError(f"Expected the last dimension of <{name}> to be of size 3, got {value.shape[-1]}.")
        if value.dtype != torch.float32:
            raise ValueError(f"Expected dtype of <{name}> to be torch.float32, got {value.dtype}.")
        if torch.any(torch.isnan(value)):
            raise RuntimeError(f"Some of the values of <{name}> are NaN.")
        if value.requires_grad == False:
            raise RuntimeError(f"Expected <{name}> to require grad.")

    def _validate_equal_shape_with_position(self, name, value):
        if value.shape != self.position.shape:
            raise ValueError(f"Expected <{name}> to have the same shape as <position>, got {value.shape})")
