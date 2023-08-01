import torch
from attrs import define, field

@define
class Elements:
    triangles: torch.Tensor = field(default=None)
    tetrahedra: torch.Tensor = field(default=None)

    @property
    def num_triangles(self):
        return self.triangles.shape[0] if self.triangles is not None else 0

    @property
    def num_tetrahedra(self):
        return self.tetrahedra.shape[0] if self.tetrahedra is not None else 0

    @triangles.validator
    def _validate_triangles(self, attribute, value):
        if value is not None:
            self._validate_common_traits_of_triangles_and_tetrahedra(name=attribute.name, value=value)
            self._validate_triangles_shape(name=attribute.name, value=value)

    @tetrahedra.validator
    def _validate_tetrahedra(self, attribute, value):
        if value is not None:
            self._validate_common_traits_of_triangles_and_tetrahedra(name=attribute.name, value=value)
            self._validate_tetrahedra_shape(name=attribute.name, value=value)

    def _validate_common_traits_of_triangles_and_tetrahedra(self, name, value):
        self._validate_type(name=name, value=value)
        self._validate_2d_tensor(name=name, value=value)
        self._validate_dtype(name=name, value=value)

    def _validate_type(self, name, value):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected <{name}> to be of type torch.Tensor, got {type(value)}.")

    def _validate_2d_tensor(self, name, value):
        if len(value.shape) != 2:
            raise ValueError(f"Expected <{name}> to be a 2D tensor, got {len(value.shape)}D tensor.")

    def _validate_dtype(self, name, value):
        if value.dtype != torch.int32:
            raise ValueError(f"Expected dtype of <{name}> to be torch.int32, got {value.dtype}")

    def _validate_triangles_shape(self, name, value):
        if value.shape[-1] != 3:
            raise ValueError(f"Expected the last dimension of <{name}> to be of size 3, got {value.shape[-1]}.")

    def _validate_tetrahedra_shape(self, name, value):
        if value.shape[-1] != 4:
            raise ValueError(f"Expected the last dimension of <{name}> to be of size 4, got {value.shape[-1]}.")
