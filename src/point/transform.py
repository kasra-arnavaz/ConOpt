import torch
from pytorch3d import transforms
from point.points import Points


class Transform:
    """Transforms points in TRS order (translate, rotate, scale)."""

    def __init__(self, translation: list = None, rotation: list = None, scale: list = None, device: str = "cuda"):
        self._translation = torch.tensor(translation or [0.0, 0.0, 0.0], device=device)
        self._rotation = torch.tensor(rotation or [1.0, 0.0, 0.0, 0.0], device=device)
        self._scale = torch.tensor(scale or [1.0, 1.0, 1.0], device=device)
        self._device = device

    def apply(self, points: Points):
        points.position = self._apply_scale(self._apply_rotation(self._apply_translate(points.position)))

    def _apply_translate(self, point):
        return point + self._translation

    def _apply_rotation(self, point):
        return transforms.quaternion_apply(quaternion=self._rotation, point=point)

    def _apply_scale(self, point):
        return point * self._scale


def get_quaternion(vector: list, angle_in_degrees: float) -> list:
    vector = torch.tensor(vector, dtype=torch.float32)
    vector = vector / vector.norm()
    angle_in_radians = torch.deg2rad(torch.tensor(angle_in_degrees / 2.0))
    real_quaternion = torch.cos(angle_in_radians)[None]
    imaginary_quaternion = torch.sin(angle_in_radians) * vector
    return torch.cat((real_quaternion[None], imaginary_quaternion[None]), dim=1).reshape(-1).tolist()
