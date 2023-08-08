from abc import ABC, abstractmethod
from typing import List
import torch
from pytorch3d import renderer


class Views(ABC):
    @abstractmethod
    def get(self) -> List[torch.Tensor]:
        pass


class ExteriorViews(Views):
    def __init__(self, distance: float = 0.5, device: str = "cuda"):
        self._distance = distance
        self._device = device


class ThreeExteriorViews(ExteriorViews):
    def get(self):
        x_view = renderer.look_at_view_transform(dist=self._distance, elev=0, azim=90, device=self._device)
        y_view = renderer.look_at_view_transform(dist=self._distance, elev=90, azim=0, device=self._device)
        z_view = renderer.look_at_view_transform(dist=self._distance, elev=0, azim=0, device=self._device)
        return [x_view, y_view, z_view]


class SixExteriorViews(ExteriorViews):
    def get(self):
        x_view = renderer.look_at_view_transform(dist=self._distance, elev=0, azim=90, device=self._device)
        y_view = renderer.look_at_view_transform(dist=self._distance, elev=90, azim=0, device=self._device)
        z_view = renderer.look_at_view_transform(dist=self._distance, elev=0, azim=0, device=self._device)
        neg_x_view = renderer.look_at_view_transform(dist=self._distance, elev=0, azim=-90, device=self._device)
        neg_y_view = renderer.look_at_view_transform(dist=self._distance, elev=-90, azim=0, device=self._device)
        neg_z_view = renderer.look_at_view_transform(dist=self._distance, elev=180, azim=180, device=self._device)
        return [x_view, y_view, z_view, neg_x_view, neg_y_view, neg_z_view]


class InteriorViews(Views):
    def __init__(self, center: torch.Tensor, device: str = "cuda"):
        self._center = center
        self._device = device


class ThreeInteriorViews(InteriorViews):
    def get(self):
        views = []
        for at, up in zip(self._at, self._up):
            views.append(
                renderer.look_at_view_transform(
                    eye=self._center[None], at=(tuple(at),), up=(tuple(up),), device=self._device
                )
            )
        return views

    @property
    def _at(self) -> list:
        EPS = 1e-5
        center = torch.tile(self._center, (3, 1)).cpu()
        eye = torch.eye(3)
        direction = EPS * eye
        return (center + direction).tolist()

    @property
    def _up(self) -> list:
        y = [0, 1, 0]
        x = [1, 0, 0]
        return [y, x, y]


class SixInteriorViews(InteriorViews):
    def get(self):
        views = []
        for at, up in zip(self._at, self._up):
            views.append(
                renderer.look_at_view_transform(
                    eye=self._center[None], at=(tuple(at),), up=(tuple(up),), device=self._device
                )
            )
        return views

    @property
    def _at(self) -> list:
        EPS = 1e-5
        center = torch.tile(self._center, (6, 1)).cpu()
        eye = torch.eye(3)
        direction = torch.cat([EPS * eye, -EPS * eye])
        return (center + direction).tolist()

    @property
    def _up(self) -> list:
        y = [0, 1, 0]
        x = [1, 0, 0]
        return [y, x, y, y, x, y]
