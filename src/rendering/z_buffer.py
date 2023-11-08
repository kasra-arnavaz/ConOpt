from pytorch3d import structures, renderer
import sys

sys.path.append("src")
from rendering.views import Views
from functools import cached_property
from mesh.mesh import Mesh
import torch


class ZBuffer:
    def __init__(self, mesh: Mesh, views: Views, device: str = "cuda"):
        self._mesh = mesh
        self._device = device
        self._views = views.get()
        self._znear = 0.001
        self._zfar = 10
        self._fov = 90

    @property
    def zbuf(self) -> torch.Tensor:
        return torch.stack([rasterizer(self.mesh).zbuf.squeeze() for rasterizer in self._rasterizers])

    @property
    def mesh(self):
        return structures.Meshes(self._mesh.nodes.position[None], self._mesh.elements.triangles[None])

    @cached_property
    def _cameras(self):
        cameras = []
        for view in self._views:
            cameras.append(
                renderer.FoVPerspectiveCameras(
                    R=view[0],
                    T=view[1],
                    znear=self._znear,
                    zfar=self._zfar,
                    fov=self._fov,
                    device=self._device,
                )
            )
        return cameras

    @cached_property
    def _rasterizers(self):
        settings = renderer.RasterizationSettings(image_size=1000, blur_radius=0.0, faces_per_pixel=1, bin_size=0)
        return [renderer.MeshRasterizer(cameras=camera, raster_settings=settings) for camera in self._cameras]
