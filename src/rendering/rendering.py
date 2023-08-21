from abc import ABC, abstractmethod
from typing import List
from pytorch3d import structures, renderer
import sys
import torch

sys.path.append("src")
from mesh.mesh import Mesh
from rendering.views import Views, ExteriorViews, InteriorViews


class DepthRendering(ABC):
    def __init__(self, meshes: List[Mesh], views: Views, device: str = "cuda"):
        self._meshes = meshes
        self._device = device
        self._views = views.get()
        self._znear = 0.001
        self._zfar = 10
        self._fov = 90

    @staticmethod
    def get_images(self) -> List[torch.Tensor]:
        pass

    def _get_meshes(self):
        meshes = [structures.Meshes(mesh.nodes.position[None], mesh.elements.triangles[None]) for mesh in self._meshes]
        return structures.join_meshes_as_batch(meshes)

    def _get_cameras(self):
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

    def _get_rasterizers(self):
        settings = renderer.RasterizationSettings(image_size=1000, blur_radius=0.0, faces_per_pixel=1, bin_size=0)
        return [renderer.MeshRasterizer(cameras=camera, raster_settings=settings) for camera in self._get_cameras()]

    def _get_zbuf(self):
        return [rasterizer(self._get_meshes()).zbuf for rasterizer in self._get_rasterizers()]


class ExteriorDepthRendering(DepthRendering):
    def __init__(self, meshes: List[Mesh], views: ExteriorViews, device: str = "cuda"):
        super().__init__(meshes, views, device)

    def get_images(self) -> List[torch.Tensor]:
        images = []
        LARGE_POSITIVE_NUMBER = 619.0
        for zbuf in self._get_zbuf():
            zbuf[zbuf == -1.0] = LARGE_POSITIVE_NUMBER
            buffers = zbuf.amin(dim=0)
            buffers[buffers == LARGE_POSITIVE_NUMBER] = -1.0
            images.append(buffers)
        return images


class InteriorDepthRendering(DepthRendering):
    def __init__(
        self,
        gripper_mesh: Mesh,
        object_mesh: Mesh,
        views: InteriorViews,
        device: str = "cuda",
    ):
        super().__init__([gripper_mesh, object_mesh], views, device)


class InteriorGapRendering(InteriorDepthRendering):
    def get_images(self):
        gaps = []
        for zbuf in self._get_zbuf():
            distance_ = zbuf[0] - zbuf[1]
            distance = distance_.clone()  # to keep gradient flow
            distance[distance < 0] = 0
            gaps.append(distance)
        return gaps


class InteriorContactRendering(InteriorDepthRendering):
    def get_images(self):
        contacts = []
        for zbuf in self._get_zbuf():
            distance = zbuf[1] - zbuf[0]
            mask_contact = distance >= 0
            mask_object = zbuf[1] > -1.0
            mask_gripper = zbuf[0] > -1.0
            mask = mask_contact * mask_gripper * mask_object * 1.0
            contacts.append(mask)
        return contacts
