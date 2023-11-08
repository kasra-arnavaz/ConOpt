from abc import ABC, abstractmethod
import sys


sys.path.append("src")
import matplotlib.pyplot as plt
from rendering.rendering import DepthRendering
import os
import torch

from pxr import Usd
import warp as wp
import warp.sim.render
from simulation.simulation_properties import SimulationProperties

wp.init()


class SceneObserver(ABC):
    @abstractmethod
    def update(self):
        pass


class SceneImages(SceneObserver):
    def __init__(self, rendering: DepthRendering, path: str, prefix: str):
        self._rendering = rendering
        self._path = path
        self._prefix = prefix
        self._name = 0
        self._mean_images = []

    def update(self) -> None:
        self._images = self._rendering.get_images()
        self._mean_images.append(torch.stack(self._images).mean().item())
        self._save_images(name=f"{self._name}")
        self._name += 1

    def save_mean_plot(self) -> None:
        plt.figure()
        plt.plot(self._mean_images)
        plt.savefig(f"{self._path}/{self._prefix}_mean.png")

    def _save_images(self, name: str):
        os.makedirs(self._path, exist_ok=True)
        figures = self._get_figures()
        for figure in figures:
            figure.savefig(f"{self._path}/{self._prefix}_{name}.png", bbox_inches="tight")

    def _get_figures(self):
        figures = []
        num_columns = 3
        num_rows = int(len(self._images) // num_columns)
        fig = plt.figure(figsize=(10 * num_columns, 10 * num_rows))
        for i, image in enumerate(self._images):
            plt.subplot(num_rows, num_columns, i + 1)
            plt.imshow(image.detach().cpu().numpy(), cmap="gray")
            plt.axis("off")
            figures.append(fig)
        return figures


class SceneViewer(SceneObserver):
    def __init__(
        self, scene: "Scene", simulation_properties: SimulationProperties, path: str, speed_factor: float = 1.0
    ):
        self._scene = scene
        os.makedirs(path, exist_ok=True)
        stage = Usd.Stage.CreateNew(f"{path}/scene.usd")
        self.usd = wp.render.UsdRenderer(stage)
        self._speed_factor = speed_factor
        self._time = 0.0
        self._segment_duration = simulation_properties.segment_duration

    def update(self):
        self._record_frame(time=self._time)
        self._time += self._segment_duration
        self.usd.save()

    def _record_frame(self, time: float):
        with wp.ScopedTimer("render", print=False):
            self.usd.begin_frame(time / self._speed_factor)
            for mesh in self._scene.all_meshes():
                self.usd.render_mesh(
                    name=mesh.properties.name,
                    points=mesh.nodes.position.detach().cpu().numpy(),
                    indices=mesh.elements.triangles.detach().cpu().numpy().flatten(),
                )
            self.usd.end_frame()
