from scene import Scene
from pxr import Usd
import warp as wp
import tqdm
from abc import ABC, abstractmethod
import os
import warp.sim.render

wp.init()


class SceneViewer(ABC):
    def __init__(self, scene: Scene, path: str):
        self._scene = scene
        self._path = path
        self._prepare_path()
        stage = Usd.Stage.CreateNew(f"{self.path}/scene_{self.scene_num}.usd")
        self.usd = wp.render.UsdRenderer(stage)

    def _prepare_path(self):
        os.makedirs(f"{self._path}/scene", exist_ok=True)
        self.scene_num = len(os.listdir(f"{self._path}/scene")) + 1
        self.path = f"{self._path}/scene/{self.scene_num}"
        os.makedirs(self.path, exist_ok=True)

    def record_frame(self, time: float = 0.0):
        with wp.ScopedTimer("render", print=False):
            self.usd.begin_frame(time)
            for mesh in [self._scene.gripper, self._scene.object]:
                self.usd.render_mesh(
                    name=mesh.properties.name,
                    points=mesh.nodes.position.detach().cpu().numpy(),
                    indices=mesh.elements.triangles.detach().cpu().numpy().flatten(),
                )
            self.usd.end_frame()

    @abstractmethod
    def record(self):
        pass

    def save(self):
        self.usd.save()


class StaticSceneViewer(SceneViewer):
    def record(self):
        self.record_frame()
        return self


class DynamicSceneViewer(SceneViewer):
    def __init__(self, path: str, scene: Scene):
        super().__init__(path, scene)

    def record(self):
        time = 0.0
            self.record_frame(time)
            self.simulation.update_one_segment()
            self.simulation.bind_states_between_segments()
            time += self.simulation._dt * self.simulation._num_substeps
        return self
