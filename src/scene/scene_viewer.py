from pxr import Usd
import warp as wp
import os
import warp.sim.render
import sys

sys.path.append("src")
from scene.scene import Scene

wp.init()


class SceneViewer:
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

    def record(self, time: float = 0.0):
        with wp.ScopedTimer("render", print=False):
            self.usd.begin_frame(time)
            for mesh in [self._scene.gripper, self._scene.object]:
                self.usd.render_mesh(
                    name=mesh.properties.name,
                    points=mesh.nodes.position.detach().cpu().numpy(),
                    indices=mesh.elements.triangles.detach().cpu().numpy().flatten(),
                )
            self.usd.end_frame()

    def save(self, time: float):
        self.record(time=time)
        self.usd.save()
