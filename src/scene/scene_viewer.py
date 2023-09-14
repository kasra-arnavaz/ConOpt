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
        os.makedirs(path, exist_ok=True)
        stage = Usd.Stage.CreateNew(f"{path}/scene.usd")
        self.usd = wp.render.UsdRenderer(stage)

    def record(self, time: float = 0.0):
        with wp.ScopedTimer("render", print=False):
            self.usd.begin_frame(time)
            for mesh in self._scene.all_meshes():
                self.usd.render_mesh(
                    name=mesh.properties.name,
                    points=mesh.nodes.position.detach().cpu().numpy(),
                    indices=mesh.elements.triangles.detach().cpu().numpy().flatten(),
                )
            self.usd.end_frame()

    def save(self, time: float):
        self.record(time=time)
        self.usd.save()
