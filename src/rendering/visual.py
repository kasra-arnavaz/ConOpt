from rendering.rendering import DepthRendering
import matplotlib.pyplot as plt
import os
from os import PathLike


class Visual:
    def __init__(self, rendering: DepthRendering, path: str, prefix: str):
        self._rendering = rendering
        self._path = path
        self._prefix = prefix

    def _get_images(self):
        return self._rendering.get_images()

    def show_images(self):
        self.get_figures()
        plt.show()

    def save_images(self, name: str):
        os.makedirs(self._path, exist_ok=True)
        figures = self.get_figures()
        for figure in figures:
            figure.savefig(f"{self._path}/{self._prefix}_{name}.png", bbox_inches="tight")

    def get_figures(self):
        figures = []
        num_columns = 3
        images = self._get_images()
        num_rows = int(len(images) // num_columns)
        fig = plt.figure(figsize=(10 * num_columns, 10*num_rows))
        for i, image in enumerate(images):
            plt.subplot(num_rows, num_columns, i + 1)
            plt.imshow(image.detach().cpu().numpy(), cmap="gray")
            plt.axis("off")
            figures.append(fig)
        return figures
