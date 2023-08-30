from rendering.rendering import DepthRendering
import matplotlib.pyplot as plt
import os


class Visual:
    def __init__(self, rendering: DepthRendering, path: str):
        self._rendering = rendering
        self._path = path
        self._prepare_path()

    def _get_images(self):
        return self._rendering.get_images()

    def _prepare_path(self):
        os.makedirs(f"{self._path}/visual", exist_ok=True)
        num = len(os.listdir(f"{self._path}/visual")) + 1
        self._visual_path = f"{self._path}/visual/{num}"
        os.makedirs(self._visual_path, exist_ok=True)

    def show_images(self):
        self.get_figures()
        plt.show()

    def save_images(self, name: str):
        figures = self.get_figures()
        for figure in figures:
            figure.savefig(f"{self._visual_path}/{name}.png", bbox_inches="tight")

    def get_figures(self):
        figures = []
        num_columns = 3
        images = self._get_images()
        num_rows = int(len(images) // num_columns)
        fig = plt.figure()
        for i, image in enumerate(images):
            plt.subplot(num_rows, num_columns, i + 1)
            plt.imshow(image.detach().cpu().numpy(), cmap="gray")
            plt.axis("off")
            figures.append(fig)
        return figures
