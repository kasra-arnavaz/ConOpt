from rendering.rendering import DepthRendering
import matplotlib.pyplot as plt
import os


class Visual:
    def __init__(self, rendering: DepthRendering, path: str):
        self._rendering = rendering
        self._path = path
        os.makedirs(path, exist_ok=True)

    def _get_images(self):
        return self._rendering.get_images()

    def show_images(self):
        self.get_figures()
        plt.show()

    def save_images(self, name: str):
        figures = self.get_figures()
        for figure in figures:
            figure.savefig(f"{self._path}/{name}.png", bbox_inches="tight")

    def get_figures(self):
        figures = []
        num_columns = 3
        images = self._get_images()
        num_rows = int(len(images) // num_columns)
        fig = plt.figure(figsize=(30,10))
        for i, image in enumerate(images):
            plt.subplot(num_rows, num_columns, i + 1)
            plt.imshow(image.detach().cpu().numpy(), cmap="gray")
            plt.axis("off")
            figures.append(fig)
        return figures
