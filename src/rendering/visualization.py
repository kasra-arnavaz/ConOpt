from rendering.rendering import DepthRendering
import matplotlib.pyplot as plt
from pathlib import Path
from os import PathLike

class Visualization:

    def __init__(self, rendering: DepthRendering):
        self._images = rendering.get_images()

    def show_images(self):
        self.get_figures()
        plt.show()

    def save_images(self, name: PathLike):
        name.parent.mkdir(exist_ok=True)
        for f, figure in enumerate(self.get_figures()):
            figure.savefig(name, bbox_inches="tight")

    def get_figures(self):
        figures = []
        num_columns = 3
        num_rows = int(len(self._images) // num_columns)
        fig = plt.figure()
        for i, image in enumerate(self._images):
            plt.subplot(num_rows, num_columns, i + 1)
            plt.imshow(image.detach().cpu().numpy(), cmap="gray")
            plt.axis("off")
            figures.append(fig)
        return figures



