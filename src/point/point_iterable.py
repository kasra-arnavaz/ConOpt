import sys

sys.path.append("src")

from point.points import Points
import copy


class PointIterable:
    def __init__(self, point: Points, num: int = 23040):
        self._points = [copy.copy(point) for _ in range(num)]
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self._points:
            raise StopIteration
        points = self._points[self._index]
        self._index = (self._index + 1) % len(self._points)
        return points
