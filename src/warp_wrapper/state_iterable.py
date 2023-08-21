from warp.sim import Model


class StateIterable:
    def __init__(self, model: Model, num: int = 23040):
        self.states = [model.state(requires_grad=True) for _ in range(num)]
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self.states:
            raise StopIteration
        state = self.states[self._index]
        self._index = (self._index + 1) % len(self.states)
        return state
