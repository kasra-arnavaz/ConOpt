from warp.sim import Model


class StateIterable:
    def __init__(self, model: Model, num: int = 23040):
        self.model = model
        self._states = [model.state(requires_grad=True) for _ in range(num)]
        self._index = 0
        self._call_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self._states:
            raise StopIteration
        
        self._index = (self._index + 1) % len(self._states)
        if self._call_count % 2 == 0:
            self._index = self._index - 1
        
        state = self._states[self._index]
        self._call_count += 1
        return state
