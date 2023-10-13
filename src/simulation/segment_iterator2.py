class ForwardBackwardIterable:
    def __init__(self, data):
        self.data = data
        self.index = 0
        self.ascending = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.data:
            raise StopIteration

        # print(self.index)
        item = self.data[self.index]

        if self.ascending:
            self.index += 1
            if self.index == len(self.data):
                self.ascending = False
                self.index -= 1
        else:
            self.index -= 1
            if self.index < 0:
                self.ascending = True
                self.index = 0

        return item
