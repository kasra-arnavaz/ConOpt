class SegmentIterator:
    def __init__(self, lst, num_segments):
        self.lst = lst
        self.num_segments = num_segments
        self.segment_length = len(lst) // num_segments
        self.segment_index = 0
        self.step_index = 0
        self.forward_done = False

    def __iter__(self):
        return self

    def __next__(self):
        value = self.lst[self.index]
        self.step_index += 1
        if self.step_index == self.segment_length:  # reached end of segment
            self.step_index = 0
            if self.forward_done:
                self.segment_index -= 1
            else:
                self.segment_index += 1
        if self.segment_index == self.num_segments:  # first forward done
            self.forward_done = True
            self.segment_index -= 1
        if self.segment_index == -1 and self.step_index != 0:
            raise StopIteration
        return value

    @property
    def index(self):
        return self.segment_length * self.segment_index + self.step_index
