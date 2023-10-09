import unittest
import sys
sys.path.append("src")
from simulation.segment_iterator import SegmentIterator
import torch

class TestSegmentIterator(unittest.TestCase):

    def tests_segment_iterator(self):
        lst = torch.arange(400).tolist()
        iterator = SegmentIterator(lst=lst, num_segments=2)
        for i in range(801):
            print(next(iterator))

if __name__ == "__main__":
    unittest.main()