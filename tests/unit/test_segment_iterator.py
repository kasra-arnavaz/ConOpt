import unittest
import sys
sys.path.append("src")
from simulation.segment_iterator import SegmentIterator
import torch

class TestSegmentIterator(unittest.TestCase):

    def tests_segment_iterator(self):
        length = 6
        lst = torch.arange(length).tolist()
        iterator = SegmentIterator(lst=lst, num_segments=3)
        actual = [next(iterator) for _ in range(length*2)]
        expected = [0, 1, 2, 3, 4, 5, 4, 5, 2, 3, 0, 1]
        self.assertEqual(actual, expected)

if __name__ == "__main__":
    unittest.main()