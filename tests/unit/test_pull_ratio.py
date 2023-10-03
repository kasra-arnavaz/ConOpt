import unittest
import torch
import sys
sys.path.append("src")
from cable.pull_ratio import TimeVariablePullRatio
from simulation.simulation_properties import SimulationProperties

class TestTimeVariantPullRatio(unittest.TestCase):

    def tests_if_linear_interpolation_is_correct_given_two_points(self):
        pull_ratio = [torch.tensor([0.]), torch.tensor([0.2])]
        sim_properties = SimulationProperties(duration=0.5, dt=0.1, segment_duration=0.1, key_timepoints=[0.0, 0.5])
        pull_ratio = TimeVariablePullRatio(pull_ratio=pull_ratio, simulation_properties=sim_properties, device="cpu")
        expected = [torch.tensor([e]) for e in [0.0, 0.04, 0.08, 0.12, 0.16, 0.2]]
        actual = pull_ratio.get()
        for e, a in zip(expected, actual):
            self.assertTrue(torch.allclose(e, a))
    
    def tests_if_linear_interpolation_is_correct_given_three_points(self):
        pull_ratio = [torch.tensor([0.]), torch.tensor([0.2]), torch.tensor([0.])]
        sim_properties = SimulationProperties(duration=0.5, dt=0.1, segment_duration=0.1, key_timepoints=[0.0, 0.5, 1.0])
        pull_ratio = TimeVariablePullRatio(pull_ratio=pull_ratio, simulation_properties=sim_properties, device="cpu")
        expected = [torch.tensor([e]) for e in [0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.16, 0.12, 0.08, 0.04, 0.0]]
        actual = pull_ratio.get()
        for e, a in zip(expected, actual):
            self.assertTrue(torch.allclose(e, a))

if __name__ == "__main__":
    unittest.main()