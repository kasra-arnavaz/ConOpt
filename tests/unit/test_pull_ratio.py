import unittest
import torch
import sys

sys.path.append("src")
from cable.pull_ratio import TimeVariablePullRatio
from simulation.simulation_properties import SimulationProperties


class TestTimeVariantPullRatio(unittest.TestCase):
    def tests_if_linear_interpolation_is_correct_given_two_points(self):
        DEVICE = "cpu"
        pull_ratio = [torch.tensor(0.0), torch.tensor(0.2)]
        sim_properties = SimulationProperties(
            duration=0.5, dt=0.1, segment_duration=0.1, key_timepoints_interval=0.5, device=DEVICE
        )
        pull_ratio = TimeVariablePullRatio(pull_ratio=pull_ratio, simulation_properties=sim_properties, device=DEVICE)
        expected = [torch.tensor(e, dtype=torch.float32) for e in [0.0, 0.04, 0.08, 0.12, 0.16]]
        actual = pull_ratio.pull_ratio
        for i in range(len(expected)):
            self.assertTrue(torch.allclose(expected[i], actual[i]))

    def tests_if_linear_interpolation_is_correct_given_three_points(self):
        DEVICE = "cpu"
        pull_ratio = [torch.tensor(0.0), torch.tensor(0.2), torch.tensor(0.0)]
        sim_properties = SimulationProperties(
            duration=1.0, dt=0.1, segment_duration=0.1, key_timepoints_interval=0.5, device=DEVICE
        )
        pull_ratio = TimeVariablePullRatio(pull_ratio=pull_ratio, simulation_properties=sim_properties, device=DEVICE)
        expected = [
            torch.tensor(e, dtype=torch.float32) for e in [0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.16, 0.12, 0.08, 0.04]
        ]
        actual = pull_ratio.pull_ratio
        for i in range(len(expected)):
            self.assertTrue(torch.allclose(expected[i], actual[i]))


if __name__ == "__main__":
    unittest.main()
