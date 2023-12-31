import unittest
import sys
import torch
import numpy as np

sys.path.append("src")
from objective.variables import Variables


class TestVariables(unittest.TestCase):
    def tests_if_parameters_is_an_empty_list_by_default(self):
        variables = Variables()
        self.assertEqual(variables.parameters, [])

    def tests_if_torch_tensor_is_added_to_parameters(self):
        variables = Variables()
        variables.add_parameter(torch.tensor([1.0, 1.0, 1.0], requires_grad=True))
        self.assertTrue(torch.equal(variables.parameters[0], torch.tensor([1.0, 1.0, 1.0])))

    def tests_if_value_error_is_raised_when_numpy_array_is_added_to_parameters(self):
        variables = Variables()
        with self.assertRaises(ValueError):
            variables.add_parameter(np.array([1.0, 1.0, 1.0]))
            variables.parameters

    def tests_if_gradients_are_correct(self):
        variables = Variables()
        p = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        variables.add_parameter(p)
        loss = (p**2).sum()
        loss.backward()
        variables.set_gradients()
        self.assertTrue(torch.equal(variables.gradients[0], 2 * p))


if __name__ == "__main__":
    unittest.main()
