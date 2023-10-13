from attrs import define, field
from decimal import Decimal
from typing import List
import torch

@define
class SimulationProperties:
    duration: float = field(metadata={"unit": "seconds"})
    segment_duration: float = field(metadata={"unit": "seconds"})
    dt: float = field(metadata={"unit": "seconds"})
    num_segments: int = field(init=False)
    num_steps_per_segment: int = field(init=False)
    num_steps: int = field(init=False)
    device: str = field(default="cuda")
    key_timepoints_interval: float = field(default=None)
    key_timepoints: torch.Tensor = field(init=False)

    @num_steps.default
    def _calc_num_steps(self):
        return self.num_segments * self.num_steps_per_segment
    
    @num_segments.default
    def _calc_num_segments(self):
        return int(self.duration / self.segment_duration)

    @num_steps_per_segment.default
    def _calc_num_steps_per_segment(self):
        return int(self.segment_duration / self.dt)
    
    @key_timepoints.default
    def _calc_key_timepoints(self):
        if self.key_timepoints_interval is None:
            return None
        steps = int(self.duration/self.key_timepoints_interval) + 1
        return torch.linspace(0.0, self.duration, steps=steps, dtype=torch.float64, device=self.device)

    @key_timepoints_interval.validator
    def _check_compatibility_of_duration(self, attribute, value):
        if value is not None:
            if self._x_not_a_multiplier_of_y(self.key_timepoints_interval, self.duration):
                raise ValueError(f"Expected <{attribute.name}> to be a multiplier of <duration>.")
    
    @segment_duration.validator
    def _check_compatibility(self, attribute, value):
        if self._x_not_a_multiplier_of_y(self.segment_duration, self.duration):
            raise ValueError(f"Expected <{attribute.name}> to be a multiplier of <duration>.")
        
    @key_timepoints.validator
    def _check_time_keypoints_increasing(self, attribute, value):
        if value is not None:
            for i in range(len(value) - 1):
                if value[i] >= value[i + 1]:
                    raise ValueError(f"Expected <{attribute.name} to be strictly increasing.>")
                
    @key_timepoints.validator
    def _check_final_time_keypoint(self, attribute, value):
        if value is not None:
            if value[-1] != self.duration:
                raise ValueError(f"Expected the final value of <{attribute.name}> to be the same as duration.")

    @staticmethod
    def _x_not_a_multiplier_of_y(x, y):
        return Decimal(str(y)) % Decimal(str(x)) != 0.0
