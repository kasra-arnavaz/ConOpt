from attrs import define, field


@define
class SimulationProperties:
    duration: float = field(metadata={"unit": "seconds"})
    segment_duration: float = field(metadata={"unit": "seconds"})
    dt: float = field(metadata={"unit": "seconds"})
    num_segments: int = field()
    num_steps_per_segment: int = field()
    device: str = field(default="cuda")

    @num_segments.default
    def _calc_num_segments(self):
        return int(self.duration / self.segment_duration)

    @num_steps_per_segment.default
    def _calc_num_steps_per_segment(self):
        return int(self.segment_duration / self.dt)

    @segment_duration.validator
    def _check_compatibility_of_duration(self, attribute, value):
        if self._segment_duration_not_a_multiplier_of_duration():
            raise ValueError(f"Expected <{attribute.name}> to be a multiplier of <duration>.")

    def _segment_duration_not_a_multiplier_of_duration(self):
        return self.duration % self.segment_duration != 0
