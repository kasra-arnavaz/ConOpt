from attrs import define, field


@define
class SimulationProperties:
    duration: float = field(metadata={"unit": "seconds"})
    segment_duration: float = field(metadata={"unit": "seconds"})
    dt: float = field(metadata={"unit": "seconds"})
    device: str = field(default="cuda")

    @segment_duration.validator
    def _check_compatibility_of_duration(self, attribute, value):
        if self._segment_duration_not_a_multiplier_of_duration():
            raise ValueError(f"Expected <{attribute.name}> to be a multiplier of <duration>.")

    def _segment_duration_not_a_multiplier_of_duration(self):
        return self.duration % self.segment_duration != 0
