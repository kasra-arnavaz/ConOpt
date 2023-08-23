from attrs import define, field


@define
class SimulationProperties:
    duration: float = field(metadata={"unit": "seconds"})
    segment_duration: float = field(metadata={"unit": "seconds"})
    dt: float = field(metadata={"unit": "seconds"})
    device: str = field(default="cuda")
