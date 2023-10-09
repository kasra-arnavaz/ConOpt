from cable.holes import Holes
from attrs import define, field
from cable.pull_ratio import PullRatio


@define
class Cable:
    holes: Holes = field()
    pull_ratio: PullRatio = field()
    stiffness: float = field(default=100.0)
    damping: float = field(default=0.01)