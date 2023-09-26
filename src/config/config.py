import os
import yaml
from attrs import define, asdict


@define
class Config:
    msh_file: str
    scad_file: str
    scad_parameters: str
    ideal_edge_length: float
    robot_density: float
    robot_youngs_modulus: float
    robot_poissons_ratio: float
    robot_damping_factor: float
    robot_frozen_bounding_box: list
    robot_translation: list
    robot_rotation_vector: list
    robot_rotation_degrees: float
    robot_scale: list
    cable_pull_ratio: list
    cable_stiffness: float
    cable_damping: float
    object_file: str
    object_density: float
    object_translation: list
    object_rotation_vector: list
    object_rotation_degrees: float
    object_scale: list
    sim_duration: float
    sim_segment_duration: float
    sim_dt: float
    contact_distance: float
    contact_ke: float
    contact_kd: float
    contact_kf: float
    ground: bool
    learning_rate: float
    num_training_iterations: int
    device: str
    use_checkpoint: bool
    out_path: str
    obstacle_file: list = None
    obstacle_density: list = None
    obstacle_translation: list = None
    obstacle_rotation_vector: list = None
    obstacle_rotation_degrees: list = None
    obstacle_scale: list = None

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, "r") as f:
            d = yaml.safe_load(f)
        return cls(**d)

    def to_dict(self):
        return asdict(self)

    def to_yaml(self, path, name="config"):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/{name}.yaml", "w") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)
