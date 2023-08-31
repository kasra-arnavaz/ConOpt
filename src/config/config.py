import os
import yaml
from attrs import define, asdict


@define
class Config:
    scad_file: str
    scad_parameters: str
    ideal_edge_length: float
    gripper_density: float
    gripper_youngs_modulus: float
    gripper_poissons_ratio: float
    gripper_damping_factor: float
    gripper_frozen_bounding_box: list
    gripper_translation: list
    gripper_rotation_vector: list
    gripper_rotation_degrees: float
    gripper_scale: list
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
    learning_rate: float
    num_training_iterations: int
    device: str
    out_path: str

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
