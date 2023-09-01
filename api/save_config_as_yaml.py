import sys

sys.path.append("src")

from config.config import Config


def main():
    config = Config(
        msh_file="tests/data/caterpillar.msh",
        scad_file="tests/data/caterpillar.scad",
        scad_parameters="tests/data/caterpillar_scad_params.json",
        ideal_edge_length=0.02,
        gripper_density=1080.0,
        gripper_youngs_modulus=149_000,
        gripper_poissons_ratio=0.40,
        gripper_damping_factor=0.4,
        gripper_frozen_bounding_box=[-float("inf"), -0.01, -float("inf"), float("inf"), float("inf"), float("inf")],
        gripper_translation=[
            0.0,
            0.0,
            0.0,
        ],
        gripper_rotation_vector=[1.0, 0.0, 0.0],
        gripper_rotation_degrees=90.0,
        gripper_scale=[0.001, 0.001, 0.001],
        cable_pull_ratio=[0.0, 0.0, 0.0],
        cable_stiffness=100,
        cable_damping=0.01,
        object_file="tests/data/cylinder.obj",
        object_density=1080.0,
        object_translation=[60.0, -60.0, -20.0],
        object_rotation_vector=[
            0.0,
            0.0,
            1.0,
        ],
        object_rotation_degrees=0.0,
        object_scale=[0.0015, 0.0015, 0.01],
        sim_duration=0.5,
        sim_segment_duration=0.1,
        sim_dt=5e-5,
        learning_rate=1e-2,
        num_training_iterations=100,
        device="cuda",
        out_path=".dev/max_grip",
    )
    config.to_yaml(path=".", name="config")


if __name__ == "__main__":
    main()
