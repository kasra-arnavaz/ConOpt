import sys

sys.path.append("src")

from config.config import Config


def main():
    config = Config(
        msh_file="data/long_caterpillar.msh",
        scad_file="data/caterpillar.scad",
        scad_parameters="data/long_caterpillar_scad_params.json",
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
        cable_pull_ratio=[0.0, 0.0, 0.0, 0.0, 0.0],
        cable_stiffness=100,
        cable_damping=0.01,
        object_file="data/sphere.obj",
        object_density=1080.0,
        object_translation=[-60.0, -60.0, -20.0],
        object_rotation_vector=[
            0.0,
            0.0,
            1.0,
        ],
        object_rotation_degrees=0.0,
        object_scale=[0.001, 0.001, 0.001],
        sim_duration=5.0,
        sim_segment_duration=0.1,
        sim_dt=5e-5,
        contact_distance=0.0,
        contact_ke=2.0,
        contact_kd=0.1,
        contact_kf=2.0,
        learning_rate=1.0e-5,
        num_training_iterations=200,
        device="cuda",
        use_checkpoint=True,
        out_path=".dev/point_touch",
    )
    config.to_yaml(path=".", name="config_long_touch")


if __name__ == "__main__":
    main()
