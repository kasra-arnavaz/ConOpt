import sys

sys.path.append("src")

from config.config import Config


def main():
    config = Config(
        msh_file="data/long_caterpillar.msh",
        scad_file="data/caterpillar.scad",
        scad_parameters="data/long_caterpillar_scad_params.json",
        ideal_edge_length=0.02,
        robot_density=1080.0,
        robot_youngs_modulus=149_000,
        robot_poissons_ratio=0.40,
        robot_damping_factor=0.4,
        robot_frozen_bounding_box=[-float("inf"), -0.01, -float("inf"), float("inf"), float("inf"), float("inf")],
        robot_translation=[
            0.0,
            0.0,
            0.0,
        ],
        robot_rotation_vector=[1.0, 0.0, 0.0],
        robot_rotation_degrees=90.0,
        robot_scale=[0.001, 0.001, 0.001],
        cable_pull_ratio=[0.0, 0.0, 0.0, 0.0, 0.0],
        cable_stiffness=100,
        cable_damping=0.01,
        object_file="data/sphere.obj",
        object_density=1080.0,
        object_translation=[-20.0, -20.0, -20.0],
        object_rotation_vector=[
            0.0,
            0.0,
            1.0,
        ],
        object_rotation_degrees=0.0,
        object_scale=[0.01, 0.01, 0.01],
        sim_duration=5.0,
        sim_segment_duration=0.1,
        sim_dt=5e-5,
        contact_distance=0.0,
        contact_ke=2.0,
        contact_kd=0.1,
        contact_kf=0.1,
        ground=False,
        learning_rate=1.0e-5,
        num_training_iterations=200,
        device="cuda",
        use_checkpoint=True,
        out_path=".dev/point_touch",
        obstacle_file=["tests/data/cylinder.obj", "tests/data/cylinder.obj"],
        obstacle_density=[1080.0, 1080.0],
        obstacle_translation=[[-20.0, -20.0, -20.0], [-20.0, -20.0, -20.0]],
        obstacle_rotation_vector=[[0,0,1], [0,0,1]],
        obstacle_rotation_degrees=[90.0, 90.0],
        obstacle_scale=[[0.0015, 0.0015, 0.01], [0.0015, 0.0015, 0.01]],
        key_timepoints_intervals = 1.0,
        target_position = [0.3144, 0.0177, 0.0282]
    )
    config.to_yaml(path=".", name="config_long_touch_with_obstacle")


if __name__ == "__main__":
    main()
