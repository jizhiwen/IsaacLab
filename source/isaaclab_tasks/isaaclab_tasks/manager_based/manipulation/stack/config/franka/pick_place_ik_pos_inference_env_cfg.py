# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
from torchvision.utils import save_image
from datetime import datetime

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, CameraCfg, RayCasterCamera, TiledCamera
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm

from ... import mdp
from . import pick_place_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_assets.robots.realman import REALMAN_HIGH_PD_CFG # isort: skip

save_image_dir = f"/mnt/data/datasets/_isaaclab_out_/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
need_save_image = False

def image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
    save_image_to_file: bool = False,
    image_path: str = "image",
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0
        elif data_type == "normals":
            images = (images + 1.0) * 0.5

    if save_image_to_file:
        dir_path, _ = os.path.split(image_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        # Get total successful episodes
        total_successes = 0
        if hasattr(env, "recorder_manager") and env.recorder_manager is not None:
            total_successes = env.recorder_manager.exported_successful_episode_count

        for tile in range(images.shape[0]):
            tile_chw = torch.swapaxes(images[tile : tile + 1].unsqueeze(1), 1, -1).squeeze(-1)
            filename = (
                f"{image_path}_{data_type}_trial_{total_successes}_tile_{tile}_step_{env.common_step_counter}.png"
            )
            save_image(tile_chw, filename)

    return images.clone()


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        object = ObsTerm(func=mdp.one_cube_object_obs)
        cube_positions = ObsTerm(func=mdp.red_cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.red_cube_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""
        global save_image_dir, need_save_image

        table_cam_normals = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "normals",
                "normalize": True,
                "save_image_to_file": False,
                "image_path": "table_cam",
            },
        )
        table_cam_segmentation = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "semantic_segmentation",
                "normalize": False,
                "save_image_to_file": False,
                "image_path": "table_cam",
            },
        )
        table_cam_rgb = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "rgb",
                "normalize": False,
                "save_image_to_file": need_save_image,
                "image_path": f"{save_image_dir}/table_cam",
            },
        )
        table_high_cam_normals = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("table_high_cam"),
                "data_type": "normals",
                "normalize": True,
                "save_image_to_file": False,
                "image_path": "table_high_cam",
            },
        )
        table_high_cam_segmentation = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("table_high_cam"),
                "data_type": "semantic_segmentation",
                "normalize": False,
                "save_image_to_file": False,
                "image_path": "table_high_cam",
            },
        )
        table_high_cam_rgb = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("table_high_cam"),
                "data_type": "rgb",
                "normalize": False,
                "save_image_to_file": need_save_image,
                "image_path": f"{save_image_dir}/table_high_cam",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")}
    )

    success = DoneTerm(func=mdp.cubes_approached, params={"gripper_open_val": torch.tensor([0.0])})


@configclass
class FrankaCubeInferencePickPlaceBlueprintEnvCfg(pick_place_joint_pos_env_cfg.FrankaCubePickPlaceEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.episode_length_s = 30

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = REALMAN_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint.*"], scale=1.0, use_default_offset=False
        )

        self.terminations: TerminationsCfg = TerminationsCfg()

        MAPPING = {
            "class:cube_2": (255, 184, 48, 255),
            "class:table": (255, 237, 218, 255),
            "class:ground": (100, 100, 100, 255),
            "class:robot": (125, 125, 125, 255),
            "class:UNLABELLED": (10, 10, 10, 255),
            "class:BACKGROUND": (10, 10, 10, 255),
        }

        # Set table view camera
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/rm65/Link6/table_cam",
            update_period=0.0666,
            height=480,
            width=640,
            data_types=["rgb", "semantic_segmentation", "normals"],
            colorize_semantic_segmentation=True,
            semantic_segmentation_mapping=MAPPING,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(-0.3934, 0, -0.11359), rot=(0.61008,-0.2652, 0.3313,-0.66911), convention="ros"),
        )

        # Set table view camera
        self.scene.table_high_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_high_cam",
            update_period=0.0666,
            height=480,
            width=640,
            data_types=["rgb", "semantic_segmentation", "normals"],
            colorize_semantic_segmentation=True,
            semantic_segmentation_mapping=MAPPING,
            # h = 1.37456192*f
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=32.989, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0, -0.85, 0.2305), rot=(-0.58779, 0.80902, 0, 0), convention="ros"),
        )