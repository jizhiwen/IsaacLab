# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates policy inference in a prebuilt USD environment.

In this example, we use a locomotion policy to control the H1 robot. The robot was trained
using Isaac-Velocity-Rough-H1-v0. The robot is commanded to move forward at a constant velocity.

.. code-block:: bash

    # Run the script
    ./isaaclab.sh -p scripts/tutorials/03_envs/policy_inference_in_usd.py --checkpoint /path/to/jit/checkpoint.pt

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on inferencing a policy on an H1 robot in a warehouse.")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", required=True)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import io
import os
import torch

import omni

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.approach_ik_pos_inference_env_cfg import FrankaCubeInferenceBlueprintEnvCfg


def main():
    """Main function."""
    # setup environment
    env_cfg = FrankaCubeInferenceBlueprintEnvCfg()
    env_cfg.scene.num_envs = 1

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    timeout_term = None
    if hasattr(env_cfg.terminations, "time_out"):
        timeout_term = env_cfg.terminations.time_out
        env_cfg.terminations.time_out = None

    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    action = [0,0,0,0,0,0,0]
    delta_pose = torch.tensor(action, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)

    # Create a policy
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy
    import numpy as np
    import time
    data_config = DATA_CONFIG_MAP["realman"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path="/home/robot/src/IsaacLab/models/checkpoint-30000",
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag="new_embodiment",
        denoising_steps=10,
    )

    grapper_closed = False

    success_step_count = 0
    total_step_count = 0

    # run inference with the policy
    obs, _ = env.reset()
    with torch.inference_mode():
        while simulation_app.is_running():
            realman_obs = {
                "state.single_arm": np.degrees(obs['policy']['joint_pos'].cpu().numpy()[:,:6]),
                "state.gripper": np.array([[0.0 if grapper_closed else 1000.0]]),
                "video.front_view": obs['rgb_camera']['table_high_cam_rgb'].cpu().numpy().astype(np.uint8),
                "video.right_view": obs['rgb_camera']['table_side_cam_rgb'].cpu().numpy().astype(np.uint8),
                "annotation.human.action.task_description": "Move above the red square.",
            }

            action_chunk = policy.get_action(realman_obs)
            single_arm = action_chunk['action.single_arm']
            gripper = action_chunk['action.gripper']

            for i in range(0, len(single_arm)):
                grapper_closed = False if gripper[i] >= 900 else True
                grapper_closed = False
                action = np.append(np.deg2rad(single_arm[i]), -1 if grapper_closed else 1)
                action = torch.tensor(action, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
                obs, _, _, _, _ = env.step(action)

                if success_term is not None:
                    if bool(success_term.func(env, **success_term.params)[0]):
                        total_step_count = total_step_count + 1
                        success_step_count = success_step_count + 1
                        print(f"[{success_step_count}/{total_step_count}] Task success ...")
                        obs, _ = env.reset()
                        break

                if timeout_term is not None:
                    if bool(timeout_term.func(env, **timeout_term.params)[0]):
                        total_step_count = total_step_count + 1
                        print(f"[{success_step_count}/{total_step_count}] Task fail ...")
                        obs, _ = env.reset()
                        break


if __name__ == "__main__":
    main()
    simulation_app.close()
