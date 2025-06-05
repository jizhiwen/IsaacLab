# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

REALMAN_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/RealMan/realman.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 1.6081,
            "joint2": 0.6089,
            "joint3": 1.1464,
            "joint4": -0.0055,
            "joint5": 1.3666,
            "joint6": 0.0628,
            "finger[1-2]_joint": 0.0,
        },
    ),
    actuators={
        "realman_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "realman_forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint[5-6]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "realman_hand": ImplicitActuatorCfg(
            joint_names_expr=["finger[1-2]_joint"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


REALMAN_HIGH_PD_CFG = REALMAN_CFG.copy()
REALMAN_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
REALMAN_HIGH_PD_CFG.actuators["realman_shoulder"].stiffness = 1047.2
REALMAN_HIGH_PD_CFG.actuators["realman_shoulder"].damping = 80.0
REALMAN_HIGH_PD_CFG.actuators["realman_forearm"].stiffness = 1047.2
REALMAN_HIGH_PD_CFG.actuators["realman_forearm"].damping = 80.0
REALMAN_HIGH_PD_CFG.actuators["realman_hand"].stiffness = 1047.2
REALMAN_HIGH_PD_CFG.actuators["realman_hand"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
