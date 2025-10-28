# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_ee_distance_and_lifted(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Combined reward for reaching the object AND lifting it."""
    # Get reaching reward
    reach_reward = object_ee_distance(env, std, object_cfg, ee_frame_cfg)
    # Get lifting reward
    lift_reward = object_is_lifted(env, minimal_height, object_cfg)
    # Combine rewards multiplicatively
    return reach_reward * lift_reward


def gripper_orientation_alignment(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the gripper for pointing straight down (perpendicular to ground).

    This encourages top-down grasping by rewarding when the gripper's Z-axis points downward.
    Uses a smooth reward function that provides gradient information at all orientations.

    Args:
        env: The environment.
        ee_frame_cfg: End-effector frame configuration.

    Returns:
        Reward value between 0.0 and 1.0, where 1.0 means perfectly vertical (looking straight down).
    """
    # Extract end-effector frame
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Get end-effector orientation (quaternion): (num_envs, 4) - [w, x, y, z]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]

    # Convert quaternion to rotation matrix and extract Z-axis (third column)
    # For quaternion [w, x, y, z], the Z-axis of the frame is:
    # z_axis = [2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x^2 + y^2)]
    w, x, y, z = ee_quat_w[:, 0], ee_quat_w[:, 1], ee_quat_w[:, 2], ee_quat_w[:, 3]

    # Z-axis of the gripper in world frame
    gripper_z_x = 2 * (x * z + w * y)
    gripper_z_y = 2 * (y * z - w * x)
    gripper_z_z = 1 - 2 * (x * x + y * y)

    # World Z-axis pointing down (for top-down grasping)
    world_down = torch.tensor([0.0, 0.0, -1.0], device=env.device)

    # Compute dot product between gripper Z-axis and downward direction
    # Values: 1.0 = perfectly vertical (down), 0.0 = horizontal, -1.0 = pointing up
    alignment = gripper_z_x * world_down[0] + gripper_z_y * world_down[1] + gripper_z_z * world_down[2]

    # Invert and map to [0, 1]: 1.0 when pointing down, 0.0 when pointing up
    # The negative sign inverts the alignment (gripper Z-axis points opposite to expected)
    reward = (-alignment + 1.0) / 2.0

    return reward
