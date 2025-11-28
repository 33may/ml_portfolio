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
from isaaclab.assets import Articulation, RigidObject
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


def gripper_close_when_near_object(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward closing the gripper when the object is within reach (< 5cm).

    This encourages the robot to actually grasp the object after positioning near it.

    Args:
        env: The environment.
        distance_threshold: Maximum distance (in meters) to trigger reward. Default 0.05m (5cm).
        object_cfg: Object configuration.
        ee_frame_cfg: End-effector frame configuration.
        robot_cfg: Robot configuration.

    Returns:
        Reward value between 0.0 and 1.0, where 1.0 means gripper is closed when object is very close.
    """
    # Extract entities
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    # Check if object is within reach (< 5cm)
    is_near = object_ee_distance < distance_threshold

    # Get gripper joint position (assuming joint name is "gripper")
    # Lower values mean more closed
    gripper_joint_idx = robot.joint_names.index("gripper")
    gripper_pos = robot.data.joint_pos[:, gripper_joint_idx]

    # Gripper closedness: 1.0 = fully closed (0.0), 0.0 = fully open (0.5)
    # Assuming gripper range is [0.0, 0.5] based on the config
    gripper_closedness = 1.0 - (gripper_pos / 0.5)
    # gripper_closedness = torch.clamp(gripper_closedness, 0.0, 1.0)

    # Reward closing gripper only when object is near
    reward = torch.where(is_near, gripper_closedness, torch.zeros_like(gripper_closedness))

    return reward


def lift_when_gripper_closed(
    env: ManagerBasedRLEnv,
    gripper_close_threshold: float = 0.15,
    minimal_height: float = 0.01,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward lifting the object when the gripper is closed enough.

    This encourages the robot to lift after grasping by only rewarding height when gripper is sufficiently closed.

    Args:
        env: The environment.
        gripper_close_threshold: Maximum gripper position to consider "closed" (default 0.15).
        minimal_height: Minimum height above ground to start rewarding (default 0.03m).
        object_cfg: Object configuration.
        robot_cfg: Robot configuration.

    Returns:
        Reward value based on object height when gripper is closed, 0.0 otherwise.
    """
    # Extract entities
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Get gripper joint position
    gripper_joint_idx = robot.joint_names.index("gripper")
    gripper_pos = robot.data.joint_pos[:, gripper_joint_idx]

    # Check if gripper is closed enough
    is_closed = gripper_pos < gripper_close_threshold

    # Get object height
    object_height = object.data.root_pos_w[:, 2]

    # Normalized height reward: 0 at minimal_height, 1 at 0.5m
    height_reward = torch.clamp((object_height - minimal_height) / 0.5, 0.0, 1.0)

    # Only reward lifting when gripper is closed
    reward = torch.where(is_closed, height_reward, torch.zeros_like(height_reward))

    return reward
