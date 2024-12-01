"""
This environment is for use with a real panda robot and polymetis.
"""

import time
from abc import abstractmethod, abstractproperty

import cv2
import gym
import numpy as np
import torch

try:
    from polymetis import GripperInterface, RobotInterface
except ImportError:
    print("[research] Could not load polymetis")

from scipy.spatial.transform import Rotation

from research.utils.cameras import RSDepthCamera, USBCamera


class FrankaPick(FrankaEnvCams):
    """
    A simple environment where the goal is for the Franka to pick up an object using
    cameras and proprioceptive state.
    """

    def __init__(self, *args, obj_position=(0.3495, -0.09632, 0.178), horizon=100, **kwargs):
        self._obj_position = np.array(obj_position)
        super().__init__(*args, horizon=horizon, **kwargs)

    def step(self, action):
        state, reward, done, info = super().step(action)
        image = self.camera.read_state()[self.img_name]
        state["image"] = image
        gripper_act = action[-1]
        goal_distance = np.linalg.norm(state["ee_pos"] - self._obj_position)
        z_coord = np.linalg.norm(state["ee_pos"][-1] - self._obj_position[-1])
        print(state["ee_pos"])
        grip_rew = 0.005 * (-np.absolute(goal_distance)) * (gripper_act)
        reward = -0.1 * goal_distance + 0.005 * (0.064 - z_coord) * (gripper_act)
        print("rewards", reward, grip_rew)
        info["goal_distance"] = goal_distance

        return state, reward, done, info


class FrankaPickCams(FrankaEnvCams):
    """
    Test environment with visual observations for RL for a pick task.
    Only using camera observations.
    """

    def __init__(self, *args, obj_position=(0.3495, -0.09632, 0.178), horizon=100, **kwargs):
        self._obj_position = np.array(obj_position)
        super().__init__(*args, horizon=horizon, **kwargs)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 63, self.img_width), dtype=np.uint8)
        print(self.observation_space)

    def reset(self):
        state = super().reset()
        frame = state["image"]
        state["image"] = torch.permute(torch.from_numpy(state["image"]), (2, 0, 1))
        state["image"] = state["image"].numpy()
        if self.render:
            self.render("human", frame)
        return state["image"]

    def step(self, action):
        state, reward, done, info = super().step(action)
        gripper_act = action[-1]
        frame = state["image"]
        goal_distance = np.linalg.norm(state["ee_pos"] - self._obj_position)
        np.linalg.norm(state["ee_pos"][-1] - self._obj_position[-1])
        print(state["ee_pos"])

        grip_reward = 0.005 * (0.02 - goal_distance) * (gripper_act)
        reward = -0.1 * goal_distance + grip_reward
        info["goal_distance"] = goal_distance

        state["image"] = torch.permute(torch.from_numpy(state["image"]), (2, 0, 1))
        state["image"] = state["image"].numpy()
        if self.render:
            self.render("human", frame)
        return state["image"], reward, done, info


class FrankaMoveBlock(FrankaEnvCams):
    """
    Franka environment for picking/moving a block.
    target_offset - how far from the tape, in pixels, we want the target position
    """

    def __init__(self, *args, horizon=500, target_offset=6, **kwargs):
        super().__init__(*args, horizon=horizon, **kwargs)
        self.target_offset = target_offset

    def compute_center(self, mask):
        # Calculates centroid of the binary mask
        moments = cv2.moments(mask)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return (cx, cy)
        else:
            return None

    def segment_image(self, image, lower_red, upper_red, lower_blue, upper_blue):
        # Find the colors in the image
        red_mask = cv2.inRange(image, lower_red, upper_red)
        blue_mask = cv2.inRange(image, lower_blue, upper_blue)

        red_center = self.compute_center(red_mask)
        blue_center = self.compute_center(blue_mask)

        return red_center, blue_center

    def determine_target_position(self, red_center, blue_center):
        # Sets the target position of where we want the block to be on either side
        if red_center[0] < blue_center[0]:  # Red block is on the left side
            return (blue_center[0] + self.target_offset, blue_center[1])
        else:
            return (blue_center[0] - self.target_offset, blue_center[1])

    def reward_function(self, image, target_position, lower_red, upper_red, lower_blue, upper_blue):
        red_center, blue_center = self.segment_image(image, lower_red, upper_red, lower_blue, upper_blue)
        if not red_center or not blue_center:
            print("Could not find red block or blue tape.")
            return None, False

        D = np.linalg.norm(np.array(red_center) - np.array(target_position))
        D_max = np.linalg.norm(np.array([0, 0]) - np.array([image.shape[1] - 1, image.shape[0] - 1]))

        if (red_center[0] < blue_center[0] and target_position[0] > blue_center[0]) or (
            red_center[0] > blue_center[0] and target_position[0] < blue_center[0]
        ):
            return 1, True  # max reward, task successful

        return -D / D_max, False

    def reset(self):
        state = super().reset()
        reward = 0
        done = 0
        info = {}
        info["success"] = False
        return state, reward, done, info

    def step(self, action):
        state, reward, done, info = super().step(action)
        # Calculate negative distance from red block to fixed point on other side
        lower_red = np.array([0, 0, 128])
        upper_red = np.array([80, 80, 255])
        lower_blue = np.array([100, 0, 0])
        upper_blue = np.array([255, 100, 100])

        red_center, blue_center = self.segment_image(state["image"], lower_red, upper_red, lower_blue, upper_blue)
        if not red_center or not blue_center:
            print("Could not detect red block or blue tape")
        else:
            target_position = self.determine_target_position(red_center, blue_center)
            reward, info["success"] = self.reward_function(
                state["image"], target_position, lower_red, upper_red, lower_blue, upper_blue
            )
        return state, reward, done, info
