from typing import Type, List

import numpy as np
from rlbench import ObservationConfig, ActionMode
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.observation import (
    BimanualObservation,
    Observation,
    UnimanualObservation,
)
from rlbench.backend.task import Task
from yarr.agents.agent import ActResult, VideoSummary, TextSummary
from yarr.envs.rlbench_env import RLBenchEnv, MultiTaskRLBenchEnv
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition
from yarr.utils.process_str import change_case

from pyrep.const import RenderMode
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.objects import VisionSensor, Dummy

import logging

class CustomRLBenchEnv(RLBenchEnv):
    def __init__(
        self,
        task_class: Type[Task],
        observation_config: ObservationConfig,
        action_mode: ActionMode,
        episode_length: int,
        dataset_root: str = "",
        channels_last: bool = False,
        reward_scale=100.0,
        headless: bool = True,
        time_in_state: bool = False,
        include_lang_goal_in_obs: bool = False,
        record_every_n: int = 20,
    ):
        super(CustomRLBenchEnv, self).__init__(
            task_class,
            observation_config,
            action_mode,
            dataset_root,
            channels_last,
            headless=headless,
            include_lang_goal_in_obs=include_lang_goal_in_obs,
        )
        self._reward_scale = reward_scale
        self._episode_index = 0
        self._record_current_episode = False
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._recorded_images = []
        self._episode_length = episode_length
        self._time_in_state = time_in_state
        self._record_every_n = record_every_n
        self._i = 0
        self._error_type_counts = {
            "IKError": 0,
            "ConfigurationPathError": 0,
            "InvalidActionError": 0,
        }
        self._last_exception = None

    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(CustomRLBenchEnv, self).observation_elements
        for oe in obs_elems:
            if "low_dim_state" in oe.name:
                oe.shape = (
                    oe.shape[0] - 7 * 3 + int(self._time_in_state),
                )  # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]

        return obs_elems

    def extract_obs(self, obs: Observation, t=None, prev_action=None):
        if obs.is_bimanual:
            return self.extract_obs_bimanual(obs, t, prev_action)
        else:
            return self.extract_obs_unimanual(obs, t, prev_action)

    def extract_obs_bimanual(self, obs: BimanualObservation, t=None, prev_action=None):
        obs.right.joint_velocities = None
        right_grip_mat = obs.right.gripper_matrix
        right_grip_pose = obs.right.gripper_pose
        right_joint_pos = obs.right.joint_positions
        obs.right.gripper_pose = None
        obs.right.gripper_matrix = None
        obs.right.joint_positions = None

        obs.left.joint_velocities = None
        left_grip_mat = obs.left.gripper_matrix
        left_grip_pose = obs.left.gripper_pose
        left_joint_pos = obs.left.joint_positions
        obs.left.gripper_pose = None
        obs.left.gripper_matrix = None
        obs.left.joint_positions = None

        if obs.right.gripper_joint_positions is not None:
            obs.right.gripper_joint_positions = np.clip(
                obs.right.gripper_joint_positions, 0.0, 0.04
            )
            obs.left.gripper_joint_positions = np.clip(
                obs.left.gripper_joint_positions, 0.0, 0.04
            )

        # 在eval的时候，会用到这个
        # print("custom obs.right.gripper_joint_positions",obs.right.gripper_joint_positions)
        # print("obs.left.gripper_joint_positions",obs.left.gripper_joint_positions)

        obs_dict = super(CustomRLBenchEnv, self).extract_obs(obs)
        # for key in obs_dict:
            # print(f"key in obs dict: {key}")
        if self._time_in_state:
            time = (
                1.0 - ((self._i if t is None else t) / float(self._episode_length - 1))
            ) * 2.0 - 1.0

            if "low_dim_state" in obs_dict:
                obs_dict["low_dim_state"] = np.concatenate(
                    [obs_dict["low_dim_state"], [time]]
                ).astype(np.float32)
            else:
                obs_dict["right_low_dim_state"] = np.concatenate(
                    [obs_dict["right_low_dim_state"], [time]]
                ).astype(np.float32)
                obs_dict["left_low_dim_state"] = np.concatenate(
                    [obs_dict["left_low_dim_state"], [time]]
                ).astype(np.float32)

        obs.right.gripper_matrix = right_grip_mat
        obs.right.joint_positions = right_joint_pos
        obs.right.gripper_pose = right_grip_pose
        obs.left.gripper_matrix = left_grip_mat
        obs.left.joint_positions = left_joint_pos
        obs.left.gripper_pose = left_grip_pose

        obs_dict['left_joint_positions'] = obs.left.joint_positions
        obs_dict['left_gripper_joint_positions'] = obs.left.gripper_joint_positions
        obs_dict['right_joint_positions'] = obs.right.joint_positions
        obs_dict['right_gripper_joint_positions'] = obs.right.gripper_joint_positions
        # print("obs.right.gripper_joint_positions",obs_dict['right_gripper_joint_positions'])
        # print("obs.left.gripper_joint_positions",obs_dict['left_gripper_joint_positions'])

        # for key in obs_dict:
            # print("obs_dict",key)

        return obs_dict

    def extract_obs_unimanual(self, obs: UnimanualObservation, t=None, prev_action=None):
        obs.joint_velocities = None
        grip_mat = obs.gripper_matrix
        grip_pose = obs.gripper_pose
        joint_pos = obs.joint_positions
        obs.gripper_pose = None
        # obs.gripper_pose = None
        obs.gripper_matrix = None
        obs.joint_positions = None
        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0.0, 0.04
            )

        obs_dict = super(CustomRLBenchEnv, self).extract_obs(obs)

        if self._time_in_state:
            time = (
                1.0 - ((self._i if t is None else t) / float(self._episode_length - 1))
            ) * 2.0 - 1.0
            obs_dict["low_dim_state"] = np.concatenate(
                [obs_dict["low_dim_state"], [time]]
            ).astype(np.float32)

        obs.gripper_matrix = grip_mat
        # obs.gripper_pose = grip_pose
        obs.joint_positions = joint_pos
        obs.gripper_pose = grip_pose
        # obs_dict['gripper_pose'] = grip_pose

        obs_dict['joint_positions'] = obs.joint_positions
        obs_dict['gripper_joint_positions'] = obs.gripper_joint_positions

        return obs_dict

    def launch(self):
        super(CustomRLBenchEnv, self).launch()
        self._task._scene.register_step_callback(self._my_callback)
        if self.eval:
            cam_placeholder = Dummy("cam_cinematic_placeholder")
            cam_base = Dummy("cam_cinematic_base")
            cam_base.rotate([0, 0, np.pi * 0.75])
            self._record_cam = VisionSensor.create([320, 180])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)

    def reset(self, novel_command=None) -> dict:
        self._i = 0
        self._previous_obs_dict = super(CustomRLBenchEnv, self).reset()
        self._record_current_episode = (
            self.eval and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()
        self._lang_goal = self._task.get_task_descriptions()[0] if novel_command is None else novel_command # new for mani nerf
        return self._previous_obs_dict

    def register_callback(self, func):
        self._task._scene.register_step_callback(func)

    def _my_callback(self):
        if self._record_current_episode:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self._recorded_images.append(cap)

    def _append_final_frame(self, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10,) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))

    def step(self, act_result: ActResult) -> Transition:
        action = act_result.action
        success = False
        obs = self._previous_obs_dict  # in case action fails.

        try:
            obs, reward, terminal = self._task.step(action)
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = 0.0

            if isinstance(e, IKError):
                self._error_type_counts["IKError"] += 1
            elif isinstance(e, ConfigurationPathError):
                self._error_type_counts["ConfigurationPathError"] += 1
            elif isinstance(e, InvalidActionError):
                self._error_type_counts["InvalidActionError"] += 1

            self._last_exception = e

        summaries = []
        self._i += 1
        if (
            terminal or self._i == self._episode_length
        ) and self._record_current_episode:
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            summaries.append(
                VideoSummary(
                    "episode_rollout_" + ("success" if success else "fail"), vid, fps=30
                )
            )

            # error summary
            error_str = (
                f"Errors - IK : {self._error_type_counts['IKError']}, "
                f"ConfigPath : {self._error_type_counts['ConfigurationPathError']}, "
                f"InvalidAction : {self._error_type_counts['InvalidActionError']}"
            )
            if not success and self._last_exception is not None:
                error_str += f"\n Last Exception: {self._last_exception}"
                self._last_exception = None

            summaries.append(
                TextSummary("errors", f"Success: {success} | " + error_str)
            )
        return Transition(obs, reward, terminal, summaries=summaries)

    def reset_to_demo(self, i, novel_command=None):
        self._i = 0
        # super(CustomRLBenchEnv, self).reset()

        self._task.set_variation(-1)
        (d,) = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i
        )

        # for key,v in d[0].perception_data.items(): # 在这里还有depth
            # if key.endswith("depth"):
                # print("Cusrim_rlbench_env.py CustomRLBenchEnv",key,v.shape)

        self._task.set_variation(d.variation_number)
        _, obs = self._task.reset_to_demo(d)  #战犯？
        # print("reset to demo运行过 d",d) # <rlbench.demo.Demo object at 0x7f8e1549b2e0>
        # for key,v in obs.perception_data.items(): # 在这里也还有depth
            # if key.endswith("_depth"):
                # print("Cusrim_rlbench_env.py CustomRLBenchEnv",key,v) # 坏了全部出在这，都是None

        self._lang_goal = self._task.get_task_descriptions()[0] if novel_command is None else novel_command
        # print("分割线-----以下有问题就说明是 self._previous_obs_dict = self.extract_obs(obs)的问题")
        self._previous_obs_dict = self.extract_obs(obs)
        # print("reset to demo运行过")

        # for key in self._previous_obs_dict:
            # print(f"reset to  demo key={key}") # 缺depth
        self._record_current_episode = (
            self.eval and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict


class CustomMultiTaskRLBenchEnv(MultiTaskRLBenchEnv):
    def __init__(
        self,
        task_classes: List[Type[Task]],
        observation_config: ObservationConfig,
        action_mode: ActionMode,
        episode_length: int,
        dataset_root: str = "",
        channels_last: bool = False,
        reward_scale=100.0,
        headless: bool = True,
        swap_task_every: int = 1,
        time_in_state: bool = False,
        include_lang_goal_in_obs: bool = False,
        record_every_n: int = 20,
    ):
        super(CustomMultiTaskRLBenchEnv, self).__init__(
            task_classes,
            observation_config,
            action_mode,
            dataset_root,
            channels_last,
            headless=headless,
            swap_task_every=swap_task_every,
            include_lang_goal_in_obs=include_lang_goal_in_obs,
        )
        self._reward_scale = reward_scale
        self._episode_index = 0
        self._record_current_episode = False
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._recorded_images = []
        self._episode_length = episode_length
        self._time_in_state = time_in_state
        self._record_every_n = record_every_n
        self._i = 0
        self._error_type_counts = {
            "IKError": 0,
            "ConfigurationPathError": 0,
            "InvalidActionError": 0,
        }
        self._last_exception = None

    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(CustomMultiTaskRLBenchEnv, self).observation_elements
        for oe in obs_elems:
            if "low_dim_state" in oe.name:
                # ..todo:: since we have the low_dimensional state separate for both robots this will also work
                oe.shape = (
                    oe.shape[0] - 7 * 3 + int(self._time_in_state),
                )  # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]
        return obs_elems

    def extract_obs(self, obs: Observation, t=None, prev_action=None):
        if obs.is_bimanual:
            return self.extract_obs_bimanual(obs, t, prev_action)
        else:
            return self.extract_obs_unimanual(obs, t, prev_action)

    def extract_obs_bimanual(self, obs: BimanualObservation, t=None, prev_action=None):
        obs.right.joint_velocities = None
        right_grip_mat = obs.right.gripper_matrix
        right_grip_pose = obs.right.gripper_pose
        right_joint_pos = obs.right.joint_positions
        obs.right.gripper_pose = None
        obs.right.gripper_matrix = None
        obs.right.joint_positions = None

        obs.left.joint_velocities = None
        left_grip_mat = obs.left.gripper_matrix
        left_grip_pose = obs.left.gripper_pose
        left_joint_pos = obs.left.joint_positions
        obs.left.gripper_pose = None
        obs.left.gripper_matrix = None
        obs.left.joint_positions = None

        if obs.right.gripper_joint_positions is not None:
            obs.right.gripper_joint_positions = np.clip(
                obs.right.gripper_joint_positions, 0.0, 0.04
            )
            obs.left.gripper_joint_positions = np.clip(
                obs.left.gripper_joint_positions, 0.0, 0.04
            )
        # print("custom ## obs.right.gripper_joint_positions",obs.right.gripper_joint_positions)
        # print("custom ## obs.left.gripper_joint_positions",obs.left.gripper_joint_positions)

        obs_dict = super(CustomMultiTaskRLBenchEnv, self).extract_obs(obs)

        if self._time_in_state:
            time = (
                1.0 - ((self._i if t is None else t) / float(self._episode_length - 1))
            ) * 2.0 - 1.0
            obs_dict["right_low_dim_state"] = np.concatenate(
                [obs_dict["right_low_dim_state"], [time]]
            ).astype(np.float32)
            obs_dict["left_low_dim_state"] = np.concatenate(
                [obs_dict["left_low_dim_state"], [time]]
            ).astype(np.float32)

        obs.right.gripper_matrix = right_grip_mat
        obs.right.joint_positions = right_joint_pos
        obs.right.gripper_pose = right_grip_pose
        obs.left.gripper_matrix = left_grip_mat
        obs.left.joint_positions = left_joint_pos
        obs.left.gripper_pose = left_grip_pose

        obs_dict['left_joint_positions'] = obs.left.joint_positions
        obs_dict['left_gripper_joint_positions'] = obs.left.gripper_joint_positions
        obs_dict['right_joint_positions'] = obs.right.joint_positions
        obs_dict['right_gripper_joint_positions'] = obs.right.gripper_joint_positions

        # print("obs.right._joint_positions",obs_dict['right_joint_positions'])
        # print("obs.right.gripper_joint_positions",obs_dict['right_gripper_joint_positions'])
        # print("obs.left._joint_positions",obs_dict['left_joint_positions'])
        # print("obs.left.gripper_joint_positions",obs_dict['left_gripper_joint_positions'])
        return obs_dict

    def extract_obs_unimanual(self, obs: Observation, t=None, prev_action=None):
        obs.joint_velocities = None
        grip_mat = obs.gripper_matrix
        grip_pose = obs.gripper_pose
        joint_pos = obs.joint_positions
        obs.gripper_pose = None
        # obs.gripper_pose = None
        obs.gripper_matrix = None
        obs.wrist_camera_matrix = None
        obs.joint_positions = None
        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0.0, 0.04
            )
        # print("obs.gripper_joint_positions",obs.gripper_joint_positions)

        obs_dict = super(CustomMultiTaskRLBenchEnv, self).extract_obs(obs)

        if self._time_in_state:
            time = (
                1.0 - ((self._i if t is None else t) / float(self._episode_length - 1))
            ) * 2.0 - 1.0
            obs_dict["low_dim_state"] = np.concatenate(
                [obs_dict["low_dim_state"], [time]]
            ).astype(np.float32)

        obs.gripper_matrix = grip_mat
        # obs.gripper_pose = grip_pose
        obs.joint_positions = joint_pos
        obs.gripper_pose = grip_pose
        # obs_dict['gripper_pose'] = grip_pose

        obs_dict['joint_positions'] = obs.joint_positions
        obs_dict['gripper_joint_positions'] = obs.gripper_joint_positions

        return obs_dict

    def launch(self):
        super(CustomMultiTaskRLBenchEnv, self).launch()
        self._task._scene.register_step_callback(self._my_callback)
        if self.eval:
            cam_placeholder = Dummy("cam_cinematic_placeholder")
            cam_base = Dummy("cam_cinematic_base")
            cam_base.rotate([0, 0, np.pi * 0.75])
            self._record_cam = VisionSensor.create([320, 180])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)

    def reset(self) -> dict:
        self._i = 0
        self._previous_obs_dict = super(CustomMultiTaskRLBenchEnv, self).reset()
        self._record_current_episode = (
            self.eval and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()
        return self._previous_obs_dict

    def register_callback(self, func):
        self._task._scene.register_step_callback(func)

    def _my_callback(self):
        if self._record_current_episode:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self._recorded_images.append(cap)

    def _append_final_frame(self, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10,) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))

    def step(self, act_result: ActResult) -> Transition:
        action = act_result.action
        success = False
        obs = self._previous_obs_dict  # in case action fails.

        try:
            obs, reward, terminal = self._task.step(action)
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = 0.0

            if isinstance(e, IKError):
                self._error_type_counts["IKError"] += 1
            elif isinstance(e, ConfigurationPathError):
                self._error_type_counts["ConfigurationPathError"] += 1
            elif isinstance(e, InvalidActionError):
                self._error_type_counts["InvalidActionError"] += 1

            self._last_exception = e

        summaries = []
        self._i += 1
        if (
            terminal or self._i == self._episode_length
        ) and self._record_current_episode:
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            task_name = change_case(self._task._task.__class__.__name__)
            summaries.append(
                VideoSummary(
                    "episode_rollout_"
                    + ("success" if success else "fail")
                    + f"/{task_name}",
                    vid,
                    fps=30,
                )
            )

            # error summary
            error_str = (
                f"Errors - IK : {self._error_type_counts['IKError']}, "
                f"ConfigPath : {self._error_type_counts['ConfigurationPathError']}, "
                f"InvalidAction : {self._error_type_counts['InvalidActionError']}"
            )
            if not success and self._last_exception is not None:
                error_str += f"\n Last Exception: {self._last_exception}"
                self._last_exception = None

            summaries.append(
                TextSummary("errors", f"Success: {success} | " + error_str)
            )
        return Transition(obs, reward, terminal, summaries=summaries)

    def reset_to_demo(self, i, variation_number=-1, 
                      novel_command=None # new 
                      ):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0
        # super(CustomMultiTaskRLBenchEnv, self).reset()

        # if variation_number == -1:
        #     self._task.sample_variation()
        # else:
        #     self._task.set_variation(variation_number)

        self._task.set_variation(-1)
        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i
        )[0]

        self._task.set_variation(d.variation_number)
        _, obs = self._task.reset_to_demo(d)
        self._lang_goal = self._task.get_task_descriptions()[0] if novel_command is None else novel_command

        self._previous_obs_dict = self.extract_obs(obs)
        # 运行的是上面不带muti的
        # print("reset to demo运行过")
        # for key in obs:
        #     print(f"reset to  demo key={key}") # 都缺depth
        # for key in self._previous_obs_dict:
        #     print(f"reset to  demo key={key}") # 都缺depth
        self._record_current_episode = (
            self.eval and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict
