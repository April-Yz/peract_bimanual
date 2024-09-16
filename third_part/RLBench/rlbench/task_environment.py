import logging
from typing import List, Callable

import numpy as np
from pyrep import PyRep
from pyrep.const import ObjectType
from rlbench import utils
from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.exceptions import BoundaryError, WaypointError, \
    TaskEnvironmentError
from rlbench.backend.observation import Observation
from rlbench.backend.robot import Robot, UnimanualRobot, BimanualRobot
from rlbench.backend.scene import Scene
from rlbench.backend.task import Task
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig

_DT = 0.05
_MAX_RESET_ATTEMPTS = 40
_MAX_DEMO_ATTEMPTS = 50 # 10  only for lift ball


class TaskEnvironment(object):

    def __init__(self,
                 pyrep: PyRep,
                 robot: Robot,
                 scene: Scene,
                 task: Task,
                 action_mode: ActionMode,
                 dataset_root: str,
                 obs_config: ObservationConfig,
                 static_positions: bool = False,
                 attach_grasped_objects: bool = True,
                 #  !!新的
                 shaped_rewards: bool = False
                 ):
        self._pyrep = pyrep
        self._robot = robot
        self._scene = scene
        self._task = task
        self._variation_number = 0
        self._action_mode = action_mode
        self._dataset_root = dataset_root
        self._obs_config = obs_config
        self._static_positions = static_positions
        self._attach_grasped_objects = attach_grasped_objects
        self._shaped_rewards = shaped_rewards
        self._reset_called = False
        self._prev_ee_velocity = None
        self._enable_path_observations = False

        self._scene.load(self._task)
        self._pyrep.start()

        if self._robot.is_bimanual:
             #..fixme !! 大多数都是这里出现的
            logging.warning("not sure how _robot_shapes are used is used.")
            self._robot_shapes = self._robot.right_arm.get_objects_in_tree(object_type=ObjectType.SHAPE) + self._robot.left_arm.get_objects_in_tree(object_type=ObjectType.SHAPE)
        else:
            self._robot_shapes = self._robot.arm.get_objects_in_tree(object_type=ObjectType.SHAPE)
        
  
    def get_name(self) -> str:
        return self._task.get_name()

    def sample_variation(self) -> int:
        self._variation_number = np.random.randint(
            0, self._task.variation_count())
        return self._variation_number

    def set_variation(self, v: int) -> None:
        if v >= self.variation_count():
            raise TaskEnvironmentError(
                'Requested variation %d, but there are only %d variations.' % (
                    v, self.variation_count()))
        self._variation_number = v

    def variation_count(self) -> int:
        return self._task.variation_count()

    def reset(self) -> (List[str], Observation):
        self._scene.reset()
        try:
            desc = self._scene.init_episode(
                self._variation_number, max_attempts=_MAX_RESET_ATTEMPTS,
                randomly_place=not self._static_positions)
        except (BoundaryError, WaypointError) as e:
            logging.error("error while initializing episode: %s", e)
            raise TaskEnvironmentError(
                'Could not place the task %s in the scene. This should not '
                'happen, please raise an issues on this task.'
                % self._task.get_name()) from e

        self._reset_called = True
        # Returns a list of descriptions and the first observation
        return desc, self._scene.get_observation()

    def get_task_descriptions(self) -> List[str]:
        return self._scene.task.init_episode(self._variation_number)

    def get_observation(self) -> Observation:
        return self._scene.get_observation()

    def step(self, action) -> (Observation, int, bool):
        # returns observation, reward, done, info  观察， 奖励， 完成， 信息
        if not self._reset_called:
            raise RuntimeError(
                "Call 'reset' before calling 'step' on a task.") # "在对任务调用“step”之前调用“reset”。"
        self._action_mode.action(self._scene, action)
        success, terminate = self._task.success()
        reward = float(success)
        if self._shaped_rewards:
            reward = self._task.reward()
            if reward is None:
                raise RuntimeError(
                    'User requested shaped rewards, but task %s does not have '
                    'a defined reward() function.' % self._task.get_name())
        return self._scene.get_observation(), reward, terminate

    def get_demos(self, amount: int, live_demos: bool = False,
                  image_paths: bool = False,
                #   !! 指定了callable_each_step变量所存储的可调用对象的参数类型和返回类型。参数的类型是Observation  None返回类型为None  = None 初始值
                  callable_each_step: Callable[[Observation], None] = None,
                  max_attempts: int = _MAX_DEMO_ATTEMPTS,
                  random_selection: bool = True,
                  from_episode_number: int = 0
                  ) -> List[Demo]:
        """Negative means all demos"""

        if not live_demos and (self._dataset_root is None
                               or len(self._dataset_root) == 0):
            raise RuntimeError(
                "Can't ask for a stored demo when no dataset root provided.")


        if not live_demos:
            if self._dataset_root is None or len(self._dataset_root) == 0:
                raise RuntimeError(
                    "Can't ask for stored demo when no dataset root provided.")
            # print("taskEnvironment get_demos case1")
            demos = utils.get_stored_demos(
                amount, image_paths, self._dataset_root, self._variation_number,
                self._task.get_name(), self._obs_config,
                random_selection, from_episode_number)
        elif not self._robot.is_bimanual:
            # print("taskEnvironment get_demos case2")
            ctr_loop = self._robot.arm.joints[0].is_control_loop_enabled()
            self._robot.arm.set_control_loop_enabled(True)
            demos = self._get_live_demos(
                amount, callable_each_step, max_attempts)
            self._robot.arm.set_control_loop_enabled(ctr_loop)
        elif self._robot.is_bimanual:
            # print("taskEnvironment get_demos case3")
            # logging.info("11**try self._robot.is_bimanual **11")
            ctr_loop_right = self._robot.right_arm.joints[0].is_control_loop_enabled()
            ctr_loop_left = self._robot.left_arm.joints[0].is_control_loop_enabled()
            self._robot.right_arm.set_control_loop_enabled(True)
            self._robot.left_arm.set_control_loop_enabled(True)
            # logging.info("12**try self._robot.is_bimanual **12")
            demos = self._get_live_demos(
                amount, callable_each_step, max_attempts)
            # logging.info("13**try self._robot.is_bimanual **13")
            self._robot.right_arm.set_control_loop_enabled(ctr_loop_right)
            self._robot.left_arm.set_control_loop_enabled(ctr_loop_left)
        # for key,v in demos[0][1].perception_data.items():
        #     if key.endswith("_depth"): # 在这里还有depth
        #         print("taskenvironment",key,v.shape)   # 还是有数据的
        return demos

    def _get_live_demos(self, amount: int,
                        callable_each_step: Callable[
                            [Observation], None] = None,
                        max_attempts: int = _MAX_DEMO_ATTEMPTS) -> List[Demo]:
        demos = []
        for i in range(amount):
            attempts = max_attempts
            # logging.info("111**start try live demo **111")
            while attempts > 0:
                random_seed = np.random.get_state()
                self.reset()
                # logging.info("112**try try live demo **112")
                try:
                    # logging.info("113 demo start 113")
                    demo = self._scene.get_demo(
                        callable_each_step=callable_each_step)
                    # logging.info("114 demo finish 114")
                    demo.random_seed = random_seed
                    demos.append(demo)
                    break
                except Exception as e:
                    attempts -= 1
                    logging.warning('Bad demo. ' + str(e) + ' Attempts left: ' + str(attempts))
            if attempts <= 0:
                raise RuntimeError(
                    # !!problem
                    'Could not collect demos. Maybe a problem with the task?')
        return demos

    def reset_to_demo(self, demo: Demo) -> (List[str], Observation):
        demo.restore_state()
        # do not set variation as suggested in commit 6e79c5bac. This version
        # of RLBench already stores the variation index
        #variation_index = demo._observations[0].misc["variation_index"]
        #self.set_variation(variation_index)
        return self.reset()
