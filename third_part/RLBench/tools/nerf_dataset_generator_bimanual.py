#!/usr/bin/env python3

import sys
import os
import logging
from functools import partial
import multiprocessing as mp
import pickle
import numpy as np

from rlbench import ObservationConfig
from rlbench.observation_config import CameraConfig

from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.arm_action_modes import BimanualJointVelocity
from rlbench.action_modes.arm_action_modes import BimanualJointPosition
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete

from rlbench.backend.exceptions import BoundaryError, InvalidActionError, TaskEnvironmentError, WaypointError
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as task
# ---------------------------------------------
from pyrep.const import RenderMode
from yarr.utils.video_utils import CircleCameraMotion, TaskRecorder, NeRFTaskRecorder
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
#------------------------------------------

from PIL import Image
from rlbench.backend import utils
from rlbench.backend.const import *
from rlbench.backend.task import BIMANUAL_TASKS_PATH


import rich_click as click
from rich.logging import RichHandler
from click_prompt import choice_option
from click_prompt import filepath_option


camera_names = ["over_shoulder_left", "over_shoulder_right", "overhead", "wrist_right", "wrist_left", "front"]


def save_demo(demo, example_path, variation):
    data_types = ["rgb", "depth", "point_cloud", "mask"]
    #full_camera_names = list(map(lambda x: ('_'.join(x), x[-1]), product(camera_names, data_types)))

    # Save image data first, and then None the image data, and pickle
    for i, obs in enumerate(demo):
        for camera_name in camera_names: # ["over_shoulder_left", "over_shoulder_right",
            for dtype in data_types:     #  ["rgb", "depth", "point_cloud", "mask"]

                camera_full_name = f"{camera_name}_{dtype}"
                data_path = os.path.join(example_path, camera_full_name)
                # ..todo:: actually I prefer to abort if this one exists
                os.makedirs(data_path, exist_ok=True)

                data = obs.perception_data.get(camera_full_name, None)

                if data is not None:
                    if dtype == 'rgb':                
                        data = Image.fromarray(data)
                    elif dtype == 'depth':
                        data = utils.float_array_to_rgb_image(data, scale_factor=DEPTH_SCALE)
                    elif dtype == 'point_cloud':
                        continue
                    elif dtype == 'mask':
                        data = Image.fromarray((data * 255).astype(np.uint8))
                    else:
                        raise Exception('Invalid data type')    
                    logging.debug("saving %s", camera_full_name)
                    data.save(os.path.join(data_path, f"{dtype}_{i:04d}.png"))
                    
        # ..why don't we put everything into a pickle file?
        obs.perception_data.clear()

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)

    with open(os.path.join(example_path, VARIATION_NUMBER), 'wb') as f:
        pickle.dump(variation, f)


def run_all_variations(task_name, headless, save_path, episodes_per_task, image_size):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation.
    每个线程都会选择一个任务和变体，然后收集 all 该变体的 Episodes_per_task"""

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    from rich.logging import RichHandler
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    logging.root.name = task_name

    logging.info("Collecting data for %s", task_name)

    # 根据任务文件名动态地导入并获取相应的任务类
    tasks = [task_file_to_task_class(task_name, True)]

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    # 生成的时候也不是meters
    default_config_params = {"image_size": image_size, "depth_in_meters": False, "masks_as_one_channel": False}
    camera_configs = {camera_name: CameraConfig(**default_config_params) for camera_name in camera_names}
    obs_config.camera_configs = camera_configs


    # ..record date with BimanualJointPosition
    robot_setup = 'dual_panda'
    rlbench_env = Environment(
        action_mode=BimanualMoveArmThenGripper(BimanualJointPosition(), BimanualDiscrete()),
        obs_config=obs_config,
        robot_setup=robot_setup,
        headless=headless)

    rlbench_env.launch()

    tasks_with_problems = ""

    # ========================================================================
    # nerf data generation
    # ========================================================================
    camera_resolution = [128, 128]
    rotate_speed = 0.1
    fps = 20
    num_views = 50      # !! yzj需要改到下面 FLAGS.num_views

    cam_placeholder = Dummy('cam_cinematic_placeholder')
    cam = VisionSensor.create(resolution=camera_resolution, render_mode=RenderMode.OPENGL)

    
    cam_front = VisionSensor('cam_front')
    pose_front = cam_front.get_pose()
    cam.set_pose(pose_front)
    
    # pose = cam_placeholder.get_pose()
    # pose[2] -= 0.2
    # cprint("camera pose: {}".format(pose), "green")
    # cam.set_pose(pose)

    cam.set_parent(cam_placeholder)

    # !!旋转
    cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), rotate_speed)
    # ========================================================================



    for task in tasks:
        
        task_env = rlbench_env.get_task(task)
        possible_variations = task_env.variation_count()

        logging.info("Task has %s possible variations", possible_variations)

        variation_path = os.path.join(save_path, task_env.get_name(), VARIATIONS_ALL_FOLDER)
        os.makedirs(variation_path, exist_ok=True)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        os.makedirs(episodes_path, exist_ok=True)

        # -------------------------continue generation 从上次结束的次数开始运行-------------------------
        file_names = os.listdir(episodes_path)
        import re
        episode_files = [f for f in file_names if f.startswith('episode') and re.match(r'episode\d+', f)]
        episode_numbers = [int(f[len('episode'):]) for f in episode_files]
        max_episode_number = max(episode_numbers) if episode_numbers else -1
        # print("max_episode_number", max_episode_number) # 目前文件夹中最大的情况
        # -------------------------continue generation-------------------------
        abort_variation = False
        for ex_idx in range(max_episode_number+1,episodes_per_task): # 改了
            attempts = 20 # 10
            # yzj !! nerf data generation---------------
            task_recorder = NeRFTaskRecorder(task_env, cam_motion, fps=fps, num_views=num_views)
            task_recorder._cam_motion.save_pose()


            while attempts > 0:
                try:
                    variation = np.random.randint(possible_variations)

                    task_env = rlbench_env.get_task(task)

                    task_env.set_variation(variation)
                    descriptions, obs = task_env.reset()
                    # !!
                    task_recorder.record_task_description(descriptions)

                    logging.info("// Task: %s Variation %s Demo %s", task_env.get_name(), variation, ex_idx)
                    # logging.info("1**try nerf data generation1")

                    # !! TODO: for now we do the explicit looping. 现在我们做显式循环。
                    demo, = task_env.get_demos(amount=1, live_demos=True,
                                               #!!新参数
                                               callable_each_step=task_recorder.take_snap
                                               )
                    logging.info("nerf gen完成get_demos")
                #  NoWaypointsError, DemoError,
                except (BoundaryError, WaypointError, InvalidActionError, TaskEnvironmentError) as e:
                    logging.warning("Exception %s", e)
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        'Failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            task_env.get_name(), variation, ex_idx,str(e))
                    )
                    logging.error(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break

                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
               
                # logging.info("3** save try nerf data generation1")
                save_demo(demo, episode_path, variation)

                with open(os.path.join( episode_path, VARIATION_DESCRIPTIONS), 'wb') as f:
                    pickle.dump(descriptions, f)

                # ========================================================================
                # yzj 新增nerf data generation
                # ========================================================================
                record_file_path = os.path.join(episode_path, 'nerf_data')
                task_recorder.save(record_file_path)
                task_recorder._cam_motion.restore_pose()

                break
            if abort_variation:
                break

    
    rlbench_env.shutdown()

    return tasks_with_problems



def get_bimanual_tasks():
    tasks =  [t.replace('.py', '') for t in
    os.listdir(BIMANUAL_TASKS_PATH) if t != '__init__.py' and t.endswith('.py')]
    return sorted(tasks)


# yzj!!原来写在这改image-size->image_size
@click.command()
@filepath_option("--save_path", default="/tmp/rlbench_data/",  help="Where to save the demos.")
@choice_option('--tasks', type=click.Choice(get_bimanual_tasks()), multiple=True, help='The tasks to collect. If empty, all tasks are collected.')
@click.option("--episodes_per_task", default=10, help="The number of episodes to collect per task.", prompt="Number of episodes")
# @click.option("--all_variations", is_flag=True, default=True, help="Include all variations when sampling epsiodes")
@click.option("--all_variations", default=True, help="Include all variations when sampling epsiodes")
#@click.option("--variations", default=-1, help="Number of variations to collect per task. -1 for all.")
@click.option("--headless/--no-headless", default=True, is_flag=True, help='Hide the simulator window')
#@click.option("--color-robot/--no-color-robot", default=False, is_flag=True, help='Colorize')
@choice_option('--image_size', type=click.Choice(["128x128", "256x256", "640x480"]), multiple=False, help='Select the image_size (width, height)')
def main(save_path, tasks, episodes_per_task, all_variations, headless, image_size):

    # ..todo check if already exits

    mp.set_start_method("spawn")

    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

    np.random.seed(None)

    ctx = mp.get_context('spawn')

    if not tasks:
        logging.error("No tasks selected!")


    logging.info("Generating %s episodes for each tasks %s with image size %s", episodes_per_task, tasks, image_size)


    image_size = list(map(int, image_size.split("x")))

    os.makedirs(save_path, exist_ok=True)

    if not all_variations:
        logging.error("Variations not supported")
        sys.exit(-1)

    logging.debug("Selected tasks %s", tasks)

    fn = partial(run_all_variations, headless=headless, save_path=save_path, episodes_per_task=episodes_per_task, image_size=image_size)
    # 创建进程池（最多4个进程）
    with ctx.Pool(processes=4) as pool:
        pool.map(fn, tasks)


if __name__ == '__main__':
  main()

