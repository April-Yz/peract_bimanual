# Adapted from ARM
# Source: https://github.com/stepjam/ARM   

# License: https://github.com/stepjam/ARM/LICENSE
"""
比传统的多了5个函数 + 加了一个checkpoint_name_prefix
Q-attention及其变体的代码库   ARM 使用 YARR 框架进行训练，并在 RLBench 1.1.0 上进行评估。
尽管强化学习方法取得了成功,我们提出了注意力驱动的机器人操纵(ARM)算法,
这是一种通用操纵算法，只需少量演示即可应用于一系列稀疏奖励任务。'
(1) Q-attention 代理从 RGB 和点云输入中提取相关像素位置，
(2) 次佳姿势代理，接受来自 Q-attention 代理的裁剪并输出姿势，
(3) 一个控制代理，它采用目标姿势并输出联合动作。
我们表明，当前的学习算法在一系列 RLBench 任务上都失败了，而 ARM 却成功了。"""

import logging
from typing import List

import numpy as np
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
import rlbench.utils as rlbench_utils
from rlbench.demo import Demo
from yarr.replay_buffer.prioritized_replay_buffer import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from yarr.replay_buffer.task_uniform_replay_buffer import TaskUniformReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer_single_process import UniformReplayBufferSingleProcess
from helpers import demo_loading_utils, utils
from helpers.preprocess_agent import PreprocessAgent
from helpers.clip.core.clip import tokenize
from helpers.language_model import create_language_model
from helpers import observation_utils # new

from agents.manigaussian_bc2.perceiver_lang_io import PerceiverVoxelLangEncoder
from agents.manigaussian_bc2.qattention_manigaussian_bc_agent import QAttentionPerActBCAgent
from agents.manigaussian_bc2.qattention_stack_agent import QAttentionStackAgent

import torch
import torch.nn as nn
import multiprocessing as mp
from torch.multiprocessing import Process, Value, Manager
from omegaconf import DictConfig
from termcolor import colored, cprint
from lightning.fabric import Fabric


REWARD_SCALE = 100.0
LOW_DIM_SIZE = 4

def create_replay(cfg, replay_path):
    
    if cfg.method.robot_name == "bimanual":
        return create_bimanual_replay(            
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cfg.rlbench.cameras,
            cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution,
            cfg=cfg
        )
    else:
        return create_unimanual_replay(            
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cfg.rlbench.cameras,
            cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution,
        )


def create_bimanual_replay(batch_size: int, timesteps: int,
                  prioritisation: bool, task_uniform: bool,
                  save_dir: str, cameras: list,
                  voxel_sizes,
                  image_size=[128, 128],
                  replay_size=3e5,
                  single_process=False,
                  cfg=None,):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = (3 + 1)
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = cfg.method.language_model_dim # !!(原来是512)要不要改成512
    # 绿色输出[create_replay] lang_emb_dim:值
    cprint(f"[create_replay] lang_emb_dim: {lang_emb_dim}", "green")

    # !! --Gaussian特有---------------------------------------------
    num_view_for_nerf = cfg.rlbench.num_view_for_nerf
    # !! -----------------------------------------------
    
    # low_dim_state-------bimanual特有-------------------------------------------
    # observation_elements = []
    # observation_elements.append(
        # ObservationElement('low_dim_state', (LOW_DIM_SIZE,), np.float32))
    # observation_elements 增加一个， 下面增加了depth参数
    observation_elements = []
    observation_elements.append(
        ObservationElement("right_low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )
    #--------------------------------------------------------------------------------------
    # for store_element in observation_elements:
    #     if store_element.name == "right_low_dim_state":
    #         print("--------------------------------manigaussian:launch_utils:create_replay-----------------------")
    #         print("store_element.name=",store_element.name)
    #         print("store_element=",store_element)
    #         print("store_element.shape=",store_element.shape)
    #-----------------------------------------------------------------------------------
    observation_elements.append(
        ObservationElement("left_low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )
    
    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            # color, height, width 新增depth
            ObservationElement(
                "%s_rgb" % cname,
                (3,image_size[1],image_size[0],),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement('%s_depth' % cname, (1, image_size[1], image_size[0]), np.float32))
        observation_elements.append(
            ObservationElement("%s_point_cloud" % cname, (3, image_size[1], image_size[0]), np.float16)
        )  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement(
                "%s_camera_extrinsics" % cname,
                (4,4,),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_camera_intrinsics" % cname,
                (3,3,),
                np.float32,
            )
        )        


        """ Mani单臂时不区分左右的写法
        observation_elements.append(
            ObservationElement('%s_rgb' % cname, (3, *image_size,), np.float32))
        observation_elements.append(
            ObservationElement('%s_depth' % cname, (1, *image_size,), np.float32))
        observation_elements.append(
            ObservationElement('%s_point_cloud' % cname, (3, *image_size),
                               np.float32))  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement('%s_camera_extrinsics' % cname, (4, 4,), np.float32))
        observation_elements.append(
            ObservationElement('%s_camera_intrinsics' % cname, (3, 3,), np.float32))
        """
        #------加了nerf和-------------------------------------------------------------

    #-------------------------------NERF----------------------------------------------
    # for nerf img, exs, ins
    # 使用 append 时，添加的对象会作为一个单一的元素（无论它本身是一个列表还是其他类型）。
    # 使用 extend 时，添加的对象（必须是可迭代的）会被拆分，其元素会被逐个添加到列表中。
    observation_elements.append(
        ObservationElement('nerf_multi_view_rgb', (num_view_for_nerf,), np.object_))
    # ------------------------2024.8.5-----------------------------------------------------------------
    # for store_element in observation_elements:
    #     if store_element.name == "nerf_multi_view_rgb":
    #         print("--------------------Manigaussian_BC2/launch_utils-----nerf_multi_view_rgb-----------------------")
    #         print("store_element.name=",store_element.name)
    #         print("store_element=",store_element)
    #         print("store_element.shape=",store_element.shape)
    # ------------------------2024.8.5-----------------------------------------------------------------
    observation_elements.append(
        ObservationElement('nerf_multi_view_depth', (num_view_for_nerf,), np.object_))
    observation_elements.append(
        ObservationElement('nerf_multi_view_camera', (num_view_for_nerf,), np.object_))
    
    # for next nerf
    observation_elements.append(
        ObservationElement('nerf_next_multi_view_rgb', (num_view_for_nerf,), np.object_))
    observation_elements.append(
        ObservationElement('nerf_next_multi_view_depth', (num_view_for_nerf,), np.object_))
    observation_elements.append(
        ObservationElement('nerf_next_multi_view_camera', (num_view_for_nerf,), np.object_))
    #-------------------------------NERF----------------------------------------------



    #-------------------------------bimanual双臂----------------------------------------------
    # 离散化翻译、离散化旋转、离散忽略碰撞、6-DoF 抓手姿势和预训练语言嵌入
    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    for robot_name in ["right", "left"]:
        observation_elements.extend(
            [
                ReplayElement(
                    f"{robot_name}_trans_action_indicies",
                    (trans_indicies_size,),
                    np.int32,
                ),
                ReplayElement(
                    f"{robot_name}_rot_grip_action_indicies",
                    (rot_and_grip_indicies_size,),
                    np.int32,
                ),
                ReplayElement(
                    f"{robot_name}_ignore_collisions",
                    (ignore_collisions_size,),
                    np.int32,
                ),
                ReplayElement(
                    f"{robot_name}_gripper_pose", (gripper_pose_size,), np.float32
                ),
            ]
        )
    # observation_elements.extend([
    #     ReplayElement('trans_action_indicies', (trans_indicies_size,),
    #                   np.int32),
    #     ReplayElement('rot_grip_action_indicies', (rot_and_grip_indicies_size,),
    #                   np.int32),
    #     ReplayElement('ignore_collisions', (ignore_collisions_size,),
    #                   np.int32),
    #     ReplayElement('gripper_pose', (gripper_pose_size,),
    #                   np.float32),
    # -------上面在双臂中改了一部分----------------------------------------------------------
    #     ReplayElement('lang_goal_emb', (lang_feat_dim,),
    #                   np.float32),
    #     ReplayElement('lang_token_embs', (max_token_seq_len, lang_emb_dim,),
    #                   np.float32), # extracted from CLIP's language encoder
    #     ReplayElement('task', (),str),
    #     ReplayElement('lang_goal', (1,),object),  # language goal string for debugging and visualization
    # ])
    observation_elements.extend(
        [
            ReplayElement("lang_goal_emb", (lang_feat_dim,), np.float32),
            ReplayElement(
                "lang_token_embs",
                (
                    max_token_seq_len,
                    lang_emb_dim,
                ),
                np.float32,
            ),  # extracted from CLIP's language encoder
            ReplayElement("task", (), str),
            ReplayElement(
                "lang_goal", (1,), object
            ),  # language goal string for debugging and visualization
        ]
    )

    extra_replay_elements = [
        ReplayElement('demo', (), np.bool),
    ]
    if not single_process:  # default: False
        # TaskUniformReplayBuffer不用改
        replay_buffer = TaskUniformReplayBuffer(
            save_dir=save_dir,
            batch_size=batch_size,
            timesteps=timesteps,
            replay_capacity=int(replay_size),
            # action_shape=(8,), # 单臂时的大小
            action_shape=(8 * 2,),
            action_dtype=np.float32,
            reward_shape=(),
            reward_dtype=np.float32,
            update_horizon=1,
            observation_elements=observation_elements,
            extra_replay_elements=extra_replay_elements,
        )
    else:
        # 奇怪的地方
        cprint(f"[create_replay] single process:单进程了(peract都没这个选项)", "red")
        replay_buffer = UniformReplayBufferSingleProcess(
            save_dir=save_dir,
            batch_size=batch_size,
            timesteps=timesteps,
            replay_capacity=int(replay_size),
            # action_shape=(8,),
            action_shape=(8 * 2,),
            action_dtype=np.float32,
            reward_shape=(),
            reward_dtype=np.float32,
            update_horizon=1,
            observation_elements=observation_elements,
            extra_replay_elements=extra_replay_elements
        )
    return replay_buffer

# 还是双臂相似代码（未改）
def create_unimanual_replay(batch_size: int, timesteps: int,
                  prioritisation: bool, task_uniform: bool,
                  save_dir: str, cameras: list,
                  voxel_sizes,
                  image_size=[128, 128],
                  replay_size=3e5,
                  single_process=False,
                  cfg=None,):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = (3 + 1)
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = cfg.method.language_model_dim # !!(原来是512)要不要改成512
    # 绿色输出[create_replay] lang_emb_dim:值
    cprint(f"[create_replay] lang_emb_dim: {lang_emb_dim}", "green")

    # !! --Gaussian特有---------------------------------------------
    num_view_for_nerf = cfg.rlbench.num_view_for_nerf
    # !! -----------------------------------------------
    
    # low_dim_state-------bimanual特有-------------------------------------------
    # observation_elements = []
    # observation_elements.append(
        # ObservationElement('low_dim_state', (LOW_DIM_SIZE,), np.float32))
    # observation_elements 增加一个， 下面增加了depth参数
    observation_elements = []
    observation_elements.append(
        ObservationElement("right_low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )
    #--------------------------------------------------------------------------------------
    # for store_element in observation_elements:
    #     if store_element.name == "right_low_dim_state":
    #         print("--------------------------------manigaussian:launch_utils:create_replay-----------------------")
    #         print("store_element.name=",store_element.name)
    #         print("store_element=",store_element)
    #         print("store_element.shape=",store_element.shape)
    #-----------------------------------------------------------------------------------
    observation_elements.append(
        ObservationElement("left_low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )
    
    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            # color, height, width 新增depth
            ObservationElement(
                "%s_rgb" % cname,
                (3,image_size[1],image_size[0],),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement('%s_depth' % cname, (1, image_size[1], image_size[0]), np.float32))
        observation_elements.append(
            ObservationElement("%s_point_cloud" % cname, (3, image_size[1], image_size[0]), np.float16)
        )  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement(
                "%s_camera_extrinsics" % cname,
                (4,4,),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_camera_intrinsics" % cname,
                (3,3,),
                np.float32,
            )
        )        


        """
        observation_elements.append(
            ObservationElement('%s_rgb' % cname, (3, *image_size,), np.float32))
        observation_elements.append(
            ObservationElement('%s_depth' % cname, (1, *image_size,), np.float32))
        observation_elements.append(
            ObservationElement('%s_point_cloud' % cname, (3, *image_size),
                               np.float32))  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement('%s_camera_extrinsics' % cname, (4, 4,), np.float32))
        observation_elements.append(
            ObservationElement('%s_camera_intrinsics' % cname, (3, 3,), np.float32))
        """
        #------加了nerf和-------------------------------------------------------------

    #-------------------------------NERF----------------------------------------------
    # for nerf img, exs, ins
    # 使用 append 时，添加的对象会作为一个单一的元素（无论它本身是一个列表还是其他类型）。
    # 使用 extend 时，添加的对象（必须是可迭代的）会被拆分，其元素会被逐个添加到列表中。
    observation_elements.append(
        ObservationElement('nerf_multi_view_rgb', (num_view_for_nerf,), np.object_))
    # ------------------------2024.8.5-----------------------------------------------------------------
    # for store_element in observation_elements:
    #     if store_element.name == "nerf_multi_view_rgb":
    #         print("--------------------Manigaussian_BC2/launch_utils-----nerf_multi_view_rgb-----------------------")
    #         print("store_element.name=",store_element.name)
    #         print("store_element=",store_element)
    #         print("store_element.shape=",store_element.shape)
    # ------------------------2024.8.5-----------------------------------------------------------------
    observation_elements.append(
        ObservationElement('nerf_multi_view_depth', (num_view_for_nerf,), np.object_))
    observation_elements.append(
        ObservationElement('nerf_multi_view_camera', (num_view_for_nerf,), np.object_))
    
    # for next nerf
    observation_elements.append(
        ObservationElement('nerf_next_multi_view_rgb', (num_view_for_nerf,), np.object_))
    observation_elements.append(
        ObservationElement('nerf_next_multi_view_depth', (num_view_for_nerf,), np.object_))
    observation_elements.append(
        ObservationElement('nerf_next_multi_view_camera', (num_view_for_nerf,), np.object_))
    #-------------------------------NERF----------------------------------------------



    #-------------------------------bimanual双臂----------------------------------------------
    # 离散化翻译、离散化旋转、离散忽略碰撞、6-DoF 抓手姿势和预训练语言嵌入
    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    for robot_name in ["right", "left"]:
        observation_elements.extend(
            [
                ReplayElement(
                    f"{robot_name}_trans_action_indicies",
                    (trans_indicies_size,),
                    np.int32,
                ),
                ReplayElement(
                    f"{robot_name}_rot_grip_action_indicies",
                    (rot_and_grip_indicies_size,),
                    np.int32,
                ),
                ReplayElement(
                    f"{robot_name}_ignore_collisions",
                    (ignore_collisions_size,),
                    np.int32,
                ),
                ReplayElement(
                    f"{robot_name}_gripper_pose", (gripper_pose_size,), np.float32
                ),
            ]
        )
    # observation_elements.extend([
    #     ReplayElement('trans_action_indicies', (trans_indicies_size,),
    #                   np.int32),
    #     ReplayElement('rot_grip_action_indicies', (rot_and_grip_indicies_size,),
    #                   np.int32),
    #     ReplayElement('ignore_collisions', (ignore_collisions_size,),
    #                   np.int32),
    #     ReplayElement('gripper_pose', (gripper_pose_size,),
    #                   np.float32),
    # -------上面在双臂中改了一部分----------------------------------------------------------
    #     ReplayElement('lang_goal_emb', (lang_feat_dim,),
    #                   np.float32),
    #     ReplayElement('lang_token_embs', (max_token_seq_len, lang_emb_dim,),
    #                   np.float32), # extracted from CLIP's language encoder
    #     ReplayElement('task', (),str),
    #     ReplayElement('lang_goal', (1,),object),  # language goal string for debugging and visualization
    # ])
    observation_elements.extend(
        [
            ReplayElement("lang_goal_emb", (lang_feat_dim,), np.float32),
            ReplayElement(
                "lang_token_embs",
                (
                    max_token_seq_len,
                    lang_emb_dim,
                ),
                np.float32,
            ),  # extracted from CLIP's language encoder
            ReplayElement("task", (), str),
            ReplayElement(
                "lang_goal", (1,), object
            ),  # language goal string for debugging and visualization
        ]
    )

    extra_replay_elements = [
        ReplayElement('demo', (), np.bool),
    ]
    if not single_process:  # default: False
        # TaskUniformReplayBuffer不用改
        replay_buffer = TaskUniformReplayBuffer(
            save_dir=save_dir,
            batch_size=batch_size,
            timesteps=timesteps,
            replay_capacity=int(replay_size),
            # action_shape=(8,), # 单臂时的大小
            action_shape=(8 * 2,),
            action_dtype=np.float32,
            reward_shape=(),
            reward_dtype=np.float32,
            update_horizon=1,
            observation_elements=observation_elements,
            extra_replay_elements=extra_replay_elements,
        )
    else:
        replay_buffer = UniformReplayBufferSingleProcess(
            save_dir=save_dir,
            batch_size=batch_size,
            timesteps=timesteps,
            replay_capacity=int(replay_size),
            action_shape=(8,),
            action_dtype=np.float32,
            reward_shape=(),
            reward_dtype=np.float32,
            update_horizon=1,
            observation_elements=observation_elements,
            extra_replay_elements=extra_replay_elements
        )
    return replay_buffer


def _get_action(
        obs_tp1: Observation,
        obs_tm1: Observation,
        rlbench_scene_bounds: List[float], # metric 3D bounds of the scene
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool):
    '''
    obs_tp1: current observation    obs_tp1：当前观测值
    obs_tm1: previous observation    obs_tm1：先前的观察
    '''
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    disc_rot = utils.correct_rotation_instability(disc_rot, rotation_resolution)

    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(voxel_sizes): # only single voxelization-level is used in PerAct
        if depth > 0:
            if crop_augmentation:
                shift = bounds_offset[depth - 1] * 0.75
                attention_coordinate += np.random.uniform(-shift, shift, size=(3,))
            bounds = np.concatenate([attention_coordinate - bounds_offset[depth - 1],
                                     attention_coordinate + bounds_offset[depth - 1]])
        index = utils.point_to_voxel_index(
            obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return trans_indicies, rot_and_grip_indicies, ignore_collisions, np.concatenate(
        [obs_tp1.gripper_pose, np.array([grip])]), attention_coordinates


def _add_keypoints_to_replay(
        cfg: DictConfig,
        task: str,
        replay: ReplayBuffer,
        inital_obs: Observation,
        demo: Demo,
        episode_keypoints: List[int],
        cameras: List[str],
        rlbench_scene_bounds: List[float],
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool,
        description: str = '',
        language_model = None,
        device = 'cpu'):
    #  bimanual!! 双臂必须要加（其他参数两边类似）-------------------
    robot_name = cfg.method.robot_name

    prev_action = None
    obs = inital_obs    # initial observation is 0

   
    for k, keypoint in enumerate(episode_keypoints):    # demo[-1].nerf_multi_view_rgb is None
        obs_tp1 = demo[keypoint]    # e.g, 44
        obs_tm1 = demo[max(0, keypoint - 1)]    # previous observation, e.g., 43
        # -----bimanual----------
        if obs_tp1.is_bimanual and robot_name == "bimanual":
            #assert isinstance(obs_tp1, BimanualObservation)
            (
                right_trans_indicies,
                right_rot_grip_indicies,
                right_ignore_collisions,
                right_action,
                right_attention_coordinates,
            ) = _get_action(
                obs_tp1.right,
                obs_tm1.right,
                rlbench_scene_bounds,
                voxel_sizes,
                bounds_offset,
                rotation_resolution,
                crop_augmentation,
            )

            (
                left_trans_indicies,
                left_rot_grip_indicies,
                left_ignore_collisions,
                left_action,
                left_attention_coordinates,
            ) = _get_action(
                obs_tp1.left,
                obs_tm1.left,
                rlbench_scene_bounds,
                voxel_sizes,
                bounds_offset,
                rotation_resolution,
                crop_augmentation,
            )
            #下面三行不懂(好像是action整合+格式转换)
            action = np.append(right_action, left_action)

            right_ignore_collisions = np.array([right_ignore_collisions])
            left_ignore_collisions = np.array([left_ignore_collisions])

        # 不用看（双臂在上面if中）    
        elif robot_name == "unimanual":
            (
                trans_indicies,
                rot_grip_indicies,
                ignore_collisions,
                action,
                attention_coordinates,
            ) = _get_action(
                obs_tp1,
                obs_tm1,
                rlbench_scene_bounds,
                voxel_sizes,
                bounds_offset,
                rotation_resolution,
                crop_augmentation,
            )
            gripper_pose = obs_tp1.gripper_pose
        elif obs_tp1.is_bimanual and robot_name == "right":
            (
                trans_indicies,
                rot_grip_indicies,
                ignore_collisions,
                action,
                attention_coordinates,
            ) = _get_action(
                obs_tp1.right,
                obs_tm1.right,
                rlbench_scene_bounds,
                voxel_sizes,
                bounds_offset,
                rotation_resolution,
                crop_augmentation,
            )
            gripper_pose = obs_tp1.right.gripper_pose
        elif obs_tp1.is_bimanual and robot_name == "left":
            (
                trans_indicies,
                rot_grip_indicies,
                ignore_collisions,
                action,
                attention_coordinates,
            ) = _get_action(
                obs_tp1.left,
                obs_tm1.left,
                rlbench_scene_bounds,
                voxel_sizes,
                bounds_offset,
                rotation_resolution,
                crop_augmentation,
            )
            gripper_pose = obs_tp1.left.gripper_pose
        else:
            logging.error("Invalid robot name %s", cfg.method.robot_name)
            raise Exception("Invalid robot name.")

        # 原本的单臂mani代码
        # trans_indicies, rot_grip_indicies, ignore_collisions, action, attention_coordinates = _get_action(
        #     obs_tp1, obs_tm1, rlbench_scene_bounds, voxel_sizes, bounds_offset,
        #     rotation_resolution, crop_augmentation)
        #------------------------------------------------------------


        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) * REWARD_SCALE if terminal else 0

        # ---新增的双臂方法--------------------和原来的NERF（clip用的原来的）--------------------------------------
        obs_dict = observation_utils.extract_obs(
            obs,
            t=k,
            prev_action=prev_action,
            cameras=cameras,
            episode_length=cfg.rlbench.episode_length,
            robot_name=robot_name,
            next_obs=obs_tp1 if not terminal else obs_tm1   #新增的
        )
        # 以下三行均是针对language做的预处理
        # tokens = tokenize([description]).numpy()
        # token_tensor = torch.from_numpy(tokens).to(device)
        # sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)
        # ----------下面是原来mani的-------------------------------------------------------
        # obs_dict = utils.extract_obs(obs, t=k, prev_action=prev_action,
        #                              cameras=cameras, episode_length=cfg.rlbench.episode_length,
        #                              next_obs=obs_tp1 if not terminal else obs_tm1,
        #                              )
        # FIXME: better way to use the last sample for next frame prediction?
        # FIXME：使用最后一个样本进行下一帧预测的更好方法？
        sentence_emb, token_embs = language_model.extract(description)
        #---------------------------------------------------------------

        obs_dict['lang_goal_emb'] = sentence_emb[0].float().detach().cpu().numpy()
        obs_dict['lang_token_embs'] = token_embs[0].float().detach().cpu().numpy()
        # 下一行mani多的
        obs_dict['lang_goal'] = np.array([description], dtype=object) # add this for usage in diffusion model

        prev_action = np.copy(action)

        others = {'demo': True}
        if robot_name == "bimanual":

            final_obs = {
                "right_trans_action_indicies": right_trans_indicies,
                "right_rot_grip_action_indicies": right_rot_grip_indicies,
                "right_gripper_pose": obs_tp1.right.gripper_pose,
                "left_trans_action_indicies": left_trans_indicies,
                "left_rot_grip_action_indicies": left_rot_grip_indicies,
                "left_gripper_pose": obs_tp1.left.gripper_pose,
                "task": task,
                "lang_goal": np.array([description], dtype=object),
            }
        else:
            final_obs = {
                "trans_action_indicies": trans_indicies,
                "rot_grip_action_indicies": rot_grip_indicies,
                "gripper_pose": gripper_pose,
                "task": task,
                "lang_goal": np.array([description], dtype=object),
            }
        # mani原有的
        # final_obs = {
        #     'trans_action_indicies': trans_indicies,
        #     'rot_grip_action_indicies': rot_grip_indicies,
        #     'gripper_pose': obs_tp1.gripper_pose,
        #     'task': task,
        #     'lang_goal': np.array([description], dtype=object),
        # }

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        replay.add(action, reward, terminal, timeout, **others)
        obs = obs_tp1

    # final step
    obs_dict_tp1 = observation_utils.extract_obs(
        obs_tp1,
        t=k + 1,
        prev_action=prev_action,
        cameras=cameras,
        episode_length=cfg.rlbench.episode_length,
        robot_name=cfg.method.robot_name,
        next_obs=obs_tp1 # 新增的
    )
    # obs_dict_tp1 = utils.extract_obs(obs_tp1, t=k + 1, prev_action=prev_action,
    #                                  cameras=cameras, episode_length=cfg.rlbench.episode_length,
    #                                  next_obs=obs_tp1,  
    #                                  )

    # nerf_multi_view_rgb is None
    obs_dict_tp1['lang_goal_emb'] = sentence_emb[0].float().detach().cpu().numpy()
    obs_dict_tp1['lang_token_embs'] = token_embs[0].float().detach().cpu().numpy()
    # mani new!!
    obs_dict_tp1['lang_goal'] = np.array([description], dtype=object) # add this for usage in diffusion model

    obs_dict_tp1.pop('wrist_world_to_cam', None)
    obs_dict_tp1.update(final_obs)
    # check nerf data here. find None 在此处查看 NERF 数据。查找 无
    replay.add_final(**obs_dict_tp1)


def fill_replay(cfg: DictConfig,
                obs_config: ObservationConfig,
                rank: int,
                replay: ReplayBuffer,
                task: str,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
                voxel_sizes: List[int],
                bounds_offset: List[float],
                rotation_resolution: int,
                crop_augmentation: bool,
                language_model = None,
                device = 'cpu',
                keypoint_method = 'heuristic'):
    logging.getLogger().setLevel(cfg.framework.logging_level)    
    # ---bimanual的language modle（）clip------------------------
    # if clip_model is None:
    #     model, _ = load_clip("RN50", jit=False, device=device)
    #     clip_model = build_model(model.state_dict())
    #     clip_model.to(device)
    #     del model    
    # ---bimanual的language modle（）clip------------------------

    logging.debug('Filling %s replay ...' % task)
    for d_idx in range(num_demos):
        # load demo from disk
        demo = rlbench_utils.get_stored_demos(
            amount=1, image_paths=False,
            dataset_root=cfg.rlbench.demo_path,
            variation_number=-1, task_name=task,
            obs_config=obs_config,
            random_selection=False,
            from_episode_number=d_idx)[0]

        
        descs = demo._observations[0].misc['descriptions']

        # extract keypoints (a.k.a keyframes)
        episode_keypoints = demo_loading_utils.keypoint_discovery(demo, method=keypoint_method)

        if rank == 0:   # always 0
            logging.info(f"Loading Demo({d_idx}) - found {len(episode_keypoints)} keypoints - {task}")

        for i in range(len(demo) - 1):
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0:
                continue

            obs = demo[i]

            desc = descs[0]
            # if our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                episode_keypoints = episode_keypoints[1:]
            if len(episode_keypoints) == 0:
                break
            
            _add_keypoints_to_replay(
                cfg, task, replay, obs, demo, episode_keypoints,
                # 中间mani新的
                cameras,
                rlbench_scene_bounds, voxel_sizes, bounds_offset,
                rotation_resolution, crop_augmentation, 
                # 下面共有的（language和clip不一样）
                description=desc,
                language_model=language_model, device=device)
    logging.debug('Replay %s filled with demos.' % task)


def fill_multi_task_replay(cfg: DictConfig,
                           obs_config: ObservationConfig,
                           rank: int,   # non-sense
                           replay: ReplayBuffer,
                           tasks: List[str],
                        #  clip_model=None,(bimanual特有参数但是无传参，so None)
                        # 下面都是mani新参数    
                           num_demos: int,
                           demo_augmentation: bool,
                           demo_augmentation_every_n: int,
                           cameras: List[str],
                           rlbench_scene_bounds: List[float],
                           voxel_sizes: List[int],
                           bounds_offset: List[float],
                           rotation_resolution: int,
                           crop_augmentation: bool,
                           keypoint_method = 'heuristic',
                           fabric: Fabric = None):
    # tasks = cfg.rlbench.tasks bimanual特有
    manager = Manager()
    store = manager.dict()

    # create a MP dict for storing indicies
    # TODO(mohit): this shouldn't be initialized here
    # mani加了个判断
    if hasattr(replay, '_task_idxs'):
        del replay._task_idxs
    task_idxs = manager.dict()

    replay._task_idxs = task_idxs
    replay._create_storage(store)
    replay.add_count = Value('i', 0)

    # fill replay buffer in parallel across tasks
    # 跨任务并行填充重放缓冲区
    max_parallel_processes = cfg.replay.max_parallel_processes # 32
    processes = []
    n = np.arange(len(tasks))
    # 把n分成每个最多32个的list？
    split_n = utils.split_list(n, max_parallel_processes)
    
    # ------------------mani language_model-----------------------------------
    device = fabric.device if fabric is not None else None
    language_model = create_language_model(name=cfg.method.language_model, device=device)
    # ------------------mani language_model-----------------------------------

    for split in split_n:
        for e_idx, task_idx in enumerate(split):
            task = tasks[int(task_idx)]
            model_device = torch.device('cuda:%s' % (e_idx % torch.cuda.device_count())
                                        if torch.cuda.is_available() else 'cpu')    # NOT USED
            p = Process(target=fill_replay, args=(cfg,
                                                  obs_config,
                                                  rank,
                                                  replay,
                                                  task,
                                                # clip_model,(bimanual)
                                                # 以下除了device mani特有
                                                  num_demos,
                                                  demo_augmentation,
                                                  demo_augmentation_every_n,
                                                  cameras,
                                                  rlbench_scene_bounds,
                                                  voxel_sizes,
                                                  bounds_offset,
                                                  rotation_resolution,
                                                  crop_augmentation,
                                                  language_model,
                                                  model_device,
                                                  keypoint_method))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()



def create_agent(cfg: DictConfig):
    LATENT_SIZE = 64
    # 场景边界信息
    depth_0bounds = cfg.rlbench.scene_bounds
    # 相机分辨率
    cam_resolution = cfg.rlbench.camera_resolution

    # 根据旋转分辨率计算旋转类别的数量。
    num_rotation_classes = int(360. // cfg.method.rotation_resolution)
    qattention_agents = []
    # 遍历配置中定义的体素大小列表
    for depth, vox_size in enumerate(cfg.method.voxel_sizes):   # default: [100]
        # 是否是最后一个
        last = depth == len(cfg.method.voxel_sizes) - 1
        perceiver_encoder = PerceiverVoxelLangEncoder(
            depth=cfg.method.transformer_depth, # 6
            iterations=cfg.method.transformer_iterations,
            voxel_size=vox_size,
            initial_dim=3 + 3 + 1 + 3,
            low_dim_size=cfg.method.low_dim_size, # 4,
            layer=depth,
            num_rotation_classes=num_rotation_classes if last else 0,
            num_grip_classes=2 if last else 0,
            num_collision_classes=2 if last else 0,
            input_axis=3,
            num_latents = cfg.method.num_latents,
            latent_dim = cfg.method.latent_dim,
            cross_heads = cfg.method.cross_heads,
            latent_heads = cfg.method.latent_heads,
            cross_dim_head = cfg.method.cross_dim_head,
            latent_dim_head = cfg.method.latent_dim_head,
            weight_tie_layers = False,
            activation = cfg.method.activation,
            pos_encoding_with_lang=cfg.method.pos_encoding_with_lang,
            input_dropout=cfg.method.input_dropout,
            attn_dropout=cfg.method.attn_dropout,
            decoder_dropout=cfg.method.decoder_dropout,
            lang_fusion_type=cfg.method.lang_fusion_type,
            voxel_patch_size=cfg.method.voxel_patch_size,
            voxel_patch_stride=cfg.method.voxel_patch_stride,
            no_skip_connection=cfg.method.no_skip_connection,
            no_perceiver=cfg.method.no_perceiver,
            no_language=cfg.method.no_language,
            final_dim=cfg.method.final_dim,
            # yzj下面两个新增
            im_channels=cfg.method.final_dim,
            cfg=cfg,
        )

        qattention_agent = QAttentionPerActBCAgent(
            layer=depth,
            coordinate_bounds=depth_0bounds,
            perceiver_encoder=perceiver_encoder,
            camera_names=cfg.rlbench.cameras,
            voxel_size=vox_size,
            bounds_offset=cfg.method.bounds_offset[depth - 1] if depth > 0 else None,
            image_crop_size=cfg.method.image_crop_size,
            lr=cfg.method.lr,
            training_iterations=cfg.framework.training_iterations,
            lr_scheduler=cfg.method.lr_scheduler,
            num_warmup_steps=cfg.method.num_warmup_steps,
            trans_loss_weight=cfg.method.trans_loss_weight,
            rot_loss_weight=cfg.method.rot_loss_weight,
            grip_loss_weight=cfg.method.grip_loss_weight,
            collision_loss_weight=cfg.method.collision_loss_weight,
            include_low_dim_state=True,
            image_resolution=cam_resolution,
            batch_size=cfg.replay.batch_size,
            voxel_feature_size=3,
            lambda_weight_l2=cfg.method.lambda_weight_l2,
            num_rotation_classes=num_rotation_classes,
            rotation_resolution=cfg.method.rotation_resolution,
            transform_augmentation=cfg.method.transform_augmentation.apply_se3,
            transform_augmentation_xyz=cfg.method.transform_augmentation.aug_xyz,
            transform_augmentation_rpy=cfg.method.transform_augmentation.aug_rpy,
            transform_augmentation_rot_resolution=cfg.method.transform_augmentation.aug_rot_resolution,
            optimizer_type=cfg.method.optimizer,
            num_devices=cfg.ddp.num_devices,
            # yzj !! 隔壁有的
            # checkpoint_name_prefix=cfg.framework.checkpoint_name_prefix,
            # cfg这里特有的
            cfg=cfg.method,
        )
        qattention_agents.append(qattention_agent)

    rotation_agent = QAttentionStackAgent(
        qattention_agents=qattention_agents,
        rotation_resolution=cfg.method.rotation_resolution,
        camera_names=cfg.rlbench.cameras,
    )
    preprocess_agent = PreprocessAgent(
        pose_agent=rotation_agent
    )
    return preprocess_agent
