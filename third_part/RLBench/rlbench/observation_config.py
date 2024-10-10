from typing import Dict
from dataclasses import dataclass

from pyrep.const import RenderMode
from rlbench.noise_model import NoiseModel, Identity


#@dataclass
class CameraConfig(object):
    def __init__(self,
                 rgb=True,
                 rgb_noise: NoiseModel=Identity(),
                 depth=True,
                 depth_noise: NoiseModel=Identity(),
                 point_cloud=True,
                 mask=True,
                 image_size=(128, 128),
                 render_mode=RenderMode.OPENGL3,
                 masks_as_one_channel=True, 
                 depth_in_meters=False,
                 # nerf_multi_view_mask =nerf_multi_view_mask,
                 ):
        self.rgb = rgb
        self.rgb_noise = rgb_noise
        self.depth = depth
        self.depth_noise = depth_noise
        self.point_cloud = point_cloud
        self.mask = mask
        self.image_size = image_size
        self.render_mode = render_mode
        self.masks_as_one_channel = masks_as_one_channel # 指定掩码是否作为单通道图像输出。默认值为 True，表示生成单通道的掩码图像
        self.depth_in_meters = depth_in_meters
        # self.nerf_multi_view_mask =nerf_multi_view_mask # 都是布尔型之类的

    def set_all(self, value: bool):
        """全部设为 value 值"""
        self.rgb = value
        self.depth = value
        self.point_cloud = value
        self.mask = value
        # # for nerf (mani里面也加了注释)
        # self.point_cloud_nerf = value


#@dataclass
class ObservationConfig(object):

    
    def __init__(self,
                 camera_configs: Dict[str, CameraConfig] = None,
                 joint_velocities=True, # for nerf                  # 是否记录机器人关节的速度
                 joint_velocities_noise: NoiseModel=Identity(),     # 噪声模型对象，用于给关节速度数据添加噪声。默认为 Identity()，表示不添加噪声。
                 joint_positions=True,                              # 是否记录机器人关节的位置
                 joint_positions_noise: NoiseModel=Identity(),      # 噪声模型对象，用于给关节位置数据添加噪声
                 joint_forces=True,                                 # 是否记录机器人关节的力
                 joint_forces_noise: NoiseModel=Identity(),
                 gripper_open=True,                                 # 可能指示是否记录夹爪的开合状态
                 gripper_pose=True,                                 # 是否记录夹爪的姿态                                
                 gripper_matrix=False,                              # 是否记录夹爪的变换矩阵       
                 gripper_joint_positions=False,                     # 是否记录夹爪关节的位置
                 gripper_touch_forces=False,                        # 是否记录夹爪的接触力
                 wrist_camera_matrix=False,                         # 是否记录手腕相机的变换矩阵
                 record_gripper_closing=False,                      # 是否记录夹爪关闭的状态
                 task_low_dim_state=True,                           # 是否记录任务的低维状态
                 record_ignore_collisions=True,                     # 是否记录忽略碰撞的状态
                 robot_name='',                                     # 指定机器人的名称        
                 nerf_multi_view=True, # for nerf（Mani）
                 ):
        # -------还有一下左右臂的设置没有改-----------nerf mani------------------------------------------------------------
        self.nerf_multi_view = nerf_multi_view
        # print(colored("[ObservationConfig] nerf_multi_view: {}".format(nerf_multi_view), "green"))
        # nerf mani------------------------------------------------------------
        self.camera_configs = camera_configs or dict()
        self.joint_velocities = joint_velocities
        self.joint_velocities_noise = joint_velocities_noise
        self.joint_positions = joint_positions
        self.joint_positions_noise = joint_positions_noise
        self.joint_forces = joint_forces
        self.joint_forces_noise = joint_forces_noise
        self.gripper_open = gripper_open
        self.gripper_pose = gripper_pose
        self.gripper_matrix = gripper_matrix
        self.gripper_joint_positions = gripper_joint_positions
        self.gripper_touch_forces = gripper_touch_forces
        self.wrist_camera_matrix = wrist_camera_matrix
        self.record_gripper_closing = record_gripper_closing
        self.task_low_dim_state = task_low_dim_state
        self.record_ignore_collisions = record_ignore_collisions
        self.robot_name = robot_name

    def set_all(self, value: bool):
        self.set_all_high_dim(value)
        self.set_all_low_dim(value)

    def set_all_high_dim(self, value: bool):
        for _, config in self.camera_configs:
            config.set_all(value)

    def set_all_low_dim(self, value: bool):
        self.joint_velocities = value
        self.joint_positions = value
        self.joint_forces = value
        self.gripper_open = value
        self.gripper_pose = value
        self.gripper_matrix = value
        self.gripper_joint_positions = value
        self.gripper_touch_forces = value
        self.wrist_camera_matrix = value
        self.task_low_dim_state = value
