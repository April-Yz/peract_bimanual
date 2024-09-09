import logging
import os
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from yarr.agents.agent import Agent, ActResult, ScalarSummary, \
    HistogramSummary, ImageSummary, Summary
from termcolor import colored, cprint
import io

from helpers.utils import visualise_voxel
from voxel.voxel_grid import VoxelGrid
from voxel.augmentation import apply_se3_augmentation_with_camera_pose
from helpers.clip.core.clip import build_model, load_clip
import PIL.Image as Image
import transformers
from helpers.optim.lamb import Lamb
from torch.nn.parallel import DistributedDataParallel as DDP
from agents.manigaussian_bc2.neural_rendering import NeuralRenderer
from agents.manigaussian_bc2.utils import visualize_pcd
from helpers.language_model import create_language_model

import wandb
import visdom
from lightning.fabric import Fabric


NAME = 'QAttentionAgent'


def visualize_feature_map_by_clustering(features, num_cluster=4, return_cluster_center=False):
    """未变(辅助函数)聚类对特征图进行可视化，将不同的特征区域着色"""
    from sklearn.cluster import KMeans
    features = features.cpu().detach().numpy()
    B, D, H, W = features.shape
    features_1d = features.reshape(B, D, H*W).transpose(0, 2, 1).reshape(-1, D)
    kmeans = KMeans(n_clusters=num_cluster, random_state=0, n_init=10).fit(features_1d)
    labels = kmeans.labels_
    labels = labels.reshape(H, W)

    cluster_colors = [
        np.array([255, 0, 0]),   # red
        np.array([0, 255, 0]),       # green
        np.array([0, 0, 255]),      # blue
        np.array([255, 255, 0]),   # yellow
        np.array([255, 0, 255]),  # magenta
        np.array([0, 255, 255]),    # cyan
    ]

    segmented_img = np.zeros((H, W, 3))
    for i in range(num_cluster):
        segmented_img[labels==i] = cluster_colors[i]
        
    if return_cluster_center:
        cluster_centers = []
        for i in range(num_cluster):
            cluster_pixels = np.argwhere(labels == i)
            cluster_center = cluster_pixels.mean(axis=0)
            cluster_centers.append(cluster_center)
        return labels, cluster_centers
        
    return segmented_img

# (新增) 
def visualize_feature_map_by_normalization(features):
    '''
    (新增)通过归一化对特征图进行可视化，以便在绘图时更容易观察。
    将特征图归一化为 [0, 1] for plt.show()
    Normalize feature map to [0, 1] for plt.show()
    :features: (B, 3, H, W)
    Return: (H, W, 3)
    '''
    MIN_DENOMINATOR = 1e-12
    features = features[0].cpu().detach().numpy()
    features = features.transpose(1, 2, 0)  # [H, W, 3]
    features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + MIN_DENOMINATOR)
    return features
 
#(未变)
def PSNR_torch(img1, img2, max_val=1):
    """(未变)计算两个图像之间的峰值信噪比(PSNR),用于衡量图像质量。"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = max_val
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

# 未变
def parse_camera_file(file_path):
    """
    Parse our camera format.
    解析我们的相机格式。
    The format is (*.txt):格式为 (*.txt):
     4x4 矩阵（摄像机外特性）
    4x4 matrix (camera extrinsic)
    space 空间
    3x3 矩阵（摄像机内部）
    3x3 matrix (camera intrinsic)
    从内在矩阵中提取焦点
    focal is extracted from the intrinsc matrix
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    camera_extrinsic = []
    for x in lines[0:4]:
        camera_extrinsic += [float(y) for y in x.split()]
    camera_extrinsic = np.array(camera_extrinsic).reshape(4, 4)

    camera_intrinsic = []
    for x in lines[5:8]:
        camera_intrinsic += [float(y) for y in x.split()]
    camera_intrinsic = np.array(camera_intrinsic).reshape(3, 3)

    focal = camera_intrinsic[0, 0]

    return camera_extrinsic, camera_intrinsic, focal

def parse_img_file(file_path, mask_gt_rgb=False, bg_color=[0,0,0,255]):
    """
    解析一个图像文件（如JPEG或PNG），并将其转换为一个NumPy数组。
    return np.array of RGB image with range [0, 1]
    """
    rgb = Image.open(file_path).convert('RGB')
    rgb = np.asarray(rgb).astype(np.float32) / 255.0    # [0, 1]
    return rgb

# 新增
def parse_depth_file(file_path):
    """
    （新增）解析深度图文件
    return np.array of depth image
    """
    depth = Image.open(file_path).convert('L')
    depth = np.asarray(depth).astype(np.float32)
    return depth


class QFunction(nn.Module):

    def __init__(self,
                 perceiver_encoder: nn.Module,   # 神经网络编码器，用于提取输入数据的特征
                 voxelizer: VoxelGrid,           # 个体素化对象，用于将点云数据转换为体素网格       
                 bounds_offset: float,           #  确定动作选择和体素化过程的参数
                 rotation_resolution: float,
                 device,
                 training,
                #  !! Mani多了两个
                 use_ddp=True,  # default: True
                 cfg=None,
                 fabric=None,):
        super(QFunction, self).__init__()
        self._rotation_resolution = rotation_resolution
        self._voxelizer = voxelizer
        self._bounds_offset = bounds_offset
        self._qnet = perceiver_encoder.to(device)
        # print("Manigs bc agent Qfunction self._qnet",self._qnet.shape)
        self._coord_trans = torch.diag(torch.tensor([1, 1, 1, 1], dtype=torch.float32)).to(device)
        
        self.cfg = cfg
        # use fabric=false时后面可以加上 and  fabric!=None
        if cfg.use_neural_rendering:
            self._neural_renderer = NeuralRenderer(cfg.neural_renderer).to(device)
            # print("use_neural_rendering maybe use ddp")
            if training and use_ddp:
                # print("useddp--------------")
                self._neural_renderer = fabric.setup(self._neural_renderer)
        else:
            self._neural_renderer = None
        print(colored(f"[NeuralRenderer]: {cfg.use_neural_rendering}", "cyan"))
        
        # distributed training
        if training and use_ddp:
            print(colored(f"[QFunction] use DDP: True", "cyan"))
            self._qnet = fabric.setup(self._qnet)
        
        self.device = device

    def _argmax_3d(self, tensor_orig):
        '''
        找到3D张量中的最大值索引
        '''
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices


    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        '''
        选择最高值的动作 根据 Q 值选择最佳动作
        '''
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies = None
        ignore_collision = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision
    
    ## !!yzj----------------好多参数------------------------------------------
    def forward(self, rgb_pcd, depth, proprio, pcd, camera_extrinsics, camera_intrinsics, lang_goal_emb, lang_token_embs,
                bounds=None, prev_bounds=None, prev_layer_voxel_grid=None,
                use_neural_rendering=False, nerf_target_rgb=None, nerf_target_depth=None,
                nerf_target_pose=None, nerf_target_camera_intrinsic=None,
                lang_goal=None,
                nerf_next_target_rgb=None, nerf_next_target_pose=None, nerf_next_target_depth=None,
                nerf_next_target_camera_intrinsic=None,
                gt_embed=None, step=None, action=None):
        '''
        Return Q-functions and neural rendering loss
        前向传播 返回Q函数和神经渲染损失
        '''

        b = rgb_pcd[0][0].shape[0]

        pcd_flat = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, 3) for p in pcd], 1)  # [1, 16384, 3]
        
        # flatten RGBs and Pointclouds 扁平化 RGB 和点云
        rgb = [rp[0] for rp in rgb_pcd]
        feat_size = rgb[0].shape[1] # 3

        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, feat_size) for p in rgb], 1)  # [1, 16384, 3]

        # print("coord_bounds",bounds) # tensor([[-0.3000, -0.5000,  0.6000,  0.7000,  0.5000,  1.6000]],device='cuda:0')
        # construct voxel grid 构建体元网格
        voxel_grid, voxel_density = self._voxelizer.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds, return_density=True)

        # swap to channels fist 换频道
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach() # Bx10x100x100x100

        # batch bounds if necessary 必要时，批量界值
        if bounds.shape[0] != b:
            bounds = bounds.repeat(b, 1)

        # forward pass 前传
        #  voxel_grid_feature,multi_scale_voxel_list是多的 
        # 双臂中用split_pred代替所有参数
        # new yzj第一行默认用左臂的观察!! 后续要看如何使用左手还是右手
        # right_trans, right_rot_and_grip,right_ignore_collisions,\
        # left_trans,left_rot_and_grip_out,left_collision_out,\
        # print("--- 每次运行 QFunction forward前报告 ---")
        # multi_scale_voxel_list, \ 暂时移除
        split_pred,voxel_grid_feature, \
        lang_embedd = self._qnet(voxel_grid,  # [1,10,100^3]
                                proprio, # [1,4]  好像不一样（【1,8】）
                                lang_goal_emb, # [1,1024]
                                lang_token_embs, # [1,77,512]
                                prev_layer_voxel_grid, #None,   # 一样，双手中prev_layer_voxel_grid=None,
                                bounds, # [1,6]
                                prev_bounds, # None    # 一样，双手中prev_bounds=None,
                                )
        right_trans, right_rot_and_grip,right_ignore_collisions,\
        left_trans,left_rot_and_grip_out,left_collision_out= split_pred
        q_trans=torch.cat((right_trans, left_trans), dim=1) 
        # left 的取名代表bimanual风格可能代表的东西不一样
        q_rot_and_grip=torch.cat((right_rot_and_grip, left_rot_and_grip_out), dim=1)
        q_ignore_collisions=torch.cat((right_ignore_collisions, left_collision_out), dim=1)
        # voxel_grid_feature=torch.cat((voxel_grid_feature, voxel_grid_feature),dim=1)
        # print("在qfunction中出现的q_trans=",q_trans.shape) #[1,2,100,100,100]
        # print("q_rot_and_grip=",q_rot_and_grip.shape)     # [1,436]
        # print("q_ignore_collisions=",q_ignore_collisions.shape) # [1,4]
        # print("---------voxel_grid_feature=",voxel_grid_feature.shape) # [1,128,100,100]
        # print("multi_scale_voxel_list=",multi_scale_voxel_list.shape)
        # print("lang_embedd=",lang_embedd.shape)
        # print("voxel_grid=",voxel_grid.shape)
        # print("proprio=",proprio.shape)
        # print("lang_goal_emb=",lang_goal_emb.shape)
        # print("lang_token_embs=",lang_token_embs.shape)
        # 但是好像是None print("-------------prev_layer_voxel_grid=",prev_layer_voxel_grid.shape)
        # print("bounds=",bounds.shape)
        # 但是好像是None print("prev_bounds=",prev_bounds.shape)
        # !!新 ----------------------------------------------------------------
        # neural rendering as an auxiliary loss # 神经渲染作为辅助损失
        rendering_loss_dict = {}
        if use_neural_rendering:    # train default: True; eval default: False
            # prepare nerf rendering
            focal = camera_intrinsics[0][:, 0, 0]  # [SB]
            cx = 128 / 2
            cy = 128 / 2
            c = torch.tensor([cx, cy], dtype=torch.float32).unsqueeze(0)

            if nerf_target_rgb is not None:
                gt_rgb = nerf_target_rgb    # [1,128,128,3]
                gt_pose = nerf_target_pose @ self._coord_trans # remember to do this
                gt_depth = nerf_target_depth

                # (新增！！！改变)We only use the front camera  我们只使用前置摄像头
                rgb_0 = rgb[0]
                depth_0 = depth[0]
                pcd_0 = pcd[0]

                # print("qfunction中的已经开始错了！！！哈哈哈终于找到你了",voxel_grid_feature.shape)
                # render loss（改变）神经渲染 后面是一些相关数据字典通过dotmap处理后的数据，可用.访问，但是无用
                # 所以就是在这里加入grounded sam结果了
                print("\033[0;32;40maction in bc agent.py\033[0m",action)
                print(action.shape) # torch.Size([1, 16])

                rendering_loss_dict, _ = self._neural_renderer(
                    rgb=rgb_0, pcd=pcd_0, depth=depth_0, # 第一个视角下的颜色 点云数据 深度
                    language=lang_embedd, # 语言目标
                    dec_fts=voxel_grid_feature, # 体素网格特征，可能用于3D体素渲染
                    gt_rgb=gt_rgb,  gt_depth=gt_depth, focal=focal,  # 真实（Ground Truth）的图像
                    c=c, 
                    gt_pose=gt_pose, gt_intrinsic=nerf_target_camera_intrinsic, \
                    lang_goal=lang_goal, 
                    next_gt_pose=nerf_next_target_pose, next_gt_intrinsic=nerf_next_target_camera_intrinsic, 
                    next_gt_rgb=nerf_next_target_rgb, step=step, action=action,
                    training=True,
                    )

            else:
                # # if we do not have additional multi-view data, we use input view as reconstruction target
                #  如果没有额外的多视图数据，我们就使用输入视图作为重建目标
                rendering_loss_dict = {
                    'loss': 0.,
                    'loss_rgb': 0.,
                    'loss_embed': 0.,
                    'l1': 0.,
                    'psnr': 0.,
                    }
        # return split_pred, voxel_grid
        # return q_trans, q_rot_and_grip, q_ignore_collisions, voxel_grid, rendering_loss_dict
        # return (right_trans, right_rot_and_grip,right_ignore_collisions,left_trans,left_rot_and_grip_out,left_collision_out), voxel_grid, rendering_loss_dict
        return split_pred, voxel_grid, rendering_loss_dict

    @torch.no_grad()
    def render(self, rgb_pcd, proprio, pcd, 
               lang_goal_emb, lang_token_embs,
                tgt_pose, tgt_intrinsic,
               camera_extrinsics=None, camera_intrinsics=None,
                depth=None, bounds=None, prev_bounds=None, prev_layer_voxel_grid=None,
                nerf_target_rgb=None, lang_goal=None,
                nerf_next_target_rgb=None, nerf_next_target_depth=None,
                nerf_next_target_pose=None, nerf_next_target_camera_intrinsic=None, 
                action=None, step=None,
                ):
        """
        Render the novel view and the next novel view during the training process
        在训练过程中渲染新视图和下一个新视图
        """
        # rgb_pcd will be list of [rgb, pcd]
        b = rgb_pcd[0][0].shape[0]
        pcd_flat = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, 3) for p in pcd], 1)

        # flatten RGBs and Pointclouds
        rgb = [rp[0] for rp in rgb_pcd]
        feat_size = rgb[0].shape[1]
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, feat_size) for p in rgb], 1)

        # construct voxel grid
        voxel_grid, voxel_density = self._voxelizer.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds, return_density=True)

        # swap to channels fist
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()

        # batch bounds if necessary
        if bounds.shape[0] != b:
            bounds = bounds.repeat(b, 1)

        # forward pass 其中上一部分类似，下面不同
        # left_trans,left_rot_and_grip_out,left_collision_out,\
        # right_trans, right_rot_and_grip,right_ignore_collisions,\
        # left_trans,left_rot_and_grip_out,left_collision_out,\
        # multi_scale_voxel_list, \暂时移除
        split_pred, \
        voxel_grid_feature, \
        lang_embedd = self._qnet(voxel_grid,
        # q_trans, \
        # q_rot_and_grip,\
        # q_ignore_collisions,\
        # voxel_grid_feature,\
        # multi_scale_voxel_list,\
        # lang_embedd = self._qnet(voxel_grid, 
                                        proprio,
                                        lang_goal_emb, 
                                        lang_token_embs,
                                        prev_layer_voxel_grid,
                                        bounds, 
                                        prev_bounds)
        # new 给两臂合起来
        # voxel_grid_feature = torch.cat((voxel_grid_feature, voxel_grid_feature), dim=1)
        # prepare nerf rendering            准备Nerf渲染?
        # We only use the front camera      我们只使用前置摄像头
        _, ret_dict = self._neural_renderer(
                pcd=pcd[0],                 # 点云数据（Point Cloud Data）的第一个元素，可能是一个3D点云表示
                rgb=rgb[0],                 # ：RGB图像数据的第一个元素，表示场景的颜色信息。
                dec_fts=voxel_grid_feature, # 解码特征，可能是指从体素网格中提取的特征。
                language=lang_embedd,       # 语言嵌入，可能表示与任务相关的自然语言处理结果。
                gt_pose=tgt_pose,           # 目标姿态（Ground Truth Pose），真实的姿态信息，用于训练或评估
                gt_intrinsic=tgt_intrinsic, # 相机的内在参数，描述相机的光学特性。
                # for providing gt embed
                gt_rgb=nerf_target_rgb,     # 神经辐射场（Neural Radiance Fields, NeRF）的目标RGB图像，用作渲染的参考。
                lang_goal=lang_goal,        # 语言目标，可能表示任务的语言描述或指令。    
                next_gt_rgb=nerf_next_target_rgb,       # 下一个目标RGB图像，可能用于生成序列中的下一帧。
                next_gt_pose=nerf_next_target_pose,     # 下一个目标姿态。
                next_gt_intrinsic=nerf_next_target_camera_intrinsic,        # 下一个目标相机的内在参数。
                step=step,
                action=action,                          # 当前执行的动作。
                training=False,
                )

        # ----------------------------------------------------------------------    
        return ret_dict.render_novel, ret_dict.next_render_novel, ret_dict.render_embed, ret_dict.gt_embed


class QAttentionPerActBCAgent(Agent):
# 基于感知器（Perceiver）模型的行为克隆（Behavioral Cloning, BC）
    def __init__(self,
                 layer: int,
                 coordinate_bounds: list,
                 perceiver_encoder: nn.Module,
                 camera_names: list,
                 batch_size: int,
                 voxel_size: int,
                 bounds_offset: float,
                 voxel_feature_size: int,
                 image_crop_size: int,
                 num_rotation_classes: int,
                 rotation_resolution: float,
                 lr: float = 0.0001,
                 lr_scheduler: bool = False,
                 training_iterations: int = 100000,
                 num_warmup_steps: int = 20000,
                 trans_loss_weight: float = 1.0,
                 rot_loss_weight: float = 1.0,
                 grip_loss_weight: float = 1.0,
                 collision_loss_weight: float = 1.0,
                 include_low_dim_state: bool = False,
                 image_resolution: list = None,
                 lambda_weight_l2: float = 0.0,
                 transform_augmentation: bool = True,   # True
                 transform_augmentation_xyz: list = [0.0, 0.0, 0.0],
                 transform_augmentation_rpy: list = [0.0, 0.0, 180.0],
                 transform_augmentation_rot_resolution: int = 5,
                 optimizer_type: str = 'adam',
                 num_devices: int = 1,
                 cfg = None,
                 ):
        self._layer = layer
        self._coordinate_bounds = coordinate_bounds
        self._perceiver_encoder = perceiver_encoder
        # print("mani bc agent init perceiver_encoder", perceiver_encoder.shape)
        self._voxel_feature_size = voxel_feature_size
        self._bounds_offset = bounds_offset
        self._image_crop_size = image_crop_size
        self._lr = lr
        self._lr_scheduler = lr_scheduler
        self._training_iterations = training_iterations
        self._num_warmup_steps = num_warmup_steps
        self._trans_loss_weight = trans_loss_weight
        self._rot_loss_weight = rot_loss_weight
        self._grip_loss_weight = grip_loss_weight
        self._collision_loss_weight = collision_loss_weight
        self._include_low_dim_state = include_low_dim_state
        self._image_resolution = image_resolution or [128, 128]
        # print("image_resolution",image_resolution)
        self._voxel_size = voxel_size
        self._camera_names = camera_names
        self._num_cameras = len(camera_names)
        self._batch_size = batch_size
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._transform_augmentation_xyz = torch.from_numpy(np.array(transform_augmentation_xyz))
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = transform_augmentation_rot_resolution
        self._optimizer_type = optimizer_type
        self._num_devices = num_devices
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution

        self.cfg = cfg
        
        self.use_neural_rendering = self.cfg.use_neural_rendering
        print(colored(f"use_neural_rendering: {self.use_neural_rendering}", "red"))

        if self.use_neural_rendering:
            print(colored(f"[agent] nerf weight step: {self.cfg.neural_renderer.lambda_nerf}", "red"))

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')  # if batch size>1
        
        # 增加改变目标姿态（Ground Truth Pose）
        self._mask_gt_rgb = cfg.neural_renderer.dataset.mask_gt_rgb
        print(colored(f"[NeuralRenderer] mask_gt_rgb: {self._mask_gt_rgb}", "cyan"))
        
        self._name = NAME + '_layer' + str(self._layer)

    def build(self, training: bool, device: torch.device = None, use_ddp=True, fabric: Fabric = None):
        """ 根据是否训练和设备信息构建代理。
            初始化体素化器 VoxelGrid 和 Q 网络 QFunction。
            设置优化器和学习率调度器。"""
        self._training = training
        self._device = device

        if device is None:
            device = torch.device('cpu')

        print(f"device: {device}")

        # print("qattention_peract_bc_agent max_num_coords",np.prod(self._image_resolution) * self._num_cameras)  # 98304 (在mani里39316)
        # print("np.prod(self._image_resolution)",np.prod(self._image_resolution)) # 16384 （65536）
        # print("self._num_cameras",self._num_cameras) # 6 （6）

        self._voxelizer = VoxelGrid(
            coord_bounds=self._coordinate_bounds.cpu() if isinstance(self._coordinate_bounds, torch.Tensor) else self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=device,  # 0
            batch_size=self._batch_size if training else 1,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod(self._image_resolution) * self._num_cameras,
        )

        # 有些不同
        # print(f"qatten mani bc agent build self._perceiver_encoder: {self._perceiver_encoder.shape}")
        # print("type(self._perceiver_encoder)",type(self._perceiver_encoder))
        self._q = QFunction(self._perceiver_encoder,
                            self._voxelizer,
                            self._bounds_offset,
                            self._rotation_resolution,
                            device,
                            training,
                            # 以下3个参数增加的
                            use_ddp,
                            self.cfg,
                            fabric=fabric
                            ).to(device).train(training)

        grid_for_crop = torch.arange(0,
                                     self._image_crop_size,
                                     device=device).unsqueeze(0).repeat(self._image_crop_size, 1).unsqueeze(-1)
        self._grid_for_crop = torch.cat([grid_for_crop.transpose(1, 0),
                                         grid_for_crop], dim=2).unsqueeze(0)

        self._coordinate_bounds = torch.tensor(self._coordinate_bounds,
                                               device=device).unsqueeze(0)

        if self._training:
            # optimizer
            if self._optimizer_type == 'lamb':
                self._optimizer = Lamb(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
            elif self._optimizer_type == 'adam':
                self._optimizer = torch.optim.Adam(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                )
            else:
                raise Exception('Unknown optimizer type')
            
            # （增加 改变）DDP optimizer
            self._optimizer = fabric.setup_optimizers(self._optimizer)

            # learning rate scheduler
            if self._lr_scheduler:
                self._scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                    self._optimizer,
                    num_warmup_steps=self._num_warmup_steps,
                    num_training_steps=self._training_iterations,
                    num_cycles=self._training_iterations // 10000,
                )

            # one-hot zero tensors
            self._action_trans_one_hot_zeros = torch.zeros((self._batch_size,
                                                            1,
                                                            self._voxel_size,
                                                            self._voxel_size,
                                                            self._voxel_size),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_x_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_y_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_z_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_grip_one_hot_zeros = torch.zeros((self._batch_size,
                                                           2),
                                                           dtype=int,
                                                           device=device)
            self._action_ignore_collisions_one_hot_zeros = torch.zeros((self._batch_size,
                                                                        2),
                                                                        dtype=int,
                                                                        device=device)

            # print total params
            logging.info('# Q Params: %d M' % (sum(
                p.numel() for name, p in self._q.named_parameters() \
                if p.requires_grad and 'clip' not in name)/1e6) )
        else:
            for param in self._q.parameters():
                param.requires_grad = False

            # 增加
            self.language_model =  create_language_model(self.cfg.language_model)
            self._voxelizer.to(device)
            self._q.to(device)

    # nerf[6]---
    #  _preprocess_inputs和下面一模一样，有机会应该删一个的
    def _preprocess_inputs(self, replay_sample, sample_id=None):
        """用于预处理从回放缓冲区获取的输入数据，包括 RGB 图像、深度图、点云、相机内外参等。
        用于预处理从回放缓冲区(eplay buffer)获取的输入数据"""
        # 存储观察结果（obs）、深度信息（depths）、点云数据（pcds）、
        # 相机外参（exs，代表 extrinsics）、相机内参（ins，代表 intrinsics）
        obs = []
        depths = [] # 深度信息是多的
        pcds = []
        exs = []
        ins = []
        self._crop_summary = []
        # 遍历 self._camera_names 中定义的所有相机名称。这些名称可能是 ['front', 'left_shoulder', 'right_shoulder', 'wrist'] 或其他配置。
        for n in self._camera_names:    # default: [front,left_shoulder,right_shoulder,wrist] or [front]
            if sample_id is not None:   # default: None
                rgb = replay_sample['%s_rgb' % n][sample_id:sample_id+1]
                depth = replay_sample['%s_depth' % n][sample_id:sample_id+1]
                pcd = replay_sample['%s_point_cloud' % n][sample_id:sample_id+1]
                extin = replay_sample['%s_camera_extrinsics' % n][sample_id:sample_id+1]
                intin = replay_sample['%s_camera_intrinsics' % n][sample_id:sample_id+1]
            else:
                rgb = replay_sample['%s_rgb' % n]
                depth = replay_sample['%s_depth' % n]
                pcd = replay_sample['%s_point_cloud' % n]
                extin = replay_sample['%s_camera_extrinsics' % n]
                intin = replay_sample['%s_camera_intrinsics' % n]
            obs.append([rgb, pcd])
            depths.append(depth)
            pcds.append(pcd)
            exs.append(extin)
            ins.append(intin)
        return obs, depths, pcds, exs, ins

    def _mani_preprocess_inputs(self, replay_sample, sample_id=None):
        """用于预处理从回放缓冲区获取的输入数据，包括 RGB 图像、深度图、点云、相机内外参等。
        用于预处理从回放缓冲区(eplay buffer)获取的输入数据"""
        # 存储观察结果（obs）、深度信息（depths）、点云数据（pcds）、
        # 相机外参（exs，代表 extrinsics）、相机内参（ins，代表 intrinsics）
        obs = []
        depths = [] # 深度信息是多的
        pcds = []
        exs = []
        ins = []
        self._crop_summary = []
        # 遍历 self._camera_names 中定义的所有相机名称。这些名称可能是 ['front', 'left_shoulder', 'right_shoulder', 'wrist'] 或其他配置。
        for n in self._camera_names:    # default: [front,left_shoulder,right_shoulder,wrist] or [front]
            if sample_id is not None:   # default: None
                rgb = replay_sample['%s_rgb' % n][sample_id:sample_id+1]
                depth = replay_sample['%s_depth' % n][sample_id:sample_id+1]
                pcd = replay_sample['%s_point_cloud' % n][sample_id:sample_id+1]
                extin = replay_sample['%s_camera_extrinsics' % n][sample_id:sample_id+1]
                intin = replay_sample['%s_camera_intrinsics' % n][sample_id:sample_id+1]
            else:
                rgb = replay_sample['%s_rgb' % n]
                depth = replay_sample['%s_depth' % n]
                pcd = replay_sample['%s_point_cloud' % n]
                extin = replay_sample['%s_camera_extrinsics' % n]
                intin = replay_sample['%s_camera_intrinsics' % n]
            obs.append([rgb, pcd])
            depths.append(depth)
            pcds.append(pcd)
            exs.append(extin)
            ins.append(intin)
        return obs, depths, pcds, exs, ins
  
    # nerf[6]---

    def _act_preprocess_inputs(self, observation):
        """Mani中原有方法 用于预处理来自环境的实时观察数据,以便进行动作选择(acting)"""
        obs, depths, pcds, exs, ins = [], [], [], [], []
        for n in self._camera_names:
            rgb = observation['%s_rgb' % n]
            # [-1,1] to [0,1]
            # rgb = (rgb + 1) / 2
            depth = observation['%s_depth' % n]
            pcd = observation['%s_point_cloud' % n]
            extin = observation['%s_camera_extrinsics' % n].squeeze(0)
            intin = observation['%s_camera_intrinsics' % n].squeeze(0)

            obs.append([rgb, pcd])
            depths.append(depth)
            pcds.append(pcd)
            exs.append(extin)
            ins. append(intin)
        return obs, depths, pcds, exs, ins

    def _get_value_from_voxel_index(self, q, voxel_idx):
        """根据体素索引从体素化的特征表示中检索值。
        获取 q 的形状，其中 b 是批次大小,c 是通道数,d 是体素的深度,h 和 w 分别是体素的高度和宽度。
        """
        b, c, d, h, w = q.shape
        q_trans_flat = q.view(b, c, d * h * w)
        flat_indicies = (voxel_idx[:, 0] * d * h + voxel_idx[:, 1] * h + voxel_idx[:, 2])[:, None].int()
        highest_idxs = flat_indicies.unsqueeze(-1).repeat(1, c, 1)
        chosen_voxel_values = q_trans_flat.gather(2, highest_idxs)[..., 0]  # (B, trans + rot + grip)
        return chosen_voxel_values

    def _get_value_from_rot_and_grip(self, rot_grip_q, rot_and_grip_idx):
        """从旋转和抓取动作的表示中检索值。"""
        q_rot = torch.stack(torch.split(
            rot_grip_q[:, :-2], int(360 // self._rotation_resolution),
            dim=1), dim=1)  # B, 3, 72
        q_grip = rot_grip_q[:, -2:]
        rot_and_grip_values = torch.cat(
            [q_rot[:, 0].gather(1, rot_and_grip_idx[:, 0:1]),
             q_rot[:, 1].gather(1, rot_and_grip_idx[:, 1:2]),
             q_rot[:, 2].gather(1, rot_and_grip_idx[:, 2:3]),
             q_grip.gather(1, rot_and_grip_idx[:, 3:4])], -1)
        return rot_and_grip_values

    def _celoss(self, pred, labels):
        """计算交叉熵损失(Cross-Entropy Loss),通常用于分类问题。"""
        return self._cross_entropy_loss(pred, labels.argmax(-1))

    def _softmax_q_trans(self, q):
        """对平移动作的Q值应用softmax函数,以获取归一化的概率分布"""
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)

    def _softmax_q_rot_grip(self, q_rot_grip):
        """对旋转和抓取动作的Q值分别应用softmax函数,并将它们合并成单一的概率分布。"""
        q_rot_x_flat = q_rot_grip[:, 0*self._num_rotation_classes: 1*self._num_rotation_classes]
        q_rot_y_flat = q_rot_grip[:, 1*self._num_rotation_classes: 2*self._num_rotation_classes]
        q_rot_z_flat = q_rot_grip[:, 2*self._num_rotation_classes: 3*self._num_rotation_classes]
        q_grip_flat  = q_rot_grip[:, 3*self._num_rotation_classes:]

        q_rot_x_flat_softmax = F.softmax(q_rot_x_flat, dim=1)
        q_rot_y_flat_softmax = F.softmax(q_rot_y_flat, dim=1)
        q_rot_z_flat_softmax = F.softmax(q_rot_z_flat, dim=1)
        q_grip_flat_softmax = F.softmax(q_grip_flat, dim=1)

        return torch.cat([q_rot_x_flat_softmax,
                          q_rot_y_flat_softmax,
                          q_rot_z_flat_softmax,
                          q_grip_flat_softmax], dim=1)

    def _softmax_ignore_collision(self, q_collision):
        """对碰撞忽略动作的Q值应用softmax函数,以获取归一化的概率分布。"""
        q_collision_softmax = F.softmax(q_collision, dim=1)
        return q_collision_softmax

    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        保存模型某一层的梯度输出,
        通常用于可视化技术如Grad-CAM(Gradient-Weighted Class Activation Mapping)。
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        保存模型某一层的输出特征，通常也用于Grad-CAM。
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
            
    # new86--- bimanual中的新增的下面是Mani上面是bimanual
    def update(self, step: int, replay_sample: dict, fabric: Fabric) -> dict:
        right_action_trans = replay_sample["right_trans_action_indicies"][
            :, self._layer * 3 : self._layer * 3 + 3
        ].int()
        right_action_rot_grip = replay_sample["right_rot_grip_action_indicies"].int()
        right_action_gripper_pose = replay_sample["right_gripper_pose"]
        right_action_ignore_collisions = replay_sample["right_ignore_collisions"].int()

        left_action_trans = replay_sample["left_trans_action_indicies"][
            :, self._layer * 3 : self._layer * 3 + 3
        ].int()
        left_action_rot_grip = replay_sample["left_rot_grip_action_indicies"].int()
        left_action_gripper_pose = replay_sample["left_gripper_pose"]
        left_action_ignore_collisions = replay_sample["left_ignore_collisions"].int()

        lang_goal_emb = replay_sample["lang_goal_emb"].float()
        lang_token_embs = replay_sample["lang_token_embs"].float()
        prev_layer_voxel_grid = replay_sample.get("prev_layer_voxel_grid", None)
        prev_layer_bounds = replay_sample.get("prev_layer_bounds", None)
        lang_goal = replay_sample['lang_goal'] # mani
        action_gt = replay_sample['action'] # [bs, 8] # mani
        device = self._device

        rank = device

        # nerf[2]---------------------------
        # rank = self._q.device
        # obs, depth, pcd, extrinsics, intrinsics = self._preprocess_inputs(replay_sample)
        # batch size
        # bs = pcd[0].shape[0]
        # 整体上移后
        # obs, pcd = self._preprocess_inputs(replay_sample)
        # 其实不用带Mani的也一样，都是5个返回值
        obs, depth, pcd, extrinsics, intrinsics = self._mani_preprocess_inputs(replay_sample)

        # # batch size
        bs = pcd[0].shape[0]
        # nerf[2]-----------------------------

        # nerf[1]---------------------------
        # !! for nerf multi-view training
        nerf_multi_view_rgb_path = replay_sample['nerf_multi_view_rgb'] # only succeed to get path sometime
        nerf_multi_view_depth_path = replay_sample['nerf_multi_view_depth']
        nerf_multi_view_camera_path = replay_sample['nerf_multi_view_camera']

        nerf_next_multi_view_rgb_path = replay_sample['nerf_next_multi_view_rgb']
        nerf_next_multi_view_depth_path = replay_sample['nerf_next_multi_view_depth']
        nerf_next_multi_view_camera_path = replay_sample['nerf_next_multi_view_camera']

        if nerf_multi_view_rgb_path is None or nerf_multi_view_rgb_path[0,0] is None:
            cprint(nerf_multi_view_rgb_path, 'red')
            cprint(replay_sample['indices'], 'red')
            nerf_target_rgb = None
            nerf_target_camera_extrinsic = None
            print(colored('warn: one iter not use additional multi view', 'cyan'))
            raise ValueError('nerf_multi_view_rgb_path is None')
        else:
            # control the number of views by the following code 
            # 通过以下代码控制查看次数
            num_view = nerf_multi_view_rgb_path.shape[-1]
            num_view_by_user = self.cfg.num_view_for_nerf
            # compute interval first
            # 先算区间
            assert num_view_by_user <= num_view, f'num_view_by_user {num_view_by_user} should be less than num_view {num_view}'
            interval = num_view // num_view_by_user # 21//20?
            # !! 第一个冒号:表示选择所有行（第一个维度）。
            # !! 第二个::interval表示从第二列开始，每隔interval列选择一列。例如，如果interval是3，那么会选择第1、4、7、...列。
            nerf_multi_view_rgb_path = nerf_multi_view_rgb_path[:, ::interval]
            
            # sample one target img
            # 一个目标图像样本 !! 
            view_dix = np.random.randint(0, num_view_by_user)
            # !! [:, view_dix]：这是NumPy的切片语法，用于从数组中选择一个子集。冒号:表示选择所有行，而view_dix是一个索引，指定了要选择的列（或视角）。
            # !! nerf_multi_view_rgb_path[:, view_dix]：这行代码的结果是一个新数组，只包含原始数组中第view_dix列的数据，即特定视角的所有图像路径。
            nerf_multi_view_rgb_path = nerf_multi_view_rgb_path[:, view_dix]
            nerf_multi_view_depth_path = nerf_multi_view_depth_path[:, view_dix]
            nerf_multi_view_camera_path = nerf_multi_view_camera_path[:, view_dix]

            next_view_dix = np.random.randint(0, num_view_by_user)
            nerf_next_multi_view_rgb_path = nerf_next_multi_view_rgb_path[:, next_view_dix]
            nerf_next_multi_view_depth_path = nerf_next_multi_view_depth_path[:, next_view_dix]
            nerf_next_multi_view_camera_path = nerf_next_multi_view_camera_path[:, next_view_dix]

            # load img and camera (support bs>1)
            nerf_target_rgbs, nerf_target_depths, nerf_target_camera_extrinsics, nerf_target_camera_intrinsics = [], [], [], []
            nerf_next_target_rgbs, nerf_next_target_depths, nerf_next_target_camera_extrinsics, nerf_next_target_camera_intrinsics = [], [], [], []
            # 增加！！！ 莫非就是这个循环  bs 代表批量大小（batch size）
            for i in range(bs):
                # 图像文件解析 对于每个视角,调用 parse_img_file 函数解析图像文件，并将解析后的RGB图像数据添加到 nerf_target_rgbs 列表中。
                # parse_img_file 函数来解析位于 nerf_multi_view_rgb_path[i] 路径的图像文件。这个函数可能读取图像文件并对其进行处理。
                # mask_gt_rgb=self._mask_gt_rgb：传递一个名为 mask_gt_rgb 的参数给 parse_img_file 函数
                nerf_target_rgbs.append(parse_img_file(nerf_multi_view_rgb_path[i], mask_gt_rgb=self._mask_gt_rgb))#, session=self._rembg_session))    # FIXME: file_path 'NoneType' object has no attribute 'read'
                # 深度文件解析： 函数解析深度图文件，并将解析后的深度数据添加到 nerf_target_depths 列表中。
                nerf_target_depths.append(parse_depth_file(nerf_multi_view_depth_path[i]))
                # 相机文件解析：解析相机文件，获取相机的外参、内参和焦距值。
                nerf_target_camera_extrinsic, nerf_target_camera_intrinsic, nerf_target_focal = parse_camera_file(nerf_multi_view_camera_path[i])
                # 将解析得到的相机外参和内参分别添加到相应的列表中。
                nerf_target_camera_extrinsics.append(nerf_target_camera_extrinsic)
                nerf_target_camera_intrinsics.append(nerf_target_camera_intrinsic)

                # 处理下一视角的图像和相机数据（与上述步骤类似，但是用于 nerf_next_target_*）
                nerf_next_target_rgbs.append(parse_img_file(nerf_next_multi_view_rgb_path[i], mask_gt_rgb=self._mask_gt_rgb))#, session=self._rembg_session))    # FIXME: file_path 'NoneType' object has no attribute 'read'
                nerf_next_target_depths.append(parse_depth_file(nerf_next_multi_view_depth_path[i]))
                nerf_next_target_camera_extrinsic, nerf_next_target_camera_intrinsic, nerf_next_target_focal = parse_camera_file(nerf_next_multi_view_camera_path[i])
                nerf_next_target_camera_extrinsics.append(nerf_next_target_camera_extrinsic)
                nerf_next_target_camera_intrinsics.append(nerf_next_target_camera_intrinsic)

            # 转换为张量： 使用 numpy 的 stack 函数将 nerf_target_rgbs 列表中的所有图像堆叠成一个数组，然后转换为PyTorch张量，并将其移动到指定的设备（如GPU）上。
            nerf_target_rgb = torch.from_numpy(np.stack(nerf_target_rgbs)).float().to(device) # [bs, H, W, 3], [0,1]
            nerf_target_depth = torch.from_numpy(np.stack(nerf_target_depths)).float().to(device) # [bs, H, W, 1], no normalization
            nerf_target_camera_extrinsic = torch.from_numpy(np.stack(nerf_target_camera_extrinsics)).float().to(device)
            nerf_target_camera_intrinsic = torch.from_numpy(np.stack(nerf_target_camera_intrinsics)).float().to(device)

            nerf_next_target_rgb = torch.from_numpy(np.stack(nerf_next_target_rgbs)).float().to(device) # [bs, H, W, 3], [0,1]
            nerf_next_target_depth = torch.from_numpy(np.stack(nerf_next_target_depths)).float().to(device) # [bs, H, W, 1], no normalization
            nerf_next_target_camera_extrinsic = torch.from_numpy(np.stack(nerf_next_target_camera_extrinsics)).float().to(device)
            nerf_next_target_camera_intrinsic = torch.from_numpy(np.stack(nerf_next_target_camera_intrinsics)).float().to(device)

        # nerf[1]-------------------------




        bounds = self._coordinate_bounds.to(device)
        if self._layer > 0:
            right_cp = replay_sample[
                "right_attention_coordinate_layer_%d" % (self._layer - 1)
            ]

            left_cp = replay_sample[
                "left_attention_coordinate_layer_%d" % (self._layer - 1)
            ]

            right_bounds = torch.cat(
                [right_cp - self._bounds_offset, right_cp + self._bounds_offset], dim=1
            )
            left_bounds = torch.cat(
                [left_cp - self._bounds_offset, left_cp + self._bounds_offset], dim=1
            )
        else:
            right_bounds = bounds
            left_bounds = bounds

        right_proprio = None
        left_proprio = None
        if self._include_low_dim_state:
            right_proprio = replay_sample["right_low_dim_state"]
            left_proprio = replay_sample["left_low_dim_state"]

        # ..TODO::
        # Can we add the coordinates of both robots?
        #

        # 整体上移
        # obs, pcd = self._preprocess_inputs(replay_sample)
        # 其实不用带Mani的也一样，都是5个返回值
        # obs, depth, pcd, extrinsics, intrinsics = self._mani_preprocess_inputs(replay_sample)

        # # batch size
        # bs = pcd[0].shape[0]

        # We can move the point cloud w.r.t to the other robot's cooridinate system
        # similar to apply_se3_augmentation
        #

        # SE(3) augmentation of point clouds and actions   SE(3) 增强点云和action
        if self._transform_augmentation:
            from voxel import augmentation
            # nerf[4]---得新写一个函数-------------
            (
                right_action_trans,
                right_action_rot_grip,
                left_action_trans,
                left_action_rot_grip,
                pcd,
                extrinsics   # 无用 right camera_pose应该无用先不写左臂了
            ) = augmentation.bimanual_apply_se3_augmentation_with_camera_pose(
                pcd,
                extrinsics, # nerf new
                right_action_gripper_pose,
                right_action_trans,
                right_action_rot_grip,
                left_action_gripper_pose,
                left_action_trans,
                left_action_rot_grip,
                bounds,
                self._layer,
                self._transform_augmentation_xyz,
                self._transform_augmentation_rpy,
                self._transform_augmentation_rot_resolution,
                self._voxel_size,
                self._rotation_resolution,
                self._device,
            )
            # (
            #     right_action_trans,
            #     right_action_rot_grip,
            #     left_action_trans,
            #     left_action_rot_grip,
            #     pcd,
            # ) = augmentation.bimanual_apply_se3_augmentation(
            #     pcd,
            #     right_action_gripper_pose,
            #     right_action_trans,
            #     right_action_rot_grip,
            #     left_action_gripper_pose,
            #     left_action_trans,
            #     left_action_rot_grip,
            #     bounds,
            #     self._layer,
            #     self._transform_augmentation_xyz,
            #     self._transform_augmentation_rpy,
            #     self._transform_augmentation_rot_resolution,
            #     self._voxel_size,
            #     self._rotation_resolution,
            #     self._device,
            # )
            # nerf[4]----------------
        else:
            right_action_trans = right_action_trans.int()
            left_action_trans = left_action_trans.int()

        proprio = torch.cat((right_proprio, left_proprio), dim=1)

        right_action = (
            right_action_trans,
            right_action_rot_grip,
            right_action_ignore_collisions,
        )
        print(" ### in update  ###  ")
        print("right_action_trans", right_action_trans) # tensor([[55, 51, 27]], device='cuda:0'
        print("right_action_rot_grip", right_action_rot_grip) #  tensor([[ 0, 36,  6,  0]], device='cuda:0')
        print("right_action", right_action) # 连接起来 ignore collision (tensor([[1]], device='cuda:0', dtype=torch.int32)
        # print("right_action.shape",right_action.shape)
        print(" ")
        left_action = (
            left_action_trans,
            left_action_rot_grip,
            left_action_ignore_collisions,
        )
        # forward pass
        # q在下面会被分别赋值到以下三个参数的左右手
        # q_trans, q_rot_grip, q_collision,voxel_grid, rendering_loss_dict=
        q, voxel_grid, rendering_loss_dict = self._q(
        # (
        #     right_q_trans,
        #     right_q_rot_grip,
        #     right_q_collision,
        #     left_q_trans,
        #     left_q_rot_grip,
        #     left_q_collision,
        #     voxel_grid, 
        #     rendering_loss_dict
        # ) = self._q(
            obs,
            depth, # nerf new
            proprio,
            pcd,
            extrinsics, # nerf new although augmented, not used
            intrinsics, # nerf new
            lang_goal_emb,
            lang_token_embs,
            bounds,
            prev_layer_bounds,
            prev_layer_voxel_grid,
            # nerf[3]---------------------
            use_neural_rendering=self.use_neural_rendering,
            nerf_target_rgb=nerf_target_rgb,
            nerf_target_depth=nerf_target_depth,
            nerf_target_pose=nerf_target_camera_extrinsic,
            nerf_target_camera_intrinsic=nerf_target_camera_intrinsic,
            lang_goal=lang_goal,
            nerf_next_target_rgb=nerf_next_target_rgb,
            nerf_next_target_depth=nerf_next_target_depth,
            nerf_next_target_pose=nerf_next_target_camera_extrinsic,
            nerf_next_target_camera_intrinsic=nerf_next_target_camera_intrinsic,
            step=step,
            action=action_gt,
            # nerf[3]---------------------
        )

        (
            right_q_trans,
            right_q_rot_grip,
            right_q_collision,
            left_q_trans,
            left_q_rot_grip,
            left_q_collision,
        ) = q

        # argmax to choose best action 选择最佳操作
        (
            right_coords,
            right_rot_and_grip_indicies,
            right_ignore_collision_indicies,
        ) = self._q.choose_highest_action(
            right_q_trans, right_q_rot_grip, right_q_collision
        )
        print(" ")
        print("\033[0;32;40mright_coords\033[0m",right_coords) #[41,58,21]
        print("right_rot_and_grip_indicies",right_rot_and_grip_indicies) #[0,36,70,0]
        print("right_ignore_collision_indicies",right_ignore_collision_indicies) #[[1]]
        print(" ")

        (
            left_coords,
            left_rot_and_grip_indicies,
            left_ignore_collision_indicies,
        ) = self._q.choose_highest_action(
            left_q_trans, left_q_rot_grip, left_q_collision
        )


        right_q_trans_loss, right_q_rot_loss, right_q_grip_loss, right_q_collision_loss = 0.0, 0.0, 0.0, 0.0
        left_q_trans_loss, left_q_rot_loss, left_q_grip_loss, left_q_collision_loss = 0.0, 0.0, 0.0, 0.0

        # translation one-hot   one-hot编码
        right_action_trans_one_hot = self._action_trans_one_hot_zeros.clone().detach()
        left_action_trans_one_hot = self._action_trans_one_hot_zeros.clone().detach()
        for b in range(bs):
            right_gt_coord = right_action_trans[b, :].int()
            right_action_trans_one_hot[
                b, :, right_gt_coord[0], right_gt_coord[1], right_gt_coord[2]
            ] = 1
            left_gt_coord = left_action_trans[b, :].int()
            left_action_trans_one_hot[
                b, :, left_gt_coord[0], left_gt_coord[1], left_gt_coord[2]
            ] = 1

        # translation loss
        right_q_trans_flat = right_q_trans.view(bs, -1)
        right_action_trans_one_hot_flat = right_action_trans_one_hot.view(bs, -1)
        right_q_trans_loss = self._celoss(
            right_q_trans_flat, right_action_trans_one_hot_flat
        )
        left_q_trans_flat = left_q_trans.view(bs, -1)
        left_action_trans_one_hot_flat = left_action_trans_one_hot.view(bs, -1)
        left_q_trans_loss = self._celoss(
            left_q_trans_flat, left_action_trans_one_hot_flat
        )

        q_trans_loss = right_q_trans_loss + left_q_trans_loss

        with_rot_and_grip = (
            len(right_rot_and_grip_indicies) > 0 and len(left_rot_and_grip_indicies) > 0
        )
        if with_rot_and_grip:
            # rotation, gripper, and collision one-hots
            right_action_rot_x_one_hot = self._action_rot_x_one_hot_zeros.clone()
            right_action_rot_y_one_hot = self._action_rot_y_one_hot_zeros.clone()
            right_action_rot_z_one_hot = self._action_rot_z_one_hot_zeros.clone()
            right_action_grip_one_hot = self._action_grip_one_hot_zeros.clone()
            right_action_ignore_collisions_one_hot = (
                self._action_ignore_collisions_one_hot_zeros.clone()
            )

            left_action_rot_x_one_hot = self._action_rot_x_one_hot_zeros.clone()
            left_action_rot_y_one_hot = self._action_rot_y_one_hot_zeros.clone()
            left_action_rot_z_one_hot = self._action_rot_z_one_hot_zeros.clone()
            left_action_grip_one_hot = self._action_grip_one_hot_zeros.clone()
            left_action_ignore_collisions_one_hot = (
                self._action_ignore_collisions_one_hot_zeros.clone()
            )

            for b in range(bs):
                right_gt_rot_grip = right_action_rot_grip[b, :].int()
                right_action_rot_x_one_hot[b, right_gt_rot_grip[0]] = 1
                right_action_rot_y_one_hot[b, right_gt_rot_grip[1]] = 1
                right_action_rot_z_one_hot[b, right_gt_rot_grip[2]] = 1
                right_action_grip_one_hot[b, right_gt_rot_grip[3]] = 1

                right_gt_ignore_collisions = right_action_ignore_collisions[b, :].int()
                right_action_ignore_collisions_one_hot[
                    b, right_gt_ignore_collisions[0]
                ] = 1

                left_gt_rot_grip = left_action_rot_grip[b, :].int()
                left_action_rot_x_one_hot[b, left_gt_rot_grip[0]] = 1
                left_action_rot_y_one_hot[b, left_gt_rot_grip[1]] = 1
                left_action_rot_z_one_hot[b, left_gt_rot_grip[2]] = 1
                left_action_grip_one_hot[b, left_gt_rot_grip[3]] = 1

                left_gt_ignore_collisions = left_action_ignore_collisions[b, :].int()
                left_action_ignore_collisions_one_hot[
                    b, left_gt_ignore_collisions[0]
                ] = 1

            # flatten predictions
            right_q_rot_x_flat = right_q_rot_grip[
                :, 0 * self._num_rotation_classes : 1 * self._num_rotation_classes
            ]
            right_q_rot_y_flat = right_q_rot_grip[
                :, 1 * self._num_rotation_classes : 2 * self._num_rotation_classes
            ]
            right_q_rot_z_flat = right_q_rot_grip[
                :, 2 * self._num_rotation_classes : 3 * self._num_rotation_classes
            ]
            right_q_grip_flat = right_q_rot_grip[:, 3 * self._num_rotation_classes :]
            right_q_ignore_collisions_flat = right_q_collision

            left_q_rot_x_flat = left_q_rot_grip[
                :, 0 * self._num_rotation_classes : 1 * self._num_rotation_classes
            ]
            left_q_rot_y_flat = left_q_rot_grip[
                :, 1 * self._num_rotation_classes : 2 * self._num_rotation_classes
            ]
            left_q_rot_z_flat = left_q_rot_grip[
                :, 2 * self._num_rotation_classes : 3 * self._num_rotation_classes
            ]
            left_q_grip_flat = left_q_rot_grip[:, 3 * self._num_rotation_classes :]
            left_q_ignore_collisions_flat = left_q_collision


            # rotation loss
            right_q_rot_loss += self._celoss(right_q_rot_x_flat, right_action_rot_x_one_hot)
            right_q_rot_loss += self._celoss(right_q_rot_y_flat, right_action_rot_y_one_hot)
            right_q_rot_loss += self._celoss(right_q_rot_z_flat, right_action_rot_z_one_hot)

            left_q_rot_loss += self._celoss(left_q_rot_x_flat, left_action_rot_x_one_hot)
            left_q_rot_loss += self._celoss(left_q_rot_y_flat, left_action_rot_y_one_hot)
            left_q_rot_loss += self._celoss(left_q_rot_z_flat, left_action_rot_z_one_hot)

            # gripper loss
            right_q_grip_loss += self._celoss(right_q_grip_flat, right_action_grip_one_hot)
            left_q_grip_loss += self._celoss(left_q_grip_flat, left_action_grip_one_hot)

            # collision loss
            right_q_collision_loss += self._celoss(
                right_q_ignore_collisions_flat, right_action_ignore_collisions_one_hot
            )
            left_q_collision_loss += self._celoss(
                left_q_ignore_collisions_flat, left_action_ignore_collisions_one_hot
            )


        q_trans_loss = right_q_trans_loss + left_q_trans_loss
        q_rot_loss = right_q_rot_loss + left_q_rot_loss
        q_grip_loss = right_q_grip_loss + left_q_grip_loss
        q_collision_loss = right_q_collision_loss + left_q_collision_loss

        combined_losses = (
            (q_trans_loss * self._trans_loss_weight)
            + (q_rot_loss * self._rot_loss_weight)
            + (q_grip_loss * self._grip_loss_weight)
            + (q_collision_loss * self._collision_loss_weight)
        )
        total_loss = combined_losses.mean()
        # argmax to choose best action 到此三个步骤 都一模一样（就是单纯的左右分别算，然后简单相加）

        # nerf[3]-----
        if self.use_neural_rendering:   # eval default: False; train default: True
            # print("use_neual_rendering and rendering loss start")
            lambda_nerf = self.cfg.neural_renderer.lambda_nerf
            lambda_BC = self.cfg.lambda_bc

            total_loss = lambda_BC * total_loss + lambda_nerf * rendering_loss_dict['loss']

            # for print
            loss_rgb_item = rendering_loss_dict['loss_rgb']
            loss_embed_item = rendering_loss_dict['loss_embed']
            loss_dyna_item = rendering_loss_dict['loss_dyna']
            loss_reg_item = rendering_loss_dict['loss_reg']
            psnr = rendering_loss_dict['psnr']

            lambda_embed = self.cfg.neural_renderer.lambda_embed * lambda_nerf  # 0.0001
            lambda_rgb = self.cfg.neural_renderer.lambda_rgb * lambda_nerf  # 0.01
            lambda_dyna = (self.cfg.neural_renderer.lambda_dyna if step >= self.cfg.neural_renderer.next_mlp.warm_up else 0.) * lambda_nerf  # 0.01
            lambda_reg = (self.cfg.neural_renderer.lambda_reg if step >= self.cfg.neural_renderer.next_mlp.warm_up else 0.) * lambda_nerf  # 0.01

            # 输出操作
            if step % 10 == 0 and rank == 0:
                # 输出操作
                cprint(f'total L: {total_loss.item():.4f} | \
                    L_BC: {combined_losses.item():.3f} x {lambda_BC:.3f} | \
                    L_trans: {q_trans_loss.item():.3f} x {(self._trans_loss_weight * lambda_BC):.3f} | \
                    L_rot: {q_rot_loss.item():.3f} x {(self._rot_loss_weight * lambda_BC):.3f} | \
                    L_grip: {q_grip_loss.item():.3f} x {(self._grip_loss_weight * lambda_BC):.3f} | \
                    L_col: {q_collision_loss.item():.3f} x {(self._collision_loss_weight * lambda_BC):.3f} | \
                    L_rgb: {loss_rgb_item:.3f} x {lambda_rgb:.3f} | \
                    L_embed: {loss_embed_item:.3f} x {lambda_embed:.4f} | \
                    L_dyna: {loss_dyna_item:.3f} x {lambda_dyna:.4f} | \
                    L_reg: {loss_reg_item:.3f} x {lambda_reg:.4f} | \
                    psnr: {psnr:.3f}', 'green')
                if self.cfg.use_wandb:
                    wandb.log({
                        'train/BC_loss':combined_losses.item(), 
                        'train/psnr':psnr, 
                        'train/rgb_loss':loss_rgb_item,
                        'train/embed_loss':loss_embed_item,
                        'train/dyna_loss':loss_dyna_item,
                        }, step=step)
            
        else:   # no neural renderer
            # print("didn't use_neual_rendering and rendering loss didn't start")
            if step % 10 == 0 and rank == 0:
                lambda_BC = self.cfg.lambda_bc
                cprint(f'total L: {total_loss.item():.4f} | \
                    L_BC: {combined_losses.item():.3f} x {lambda_BC:.3f} | \
                    L_trans: {q_trans_loss.item():.3f} x {(self._trans_loss_weight * lambda_BC):.3f} | \
                    L_rot: {q_rot_loss.item():.3f} x {(self._rot_loss_weight * lambda_BC):.3f} | \
                    L_grip: {q_grip_loss.item():.3f} x {(self._grip_loss_weight * lambda_BC):.3f} | \
                    L_col: {q_collision_loss.item():.3f} x {(self._collision_loss_weight * lambda_BC):.3f}', 'green')            
                if self.cfg.use_wandb:
                    wandb.log({
                        'train/BC_loss':combined_losses.item(), 
                        }, step=step)

        # nerf[3]-----
        # nerf[3]-----原来的输出操作------------ 
        # if step % 10 == 0 and rank == 0:
        #     wandb.log({
        #         'train/grip_loss': q_grip_loss.mean(),
        #         'train/trans_loss': q_trans_loss.mean(),
        #         'train/rot_loss': q_rot_loss.mean(),
        #         'train/collision_loss': q_collision_loss.mean(),
        #         'train/total_loss': total_loss,
        #     }, step=step)
        # nerf[3]----- 在上面被替换掉了

        self._optimizer.zero_grad()
        # total_loss.backward() # 在Mani nerf中被注释了
        # Mani中用的下面fabric ddp的方式
        fabric.backward(total_loss)
        self._optimizer.step()

        # new rendering---
        ############### Render in training process #################

        # 渲染频率，决定多久进行一次神经渲染
        render_freq = self.cfg.neural_renderer.render_freq
        #  判断当前步骤是否应该进行渲染。
        to_render = (step % render_freq == 0 and self.use_neural_rendering and nerf_target_camera_extrinsic is not None)
        if to_render:
            # print("to_render start")
            print("\033[0;33;40maction_gt\033[0m",action_gt)
            rgb_render, next_rgb_render, embed_render, gt_embed_render = self._q.render(
                rgb_pcd=obs,
                proprio=proprio,
                pcd=pcd,
                lang_goal_emb=lang_goal_emb,
                lang_token_embs=lang_token_embs,
                bounds=bounds,
                prev_bounds=prev_layer_bounds,
                prev_layer_voxel_grid=prev_layer_voxel_grid,
                tgt_pose=nerf_target_camera_extrinsic,
                tgt_intrinsic=nerf_target_camera_intrinsic,
                nerf_target_rgb=nerf_target_rgb,
                lang_goal=lang_goal,
                nerf_next_target_rgb=nerf_next_target_rgb,
                nerf_next_target_depth=nerf_next_target_depth,
                nerf_next_target_pose=nerf_next_target_camera_extrinsic,
                nerf_next_target_camera_intrinsic=nerf_next_target_camera_intrinsic,
                step=step,
                action=action_gt,
                )
            
            # NOTE: [1, h, w, 3]
            rgb_gt = nerf_target_rgb[0]
            rgb_render = rgb_render[0]
            psnr = PSNR_torch(rgb_render, rgb_gt)
            if next_rgb_render is not None:
                next_rgb_gt = nerf_next_target_rgb[0]
                next_rgb_render = next_rgb_render[0]
                psnr_dyna = PSNR_torch(next_rgb_render, next_rgb_gt)
            
            # 创建目录 'recon' 用于保存可视化结果。 
            os.makedirs('recon', exist_ok=True)
            import matplotlib.pyplot as plt
            # plot three images in one row with subplots:
            # src, tgt, pred
            rgb_src =  obs[0][0].squeeze(0).permute(1, 2, 0)  / 2 + 0.5
            # 绘制源图像（rgb_src）、目标图像（rgb_gt）、渲染图像（rgb_render）、
            # 特征嵌入（embed_render 和 gt_embed_render）
            # 以及下一步骤的图像（next_rgb_gt 和 next_rgb_render）。
            fig, axs = plt.subplots(1, 7, figsize=(15, 3))
            # src
            axs[0].imshow(rgb_src.cpu().numpy())    # 在子图 axs[0] 上显示名为 rgb_src 的图像数
            axs[0].title.set_text('src')            # 设置子图 axs[0] 的标题为 'src'，这可能代表“源图像”（source image）
            # tgt
            axs[1].imshow(rgb_gt.cpu().numpy())
            axs[1].title.set_text('tgt')
            # pred rgb
            axs[2].imshow(rgb_render.cpu().numpy())
            axs[2].title.set_text('psnr={:.2f}'.format(psnr))
            # pred embed
            # embed_render = visualize_feature_map_by_clustering(embed_render.permute(0,3,1,2), num_cluster=4)
            embed_render = visualize_feature_map_by_normalization(embed_render.permute(0,3,1,2))    # range from -1 to 1
            axs[3].imshow(embed_render)
            axs[3].title.set_text('embed seg')
            # gt embed
            if gt_embed_render is not None:
                # gt_embed_render = visualize_feature_map_by_clustering(gt_embed_render, num_cluster=4)
                gt_embed_render = visualize_feature_map_by_normalization(gt_embed_render)    # range from -1 to 1
                axs[4].imshow(gt_embed_render)
                axs[4].title.set_text('gt embed seg')
            if next_rgb_render is not None:
                # gt next rgb frame
                axs[6].imshow(next_rgb_gt.cpu().numpy())
                axs[6].title.set_text('next tgt')
                # Ours
                axs[5].imshow(next_rgb_render.cpu().numpy())
                axs[5].title.set_text('next psnr={:.2f}'.format(psnr_dyna))
            # remove axis
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            
            if rank == 0:
                if self.cfg.use_wandb:
                    # save to buffer and write to wandb
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)

                    image = Image.open(buf)
                    wandb.log({"eval/recon_img": wandb.Image(image)}, step=step)

                    buf.close()
                    workdir = os.getcwd()
                    cprint(f'Saved {workdir}/recon/{step}_rgb.png to wandb', 'cyan')
                else:
                    plt.savefig(f'recon/{step}_rgb.png')
                    workdir = os.getcwd()
                    cprint(f'Saved {workdir}/recon/{step}_rgb.png locally', 'cyan')

        # new rendering---
        # TODO: _summaries内容不同相关的肯定要改
        # torch.cuda.empty_cache() # Mani中的没有，在bimanual中出现的
        self._summaries = {
            "losses/total_loss": total_loss,
            "losses/trans_loss": q_trans_loss.mean(),
            "losses/rot_loss": q_rot_loss.mean() if with_rot_and_grip else 0.0,
            "losses/grip_loss": q_grip_loss.mean() if with_rot_and_grip else 0.0,

            "losses/right/trans_loss": q_trans_loss.mean(),
            "losses/right/rot_loss": q_rot_loss.mean() if with_rot_and_grip else 0.0,
            "losses/right/grip_loss": q_grip_loss.mean() if with_rot_and_grip else 0.0,
            "losses/right/collision_loss": q_collision_loss.mean() if with_rot_and_grip else 0.0,

            "losses/left/trans_loss": q_trans_loss.mean(),
            "losses/left/rot_loss": q_rot_loss.mean() if with_rot_and_grip else 0.0,
            "losses/left/grip_loss": q_grip_loss.mean() if with_rot_and_grip else 0.0,
            "losses/left/collision_loss": q_collision_loss.mean() if with_rot_and_grip else 0.0,

            "losses/collision_loss": q_collision_loss.mean()
            if with_rot_and_grip
            else 0.0,
        }

        self._wandb_summaries = {
            'losses/total_loss': total_loss,
            'losses/trans_loss': q_trans_loss.mean(),
            'losses/rot_loss': q_rot_loss.mean() if with_rot_and_grip else 0.,
            'losses/grip_loss': q_grip_loss.mean() if with_rot_and_grip else 0.,
            'losses/collision_loss': q_collision_loss.mean() if with_rot_and_grip else 0.,
        
            # for visualization new form mani nerf
            'point_cloud': None,
            'left_coord_pred': left_coords,
            'right_coord_pred': right_coords,
            'right_coord_gt': right_gt_coord,
            'left_coord_gt': left_gt_coord,
        }

        if self._lr_scheduler:
            self._scheduler.step()
            # 记录最新学习率
            self._summaries["learning_rate"] = self._scheduler.get_last_lr()[0]

        self._vis_voxel_grid = voxel_grid[0]
        self._right_vis_translation_qvalue = self._softmax_q_trans(right_q_trans[0])
        self._right_vis_max_coordinate = right_coords[0]
        self._right_vis_gt_coordinate = right_action_trans[0]

        self._left_vis_translation_qvalue = self._softmax_q_trans(left_q_trans[0])
        self._left_vis_max_coordinate = left_coords[0]
        self._left_vis_gt_coordinate = left_action_trans[0]

        # Note: PerAct doesn't use multi-layer voxel grids like C2FARM
        # stack prev_layer_voxel_grid(s) from previous layers into a list
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [voxel_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [voxel_grid]

        # stack prev_layer_bound(s) from previous layers into a list
        if prev_layer_bounds is None:
            prev_layer_bounds = [self._coordinate_bounds.repeat(bs, 1)]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        return {
            "total_loss": total_loss,
            "prev_layer_voxel_grid": prev_layer_voxel_grid,
            "prev_layer_bounds": prev_layer_bounds,
        }
    # 基于bimanual照着Mani中的改写
    # new86--- bimanual中的新增的

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        deterministic = True
        bounds = self._coordinate_bounds
        prev_layer_voxel_grid = observation.get('prev_layer_voxel_grid', None)
        prev_layer_bounds = observation.get('prev_layer_bounds', None)
        lang_goal_tokens = observation.get('lang_goal_tokens', None).long()
        lang_goal = observation['lang_goal']

        # extract language embs
        with torch.no_grad():
            lang_goal_emb, lang_token_embs = self.language_model.extract(lang_goal)

        # voxelization resolution
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        max_rot_index = int(360 // self._rotation_resolution)
        # proprio = None

        # if self._include_low_dim_state:
        #     proprio = observation['low_dim_state']
        right_proprio = None
        left_proprio = None

        if self._include_low_dim_state:
            right_proprio = observation["right_low_dim_state"]
            left_proprio = observation["left_low_dim_state"]
            right_proprio = right_proprio[0].to(self._device)
            left_proprio = left_proprio[0].to(self._device)

        # for key in observation:
        #     print(f"key={key}") # 都缺depth
        obs, depth, pcd, extrinsics, intrinsics = self._act_preprocess_inputs(observation)

        # correct batch size and device
        obs = [[o[0][0].to(self._device), o[1][0].to(self._device)] for o in obs]
        # proprio = proprio[0].to(self._device) # 已经在上面实现了（双臂是两句话）
        pcd = [p[0].to(self._device) for p in pcd]
        lang_goal_emb = lang_goal_emb.to(self._device)
        lang_token_embs = lang_token_embs.to(self._device)
        bounds = torch.as_tensor(bounds, device=self._device)
        prev_layer_voxel_grid = prev_layer_voxel_grid.to(self._device) if prev_layer_voxel_grid is not None else None
        prev_layer_bounds = prev_layer_bounds.to(self._device) if prev_layer_bounds is not None else None

        proprio = torch.cat((right_proprio, left_proprio), dim=1) # new bimanual

        # inference !!多了一个渲染损失
        # q_trans, \
        # q_rot_grip, \
        # q_ignore_collisions, \
        # vox_grid,\
        # rendering_loss_dict  = self._q(obs,
        #                     depth,
        #                     proprio,
        #                     pcd,
        #                     extrinsics, # !! append
        #                     intrinsics, #
        #                     lang_goal_emb,
        #                     lang_token_embs,
        #                     bounds,
        #                     prev_layer_bounds,
        #                     prev_layer_voxel_grid, 
        #                     use_neural_rendering=False)     # 
        # 推理 bimanual
        # (
        #     right_q_trans,
        #     right_q_rot_grip,
        #     right_q_ignore_collisions,
        #     left_q_trans,
        #     left_q_rot_grip,
        #     left_q_ignore_collisions,
        # ), vox_grid = self._q(
        #     obs,
        #     proprio,
        #     pcd,
        #     lang_goal_emb,
        #     lang_token_embs,
        #     bounds,
        #     prev_layer_bounds,
        #     prev_layer_voxel_grid,
        # )
        # 两边叫法有些不一样，注意区别
        q, vox_grid, rendering_loss_dict = self._q(
            obs,
            depth, # nerf new
            proprio,
            pcd,
            extrinsics, # nerf new although augmented, not used
            intrinsics, # nerf new
            lang_goal_emb,
            lang_token_embs,
            bounds,
            prev_layer_bounds,
            prev_layer_voxel_grid,
            # nerf[3]---------------------
            use_neural_rendering=False
            # nerf[3]---------------------
        )

        (
            right_q_trans,
            right_q_rot_grip,
            right_q_ignore_collisions,
            left_q_trans,
            left_q_rot_grip,
            left_q_ignore_collisions,
            # left_q_collision,
        ) = q
        # softmax Q predictions
        # q_trans = self._softmax_q_trans(q_trans)
        # q_rot_grip =  self._softmax_q_rot_grip(q_rot_grip) if q_rot_grip is not None else q_rot_grip
        # q_ignore_collisions = self._softmax_ignore_collision(q_ignore_collisions) \
        #     if q_ignore_collisions is not None else q_ignore_collisions
        right_q_trans = self._softmax_q_trans(right_q_trans)
        left_q_trans = self._softmax_q_trans(left_q_trans)

        if right_q_rot_grip is not None:
            right_q_rot_grip = self._softmax_q_rot_grip(right_q_rot_grip)

        if left_q_rot_grip is not None:
            left_q_rot_grip = self._softmax_q_rot_grip(left_q_rot_grip)

        if right_q_ignore_collisions is not None:
            right_q_ignore_collisions = self._softmax_ignore_collision(
                right_q_ignore_collisions
            )

        if left_q_ignore_collisions is not None:
            left_q_ignore_collisions = self._softmax_ignore_collision(
                left_q_ignore_collisions
            )

        # argmax Q predictions 最大参数Q预测
        # coords, \
        # rot_and_grip_indicies, \
        # ignore_collisions = self._q.choose_highest_action(q_trans, q_rot_grip, q_ignore_collisions)

        # rot_grip_action = rot_and_grip_indicies if q_rot_grip is not None else None
        # ignore_collisions_action = ignore_collisions.int() if ignore_collisions is not None else None

        # coords = coords.int()
        # attention_coordinate = bounds[:, :3] + res * coords + res / 2
        (
            right_coords,
            right_rot_and_grip_indicies,
            right_ignore_collisions,
        ) = self._q.choose_highest_action(
            right_q_trans, right_q_rot_grip, right_q_ignore_collisions
        )
        (
            left_coords,
            left_rot_and_grip_indicies,
            left_ignore_collisions,
        ) = self._q.choose_highest_action(
            left_q_trans, left_q_rot_grip, left_q_ignore_collisions
        )

        if right_q_rot_grip is not None:
            right_rot_grip_action = right_rot_and_grip_indicies
        if right_q_ignore_collisions is not None:
            right_ignore_collisions_action = right_ignore_collisions.int()

        if left_q_rot_grip is not None:
            left_rot_grip_action = left_rot_and_grip_indicies
        if left_q_ignore_collisions is not None:
            left_ignore_collisions_action = left_ignore_collisions.int()

        right_coords = right_coords.int()
        left_coords = left_coords.int()

        right_attention_coordinate = bounds[:, :3] + res * right_coords + res / 2
        left_attention_coordinate = bounds[:, :3] + res * left_coords + res / 2


        # 将 prev_layer_voxel_grid(s) 叠加到列表中
        # 注意：PerAct 没有像 C2FARM 那样使用多层体素网格
        # stack prev_layer_voxel_grid(s) into a list
        # NOTE: PerAct doesn't used multi-layer voxel grids like C2FARM
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [vox_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [vox_grid]

        if prev_layer_bounds is None:
            prev_layer_bounds = [bounds]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        # observation_elements = {
        #     'attention_coordinate': attention_coordinate,
        #     'prev_layer_voxel_grid': prev_layer_voxel_grid,
        #     'prev_layer_bounds': prev_layer_bounds,
        # }
        # info = {
        #     'voxel_grid_depth%d' % self._layer: vox_grid,
        #     'q_depth%d' % self._layer: q_trans,
        #     'voxel_idx_depth%d' % self._layer: coords
        # }
        observation_elements = {
            "right_attention_coordinate": right_attention_coordinate,
            "left_attention_coordinate": left_attention_coordinate,
            "prev_layer_voxel_grid": prev_layer_voxel_grid,
            "prev_layer_bounds": prev_layer_bounds,
        }
        info = {
            "voxel_grid_depth%d" % self._layer: vox_grid,
            "right_q_depth%d" % self._layer: right_q_trans,
            "right_voxel_idx_depth%d" % self._layer: right_coords,
            "left_q_depth%d" % self._layer: left_q_trans,
            "left_voxel_idx_depth%d" % self._layer: left_coords,
        }
        # self._act_voxel_grid = vox_grid[0]
        # self._act_max_coordinate = coords[0]
        # self._act_qvalues = q_trans[0].detach()
        self._act_voxel_grid = vox_grid[0]
        self._right_act_max_coordinate = right_coords[0]
        self._right_act_qvalues = right_q_trans[0].detach()
        self._left_act_max_coordinate = left_coords[0]
        self._left_act_qvalues = left_q_trans[0].detach()
        action = (
            right_coords,
            right_rot_grip_action,
            right_ignore_collisions,
            left_coords,
            left_rot_grip_action,
            left_ignore_collisions,
        )
        # print("\033[0;34;40maction in act\033[0m",action)
        # return ActResult((coords, rot_grip_action, ignore_collisions_action),
        #                  observation_elements=observation_elements,
        #                  info=info)
        return ActResult(action, observation_elements=observation_elements, info=info)


    def update_summaries(self) -> List[Summary]:
        """收集和更新神经网络训练过程中的各种统计摘要信息 bimanual中有好多，但是都注释掉了"""
        # voxel_grid = self._vis_voxel_grid.detach().cpu().numpy()
        summaries = []
        # summaries.append(
        #     ImageSummary(
        #         "%s/right_update_qattention" % self._name,
        #         transforms.ToTensor()(
        #             visualise_voxel(
        #                 voxel_grid,
        #                 self._right_vis_translation_qvalue.detach().cpu().numpy(),
        #                 self._right_vis_max_coordinate.detach().cpu().numpy(),
        #                 self._right_vis_gt_coordinate.detach().cpu().numpy(),
        #             )
        #         ),
        #     )
        # )
        # summaries.append(
        #     ImageSummary(
        #         "%s/left_update_qattention" % self._name,
        #         transforms.ToTensor()(
        #             visualise_voxel(
        #                 voxel_grid,
        #                 self._left_vis_translation_qvalue.detach().cpu().numpy(),
        #                 self._left_vis_max_coordinate.detach().cpu().numpy(),
        #                 self._left_vis_gt_coordinate.detach().cpu().numpy(),
        #             )
        #         ),
        #     )
        # )
        for n, v in self._summaries.items():
            summaries.append(ScalarSummary('%s/%s' % (self._name, n), v))

        for (name, crop) in (self._crop_summary):
            crops = (torch.cat(torch.split(crop, 3, dim=1), dim=3) + 1.0) / 2.0
            # ImageSummary 用于记录图像数据
            summaries.extend([
                ImageSummary('%s/crops/%s' % (self._name, name), crops)])

        for tag, param in self._q.named_parameters():
            # assert not torch.isnan(param.grad.abs() <= 1.0).all()
            if param.grad is None:
                continue

            # 记录参数值的直方图分布。        
            summaries.append(
                HistogramSummary('%s/gradient/%s' % (self._name, tag),
                                 param.grad))
            summaries.append(
                HistogramSummary('%s/weight/%s' % (self._name, tag),
                                 param.data))

        return summaries
    
    
    def update_wandb_summaries(self):
        """更新和返回与Wandb(一个实验跟踪和可视化工具)相关的摘要信息"""
        summaries = dict()

        for k, v in self._wandb_summaries.items():
            summaries[k] = v
        return summaries


    def act_summaries(self) -> List[Summary]:
        """生成动作选择过程中的可视化摘要信息"""
        # voxel_grid = self._act_voxel_grid.cpu().numpy()
        # right_q_attention = self._right_act_qvalues.cpu().numpy()
        # right_highlight_coordinate = self._right_act_max_coordinate.cpu().numpy()
        # right_visualization = visualise_voxel(
        #     voxel_grid, right_q_attention, right_highlight_coordinate
        # )

        # left_q_attention = self._left_act_qvalues.cpu().numpy()
        # left_highlight_coordinate = self._left_act_max_coordinate.cpu().numpy()
        # left_visualization = visualise_voxel(
        #     voxel_grid, left_q_attention, left_highlight_coordinate
        # )

        # return [
        #     ImageSummary(
        #         f"{self._name}/right_act_Qattention",
        #         transforms.ToTensor()(right_visualization),
        #     ),
        #     ImageSummary(
        #         f"{self._name}/left_act_Qattention",
        #         transforms.ToTensor()(left_visualization),
        #     ),
        # ]
        # Mani写法
        # return [
        #     ImageSummary('%s/act_Qattention' % self._name,
        #                  transforms.ToTensor()(visualise_voxel(
        #                      self._act_voxel_grid.cpu().numpy(),
        #                      self._act_qvalues.cpu().numpy(),
        #                      self._act_max_coordinate.cpu().numpy())))]
        return []


    def load_weights(self, savedir: str):
        """加载模型权重"""
        # print("### ###  bc agent中的load weight  ### ### ")
        device = self._device if not self._training else torch.device('cuda:%d' % self._device)
        # if device is str, convert it to torch.device
        # 如果 device 为 str，则将其转换为 torch.device
        if isinstance(device, int):
            device = torch.device('cuda:%d' % self._device)

        # 模型权重文件的路径 加载模型的状态字典
        weight_file = os.path.join(savedir, '%s.pt' % self._name)
        state_dict = torch.load(weight_file, map_location=device)

        # load only keys that are in the current model
        # 代码检查并合并当前模型的状态字典与从文件中加载的状态字典，以确保只加载模型中存在的键。
        # 存储合并后的状态字典
        merged_state_dict = self._q.state_dict()
        for k, v in state_dict.items(): # 遍历存储的状态
            if not self._training:
                k = k.replace('_qnet.module', '_qnet')
                k = k.replace('_neural_renderer.module', '_neural_renderer')
            # else:   
            # 继续训练也有这个问题
                # print("training")
                # k = k.replace('_qnet', '_qnet._forward_module.module') # (old,new)
                # k = k.replace('_neural_renderer.module', '_neural_renderer')
            
            if k in merged_state_dict:
                # print("key in statr_dict:",k,"--------- ----------",k1 )
                merged_state_dict[k] = v
                # new for continue training
                # if self._training:
                #     k1 = k.replace('_qnet', '_qnet._forward_module.module') # (old,new)
                #     k2 = k.replace('_qnet', '_qnet._original_module')
                #     merged_state_dict[k1] = v
                #     merged_state_dict[k2] = v # 这一步可能没什么用
            else:
                if '_voxelizer' not in k: #and '_neural_renderer' not in k:
                    logging.warning(f"key {k} is found in checkpoint, but not found in current model.")
        # ---bimanual 特有，要不要加呢--
        if not self._training:
            # reshape voxelizer weights
            b = merged_state_dict["_voxelizer._ones_max_coords"].shape[0]
            merged_state_dict["_voxelizer._ones_max_coords"] = merged_state_dict["_voxelizer._ones_max_coords"][0:1]
            flat_shape = merged_state_dict["_voxelizer._flat_output"].shape[0]
            merged_state_dict["_voxelizer._flat_output"] = merged_state_dict["_voxelizer._flat_output"][0 : flat_shape // b]
            merged_state_dict["_voxelizer._tiled_batch_indices"] = merged_state_dict["_voxelizer._tiled_batch_indices"][0:1]
            merged_state_dict["_voxelizer._index_grid"] = merged_state_dict["_voxelizer._index_grid"][0:1]
        # ---bimanual--
        
        msg = self._q.load_state_dict(merged_state_dict, strict=False) # 运行时一直是这个，按照严格的形式debug
        # msg = self._q.load_state_dict(merged_state_dict, strict=True)
        if msg.missing_keys:
            print("missing some keys...") # True
            # for key in msg.missing_keys:
                # print("Missing key:", key)  # 打印每一个缺失的键
        if msg.unexpected_keys:
            print("unexpected some keys...")  # True
            # for key in msg.unexpected_keys:
                # print("Unexpected key:", key) 
        print("loaded weights from %s" % weight_file)


    def save_weights(self, savedir: str):
        """保存模型的权重到文件"""
        torch.save(self._q.state_dict(), os.path.join(savedir, '%s.pt' % self._name))
        # unexpect?
        # torch.save(self._qnet.state_dict(), os.path.join(savedir, '%s_qnet.pt' % self._name))
        # torch.save(self._neural_renderer.state_dict(), os.path.join(savedir, '%s_neural_renderer.pt' % self._name))
    
    def load_clip(self):
        """!! 前面出现过，还嫌他啰嗦 加载 CLIP(Contrastive Language-Image Pre-trainin)模型"""
        model, _ = load_clip("RN50", jit=False)
        self._clip_rn50 = build_model(model.state_dict())
        self._clip_rn50 = self._clip_rn50.float().to(self._device)
        self._clip_rn50.eval() 
        del model


    def unload_clip(self):
        """卸载 CLIP模型以释放资源"""
        del self._clip_rn50