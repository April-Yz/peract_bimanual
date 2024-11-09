import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import torchvision.transforms as T

from termcolor import colored, cprint
from dotmap import DotMap

import agents.manigaussian_bc2.utils as utils
from agents.manigaussian_bc2.models_embed import GeneralizableGSEmbedNet
from agents.manigaussian_bc2.loss import l1_loss, l2_loss, cosine_loss, ssim
from agents.manigaussian_bc2.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from agents.manigaussian_bc2.gaussian_renderer import render,render_mask, render_mask_gen,render1,render_rgb
from agents.manigaussian_bc2.project_hull import label_point_cloud, points_inside_convex_hull, \
    depth_mask_to_3d, project_3d_to_2d, create_2d_mask_from_convex_hull, merge_arrays, merge_tensors
import visdom
import logging
import einops
import time
import random

# for debugging 
# from PIL import Image
# import cv2 

def PSNR_torch(img1, img2, max_val=1, mask=None):
    """计算两张图像之间的峰值信噪比（Peak Signal-to-Noise Ratio，简称 PSNR）。
    PSNR 是一种常用的衡量图像质量的指标，特别是在评估图像重建、压缩或去噪算法的性能时。
    PSNR 值越高，表示两幅图像越相似，图像质量越好。"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0).to(img1.device)
    PIXEL_MAX = max_val
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


class NeuralRenderer(nn.Module):
    """
    take a voxel, camera pose, and camera intrinsics as input,
    and output a rendered image
    将体素、摄像机姿态和摄像机本征作为输入、
    并输出渲染图像
    """
    def __init__(self, cfg):
        super(NeuralRenderer, self).__init__()

        self.cfg = cfg
        self.coordinate_bounds = cfg.coordinate_bounds # bounds of voxel grid
        self.W = cfg.image_width
        self.H = cfg.image_height
        self.bg_color = cfg.dataset.bg_color
        self.bg_mask = [0,0,0] #[1,0,0]

        self.znear = cfg.dataset.znear
        self.zfar = cfg.dataset.zfar
        self.trans = cfg.dataset.trans # default: [0, 0, 0]
        self.scale = cfg.dataset.scale
        # 定义类别数
        self.num_classes = 3

        # gs regressor 应该不用改
        self.gs_model = GeneralizableGSEmbedNet(cfg, with_gs_render=True)
        print(colored("[NeuralRenderer] GeneralizableGSEmbedNet is build", "cyan"))

        self.model_name = cfg.foundation_model_name
        self.d_embed = cfg.d_embed
        self.loss_embed_fn = cfg.loss_embed_fn
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean') 
        self.criterion_nll = nn.NLLLoss()

        if self.model_name == "diffusion":
            from odise.modeling.meta_arch.ldm import LdmFeatureExtractor
            import torchvision.transforms as T
            self.feature_extractor = LdmFeatureExtractor(
                            encoder_block_indices=(5, 7),
                            unet_block_indices=(2, 5, 8, 11),
                            decoder_block_indices=(2, 5),
                            steps=(0,),
                            captioner=None,
                        )
            self.diffusion_preprocess = T.Resize(512, antialias=True)
            cprint("diffusion feature dims: "+str(self.feature_extractor.feature_dims), "yellow")
        elif self.model_name == "dinov2":
            from agents.manigaussian_bc2.dino_extractor import VitExtractor
            import torchvision.transforms as T
            self.feature_extractor = VitExtractor(
                model_name='dinov2_vitl14',
            )
            self.dino_preprocess = T.Compose([
                T.Resize(224 * 8, antialias=True),  # must be a multiple of 14
            ])
            cprint("dinov2 feature dims: "+str(self.feature_extractor.feature_dims), "yellow")
        else:
            cprint(f"foundation model {self.model_name} is not implemented", "yellow")

        self.lambda_embed = cfg.lambda_embed
        print(colored(f"[NeuralRenderer] foundation model {self.model_name} is build. loss weight: {self.lambda_embed}", "cyan"))

        self.lambda_rgb = 1.0 if cfg.lambda_rgb is None else cfg.lambda_rgb
        print(colored(f"[NeuralRenderer] rgb loss weight: {self.lambda_rgb}", "cyan"))

        self.use_dynamic_field = cfg.use_dynamic_field
        self.field_type = cfg.field_type
        self.mask_gen = cfg.mask_gen
        self.use_nerf_picture = cfg.use_nerf_picture

    def _embed_loss_fn(self, render_embed, gt_embed):
        """
        render_embed: [bs, h, w, 3]
        gt_embed: [bs, h, w, 3]
        """
        if self.loss_embed_fn == "l2_norm":
            # label normalization
            MIN_DENOMINATOR = 1e-12
            gt_embed = (gt_embed - gt_embed.min()) / (gt_embed.max() - gt_embed.min() + MIN_DENOMINATOR)
            loss_embed = l2_loss(render_embed, gt_embed)
        elif self.loss_embed_fn == "l2":
            loss_embed = l2_loss(render_embed, gt_embed)
        elif self.loss_embed_fn == "cosine":
            loss_embed = cosine_loss(render_embed, gt_embed)
        else:
            cprint(f"loss_embed_fn {self.loss_embed_fn} is not implemented", "yellow")
        return loss_embed

    def ele_multip_in_chunks(self, feat_expanded, masks_expanded, chunk_size=5):
        # 逐块元素乘法运算
        result = torch.zeros_like(feat_expanded)
        for i in range(0, feat_expanded.size(0), chunk_size):
            end_i = min(i + chunk_size, feat_expanded.size(0))
            for j in range(0, feat_expanded.size(1), chunk_size):
                end_j = min(j + chunk_size, feat_expanded.size(1))
                chunk_feat = feat_expanded[i:end_i, j:end_j]
                chunk_mask = masks_expanded[i:end_i, j:end_j].float()

                result[i:end_i, j:end_j] = chunk_feat * chunk_mask
        return result


    def _mask_ce_loss_fn(self, render_mask, gt_mask):
        """
        render_embed: [bs, h, w, 3]
        gt_embed: [bs, h, w, 3]
        """
        MIN_DENOMINATOR = 1e-12
        render_mask = (render_mask - render_mask.min()) / (render_mask.max() - render_mask.min() + MIN_DENOMINATOR)
        loss_mask = self.CrossEntropyLoss(render_mask, gt_mask)
        return loss_mask

    def _save_gradient(self, name):
        """
        用作神经网络中的钩子，以便在反向传播时捕获和检查梯度。
        钩子函数可以在梯度计算完成后执行额外的操作，例如打印信息或检查梯度的值。
        for debugging language feature rendering
        """
        def hook(grad):
            print(f"name={name}, grad={grad}")
            return grad
        return hook

    def extract_foundation_model_feature(self, gt_rgb, lang_goal):
        """
        从基础模型中提取特征，这些特征可能用于图像渲染、图像处理或其他机器学习任务
        we use the last layer of the diffusion feature extractor  我们使用扩散特征提取器的最后一层
        因为我们将 128x128 的图像重塑为 512x512，所以最后一层的特征只是 128x128
        since we reshape 128x128 img to 512x512, the last layer's feature is just 128x128
        thus, no need to resize the feature map    因此，无需调整特征图的大小
        lang_goal: numpy.ndarray, [bs, 1, 1]
        """
        
        if self.model_name == "diffusion":
            """
            we support multiple captions for batched input here
            """
            if lang_goal.shape[0] > 1:
                caption = ['a robot arm ' + cap.item() for cap in lang_goal]
            else:
                caption = "a robot arm " + lang_goal.item()
            batched_input = {'img': self.diffusion_preprocess(gt_rgb.permute(0, 3, 1, 2)), 'caption': caption}
            feature_list, lang_embed = self.feature_extractor(batched_input) # list of visual features, and 77x768 language embedding
            used_feature_idx = -1  
            gt_embed = feature_list[used_feature_idx]   # [bs,512,128,128]

            # NOTE: dimensionality reduction with PCA, which is used to satisfy the output dimension of the Gaussian Renderer
            bs = gt_rgb.shape[0]
            A = gt_embed.reshape(bs, 512, -1).permute(0, 2, 1)  # [bs, 128*128, 512]
            gt_embed_list = []
            for i in range(bs):
                U, S, V = torch.pca_lowrank(A[i], q=np.maximum(6, self.d_embed))
                reconstructed_embed = torch.matmul(A[i], V[:, :self.d_embed])
                gt_embed_list.append(reconstructed_embed)

            gt_embed = torch.stack(gt_embed_list, dim=0).permute(0, 2, 1).reshape(bs, self.d_embed, 128, 128)
            return gt_embed
        
        elif self.model_name == "dinov2":
            batched_input = self.dino_preprocess(gt_rgb.permute(0, 3, 1, 2))    # resize
            feature = self.feature_extractor(batched_input)
            gt_embed = F.interpolate(feature, size=(128, 128), mode='bilinear', align_corners=False)    # [b, 1024, 128, 128]

            # NOTE: dimensionality reduction with PCA, which is used to satisfy the output dimension of the Gaussian Renderer
            bs = gt_rgb.shape[0]
            A = gt_embed.reshape(bs, 1024, -1).permute(0, 2, 1)  # [bs, 128*128, 1024]
            gt_embed_list = []
            for i in range(bs):
                U, S, V = torch.pca_lowrank(A[i], q=np.maximum(6, self.d_embed))
                reconstructed_embed = torch.matmul(A[i], V[:, :self.d_embed])
                gt_embed_list.append(reconstructed_embed)
            gt_embed = torch.stack(gt_embed_list, dim=0).permute(0, 2, 1).reshape(bs, self.d_embed, 128, 128)
            return gt_embed
        else:
            return None

    def encode_data(self, pcd, dec_fts, lang, 
                    rgb=None, depth=None, mask=None, focal=None, c=None, lang_goal=None, tgt_pose=None, tgt_intrinsic=None,
                    next_tgt_pose=None, next_tgt_intrinsic=None, action=None, step=None, 
                    gt_mask_camera_extrinsic=None, gt_mask_camera_intrinsic=None, indx=None,
                    next_gt_mask_camera_extrinsic=None, next_gt_mask_camera_intrinsic=None,
                    gt_mask=None,next_gt_mask = None,
                    ): # 后续删掉的一行
        '''prepare data dict'''
        bs = pcd.shape[0]
        data = {}
        # format input
        data['img'] = rgb
        # print("\033[0;31;40m rgb in neural_rendering.py\033[0m",rgb)
        data['dec_fts'] = dec_fts
        # print("encode_data ----- dec_fts.shape", dec_fts.shape)
        data['depth'] = depth
        # print("\033[0;31;40m depth in neural_rendering.py\033[0m",depth)
        data['lang'] = lang
        data['action'] = action
        # maniaction 不确定rl那个在前
        # right_action, left_action = torch.split(action, split_size_or_sections=8, dim=1)
        # data['right_action'] = right_action
        # data['left_action'] = left_action
        data['step'] = step

        # novel pose
        data['novel_view'] = {}
        data['intr'] = tgt_intrinsic # nerf 相机内参通常包括焦距、主点坐标等，用于将3D坐标转换为2D图像坐标。
        data['extr'] = tgt_pose     # 相机外参通常包括旋转矩阵和平移向量，用于将世界坐标转换为相机坐标。相机外参定义了相机在世界坐标系中的位置和方向。
        data['xyz'] = einops.rearrange(pcd, 'b c h w -> b (h w) c') # bs,256*256,(xyz)
        #  einops.rearrange 函数重新排列点云数据 pcd   'b c h w -> b (h w) c' 是一个重组操作的模式，它将输入张量从四维格式（可能表示 [batch_size, channels, height, width]）转换为三维格式，  其中高度和宽度被合并为一个维度。这通常用于将3D点云数据从图像格式转换为列表格式，

        # use extrinsic pose to generate gaussain parameters
        # 使用 extrinsic pose(外部姿态) 生成 Gaussain 参数
        if data['intr'] is not None:
            data_novel = self.get_novel_calib(data, False)
            data['novel_view'].update(data_novel)
        if self.use_dynamic_field:
            if self.field_type =='bimanual':      # 双臂同时工作
                data['next'] = {
                    'extr': next_tgt_pose,
                    'intr': next_tgt_intrinsic,
                    'novel_view': {},
                }
                if data['next']['intr'] is not None:
                    data_novel = self.get_novel_calib(data['next'], False)
                    data['next']['novel_view'].update(data_novel)
        return data

    def get_novel_calib(self, data, mask=False):
        """
        True :mask          False: rgb
        get readable camera state for gaussian renderer from gt_pose
        从 gt_pose 获取 Gaussian Renderer 的可读摄像机状态
        :param data: dict
        :param data['intr']: intrinsic matrix
        :param data['extr']: c2w matrix

        :return: dict
        """
        bs = data['intr'].shape[0]
        device = data['intr'].device
        fovx_list, fovy_list, world_view_transform_list, full_proj_transform_list, camera_center_list = [], [], [], [], []
        for i in range(bs):
            intr = data['intr'][i, ...].cpu().numpy()
            # print("intr",intr)
            if mask: # mask图片需要缩放到128 128
                intr = intr / 2

            extr = data['extr'][i, ...].cpu().numpy()
            extr = np.linalg.inv(extr)  # the saved extrinsic is actually cam2world matrix, so turn it to world2cam matrix 保存的extrinsic实际上是cam2world矩阵，所以将其转为world2cam矩阵

            width, height = self.W, self.H
            # print("width, height",width, height)
            R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)    # inverse 旋转矩阵
            T = np.array(extr[:3, 3], np.float32)                                   # 移动
            FovX = focal2fov(intr[0, 0], width)     # 视场角=focal2fov（焦距，w) 
            FovY = focal2fov(intr[1, 1], height)
            projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, K=intr, h=height, w=width).transpose(0, 1)
            world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.trans), self.scale)).transpose(0, 1) # [4, 4], w2c
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)    # [4, 4]
            camera_center = world_view_transform.inverse()[3, :3]   # inverse is c2w

            fovx_list.append(FovX)
            fovy_list.append(FovY)
            world_view_transform_list.append(world_view_transform.unsqueeze(0))
            full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
            camera_center_list.append(camera_center.unsqueeze(0))

        novel_view_data = {
            'FovX': torch.FloatTensor(np.array(fovx_list)).to(device),
            'FovY': torch.FloatTensor(np.array(fovy_list)).to(device),
            'width': torch.tensor([width] * bs).to(device),
            'height': torch.tensor([height] * bs).to(device),
            'world_view_transform': torch.concat(world_view_transform_list).to(device),
            'full_proj_transform': torch.concat(full_proj_transform_list).to(device),
            'camera_center': torch.concat(camera_center_list).to(device),
        }

        return novel_view_data

    def forward(self, pcd, dec_fts, language, mask=None, gt_rgb=None, gt_pose=None, gt_intrinsic=None, rgb=None, depth=None, camera_intrinsics=None, camera_extrinsics=None, 
                focal=None, c=None, lang_goal=None, gt_depth=None,
                next_gt_pose=None, next_gt_intrinsic=None, next_gt_rgb=None, step=None, action=None,
                training=True, gt_mask=None, gt_mask_camera_extrinsic=None, gt_mask_camera_intrinsic=None, next_gt_mask = None,
                next_gt_mask_camera_extrinsic=None, next_gt_mask_camera_intrinsic=None,
                gt_maskdepth=None,next_gt_maskdepth=None,):
        '''
        main forward function
        Return:
        :loss_dict: dict, loss values
        :ret_dict: dict, rendered images
        '''
        bs = rgb.shape[0]
        # print("dec_fts.shape=",dec_fts.shape)
        # print("好吧，这里的data也有问题，但是谁干的是谁调用了我的neuralrender forward啊啊啊!!!! bs=",bs)
        # 数据预处理 return 字典对应各类信息

        data = self.encode_data(
                rgb=rgb, depth=depth, pcd=pcd,mask=mask, focal=focal, c=c, lang_goal=None, 
                tgt_pose=gt_pose, tgt_intrinsic=gt_intrinsic,next_tgt_pose=next_gt_pose, next_tgt_intrinsic=next_gt_intrinsic,   # 相机参数
                dec_fts=dec_fts, lang=language, action=action, step=step, 
                gt_mask=gt_mask,next_gt_mask=next_gt_mask, 
                gt_mask_camera_extrinsic=gt_mask_camera_extrinsic, gt_mask_camera_intrinsic=gt_mask_camera_intrinsic, 
                next_gt_mask_camera_extrinsic=next_gt_mask_camera_extrinsic, next_gt_mask_camera_intrinsic=next_gt_mask_camera_intrinsic, 
                            )


        # 渲染 novel视角
        render_novel = None
        next_render_novel = None
        render_embed = None
        gt_embed = None

        # create gt feature from foundation models 从基础模型创建 gt特征
        # 用于暂时禁用PyTorch中的梯度计算
        with torch.no_grad():
            # 提取基础模型特征 # Diffusion or dinov2
            gt_embed = self.extract_foundation_model_feature(gt_rgb, lang_goal)
        

        # if gt_rgb is not None:
        if training:
            # Gaussian Generator 高斯生成器 gs regress (g) 应该也不用改
            data = self.gs_model(data) # GeneralizableGSEmbedNet(cfg, with_gs_render=True)
            # torch.nn.utils.clip_grad_norm_(data['xyz_maps'].parameters(), max_norm=1.0)

            # Gaussian Render
            data = self.pts2render(data, bg_color=self.bg_color) # default: [0, 0, 0]

            # Loss L(GEO) 当前场景一致性损失 Current Scence Consistency Loss
            # permute置换  将张量的维度从原来的顺序重新排列为新的顺序 
            render_novel = data['novel_view']['img_pred'].permute(0, 2, 3, 1)   # [1, 128, 128, 3]            
            # visdom 视界(可视化数据用的) Manigaussian2 中是False bash中好像也没有指定
            if self.cfg.visdom: # False
                vis = visdom.Visdom()
                rgb_vis = data['img'][0].detach().cpu().numpy() * 0.5 + 0.5
                vis.image(rgb_vis, win='front_rgb', opts=dict(title='front_rgb'))
                depth_vis = data['depth'][0].detach().cpu().numpy()#/255.0
                # convert 128x128 0-255 depth map to 3x128x128 0-1 colored map 
                # 将 128x128 0-255 深度贴图转换为 3x128x128 0-1 彩色贴图
                vis.image(depth_vis, win='front_depth', opts=dict(title='front_depth'))
                vis.image(render_novel[0].permute(2, 0, 1).detach().cpu().numpy(), win='render_novel', opts=dict(title='render_novel'))
                vis.image(gt_rgb[0].permute(2, 0, 1).detach().cpu().numpy(), win='gt_novel', opts=dict(title='gt_novel'))
            
            loss = 0.
            # print("render_novel=",render_novel.shape,gt_rgb.shape)
            Ll1 = l2_loss(render_novel, gt_rgb) # loss_now_rgb
            # Lssim = 1.0 - ssim(render_novel, gt_rgb)
            Lssim = 0.
            # PSNR好像表示图片质量？
            psnr = PSNR_torch(render_novel, gt_rgb)

            # loss_rgb = self.cfg.lambda_l1 * Ll1 + self.cfg.lambda_ssim * Lssim
            loss_rgb = Ll1
            # 1 LGeo?
            loss += loss_rgb

            # 语义（optional）
            if gt_embed is not None:
                # 比较真实和render的embed 应该是语义Lsem
                gt_embed = gt_embed.permute(0, 2, 3, 1) # channel last
                render_embed = data['novel_view']['embed_pred'].permute(0, 2, 3, 1)
                # DEBUG gradient    debug 梯度
                # render_embed_grad = render_embed.register_hook(self._save_gradient('render_embed'))
                loss_embed = self._embed_loss_fn(render_embed, gt_embed)
                # 2 loss(LGeo? + embed是啥 应该是语义Lsem) = loss_rgb + self.cfg.lambda_embed * loss_embed
                loss += self.cfg.lambda_embed * loss_embed
            else:
                loss_embed = torch.tensor(0.)

            # Ll1 = l1_loss(render_novel, gt_rgb) 
            loss_dyna_mask = torch.tensor(0.) 

            # next frame prediction 下一帧预测 Ldyna(optional)
            if self.field_type == 'bimanual':
                if self.use_dynamic_field and (next_gt_rgb is not None) and ('xyz_maps' in data['next']):
                    data['next'] = self.pts2render(data['next'], bg_color=self.bg_color)
                    next_render_novel = data['next']['novel_view']['img_pred'].permute(0, 2, 3, 1)
                    loss_dyna = l2_loss(next_render_novel, next_gt_rgb)
                    lambda_dyna = self.cfg.lambda_dyna if step >= self.cfg.next_mlp.warm_up else 0.
                    loss += lambda_dyna * loss_dyna

                    loss_reg = torch.tensor(0.)
                    # TODO: regularization on deformation 
                    # 考虑加入一些正则化项来处理形变（deformation）
                    # if self.cfg.lambda_reg > 0:
                    #     loss_reg = l2_loss(data['next']['xyz_maps'], data['xyz_maps'].detach()) #detach不追踪梯度的张量？
                    #     lambda_reg = self.cfg.lambda_reg if step >= self.cfg.next_mlp.warm_up else 0.
                    #     loss += lambda_reg * loss_reg

                    # TODO: local rigid loss 局部刚性损失
                    loss_LF = torch.tensor(0.)
                    loss_dyna_mask = torch.tensor(0.)
                else:
                    loss_dyna = torch.tensor(0.)
                    loss_LF = torch.tensor(0.)
                    loss_dyna_mask = torch.tensor(0.)
                    loss_reg = torch.tensor(0.)

            loss_dict = {
                'loss': loss,
                'loss_rgb': loss_rgb.item(),
                'loss_embed': loss_embed.item(),
                'loss_dyna': loss_dyna.item(),
                'loss_LF': loss_LF.item(),
                'loss_dyna_mask': loss_dyna_mask.item(),
                'loss_reg': loss_reg.item(),
                'l1': Ll1.item(),
                'psnr': psnr.item(),
                }
        else: # not training （第0次是走这边的）
            # no ground-truth given, rendering (inference) 
            with torch.no_grad():
                # Gaussian Generator
                data = self.gs_model(data)
                # Gaussian Render
                data = self.pts2render(data, bg_color=self.bg_color) # default: [0, 0, 0]
                # 当前场景
                render_novel = data['novel_view']['img_pred'].permute(0, 2, 3, 1) # channel last
                render_embed = data['novel_view']['embed_pred'].permute(0, 2, 3, 1)
                
                # 未来预测
                if self.field_type == 'bimanual':
                    if self.use_dynamic_field and 'xyz_maps' in data['next']:
                        data['next'] = self.pts2render(data['next'], bg_color=self.bg_color)
                        next_render_novel = data['next']['novel_view']['img_pred'].permute(0, 2, 3, 1)

                loss_dict = {
                    'loss': 0.,
                    'loss_rgb': 0.,
                    'loss_embed': 0.,
                    'loss_dyna': 0.,
                    'loss_LF':  0.,
                    'loss_dyna_mask':  0.,
                    'loss_reg': 0.,
                    'l1': 0.,
                    'psnr': 0.,
                }

        # get Gaussian embedding 获得高斯嵌入
        # dotmap 允许使用点（.）符号来访问字典中的键
        ret_dict = DotMap(render_novel=render_novel, next_render_novel=next_render_novel,
                          render_embed=render_embed, gt_embed=gt_embed, 
                        )             
        return loss_dict, ret_dict
    
    def pts2render(self, data: dict, bg_color=[0,0,0]):
        '''use render function in GSZ 在GSZ 中使用渲染功能(应该就是使用先前采集的数据重建场景)'''
        bs = data['intr'].shape[0]
        assert bs == 1, "batch size should be 1"
        # 公式2中 时刻i 的状态（θ 多了f 高级语义特征）
        i = 0
        xyz_i = data['xyz_maps'][i, :, :] 
        # feature_i = data['sh_maps'][i, :, :] # [16384, 4, 3]
        feature_i = data['sh_maps'][i, :, :, :] # *0.2 + 0.2 # [16384, 4, 3]
        rot_i = data['rot_maps'][i, :, :] #*0 +0.1
        scale_i = data['scale_maps'][i, :, :] #*0 +0.01
        opacity_i = data['opacity_maps'][i, :, :] # *0 +0.1
        feature_language_i = data['feature_maps'][i, :, :]# *0 +0.1  # [B, N, 3]   [1, 65536, 3]  

        # # 随机生成一个
        # feature_i = torch.rand(65536, 4, 3)
        # rot_i = torch.rand(65536, 3)
        # scale_i = torch.rand(65536, 3) 
        # opacity_i = torch.rand(65536, 3) * 0.2 # 透明一些
        # feature_language_i = torch.rand(65536, 3)

        # 渲染返回字典  render应该是用来渲染的  from agents.manigaussian_bc2.gaussian_renderer import render
        render_return_dict = render(
            data, i, xyz_i, rot_i, scale_i, opacity_i, 
            bg_color=bg_color, pts_rgb=None, features_color=feature_i, features_language=feature_language_i
            )

        # .unsqueeze(0): 这是PyTorch张量的一个操作，用于在张量的第0个维度（即最前面）增加一个维度。如果原始张量是一维的，这个操作会将其变成二维的，其中新加的维度大小为1。
        # data['novel_view']['img_pred']: 这是在 data 字典中的 'novel_view' 键下创建或更新一个子键 'img_pred'。这个子键被赋值为 render_return_dict['render'] 张量增加一个新维度后的结果。
        data['novel_view']['img_pred'] = render_return_dict['render'].unsqueeze(0)
        data['novel_view']['embed_pred'] = render_return_dict['render_embed'].unsqueeze(0)
        return data

    def pts2render1(self, data: dict, bg_color=[0,0,0]):
        '''feature_language_i用mask赋值'''
        bs = data['intr'].shape[0]
        assert bs == 1, "batch size should be 1"
        # 公式2中 时刻i 的状态（θ 多了f 高级语义特征）
        i = 0
        xyz_i = data['xyz_maps'][i, :, :].detach()
        feature_i = data['sh_maps'][i, :, :, :].detach() # [16384, 4, 3]
        rot_i = data['rot_maps'][i, :, :].detach()
        scale_i = data['scale_maps'][i, :, :].detach()
        opacity_i = data['opacity_maps'][i, :, :].detach()
        feature_language_i = data['mask_maps'][i, :, :]  # [B, N, 3]   [1, 65536, 3]  

        # 渲染返回字典  render应该是用来渲染的  from agents.manigaussian_bc2.gaussian_renderer import render
        render_return_dict = render1(
            data, i, xyz_i, rot_i, scale_i, opacity_i, 
            bg_color=bg_color, pts_rgb=None, features_color=feature_i, features_language=feature_language_i
            )

        data['novel_view']['img_pred'] = render_return_dict['render'].unsqueeze(0)
        data['novel_view']['embed_pred'] = render_return_dict['render_embed'].unsqueeze(0)
        return data

