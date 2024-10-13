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
from agents.manigaussian_bc2.gaussian_renderer import render,render_mask

import visdom
import logging
import einops

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

        self.znear = cfg.dataset.znear
        self.zfar = cfg.dataset.zfar
        self.trans = cfg.dataset.trans # default: [0, 0, 0]
        self.scale = cfg.dataset.scale

        # gs regressor 应该不用改
        self.gs_model = GeneralizableGSEmbedNet(cfg, with_gs_render=True)
        print(colored("[NeuralRenderer] GeneralizableGSEmbedNet is build", "cyan"))

        self.model_name = cfg.foundation_model_name
        self.d_embed = cfg.d_embed
        self.loss_embed_fn = cfg.loss_embed_fn

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
    
    def _mask_loss_fn(self, render_mask, gt_mask):
        """
        render_embed: [bs, h, w, 3]
        gt_embed: [bs, h, w, 3]
        """
        MIN_DENOMINATOR = 1e-12
        gt_mask = (gt_mask - gt_mask.min()) / (gt_mask.max() - gt_mask.min() + MIN_DENOMINATOR)
        loss_mask = l2_loss(render_mask, gt_mask)
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
                    rgb=None, depth=None, focal=None, c=None, lang_goal=None, tgt_pose=None, tgt_intrinsic=None,
                    next_tgt_pose=None, next_tgt_intrinsic=None, action=None, step=None, 
                    gt_mask=None,gt_mask_camera_extrinsic=None, gt_mask_camera_intrinsic=None, next_gt_mask = None):
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
        right_action, left_action = torch.split(action, split_size_or_sections=8, dim=1)
        # print("self.field_type=",self.field_type)
        # if self.cfg.method.field_type == 'bimanual_LF':
        data['right_action'] = right_action
        data['left_action'] = left_action
        # right_action, left_action = action.chunk(2, dim=2) # agent写法
        # print("\033[0;31;40mactionr in neural_rendering.py\033[0m",right_action)
        # print("\033[0;31;40mactionl in neural_rendering.py\033[0m",left_action)
        # print("\033[0;31;40maction in neural_rendering.py\033[0m",action)
        # tensor([[ 
        #   1.8244e-01, -1.2036e-01,  7.7095e-01, 6.9350e-05,  1.0000e+00,  2.3808e-04, -1.1926e-03,  0.0000e+00, 
        #   2.8597e-01,  3.3652e-01,  7.7093e-01, -9.3490e-05, 1.0000e+00,  7.0034e-04, 1.0216e-03,  0.0000e+00]], device='cuda:0')
        # print(action.shape) # torch.Size([1, 16])
        data['step'] = step
        data['mask_view'] = {}
        data['mask_view']['intr'] = gt_mask_camera_intrinsic 
        data['mask_view']['extr'] = gt_mask_camera_extrinsic         
        if data['mask_view']['intr'] is not None:
            data_novel = self.get_novel_calib(data['mask_view'])
            data['mask_view'].update(data_novel) # 更新数据

        # novel pose
        data['novel_view'] = {}
        data['intr'] = tgt_intrinsic # 相机内参通常包括焦距、主点坐标等，用于将3D坐标转换为2D图像坐标。
        data['extr'] = tgt_pose     # 相机外参通常包括旋转矩阵和平移向量，用于将世界坐标转换为相机坐标。相机外参定义了相机在世界坐标系中的位置和方向。
        data['xyz'] = einops.rearrange(pcd, 'b c h w -> b (h w) c')
        #  einops.rearrange 函数重新排列点云数据 pcd
        # 'b c h w -> b (h w) c' 是一个重组操作的模式，它将输入张量
        # 从四维格式（可能表示 [batch_size, channels, height, width]）转换为三维格式，
        # 其中高度和宽度被合并为一个维度。这通常用于将3D点云数据从图像格式转换为列表格式，

        # use extrinsic pose to generate gaussain parameters
        # 使用 extrinsic pose(外部姿态) 生成 Gaussain 参数
        if data['intr'] is not None:
            data_novel = self.get_novel_calib(data)
            data['novel_view'].update(data_novel)

        if self.use_dynamic_field:
            if self.field_type !='LF':
                data['next'] = {
                    'extr': next_tgt_pose,
                    'intr': next_tgt_intrinsic,
                    'novel_view': {},
                }
                if data['next']['intr'] is not None:
                    data_novel = self.get_novel_calib(data['next'])
                    data['next']['novel_view'].update(data_novel)
            # ------------------------------------------------------------------------------
            else:
                data['right_next'] = {
                    'extr': next_tgt_pose,
                    'intr': next_tgt_intrinsic,
                    'novel_view': {},
                }
                if data['right_next']['intr'] is not None:
                    data_novel = self.get_novel_calib(data['right_next'])
                    data['right_next']['novel_view'].update(data_novel)
                    data['right_next']['mask_view'] = {}
                    data['right_next']['mask_view'].update(data_novel) # 更新数据

                data['left_next'] = {
                    'extr': next_tgt_pose,
                    'intr': next_tgt_intrinsic,
                    'novel_view': {},
                }
                if data['left_next']['intr'] is not None:
                    data_novel = self.get_novel_calib(data['left_next'])
                    data['left_next']['novel_view'].update(data_novel)
                    data['left_next']['mask_view'] = {}
                    # data['mask_view']['intr'] = gt_mask_camera_intrinsic 
                    # data['mask_view']['extr'] = gt_mask_camera_extrinsic         
                    # if data['mask_view']['intr'] is not None:
                    #     data_novel = self.get_novel_calib(data)
                    data['left_next']['mask_view'].update(data_novel) # 更新数据
            # ------------------------------------------------------------------------------

        return data

    def get_novel_calib(self, data):
        """
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
            extr = data['extr'][i, ...].cpu().numpy()
            extr = np.linalg.inv(extr)  # the saved extrinsic is actually cam2world matrix, so turn it to world2cam matrix

            width, height = self.W, self.H
            R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)    # inverse
            T = np.array(extr[:3, 3], np.float32)
            FovX = focal2fov(intr[0, 0], width)
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

    def forward(self, pcd, dec_fts, language, gt_rgb=None, gt_pose=None, gt_intrinsic=None, rgb=None, depth=None, camera_intrinsics=None, camera_extrinsics=None, 
                focal=None, c=None, lang_goal=None, gt_depth=None,
                next_gt_pose=None, next_gt_intrinsic=None, next_gt_rgb=None, step=None, action=None,
                training=True, gt_mask=None, gt_mask_camera_extrinsic=None, gt_mask_camera_intrinsic=None, next_gt_mask = None):
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
            rgb=rgb, depth=depth, pcd=pcd, focal=focal, c=c, lang_goal=None, tgt_pose=gt_pose, tgt_intrinsic=gt_intrinsic,
            dec_fts=dec_fts, lang=language, next_tgt_pose=next_gt_pose, next_tgt_intrinsic=next_gt_intrinsic, 
            action=action, step=step, gt_mask=gt_mask, gt_mask_camera_extrinsic=gt_mask_camera_extrinsic, gt_mask_camera_intrinsic=gt_mask_camera_intrinsic,  
            next_gt_mask=next_gt_mask,
        )

        # 渲染 novel视角
        render_novel = None
        next_render_novel = None
        render_embed = None
        gt_embed = None
        render_mask_novel = None
        next_render_mask_novel_right = None
        next_render_mask_novel = None

        # create gt feature from foundation models 从基础模型创建 gt特征
        # 用于暂时禁用PyTorch中的梯度计算
        with torch.no_grad():
            # 提取基础模型特征 # Diffusion or dinov2
            gt_embed = self.extract_foundation_model_feature(gt_rgb, lang_goal)

        # print("training=",training)
        # if gt_rgb is not None:
        if training:
            # Gaussian Generator 高斯生成器
            # print("Gaussian Generator self.gs_model这里的data已经不对了",data["dec_fts"].shape)
            # gs regress (g) 应该也不用改
            # 这里也有数据处理（一直落下了）
            data = self.gs_model(data) # GeneralizableGSEmbedNet(cfg, with_gs_render=True)

            # Gaussian Render
            # print("data = self.pts2render")
            # if self.field_type !='LF'
            data = self.pts2render(data, bg_color=self.bg_color) # default: [0, 0, 0]
            # else:
                # data = self.pts2render_mask(data, bg_color=self.bg_color) # default: [0, 0, 0]


            # Loss L(GEO) 当前场景一致性损失 Current Scence Consistency Loss
            # permute置换  将张量的维度从原来的顺序重新排列为新的顺序  
            # predicetion预测: [1, 3, 128, 128] -> [1, 128, 128, 3]
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

            # Ll1 = l1_loss(render_novel, gt_rgb)
            
            Ll1 = l2_loss(render_novel, gt_rgb)
            # Lssim = 1.0 - ssim(render_novel, gt_rgb)
            Lssim = 0.
            # PSNR好像表示图片质量？
            psnr = PSNR_torch(render_novel, gt_rgb)

            loss = 0.
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

            # next frame prediction 下一帧预测 Ldyna(optional)
            if self.field_type != 'LF':
                if self.use_dynamic_field and (next_gt_rgb is not None) and ('xyz_maps' in data['next']):
                    data['next'] = self.pts2render(data['next'], bg_color=self.bg_color)
                    next_render_novel = data['next']['novel_view']['img_pred'].permute(0, 2, 3, 1)
                    # loss_dyna = l1_loss(next_render_novel, next_gt_rgb)
                    loss_dyna = l2_loss(next_render_novel, next_gt_rgb)
                    # 预热步数（3000步以后算上了）
                    lambda_dyna = self.cfg.lambda_dyna if step >= self.cfg.next_mlp.warm_up else 0.
                    # Step 3 Loss(LGeo? + L embed/L sem + L dyna) = loss_rgb + self.cfg.lambda_embed * loss_embed + lambda_dyna * loss_dyna
                    loss += lambda_dyna * loss_dyna

                    loss_reg = torch.tensor(0.)
                    # TODO: regularization on deformation 
                    # 考虑加入一些正则化项来处理形变（deformation）
                    # if self.cfg.lambda_reg > 0:
                    #     loss_reg = l2_loss(data['next']['xyz_maps'], data['xyz_maps'].detach()) #detach不追踪梯度的张量？
                    #     lambda_reg = self.cfg.lambda_reg if step >= self.cfg.next_mlp.warm_up else 0.
                    #     loss += lambda_reg * loss_reg

                    # TODO: local rigid loss 局部刚性损失
                else:
                    loss_dyna = torch.tensor(0.)
                    loss_reg = torch.tensor(0.)
            else:    # Leader Follower condition
                if self.use_dynamic_field and (next_gt_rgb is not None and gt_mask is not None):
                    right_min = 53
                    right_max = 73
                    left_min = 94
                    left_max = 114

                    # [1,1,128,128] -> [1,128,128,1] -> [1,128,128,3]
                    gt_mask = gt_mask.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)  # 复制三次，得到 [1,128, 128, 3]
                    next_gt_mask = next_gt_mask.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
                    
                    # numpy写法
                    # gt_mask_cpu = gt_mask1.cpu().numpy()
                    # print("gt_rgb_cpu",gt_mask_cpu,gt_mask_cpu.shape)
                    # gt_mask_label = np.zeros_like(gt_mask_cpu, dtype=np.uint8)

                    # Tensor写法（mask标签归类 0：bg 1:ritght 2:left）
                    gt_mask_label = torch.zeros_like(gt_mask, dtype=torch.uint8)
                    gt_mask_label[(gt_mask > right_min-1) & (gt_mask < right_max+1)] = 1
                    gt_mask_label[(gt_mask > left_min-1) & (gt_mask < left_max+1)] = 2

                    next_gt_mask_label = torch.zeros_like(next_gt_mask, dtype=torch.uint8)
                    next_gt_mask_label[(next_gt_mask > right_min-1) & (next_gt_mask < right_max+1)] = 1
                    next_gt_mask_label[(next_gt_mask > left_min-1) & (next_gt_mask < left_max+1)] = 2


                    # print("gt_mask_label", gt_mask_label, gt_mask_label.shape) # torch.Size([1, 128, 128, 3])

                    # # 用来可视化的东西
                    # gt_mask_label1 = gt_mask_label.squeeze(0)
                    # gt_mask_label2 = (gt_mask_label1*127) # .astype(np.uint8)
                    # gt_mask_label2 =gt_mask_label2.permute(2, 0, 1)
                    # print("gt_rgb_label2", gt_mask_label2, gt_mask_label2.shape) # torch.Size([3, 128, 128])
                    # from torchvision import transforms
                    # to_pil = transforms.ToPILImage()
                    # gt_mask_label2_cpu =gt_mask_label2.cpu()
                    # img = to_pil(gt_mask_label2_cpu)
                    # img.save('/data1/zjyang/program/peract_bimanual/scripts/test_demo/mask_label/gt_mask_label2_cpu.png')
                    # # cv2.imwrite('/data1/zjyang/program/peract_bimanual/scripts/test_demo/mask_label/gt_mask_label2.png', gt_mask_label2)


                    # mask Loss 只有front camera的训练
                    # # print(gt_mask.shape) # torch.Size([1, 1, 256, 256])  新版torch.Size([1, 1, 128, 128])
                    # 可视化   保存为 PNG 图片
                    # mask_np = gt_mask.squeeze().cpu().numpy() # 形状变为 [256, 256]
                    # print("mask.np1 = ",mask_np, mask_np.shape)
                    # image = Image.fromarray(mask_np.astype(np.uint8))  # 将值从 [0, 1] 转换到 [0, 255]
                    # image.save('/data1/zjyang/program/peract_bimanual/scripts/test_demo/mask_label/mask.png')

                    # 1 当前场景的mask 训练  loss_dyna_mask_novel
                    data =self.pts2render_mask(data, bg_color=self.bg_color)
                    render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]                           
                    loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask) # mask现阶段的

                    # 2 GRB total(Follower)  先对left预测（最后的结果） loss_dyna_follower  左臂 RGB Loss   
                    # 也可以说是双手结果
                    if ('xyz_maps' in data['left_next']):
                        data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                        next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]
                        # print("4 next_gt_mask.shape = ",next_gt_mask.shape, next_render_novel.shape) # torch.Size([1, 128, 128, 3]) 
                        loss_dyna_follower = l2_loss(next_render_novel, next_gt_rgb) # 双臂结果预测


                    # 3 next mask (pre - left(mask) ) next_loss_dyna_mask_left  左臂 Mask Loss
                    # data['right_next'] =self.pts2render_mask(data['right_next'], bg_color=self.bg_color)
                    # render_mask_novel_next = data['right_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1)
                    data['left_next'] =self.pts2render_mask(data['left_next'], bg_color=self.bg_color)
                    next_render_mask_novel = data['left_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]
                    next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask_novel, next_gt_mask) # mask去左臂的mask
                    # loss_dyna_mask = loss_dyna_mask_novel  + next_loss_dyna_mask_left

                    # exclude_right_mask = (render_mask_novel_next < right_min) | (render_mask_novel_next > right_max)
                    # exclude_left_mask = (next_render_mask_novel < left_min) | (next_render_mask_novel > left_max) # 排除左臂标签 [1,128, 128, 3] [True,False]
                    exclude_left_mask = (next_render_mask_novel != 2) 
                    # background_color = torch.tensor(self.bg_color, dtype=torch.float32)  # 背景
                    result_right_image = next_gt_rgb * exclude_left_mask    # [1, 128, 128, 3] # + background_color * (~exclude_right_mask_expanded)

                    # # print("next_render_mask_novel",next_render_mask_novel,next_render_mask_novel.shape) 
                    # exclude_test_mask = (gt_mask < left_min) | (gt_mask > left_max)  # [1, 128, 128, 3] ？
                    # rtest = gt_rgb * exclude_test_mask # gt_rgb来自nerf，所以用的时候只能用训练得到的mask（这里输出是错误的）
                    # # print("exclude_test_mask.shape",exclude_test_mask.shape)
                    # test1=test1.permute(0,2,3,1)
                    # etext1 = exclude_test_mask # .permute(0,2,3,1)
                    # test1_cpu = test1.cpu().numpy()
                    # etest1_cpu = etext1.cpu().numpy()
                    # rtest1_cpu = rtest.cpu().numpy()
                    # # print("test1_cpu",test1_cpu)
                    # # print("etest1_cpu",etest1_cpu, etest1_cpu.shape) # 一个全是True的数组
                    # # print("rtest1_cpu",rtest1_cpu,rtest1_cpu.shape)
                    # test2 = np.ones_like(test1_cpu, dtype=np.uint8)
                    # # etest2 = np.ones_like(etest1_cpu, dtype=np.uint8)
                    # test2[(test1_cpu < left_min) | (test1_cpu > left_max)] = 255  # (1,1,128,128) 为什么感觉不太对劲
                    # # print("test2 = ",test2,test2.shape)
                    # # # np.savetxt('/data1/zjyang/program/peract_bimanual/scripts/test_demo/mask_label/test1.txt', test1.cpu().numpy(), fmt='%d')
                    # test3 = test2
                    # # print("test3 = ",test3)
                    # test4 = test3.squeeze(0) # (1,128,128)
                    # etest4 = etest1_cpu.squeeze(0)
                    # rtest4 = rtest1_cpu.squeeze(0)
                    # # print("test4 = ",test4,test4.shape)
                    # # print("etest4 = ",etest4,etest4.shape)
                    # print("rtest4 = ",rtest4,rtest4.shape)
                    # etest5 = (etest4.astype(np.uint8)) * 255
                    # rtest5 = (rtest4*255).astype(np.uint8)
                    # print("etest5 = ",etest5,etest5.shape)
                    # print("rtest5 = ",rtest5,rtest5.shape)
                    # cv2.imwrite('/data1/zjyang/program/peract_bimanual/scripts/test_demo/mask_label/etest5.png', etest5)
                    # cv2.imwrite('/data1/zjyang/program/peract_bimanual/scripts/test_demo/mask_label/rtest5.png', rtest5)
                    # # # test5 = test4[:,:,0]
                    # # test5 = test4.squeeze(0)
                    # # print("test5 = ",test5,test5.shape)
                    # # import json
                    # # with open('/data1/zjyang/program/peract_bimanual/scripts/test_demo/mask_label/1.json', 'w') as f:
                    # #     json.dump(test2.tolist(), f)
                    # output_path = "/data1/zjyang/program/peract_bimanual/scripts/test_demo/mask_label/test4.png"
                    # # cv2.imwrite('/data1/zjyang/program/peract_bimanual/scripts/test_demo/mask_label/gt_mask_label2.png', gt_mask_label2)
                    # cv2.imwrite(output_path, test4)
                    # np.savetxt('/data1/zjyang/program/peract_bimanual/scripts/test_demo/mask_label/test4.txt', test4, fmt='%d')

                    #  4 RGB loss_dyna_leader leader  利用前面得到的mask删去左臂
                    if ('xyz_maps' in data['right_next']):
                        # with torch.no_grad():  原来为了有的Loss没计算报无回传错误而写的
                        data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                        next_render_novel = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                        # if self.cfg.mask_type=='exclude':   # 将预测图片根据mask裁剪
                        next_render_novel_mask = next_gt_rgb * exclude_left_mask 
                        # else:
                        #     next_render_novel_mask = next_gt_rgb * (~exclude_right_mask_expanded)
                        loss_dyna_leader = l2_loss(next_render_novel_mask, result_right_image)
                        # print('loss_dyna_leader = ', loss_dyna_leader)
                    
                    # 5 Mask loss_dyna_mask_next_right 右臂mask训练（和now一样 无用）
                    data['right_next'] =self.pts2render_mask(data['right_next'], bg_color=self.bg_color)
                    next_render_mask_novel_right = data['right_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1)
                    next_loss_dyna_mask_right = self._mask_loss_fn(next_render_mask_novel_right, next_gt_mask)
                    # next_loss_dyna_mask_right = l2_loss(next_render_mask_novel_right, next_gt_mask)

                    # pre mask = right +left    
                    next_loss_dyna_mask = next_loss_dyna_mask_left * ( 1 - self.cfg.lambda_mask_right ) + next_loss_dyna_mask_right * self.cfg.lambda_mask_right  # 右臂权重小一点
                    
                    # MASK = now +pre
                    loss_dyna_mask = loss_dyna_mask_novel *0.5  + next_loss_dyna_mask * 0.5

                    # RGB pre = leader( right ) + follower
                    loss_LF = loss_dyna_leader * self.cfg.lambda_dyna_leader + loss_dyna_follower * (1-self.cfg.lambda_dyna_leader)
                    # print('loss_LF = ', loss_LF, loss_dyna_leader, loss_dyna_follower)
                    loss_dyna = loss_LF * (1-self.cfg.lambda_mask) + loss_dyna_mask * self.cfg.lambda_mask 
                    # print('loss_dyna = ', loss_dyna,loss_LF,loss_dyna_mask)
                    # 预热步数（3000步以后算上了）
                    lambda_dyna = self.cfg.lambda_dyna if step >= self.cfg.next_mlp.warm_up else 0.                    
                    
                    # Step 3 Loss(LGeo? + L embed/L sem + L dyna) = loss_rgb + self.cfg.lambda_embed * loss_embed + lambda_dyna * loss_dyna
                    loss += lambda_dyna * loss_dyna

                    loss_reg = torch.tensor(0.)
                    # TODO: regularization on deformation 
                    # 考虑加入一些正则化项来处理形变（deformation）
                    # if self.cfg.lambda_reg > 0:
                    #     loss_reg = l2_loss(data['next']['xyz_maps'], data['xyz_maps'].detach()) #detach不追踪梯度的张量？
                    #     lambda_reg = self.cfg.lambda_reg if step >= self.cfg.next_mlp.warm_up else 0.
                    #     loss += lambda_reg * loss_reg

                    # TODO: local rigid loss 局部刚性损失

                else:
                    loss_dyna = torch.tensor(0.)
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
            # 无真实数据，渲染（推理）
            # no ground-truth given, rendering (inference) 
            with torch.no_grad():
                # Gaussian Generator
                data = self.gs_model(data)
                # Gaussian Render
                data = self.pts2render(data, bg_color=self.bg_color) # default: [0, 0, 0]
                # 当前场景
                render_novel = data['novel_view']['img_pred'].permute(0, 2, 3, 1) # channel last
                # 语义特征
                render_embed = data['novel_view']['embed_pred'].permute(0, 2, 3, 1)
                
                # 未来预测
                if self.field_type != 'LF':
                    if self.use_dynamic_field and 'xyz_maps' in data['next']:
                        data['next'] = self.pts2render(data['next'], bg_color=self.bg_color)
                        next_render_novel = data['next']['novel_view']['img_pred'].permute(0, 2, 3, 1)
                else:
                    # if self.use_dynamic_field and ('xyz_maps' in data['left_next']):
                    #     data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                    #     next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1)                
                    if self.use_dynamic_field and (next_gt_rgb is not None and gt_mask is not None):
                        # 1 当前场景的mask 训练  loss_dyna_mask_novel
                        data =self.pts2render_mask(data, bg_color=self.bg_color)
                        render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]                           

                        if ('xyz_maps' in data['left_next']):
                            data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                            next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]
                        data['left_next'] =self.pts2render_mask(data['left_next'], bg_color=self.bg_color)
                        next_render_mask_novel = data['left_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]

                        #  4 RGB loss_dyna_leader leader  利用前面得到的mask删去左臂
                        if ('xyz_maps' in data['right_next']):
                            # with torch.no_grad():  原来为了有的Loss没计算报无回传错误而写的
                            data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                            next_render_novel = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                        # 5 Mask loss_dyna_mask_next_right 右臂mask训练（和now一样 无用）
                        data['right_next'] =self.pts2render_mask(data['right_next'], bg_color=self.bg_color)
                        next_render_mask_novel_right = data['right_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1)

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
                          render_mask_novel = render_mask_novel,
                        next_render_mask_novel_right = next_render_mask_novel_right, next_render_mask_novel = next_render_mask_novel,)
        # print("render_mask_novel = ",render_mask_novel.shape, render_mask_novel)

        return loss_dict, ret_dict
    
    def pts2render(self, data: dict, bg_color=[0,0,0]):
        '''use render function in GSZ 在GSZ 中使用渲染功能(应该就是使用先前采集的数据重建场景)'''
        bs = data['intr'].shape[0]
        assert bs == 1, "batch size should be 1"
        # 公式2中 时刻i 的状态（θ 多了f 高级语义特征）
        i = 0
        xyz_i = data['xyz_maps'][i, :, :]
        feature_i = data['sh_maps'][i, :, :, :] # [16384, 4, 3]
        rot_i = data['rot_maps'][i, :, :]
        scale_i = data['scale_maps'][i, :, :]
        opacity_i = data['opacity_maps'][i, :, :]
        feature_language_i = data['feature_maps'][i, :, :]  # [B, N, 3]   [1, 65536, 3]  

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

    def pts2render_mask(self, data: dict, bg_color=[0,0,0]):
        '''use render function in GSZ 在GSZ 中使用渲染功能(应该就是使用先前采集的数据重建场景)'''
        bs = data['intr'].shape[0]
        assert bs == 1, "batch size should be 1"
        # 公式2中 时刻i 的状态（θ 多了f 高级语义特征）
        i = 0
        xyz_i = data['xyz_maps'][i, :, :]
        feature_i = data['sh_maps'][i, :, :, :] # [16384, 4, 3]
        rot_i = data['rot_maps'][i, :, :]
        scale_i = data['scale_maps'][i, :, :]
        opacity_i = data['opacity_maps'][i, :, :]
        precomputed_mask_i = data['mask_maps'][i, :, :] # mask  [1, 65536, 3]
        feature_language_i = data['feature_maps'][i, :, :]  # [B, N, 3]   [1, 65536, 3]  
        

        # 渲染返回字典  render应该是用来渲染的  from agents.manigaussian_bc2.gaussian_renderer import render
        render_return_dict = render_mask(
            data, i, xyz_i, rot_i, scale_i, opacity_i, 
            bg_color=bg_color, pts_rgb=None, features_color=feature_i, features_language=feature_language_i,
            precomputed_mask = precomputed_mask_i,
            )

        # .unsqueeze(0): 这是PyTorch张量的一个操作，用于在张量的第0个维度（即最前面）增加一个维度。如果原始张量是一维的，这个操作会将其变成二维的，其中新加的维度大小为1。
        # data['novel_view']['img_pred']: 这是在 data 字典中的 'novel_view' 键下创建或更新一个子键 'img_pred'。这个子键被赋值为 render_return_dict['render'] 张量增加一个新维度后的结果。
        data['novel_view']['mask_pred'] = render_return_dict['mask'].unsqueeze(0)
        data['novel_view']['embed_pred'] = render_return_dict['render_embed'].unsqueeze(0)
        return data