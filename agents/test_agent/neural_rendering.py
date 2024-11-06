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


        self.use_CEloss =7 # 常用：0/7/1  2:两个维度 7：3维度-ignore0 5搞笑版language代替mask 6 onlyembed

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

    def mask_feature_mean(self, feat_map, gt_masks, image_mask=None, return_var=False):
        """Compute the average instance features within each mask.
        feat_map: [C=6, H, W]         the instance features of the entire image
        gt_masks: [num_mask, H, W]  num_mask boolean masks
        计算每个掩码内的平均实例特征。
        feat_map：[C=6, H, W]整张图像的实例特征
        gt_masks: [num_mask, H, W] num_mask 布尔掩码
        """
        num_mask, H, W = gt_masks.shape

        # expand feat and masks for batch processing 扩展批量处理的功能和掩码
        feat_expanded = feat_map.unsqueeze(0).expand(num_mask, *feat_map.shape)  # [num_mask, C, H, W]
        masks_expanded = gt_masks.unsqueeze(1).expand(-1, feat_map.shape[0], -1, -1)  # [num_mask, C, H, W]
        if image_mask is not None:  # image level mask 图像级掩模 (alpha) 
            image_mask_expanded = image_mask.unsqueeze(0).expand(num_mask, feat_map.shape[0], -1, -1)

        # average features within each mask 每个掩模内的平均特征
        if image_mask is not None:
            masked_feats = feat_expanded * masks_expanded.float() * image_mask_expanded.float()
            mask_counts = (masks_expanded * image_mask_expanded.float()).sum(dim=(2, 3))
        else:
            # masked_feats = feat_expanded * masks_expanded.float()  # [num_mask, C, H, W] may cause OOM
            masked_feats = self.ele_multip_in_chunks(feat_expanded, masks_expanded, chunk_size=5)   # in chuck to avoid OOM  # 按块处理以避免内存溢出
            mask_counts = masks_expanded.sum(dim=(2, 3))  # [num_mask, C]

        # the number of pixels within each mask 每个掩模内的像素数
        mask_counts = mask_counts.clamp(min=1)

        # the mean features of each mask 每个掩模的平均特征
        sum_per_channel = masked_feats.sum(dim=[2, 3])
        mean_per_channel = sum_per_channel / mask_counts    # [num_mask, C]

        if not return_var: # default
            return mean_per_channel   # [num_mask, C]
        # else:
        #     # calculate variance
        #     # masked_for_variance = torch.where(masks_expanded.bool(), masked_feats - mean_per_channel.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(masked_feats))
        #     masked_for_variance = process_in_chunks(masks_expanded, masked_feats, mean_per_channel, chunk_size=5) # in chunk to avoid OOM

        #     # variance_per_channel = (masked_for_variance ** 2).sum(dim=[2, 3]) / mask_counts    # [num_mask, 6]
        #     variance_per_channel = calculate_variance_in_chunks(masked_for_variance, mask_counts, chunk_size=5)   # in chuck to avoid OOM

        #     # mean and variance
        #     mean = mean_per_channel.mean(dim=1)          # [num_mask]，not used
        #     variance = variance_per_channel.mean(dim=1)  # [num_mask]

        #     return mean_per_channel, variance, mask_counts[:, 0]   # [num_mask, C], [num_mask], [num_mask]

    def cohesion_loss(self, feat_map, gt_mask, feat_mean_stack):
        """intra-mask smoothing loss. Eq.(1) in the paper
        Constrain the feature of each pixel within the mask to be close to the mean feature of that mask.
        掩模内平滑损失。论文中的式(1)将掩模内每个像素的特征限制为接近该掩模的平均特征。
        """
        N, H, W = gt_mask.shape
        C = feat_map.shape[0]
        # expand feat_map [6, H, W] to [N, 6, H, W]
        feat_map_expanded = feat_map.unsqueeze(0).expand(N, C, H, W)
        # expand mean feat [N, 6] to [N, 6, H, W]
        feat_mean_stack_expanded = feat_mean_stack.unsqueeze(-1).unsqueeze(-1).expand(N, C, H, W)
        
        # fature distance     计算特征距离   masked_feat：通过将扩展后的特征图与掩码相乘，得到的结果只保留了掩码为 1 的位置（即特定实例的特征），其余位置将为 0。最终形状仍为 [𝑁,6,𝐻,𝑊][N,6,H,W]。
        masked_feat = feat_map_expanded * gt_mask.unsqueeze(1)           # [N, 6, H, W]  # [N, 6, H, W]*[N,1,H,W]，只保留 gt_mask 为 1 的位置
        dist = (masked_feat - feat_mean_stack_expanded).norm(p=2, dim=1) # [N, H, W]     # 计算每个像素的特征距离，结果为 [N, H, W]   .norm(p=2, dim=1)：这个方法计算在特征维度（即通道维度，大小为6）上的 L2 范数（欧几里得距离），得到的结果是一个形状为 [𝑁,𝐻,𝑊[N,H,W] 的张量，表示每个样本的每个像素与其对应平均特征之间的距离。
        
        # per mask feature distance (loss) 每个掩模特征距离（损失）
        masked_dist = dist * gt_mask    # [N, H, W] # [N, H, W]，只保留 gt_mask 为 1 的位置
        loss_per_mask = masked_dist.sum(dim=[1, 2]) / gt_mask.sum(dim=[1, 2]).clamp(min=1) # 对每个 mask 的距离求和并归一化

        return loss_per_mask.mean()


    def separation_loss(self,feat_mean_stack):
        """ inter-mask contrastive loss Eq.(2) in the paper
        Constrain the instance features within different masks to be as far apart as possible.
        论文中的掩模间对比损失方程（2）将不同蒙版内的实例特征限制为尽可能远离
        """
        N, _ = feat_mean_stack.shape  # 获取特征均值堆栈的数量 N

        # expand feat_mean_stack[N, 6] to [N, N, C] 将 feat_mean_stack 从 [N, 6] 扩展到 [N, N, C]
        feat_expanded = feat_mean_stack.unsqueeze(1).expand(-1, N, -1)    # [N, N, C]
        feat_transposed = feat_mean_stack.unsqueeze(0).expand(N, -1, -1)  # [N, N, C]
        
        # distance  计算特征之间的平方距离
        diff_squared = (feat_expanded - feat_transposed).pow(2).sum(2) # [N, N]
        
        # Calculate the inverse of the distance to enhance discrimination 计算距离的倒数以增强区分性
        epsilon = 1     # 1e-6  # 1e-6，避免除以零的常数
        inverse_distance = 1.0 / (diff_squared + epsilon)  # [N, N]
        # Exclude diagonal elements (distance from itself) and calculate the mean inverse distance
        # 排除对角元素（自身的距离），并计算平均倒数距离
        mask = torch.eye(N, device=feat_mean_stack.device).bool() # 创建单位矩阵掩码
        inverse_distance.masked_fill_(mask, 0)                    # 将对角线元素设为 0，避免对自身距离的影响  

        # note: weight   权重计算
        # sorted by distance  根据距离排序
        sorted_indices = inverse_distance.argsort().argsort()   # [N, N]，对距离进行排序 
        loss_weight = (sorted_indices.float() / (N - 1)) * (1.0 - 0.1) + 0.1    # scale to 0.1 - 1.0, [N, N]   # 将权重缩放到 0.1 - 1.0 的范围
        # small weight
        # if iteration > 35_000:   # 如果迭代次数大于 35,000
            # loss_weight[loss_weight < 0.9] = 0.1   # 将小于 0.9 的权重设为 0.1
        # inverse_distance *= loss_weight     # [N, N] 应用权重

        # final loss
        loss = inverse_distance.sum() / (N * (N - 1))

        return loss

    def _mask_loss_fn(self, render_mask, gt_mask):
        
        if self.use_CEloss == 1:
            render_mask = torch.log(render_mask/ 2 + 0.5)
            # print("render_mask = ",render_mask)
            # render_mask = torch.log(render_mask)
            loss = self.criterion_nll (render_mask, gt_mask)
        elif self.use_CEloss == 2:
            celoss = nn.CrossEntropyLoss(ignore_index=-1)
            loss = celoss(render_mask, gt_mask-1)  
        elif self.use_CEloss == 21:
            celoss = nn.CrossEntropyLoss(reduction='mean')
            loss = celoss(render_mask, gt_mask)  
        elif self.use_CEloss == 7:
            celoss = nn.CrossEntropyLoss(ignore_index=0)
            loss = celoss(render_mask, gt_mask)  
        elif self.use_CEloss == 3:
            rendered_ins_feat = render_mask[0,:,:,:]
            mask_bool = gt_mask[0,:,:,:]
            # (0) compute the average instance features within each mask. [num_mask, 6]
            # rendered_silhouette 即透明度 需要render做的
            feat_mean_stack = self.mask_feature_mean(rendered_ins_feat, mask_bool) #, image_mask=rendered_silhouette)
            # (1) intra-mask smoothing loss. Eq.(1) in the paper
            loss_cohesion = self.cohesion_loss(rendered_ins_feat, mask_bool, feat_mean_stack)
            # (2) inter-mask contrastive loss Eq.(2) in the paper
            loss_separation = self.separation_loss(feat_mean_stack)
            # total loss, opt.loss_weight: 0.1
            loss = loss_separation + 0.1 * loss_cohesion # opt.loss_weight
        elif self.use_CEloss == 0:
            render_mask = (render_mask.permute(0, 3, 1, 2)/2 + 0.5) * (self.num_classes-1)
            gt_mask =gt_mask.permute(0, 3, 1, 2)
            loss = 0.8 * l1_loss(render_mask, gt_mask) + 0.2 * (1.0 - ssim(render_mask, gt_mask))
        elif self.use_CEloss == 4: # 无用 [-2 , 2] 忘记先归一化了
            render_mask = render_mask.permute(0, 3, 1, 2) * (self.num_classes-1)
            gt_mask =gt_mask.permute(0, 3, 1, 2)
            loss = 0.8 * l1_loss(render_mask, gt_mask) + 0.2 * (1.0 - ssim(render_mask, gt_mask))
        elif self.use_CEloss == 5: # 用render的language
            render_mask = render_mask.permute(0, 3, 1, 2) * (self.num_classes-1)
            gt_mask =gt_mask.permute(0, 3, 1, 2)
            loss = self._mask_ce_loss_fn(render_mask, gt_mask)

        return loss

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
        right_action, left_action = torch.split(action, split_size_or_sections=8, dim=1)
        # print("self.field_type=",self.field_type)
        # if self.cfg.method.field_type == 'bimanual_LF':
        data['right_action'] = right_action
        data['left_action'] = left_action
        """ # right_action, left_action = action.chunk(2, dim=2) # agent写法
        # print("\033[0;31;40mactionr in neural_rendering.py\033[0m",right_action)
        # print("\033[0;31;40mactionl in neural_rendering.py\033[0m",left_action)
        # print("\033[0;31;40maction in neural_rendering.py\033[0m",action)
        # tensor([[ 
        #   1.8244e-01, -1.2036e-01,  7.7095e-01, 6.9350e-05,  1.0000e+00,  2.3808e-04, -1.1926e-03,  0.0000e+00, 
        #   2.8597e-01,  3.3652e-01,  7.7093e-01, -9.3490e-05, 1.0000e+00,  7.0034e-04, 1.0216e-03,  0.0000e+00]], device='cuda:0')
        # print(action.shape) # torch.Size([1, 16]) """
        data['step'] = step
        # if self.field_type == 'LF':

        if self.mask_gen == 'pre':
            data['mask_view'] = {}
            data['mask_view']['intr'] = gt_mask_camera_intrinsic # [indx] 在render（）中确认了
            data['mask_view']['extr'] = gt_mask_camera_extrinsic #  [indx]         
            if data['mask_view']['intr'] is not None:
                data_novel = self.get_novel_calib(data['mask_view'], True)
                data['mask_view'].update(data_novel) # 更新数据
        elif self.mask_gen == 'nonerf':
            data['mask'] = mask

        # novel pose
        data['novel_view'] = {}
        data['intr'] = tgt_intrinsic # nerf 相机内参通常包括焦距、主点坐标等，用于将3D坐标转换为2D图像坐标。
        data['extr'] = tgt_pose     # 相机外参通常包括旋转矩阵和平移向量，用于将世界坐标转换为相机坐标。相机外参定义了相机在世界坐标系中的位置和方向。
        data['xyz'] = einops.rearrange(pcd, 'b c h w -> b (h w) c') # bs,256*256,(xyz)
        #  einops.rearrange 函数重新排列点云数据 pcd   'b c h w -> b (h w) c' 是一个重组操作的模式，它将输入张量
        # 从四维格式（可能表示 [batch_size, channels, height, width]）转换为三维格式，  其中高度和宽度被合并为一个维度。这通常用于将3D点云数据从图像格式转换为列表格式，

        # use extrinsic pose to generate gaussain parameters
        # 使用 extrinsic pose(外部姿态) 生成 Gaussain 参数
        if data['intr'] is not None:
            data_novel = self.get_novel_calib(data, True)
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
            elif self.field_type =='LF':                         # 看做leader follower学习  
                if self.mask_gen == 'pre':          # mask自己计算
                    data['right_next'] = {
                        'extr': next_tgt_pose,
                        'intr': next_tgt_intrinsic,
                        'novel_view': {},
                    }
                    if data['right_next']['intr'] is not None:
                        # 为了生成左臂的mask
                        data_novel = self.get_novel_calib(data['right_next'], False)
                        data['right_next']['novel_view'].update(data_novel)
                        # mask_view用来训练的mask参数，novel_view用来生成 分割用的mask
                        data['right_next']['mask_view'] = {}
                        # test 为了mask训练
                        data['right_next']['mask_view']['intr'] = next_gt_mask_camera_intrinsic # [indx] 
                        data['right_next']['mask_view']['extr'] = next_gt_mask_camera_extrinsic # [indx]   
                        if data['right_next']['mask_view']['intr'] is not None:
                            data_novel_test = self.get_novel_calib(data['right_next']['mask_view'], True)
                            data['right_next']['mask_view'].update(data_novel_test) # 更新数据

                    data['left_next'] = {
                        'extr': next_tgt_pose,
                        'intr': next_tgt_intrinsic,
                        'novel_view': {},
                    }
                    if data['left_next']['intr'] is not None:
                        data_novel = self.get_novel_calib(data['left_next'], False)
                        data['left_next']['novel_view'].update(data_novel)
                        data['left_next']['mask_view'] = {}
                        # data['mask_view']['intr'] = gt_mask_camera_intrinsic 
                        # data['mask_view']['extr'] = gt_mask_camera_extrinsic         
                        # if data['mask_view']['intr'] is not None:
                        #     data_novel = self.get_novel_calib(data)
                        # data['left_next']['mask_view'].update(data_novel) # 更新数据
                        data['left_next']['mask_view']['intr'] = next_gt_mask_camera_intrinsic # [indx] 
                        data['left_next']['mask_view']['extr'] = next_gt_mask_camera_extrinsic # [indx]   
                        if data['left_next']['mask_view']['intr'] is not None:
                            # print("left_next intr is not none")
                            data_novel_test = self.get_novel_calib(data['left_next']['mask_view'], True)
                            data['left_next']['mask_view'].update(data_novel_test) # 更新数据
                elif self.mask_gen == 'gt':                       # gt mask
                    data['right_next'] = {
                        'extr': next_tgt_pose,
                        'intr': next_tgt_intrinsic,
                        'novel_view': {},
                    }
                    if data['right_next']['intr'] is not None:
                        data_novel = self.get_novel_calib(data['right_next'],False)
                        data['right_next']['novel_view'].update(data_novel)
                    data['left_next'] = {
                        'extr': next_tgt_pose,
                        'intr': next_tgt_intrinsic,
                        'novel_view': {},
                    }
                    if data['left_next']['intr'] is not None:
                        data_novel = self.get_novel_calib(data['left_next'], False)
                        data['left_next']['novel_view'].update(data_novel)
                    # print("encode next gt (无需mask)")
                elif self.mask_gen == 'nonerf':          # 使用6个相机的参数
                    data['right_next'] = {
                        'extr': next_tgt_pose,
                        'intr': next_tgt_intrinsic,
                        'novel_view': {},
                    }
                    if data['right_next']['intr'] is not None:
                        # 为了生成左臂的mask
                        data_novel = self.get_novel_calib(data['right_next'], False)
                        data['right_next']['novel_view'].update(data_novel)
                        # mask_view用来训练的mask参数，novel_view用来生成 分割用的mask
                        # 准备通过该光栅化代码实现
                        # data['right_next']['mask_view'] = {}
                        # data['right_next']['mask_view'].update(data_novel) # 更新数据 

                    data['left_next'] = {
                        'extr': next_tgt_pose,
                        'intr': next_tgt_intrinsic,
                        'novel_view': {},
                    }
                    if data['left_next']['intr'] is not None:
                        data_novel = self.get_novel_calib(data['left_next'], False)
                        data['left_next']['novel_view'].update(data_novel)
                        # data['left_next']['mask_view'] = {}
                        # # #改了光栅化代码render_mask 就可以注释
                        # data['left_next']['mask_view'].update(data_novel) # 更新数据

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
            if mask: # mask图片需要缩放到128 128
                intr = intr / 2

            extr = data['extr'][i, ...].cpu().numpy()
            extr = np.linalg.inv(extr)  # the saved extrinsic is actually cam2world matrix, so turn it to world2cam matrix 保存的extrinsic实际上是cam2world矩阵，所以将其转为world2cam矩阵

            width, height = self.W, self.H
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
        indx = random.randint(0, 1) # 0 front or 1 overhead
        if self.mask_gen == 'nonerf':
            data = self.encode_data(
                rgb=rgb, depth=depth, pcd=pcd, focal=focal, c=c, lang_goal=None, tgt_pose=gt_pose, tgt_intrinsic=gt_intrinsic,
                dec_fts=dec_fts, lang=language, next_tgt_pose=next_gt_pose, next_tgt_intrinsic=next_gt_intrinsic, 
                action=action, step=step, gt_mask=gt_mask, gt_mask_camera_extrinsic=gt_mask_camera_extrinsic, gt_mask_camera_intrinsic=gt_mask_camera_intrinsic,  
                next_gt_mask=next_gt_mask, indx = indx, 
            )
        else:
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
        render_mask_gtrgb = None
        render_mask_novel = None
        next_render_mask_right = None
        next_render_mask = None
        next_render_rgb_right = None
        next_left_mask_gen, exclude_left_mask = None, None
        gt_mask_vis,next_gt_mask_vis = None, None

        # create gt feature from foundation models 从基础模型创建 gt特征
        # 用于暂时禁用PyTorch中的梯度计算
        with torch.no_grad():
            # 提取基础模型特征 # Diffusion or dinov2
            gt_embed = self.extract_foundation_model_feature(gt_rgb, lang_goal)
        
        # change the mask
        if self.mask_gen == 'pre':
            
            # print("1 origin mask",gt_mask.shape)
            gt_mask = F.interpolate(gt_mask, size=(128, 128), mode='bilinear', align_corners=False)
            # print("2 mask",gt_mask.shape)
            gt_mask =gt_mask.permute(0,2,3,1) # [1, 1, 256, 256] ->[1, 256, 256, 1]
            if (self.use_CEloss >=1 and self.use_CEloss <= 3) or (self.use_CEloss == 7) or (self.use_CEloss == 21):
                gt_mask_label = self.mask_label_onehot(gt_mask) # 1 128 128 [target 0 1 2]  这里不改，算的时候-1即可
                """ # test_onehot = self.mask_onehot(gt_mask) # [1 128 128 3]
                # device = gt_mask.device  # 获取 next_gt_rgb 的设备 print("test.shape",test.shape,test)  
                # test_onehot = test_onehot.to(device).permute(0, 3, 1, 2)  # [1 128 128 3] -> [1 3 128 128]
                # loss_test = self._mask_loss_fn(test_onehot,gt_mask_label)   
                # print("loss_test = ",loss_test)
                # test_onehot = test_onehot.repeat(1, 1, 1, 3)
                    # gt_mask_label_test = self.mask_label_onehot(gt_mask)
                    # test_onehot1 = self.one_hot_encode(gt_mask_label_test,3)
                    # loss_test1 = self.CrossEntropyLoss(test_onehot1,gt_mask_label_test)  
                    # print("loss_test1 = ",loss_test1)         
                # print("L1gt_mask_label =[000 111 222]",gt_mask_label.shape, gt_mask_label) """
            elif self.use_CEloss == 0:
                # print("3 mask",gt_mask.shape)
                gt_mask_label = self.mask_label(gt_mask) # [1 128 128 3]   
            elif self.use_CEloss == 4:
                gt_mask_label = self.mask_label(gt_mask) 
                gt_mask = gt_mask_label
            elif self.use_CEloss == 5: # 用render的language
                gt_mask_label = self.mask_label(gt_mask)    # [1 128 128 1] -> [1 128 128 3] [000 / 111 / 222]
            # 6 only rgb
            if self.use_dynamic_field:
                next_gt_mask = F.interpolate(next_gt_mask, size=(128, 128), mode='bilinear', align_corners=False)
                next_gt_mask =next_gt_mask.permute(0,2,3,1) # [1, 1, 256, 256] ->[1, 256, 256, 1]
                if (self.use_CEloss >=1 and self.use_CEloss <= 3) or (self.use_CEloss == 7) or (self.use_CEloss == 21):
                    next_gt_mask_label = self.mask_label_onehot(next_gt_mask) # 1 128 128 [target 0 1 2]   7:这里不改，算loss时-1[-1,0,1]即可
                elif self.use_CEloss == 0: #L1
                    next_gt_mask_label = self.mask_label(next_gt_mask) # [1 128 128 3]   
                elif self.use_CEloss == 4:
                    next_gt_mask_label = self.mask_label(next_gt_mask) 
                    next_gt_mask = next_gt_mask_label
                elif self.use_CEloss == 5: # 用render的language
                    next_gt_mask_label = self.mask_label(next_gt_mask)    # [1 128 128 1] -> [1 128 128 3] [000 / 111 / 222]

        # if gt_rgb is not None:
        if training:
            # Gaussian Generator 高斯生成器 gs regress (g) 应该也不用改
            data = self.gs_model(data) # GeneralizableGSEmbedNet(cfg, with_gs_render=True)

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
            if self.mask_gen == 'nonerf':
                gt_rgb = gt_rgb.permute(0,2,3,1) # [1, 3, 256, 256] ->[1, 256, 256, 3]
                gt_mask =gt_mask.permute(0,2,3,1) # [1, 1, 256, 256] ->[1, 256, 256, 1]
                # gt_mask_label = self.mask_label(gt_mask) # 100 010  001
                # # print("gt_mask_label = ",gt_mask_label.shape,gt_mask_label)
                # # 1 当前场景的mask 训练  loss_dyna_mask_novel
                # data =self.pts2render_mask_gen(data, bg_color=self.bg_mask)
                # render_mask_novel = data['novel_view']['mask_gen'] # .permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]                           
                # loss_dyna_mask_novel = self.CrossEntropyLoss(render_mask_novel, gt_mask_label) # mask现阶段的
                # render_mask_novel = render_mask_novel.permute(0, 2, 3, 1)
                # loss_dyna_mask = loss_dyna_mask_novel
                # lambda_mask = 0 # 1  if step >= 40000 else 0
                # loss += loss_dyna_mask * lambda_mask # * 0.001
                
                # print("render_novel.shape",render_novel.shape)
                # print("gt_rgb.shape",gt_rgb.shape)
                if self.use_dynamic_field:
                    next_gt_rgb = next_gt_rgb.permute(0,2,3,1)
                    next_gt_mask = next_gt_mask.permute(0,2,3,1)
            elif self.mask_gen == 'pre':
                # 1 当前场景的mask 训练  loss_dyna_mask_novel
                data =self.pts2render_mask(data, bg_color=self.bg_mask)
                # print("1 gt_mask_label = ",gt_mask_label.shape, gt_mask_label)              # [1 128 128]
                # print("2 render_mask_novel = ",render_mask_novel.shape, render_mask_novel)  # [1 3 128 128]
                if self.use_CEloss==1 or self.use_CEloss == 7 or self.use_CEloss == 21:
                    render_mask_novel = data['novel_view']['mask_pred'] # 1 3 128 128 
                    loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask_label) #gt_mask) # mask现阶段的 _mask_loss_fn
                    render_mask_novel = render_mask_novel.permute(0, 2, 3, 1)
                    # # print("render_mask_novel = ",render_mask_novel.shape)
                    # # next_render_mask_right = self.vis_labels(render_mask_novel) # debug 的时候用一下
                    # # render_mask_novel = self.generate_final_class_labels(render_mask_novel)      
                elif self.use_CEloss==2:
                    render_mask_novel = data['novel_view']['mask_pred'] # 1 3 128 128 
                    render_mask_novel = render_mask_novel[:, [0, 1], :, :]
                    loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask_label) #gt_mask) # mask现阶段的 _mask_loss_fn
                    render_mask_novel = render_mask_novel.permute(0, 2, 3, 1)
                elif self.use_CEloss == 0:
                    render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1)
                    loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask_label)  # gt_mask) # mask现阶段的 _mask_loss_fn
                # loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask_label)  # gt_mask) # mask现阶段的 _mask_loss_fn
                # print("loss_dyna_mask_novel = ",loss_dyna_mask_novel)    
                elif self.use_CEloss == 3:   # open
                    render_mask_novel = data['novel_view']['mask_pred'] # 1 3 128 128 
                    one_hot = F.one_hot(gt_mask_label.type(torch.int64), num_classes=3) #int(instance_num.item() + 1))
                    gt_mask_label = one_hot.permute(0,3, 1, 2)
                    loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask_label) #gt_mask) # mask现阶段的 _mask_loss_fn
                    render_mask_novel = render_mask_novel.permute(0, 2, 3, 1)    
                elif self.use_CEloss == 4:
                    # data =self.pts2render_mask(data, bg_color=self.bg_mask)
                    # render_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1)
                    render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1) # 1 3 128 128 
                    loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask_label) #gt_mask) # mask现阶段的 _mask_loss_fn
                    render_mask_novel = render_mask_novel.permute(0, 2, 3, 1)                    
                elif self.use_CEloss == 5:
                    data =self.pts2render1(data, bg_color=self.bg_color)
                    render_mask_novel = data['novel_view']['embed_pred'].permute(0, 2, 3, 1) 
                    loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask_label)  
       

                if not self.use_dynamic_field and self.use_CEloss != 6:
                    loss_dyna_mask = loss_dyna_mask_novel
                    lambda_mask =1   if step >= 1000 else 0
                    loss += loss_dyna_mask * lambda_mask # * 0.001
                elif self.use_CEloss != 6:
                    # loss_dyna_mask = loss_dyna_mask_novel * (1 - self.cfg.lambda_next_loss_mask)
                    lambda_mask = self.cfg.lambda_mask if step >= self.cfg.mask_warm_up else 0. # 0.4 2000
                    loss += loss_dyna_mask_novel * lambda_mask # * 0.001


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
            elif self.field_type == 'LF':    # Leader Follower condition
                if self.mask_gen == 'gt':   # mask现成的gt
                    if self.use_dynamic_field and next_gt_rgb is not None:
                        # 左手的点云
                        # time1 = time.perf_counter()
                        # time_step1 = time2 - time1
                        # print(f"1 ### time1 = {time1}")   
                        mask_3d, next_mask_3d = self.createby_gt_mask(data=data, gt_mask=gt_mask, 
                            gt_mask_camera_extrinsic=gt_mask_camera_extrinsic, gt_mask_camera_intrinsic=gt_mask_camera_intrinsic,  
                            next_gt_mask=next_gt_mask,gt_maskdepth=gt_maskdepth, next_gt_maskdepth=next_gt_maskdepth)
                        # time2 = time.perf_counter()
                        # time_step1 = time2 - time1
                        # print(f"2 ### time3 = {time2} step1 = {time_step1:.2f}s 凸包") # 0.82s -> 0.6 (计算3d点改用torch方法)

                        # 投影到二维 (二维点)
                        projected_points = project_3d_to_2d(next_mask_3d, next_gt_intrinsic)
                        
                        # time3 = time.perf_counter()
                        # time_step2 = time3 - time2
                        # print(f"3 ### time3 = {time3} step1 = {time_step2:.2f}s 凸包->2D") # 1.05/1.57s      

                        # 创建二维掩码 （二维的凸包） 其实可以不用计算？
                        # mask_shape = (256, 256)  # 假设的掩码大小 用256,256更合适 然后再缩小
                        mask_shape = (128, 128)
                        exclude_left_mask = create_2d_mask_from_convex_hull(projected_points, mask_shape)
                        exclude_left_mask = exclude_left_mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3) # [1,256,256,3]
                        # exclude_left_mask = exclude_left_mask.permute(0, 3, 1, 2) # [1,3,256,256]
                        # exclude_left_mask = F.interpolate(exclude_left_mask, size=(128, 128), mode='bilinear', align_corners=False)
                        # exclude_left_mask = exclude_left_mask.permute(0, 2, 3, 1) # [1,128,128,3]

                        # time4 = time.perf_counter()
                        # time_step3 = time4 - time3
                        # print(f"4 ### time4 = {time4} step3 = {time_step3:.2f}s 2D->2D mask(求凸包)")   

                        # print(f"exclude_left_mask.shape = {exclude_left_mask.shape}\n {exclude_left_mask}") # [1,256,256,3]
                        device = next_gt_rgb.device  # 获取 next_gt_rgb 的设备
                        # 确保 exclude_left_mask 在同一个设备上
                        exclude_left_mask = exclude_left_mask.to(device)
                        next_render_mask = exclude_left_mask
                        result_right_image = next_gt_rgb * exclude_left_mask
                        # print(f"next_gt_rgb = {next_gt_rgb.shape} {next_gt_rgb}") # [1,256,256,3]
                        render_mask_novel = result_right_image # 看看能不能可视化
                        # print(f"render_mask_novel.shape = {render_mask_novel.shape}\n {render_mask_novel}") # [1,256,256,3]
                        # time5 = time.perf_counter()
                        # time_step4 = time5 - time4
                        # print(f"5 ### time5 = {time5} step4 = {time_step4:.2f}s 匹配格式")                             

                        # 2 GRB total(Follower)  先对left预测（最后的结果） loss_dyna_follower  左臂 RGB Loss   
                        # 也可以说是双手结果
                        if ('xyz_maps' in data['left_next']):
                            data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                            next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]
                            # print("4 next_gt_mask.shape = ",next_gt_mask.shape, next_render_novel.shape) # torch.Size([1, 128, 128, 3]) 
                            loss_dyna_follower = l2_loss(next_render_novel, next_gt_rgb) # 双臂结果预测
                        
                        #  4 RGB loss_dyna_leader leader  利用前面得到的mask删去左臂
                        if ('xyz_maps' in data['right_next']):
                            # with torch.no_grad():  原来为了有的Loss没计算报无回传错误而写的
                            data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                            next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                            next_render_novel_mask = next_render_rgb_right * exclude_left_mask  # 原来用错了...  next_gt_rgb -> next_render_rgb_right
                            loss_dyna_leader = l2_loss(next_render_novel_mask, result_right_image)
                            # print('loss_dyna_leader = ', loss_dyna_leader)

                        # RGB pre = leader( right ) + follower
                        loss_LF = loss_dyna_leader * self.cfg.lambda_dyna_leader + loss_dyna_follower * (1-self.cfg.lambda_dyna_leader)

                        loss_dyna_mask = torch.tensor(0.) # 为了在那里输出
                        loss_reg = torch.tensor(0.) 
                        loss_dyna = loss_LF    # * (1-self.cfg.lambda_mask) + loss_dyna_mask * self.cfg.lambda_mask 
                        # print('loss_dyna = ', loss_dyna,loss_LF,loss_dyna_mask)
                        # 预热步数（3000步以后算上了）
                        lambda_dyna = self.cfg.lambda_dyna if step >= self.cfg.next_mlp.warm_up else 0.                    
                        
                        # Step 3 Loss(LGeo? + L embed/L sem + L dyna) = loss_rgb + self.cfg.lambda_embed * loss_embed + lambda_dyna * loss_dyna
                        loss += lambda_dyna * loss_dyna
                    else:
                        loss_dyna = torch.tensor(0.)
                        loss_reg = torch.tensor(0.)      
                        loss_LF = torch.tensor(0.)
                        loss_dyna_mask = torch.tensor(0.)
                elif self.mask_gen == 'pre':   # 需要自己训练mask  loss还有问题（mask 和mask label）
                    if self.use_dynamic_field and (next_gt_rgb is not None and gt_mask is not None):
                        # 2 GRB total(Follower)  先对left预测（最后的结果） loss_dyna_follower  左臂 RGB Loss   
                        if ('xyz_maps' in data['left_next']):
                            data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                            next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]
                            # print("4 next_gt_mask.shape = ",next_gt_mask.shape, next_render_novel.shape) # torch.Size([1, 128, 128, 3]) 
                            loss_dyna_follower = l2_loss(next_render_novel, next_gt_rgb) # 双臂结果预测

                        # -------------------------------------------------------------------------------------------------------------------
                        # 3 next mask train (pre - left(mask) ) next_loss_dyna_mask_left  左臂 Mask Loss
                        # data['left_next'] =self.pts2render_mask(data['left_next'], bg_color=self.bg_mask)
                        # next_render_mask = data['left_next']['novel_view']['mask_pred'] # .permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]
                        # next_loss_dyna_mask_left = self.CrossEntropyLoss(next_render_mask, next_gt_mask_label) # mask去左臂的mask
                        # loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask_label)
                        # # next_render_mask = next_render_mask.permute(0, 2, 3, 1)

                        data['left_next'] =self.pts2render_mask(data['left_next'], bg_color=self.bg_mask)
                        data['left_next'] =self.pts2render_mask_gen(data['left_next'], bg_color=self.bg_mask)
                        if self.use_CEloss==1 or self.use_CEloss == 7 or self.use_CEloss == 21:
                            next_render_mask = data['left_next']['novel_view']['mask_pred'] # 1 3 128 128 
                            next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask, next_gt_mask_label) #gt_mask) # mask现阶段的 _mask_loss_fn
                            next_render_mask = next_render_mask.permute(0, 2, 3, 1) 
                            # gen exclude
                            next_left_mask_gen = data['left_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1)  # 1 128 128 3   
                            exclude_left_mask = self.generate_final_class_labels(next_left_mask_gen)
                            exclude_left_mask = exclude_left_mask.unsqueeze(3).repeat(1, 1, 1, 3)
                        elif self.use_CEloss == 0:     # Ll1
                            next_render_mask = data['left_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1) # 1 128 128 3(rgb label mean-> 0 1 2)
                            next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask, next_gt_mask_label) 

                            next_left_mask_gen = data['left_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1) # 1 128 128 3
                            next_left_mask_gen = (next_left_mask_gen / 2 + 0.5  )*(self.num_classes-1)           # 1 128 128 3 [0,2]
                            exclude_left_mask = self.generate_final_class_labels_L1(next_left_mask_gen)

                        elif self.use_CEloss == 3:     # OpenGaussian ？ 2 3 要不要加注释？
                            next_render_mask = data['left_next']['novel_view']['mask_pred'] # 1 3 128 128 
                            one_hot = F.one_hot(next_gt_mask_label.type(torch.int64), num_classes=3) #int(instance_num.item() + 1)) # 可以改到前面去
                            next_gt_mask_label = one_hot.permute(0, 3, 1, 2)
                            next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask, next_gt_mask_label) #gt_mask) # mask现阶段的 _mask_loss_fn
                            next_render_mask = next_render_mask.permute(0, 2, 3, 1)  # 1 128 128 3  

                        elif self.use_CEloss == 4: # 无用 忘了归一化的L1
                            next_render_mask = data['novel_view']['mask_pred'].permute(0, 2, 3, 1) # 1 3 128 128 
                            next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask, next_gt_mask_label) 
                            next_render_mask = next_render_mask.permute(0, 2, 3, 1)                    
                        elif self.use_CEloss == 5:  # 无用 把mask作为language算
                            data =self.pts2render1(data, bg_color=self.bg_color)
                            next_render_mask = data['novel_view']['embed_pred'].permute(0, 2, 3, 1) 
                            next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask, next_gt_mask_label)          
                        elif self.use_CEloss==2: # 无用 ignore -1 [10 01] target: -1 0 1
                            next_render_mask = data['novel_view']['mask_pred'] # 1 3 128 128 
                            next_render_mask = next_render_mask[:, [0, 1], :, :]
                            next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask, next_gt_mask_label)
                            next_render_mask = next_render_mask.permute(0, 2, 3, 1)
                        # -------------------------------------------------------------------------------------------------------------------

                        # gen mask and exclude 改到前面的if中了
                            # data['left_next'] =self.pts2render_mask_gen(data['left_next'], bg_color=self.bg_mask)
                            # next_left_mask_gen = data['left_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1)                        
                            # # exclude_left_mask = (next_render_mask < left_min) | (next_render_mask > left_max) # 排除左臂标签 [1,128, 128, 3] [True,False]
                            # # exclude_left_mask = (next_left_mask_gen > 2.5) | (next_left_mask_gen < 1.5)
                            # exclude_left_mask = self.generate_final_class_labels(next_left_mask_gen)
                            # exclude_left_mask = exclude_left_mask.unsqueeze(3).repeat(1, 1, 1, 3)
                            # # background_color = torch.tensor(self.bg_color, dtype=torch.float32)  # 背景
                        result_right_image = next_gt_rgb * exclude_left_mask # + background_color * (~exclude_left_mask) # [1, 128, 128, 3] #

                        #  4 RGB loss_dyna_leader leader  利用前面得到的mask删去左臂
                        if ('xyz_maps' in data['right_next']):
                            data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                            next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                            next_render_novel_mask = next_render_rgb_right * exclude_left_mask  # 原来用错了...  next_gt_rgb -> next_render_rgb_right
                            loss_dyna_leader = l2_loss(next_render_novel_mask, result_right_image)
                        

                        # -------------------------------------------------------------------------------------------------------------------
                        # 5 Mask loss_dyna_mask_next_right 右臂mask训练（和now一样 无用）
                            # data['right_next'] =self.pts2render_mask(data['right_next'], bg_color=self.bg_mask)
                            # next_render_mask_right = data['right_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1)
                            # # next_loss_dyna_mask_right = self.CrossEntropyLoss(next_render_mask_right, next_gt_mask_label) #next_gt_mask)
                            # next_loss_dyna_mask_right = self._mask_loss_fn(next_render_mask_right, next_gt_mask_label)
                            # # next_render_mask_right = next_render_mask_right.permute(0, 2, 3, 1)

                        data['right_next'] =self.pts2render_mask(data['right_next'], bg_color=self.bg_mask)
                        if self.use_CEloss==1 or self.use_CEloss == 7 or self.use_CEloss == 21:
                            next_render_mask_right = data['right_next']['novel_view']['mask_pred'] # 1 3 128 128 
                            next_loss_dyna_mask_right = self._mask_loss_fn(next_render_mask_right, next_gt_mask_label) #gt_mask) # mask现阶段的 _mask_loss_fn
                            next_render_mask_right = next_render_mask_right.permute(0, 2, 3, 1) 
                        elif self.use_CEloss == 0:     # Ll1
                            next_render_mask_right = data['right_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1) # 1 128 128 3(rgb label mean-> 0 1 2)
                            next_loss_dyna_mask_right = self._mask_loss_fn(next_render_mask_right, next_gt_mask_label) 

                        elif self.use_CEloss == 3:     # OpenGaussian ？ 2 3 要不要加注释？
                            next_render_mask = data['right_next']['novel_view']['mask_pred'] # 1 3 128 128 
                            one_hot = F.one_hot(next_gt_mask_label.type(torch.int64), num_classes=3) #int(instance_num.item() + 1)) # 可以改到前面去
                            next_gt_mask_label = one_hot.permute(0, 3, 1, 2)
                            next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask, next_gt_mask_label) #gt_mask) # mask现阶段的 _mask_loss_fn
                            next_render_mask = next_render_mask.permute(0, 2, 3, 1)  # 1 128 128 3  

                        elif self.use_CEloss == 4: # 无用 忘了归一化的L1
                            next_render_mask = data['novel_view']['mask_pred'].permute(0, 2, 3, 1) # 1 3 128 128 
                            next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask, next_gt_mask_label) 
                            next_render_mask = next_render_mask.permute(0, 2, 3, 1)                    
                        elif self.use_CEloss == 5:  # 无用 把mask作为language算
                            data =self.pts2render1(data, bg_color=self.bg_color)
                            next_render_mask = data['novel_view']['embed_pred'].permute(0, 2, 3, 1) 
                            next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask, next_gt_mask_label)          
                        elif self.use_CEloss==2: # 无用 ignore -1 [10 01] target: -1 0 1
                            next_render_mask = data['novel_view']['mask_pred'] # 1 3 128 128 
                            next_render_mask = next_render_mask[:, [0, 1], :, :]
                            next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask, next_gt_mask_label)
                            next_render_mask = next_render_mask.permute(0, 2, 3, 1)
                        # -------------------------------------------------------------------------------------------------------------------

                        # next mask = right +left    
                        next_loss_dyna_mask = next_loss_dyna_mask_left * ( 1 - self.cfg.lambda_mask_right ) + next_loss_dyna_mask_right * self.cfg.lambda_mask_right  # 右臂权重小一点
                        
                        # MASK = now +pre
                        # loss_dyna_mask = loss_dyna_mask_novel * (1 - self.cfg.lambda_next_loss_mask)  + next_loss_dyna_mask * self.cfg.lambda_next_loss_mask
                        # loss_dyna_mask += next_loss_dyna_mask * self.cfg.lambda_next_loss_mask
                        loss_dyna_mask = next_loss_dyna_mask


                        # RGB pre = leader( right ) + follower
                        loss_LF = loss_dyna_leader * self.cfg.lambda_dyna_leader + loss_dyna_follower * (1-self.cfg.lambda_dyna_leader)
                        # print('loss_LF = ', loss_LF, loss_dyna_leader, loss_dyna_follower)

                        lambda_mask = self.cfg.lambda_mask if step >= self.cfg.next_mlp.warm_up + self.cfg.mask_warm_up else 0. # 5000
                        loss_dyna = loss_LF * (1 - lambda_mask) + loss_dyna_mask * lambda_mask

                        # print('loss_dyna = ', loss_dyna,loss_LF,loss_dyna_mask)
                        # 预热步数（3000步以后算上了）
                        lambda_dyna = self.cfg.lambda_dyna if step >= self.cfg.next_mlp.warm_up else 0.                    
                        
                        # Step 3 Loss(LGeo? + L embed/L sem + L dyna) = loss_rgb + self.cfg.lambda_embed * loss_embed + lambda_dyna * loss_dyna
                        loss += lambda_dyna * loss_dyna

                        loss_reg = torch.tensor(0.)
                    else:
                        loss_dyna = torch.tensor(0.)
                        loss_reg = torch.tensor(0.)      
                        loss_LF = torch.tensor(0.)
                        # loss_dyna_mask = torch.tensor(0.) 

                elif self.mask_gen == 'None':
                    if self.use_dynamic_field and (next_gt_rgb is not None):
                        # 2 GRB total(Follower)  先对left预测（最后的结果） loss_dyna_follower  左臂 RGB Loss   
                        # 也可以说是双手结果
                        if ('xyz_maps' in data['left_next']):
                            data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                            next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]
                            loss_dyna_follower = l2_loss(next_render_novel, next_gt_rgb) # 双臂结果预测

                        #  4 RGB loss_dyna_leader leader  利用前面得到的mask删去左臂
                        if ('xyz_maps' in data['right_next']):
                            # with torch.no_grad():  原来为了有的Loss没计算报无回传错误而写的
                            data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                            next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                            loss_dyna_leader = l2_loss(next_render_rgb_right, next_gt_rgb)

                        # RGB pre = leader( right ) + follower
                        loss_LF = loss_dyna_leader * self.cfg.lambda_dyna_leader + loss_dyna_follower * (1-self.cfg.lambda_dyna_leader)
                        loss_dyna = loss_LF 
                        # 预热步数（3000步以后算上了）
                        lambda_dyna = self.cfg.lambda_dyna if step >= self.cfg.next_mlp.warm_up else 0.                    
    
                        # Step 3 Loss(LGeo? + L embed/L sem + L dyna) = loss_rgb + self.cfg.lambda_embed * loss_embed + lambda_dyna * loss_dyna
                        loss += lambda_dyna * loss_dyna
                        loss_dyna_mask = torch.tensor(0.)
                        loss_reg = torch.tensor(0.)

                    else:
                        loss_dyna = torch.tensor(0.)
                        loss_reg = torch.tensor(0.)      
                        loss_LF = torch.tensor(0.)
                        loss_dyna_mask = torch.tensor(0.)
                elif not self.use_nerf_picture: 
                    if self.use_dynamic_field and (next_gt_rgb is not None and gt_mask is not None):
                        # 注意：这里的图片都是256*256的
                        # right_min, right_max, left_min, left_max = 53, 73, 94, 114

                        # # # [1,1,128,128] -> [1,128,128,1] -> [1,128,128,3]
                        # gt_mask1 = gt_mask.squeeze(-1) # [1,256,256]
                        # # 初始化独热编码张量 bg:[0,0,0]   right:[0,1,0]  left:[0,0,1]
                        # gt_mask_label = torch.zeros((*gt_mask1.shape, self.num_classes), dtype=torch.float32) # [1,256,256,3]
                        # # 将标签转换为独热编码
                        # bg_mask = (gt_mask1 < right_min) | ((gt_mask1 > right_max) & (gt_mask1 < left_min)) | (gt_mask1 > left_max)
                        # gt_mask_label[bg_mask] = torch.tensor([1, 0, 0], dtype=torch.float32)
                        # right_mask = (gt_mask1 > right_min - 1) & (gt_mask1 < right_max + 1)
                        # gt_mask_label[right_mask] = torch.tensor([0, 1, 0], dtype=torch.float32)  # 右手的独热编码                  
                        # left_mask = (gt_mask1 > left_min - 1) & (gt_mask1 < left_max + 1)
                        # gt_mask_label[left_mask] = torch.tensor([0, 0, 1], dtype=torch.float32) 

                        # gt_mask1 = next_gt_mask.squeeze(-1) # [1,256,256]
                        # # 初始化独热编码张量 bg:[0,0,0]   right:[0,1,0]  left:[0,0,1]
                        # next_gt_mask_label = torch.zeros((*gt_mask1.shape, self.num_classes), dtype=torch.float32) # [1,256,256,3]
                        # # 将标签转换为独热编码
                        # bg_mask = (gt_mask1 < right_min) | ((gt_mask1 > right_max) & (gt_mask1 < left_min)) | (gt_mask1 > left_max)
                        # next_gt_mask_label[bg_mask] = torch.tensor([1, 0, 0], dtype=torch.float32)
                        # right_mask = (gt_mask1 > right_min - 1) & (gt_mask1 < right_max + 1)
                        # next_gt_mask_label[right_mask] = torch.tensor([0, 1, 0], dtype=torch.float32)  # 右手的独热编码                  
                        # left_mask = (gt_mask1 > left_min - 1) & (gt_mask1 < left_max + 1)
                        # next_gt_mask_label[left_mask] = torch.tensor([0, 0, 1], dtype=torch.float32) 

                        # gt_mask_label = self.mask_onehot(gt_mask) # 100 010  001
                        # next_gt_mask_label = self.mask_onehot(next_gt_mask) # [1 256 256 3]
                        # device = gt_mask.device  # 获取 next_gt_rgb 的设备
                        # gt_mask_label = gt_mask_label.to(device)
                        # next_gt_mask_label = next_gt_mask_label.to(device)

                        next_gt_mask_label = self.mask_label(next_gt_mask)                        
                        print("next_gt_mask_label = ",next_gt_mask_label.shape,next_gt_mask_label)
                        # # 1 当前场景的mask 训练  loss_dyna_mask_novel
                        # data =self.pts2render_mask_gen(data, bg_color=self.bg_mask)
                        # render_mask_novel = data['novel_view']['mask_gen'] # .permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]                           
                        # # 忽略特定的标签（例如空白背景类）
                        # # CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
                        # # 接收两个参数：logit（模型输出的 logits）和 label（真实标签）
                        # # abel 被减去 1。这个操作是为了将背景类（通常 ID 为 0）转移到 -1，使得 CrossEntropyLoss 可以正确忽略这个类别。
                        # # crossentropy_loss = lambda logit, label: CrossEntropyLoss(logit, label-1)
                        # loss_dyna_mask_novel = self.CrossEntropyLoss(render_mask_novel, gt_mask_label) # mask现阶段的
                        # render_mask_novel = render_mask_novel.permute(0, 2, 3, 1)

                        # 2 GRB total(Follower)  先对left预测（最后的结果） loss_dyna_follower  左臂 RGB Loss   
                        # 也可以说是双手结果
                        if ('xyz_maps' in data['left_next']):
                            data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                            next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]
                            loss_dyna_follower = l2_loss(next_render_novel, next_gt_rgb) # 双臂结果预测

                        # 3 next mask train (pre - left(mask) ) next_loss_dyna_mask_left  左臂 Mask Loss
                        data['left_next'] =self.pts2render_mask_gen(data['left_next'], bg_color=self.bg_mask)
                        next_render_mask = data['left_next']['novel_view']['mask_gen'] # .permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]
                        # print('next_render_mask', next_render_mask.shape,next_gt_mask_label.shape)
                        next_loss_dyna_mask_left = self.CrossEntropyLoss(next_render_mask, next_gt_mask_label) # mask去左臂的mask
                        next_render_mask = next_render_mask.permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]

                        # gen mask and exclude
                        # data['left_next'] =self.pts2render_mask_gen(data['left_next'], bg_color=self.bg_color)
                        # next_left_mask_gen = data['left_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1)        

                        # exclude_left_mask = self.generate_final_class_labels(next_render_mask) # ?感觉有问题，用gt还是gen？
                        # [1,256,256,3]
                        # exclude_left_mask = self.generate_final_class_labels(next_gt_mask_label) # ?感觉有问题，用gt还是gen？ 用gt 因为这是对输出的计算
                        exclude_left_mask = self.generate_final_class_labels(next_render_mask) 
                        # print('exclude_left_mask', exclude_left_mask.shape) # [1 256 256]
                        # print("next_gt_rgb",next_gt_rgb.shape)              # [1 256 256 3]
                        exclude_left_mask = exclude_left_mask.unsqueeze(3).repeat(1, 1, 1, 3) # [1 256 256] -> [1 256 256 1] -> [1 256 256 3]
                        result_right_image = next_gt_rgb * exclude_left_mask # 反正背景都是False不用加

                        #  4 RGB loss_dyna_leader leader  利用前面得到的mask删去左臂
                        if ('xyz_maps' in data['right_next']):
                            data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                            next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                            next_render_novel_mask = next_render_rgb_right * exclude_left_mask  # 原来用错了...  next_gt_rgb -> next_render_rgb_right
                            loss_dyna_leader = l2_loss(next_render_novel_mask, result_right_image)
                        
                        # 5 Mask loss_dyna_mask_next_right 右臂mask训练（和now一样 无用）
                        data['right_next'] =self.pts2render_mask_gen(data['right_next'], bg_color=self.bg_mask)
                        next_render_mask_right = data['right_next']['novel_view']['mask_gen'] # .permute(0, 2, 3, 1)
                        # CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
                        next_loss_dyna_mask_right = self.CrossEntropyLoss(next_render_mask_right, next_gt_mask_label)
                        next_render_mask_right = next_render_mask_right.permute(0, 2, 3, 1) # [1,128, 128, 3]

                        # pre mask = right +left    
                        next_loss_dyna_mask = next_loss_dyna_mask_left * ( 1 - self.cfg.lambda_mask_right ) + next_loss_dyna_mask_right * self.cfg.lambda_mask_right  # 右臂权重小一点
                        
                        # MASK = now +pre
                        # loss_dyna_mask = loss_dyna_mask_novel * (1 - self.cfg.lambda_next_loss_mask)  + next_loss_dyna_mask * self.cfg.lambda_next_loss_mask
                        loss_dyna_mask = loss_dyna_mask_novel * (1 - self.cfg.lambda_next_loss_mask)  + next_loss_dyna_mask * self.cfg.lambda_next_loss_mask


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
                    else:
                        loss_dyna = torch.tensor(0.)
                        loss_reg = torch.tensor(0.)      
                        loss_LF = torch.tensor(0.)
                        # loss_dyna_mask = torch.tensor(0.) 

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
                if self.field_type == 'bimanual':
                    if self.use_dynamic_field and 'xyz_maps' in data['next']:
                        data['next'] = self.pts2render(data['next'], bg_color=self.bg_color)
                        next_render_novel = data['next']['novel_view']['img_pred'].permute(0, 2, 3, 1)
                else:
                    if self.mask_gen == 'gt':
                        if self.use_dynamic_field:
                            # start_time = time.perf_counter()
                            # print("#0 time1: ", start_time)
                            # 左手的点云
                            mask_3d, next_mask_3d = self.createby_gt_mask(data=data, 
                                gt_mask=gt_mask,next_gt_mask=next_gt_mask, 
                                gt_mask_camera_extrinsic=gt_mask_camera_extrinsic, gt_mask_camera_intrinsic=gt_mask_camera_intrinsic,  
                                gt_maskdepth=gt_maskdepth, next_gt_maskdepth=next_gt_maskdepth)
                            # start_time = time.perf_counter()
                            # print("#0 time2: ", start_time)                            
                            # 投影到二维
                            projected_points = project_3d_to_2d(next_mask_3d, next_gt_intrinsic)
                            # 创建二维掩码
                            mask_shape = (128,128) # (256, 256)  # 假设的掩码大小
                            # start_time = time.perf_counter()
                            # print("#0 time3: ", start_time)
                            exclude_left_mask = create_2d_mask_from_convex_hull(projected_points, mask_shape)
                            exclude_left_mask = exclude_left_mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3) # [1,256,256,3]
                            # exclude_left_mask = exclude_left_mask.permute(0, 3, 1, 2) # [1,3,256,256]
                            # exclude_left_mask = F.interpolate(exclude_left_mask, size=(128, 128), mode='bilinear', align_corners=False)
                            # exclude_left_mask = exclude_left_mask.permute(0, 2, 3, 1) # [1,128,128,3]
                            # final_time = time.perf_counter()
                            # print("#0 time4: ", final_time)
                            # exclude_left_mask = exclude_left_mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
                            if next_gt_rgb is not None:
                                device = next_gt_rgb.device  # 获取 next_gt_rgb 的设备
                                exclude_left_mask = exclude_left_mask.to(device)
                                next_render_mask = exclude_left_mask
                                result_right_image = next_gt_rgb * exclude_left_mask  
                                render_mask_novel = result_right_image

                            # 2 GRB total(Follower)  先对left预测（最后的结果） loss_dyna_follower  左臂 RGB Loss   
                            # 也可以说是双手结果
                            if ('xyz_maps' in data['left_next']):
                                data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                                next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]
                            
                            #  4 RGB loss_dyna_leader leader  利用前面得到的mask删去左臂
                            if ('xyz_maps' in data['right_next']):
                                # with torch.no_grad():  原来为了有的Loss没计算报无回传错误而写的
                                data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                                next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]                                
                    elif self.mask_gen == 'pre':   # LF + 自己 train mask   
                        data =self.pts2render_mask(data, bg_color=self.bg_mask)
                        if self.use_CEloss==1 or self.use_CEloss==3 or self.use_CEloss == 7 or self.use_CEloss == 21:
                            render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1)
                            ## !! render_mask_novel = render_novel * render_mask_gtrgb
                            render_mask_gtrgb = self.generate_final_class_labels(render_mask_novel).unsqueeze(3).repeat(1, 1, 1, 3)     # 1 128 128 3 [TTT/FFF]                        
                            render_mask_gtrgb = gt_rgb * render_mask_gtrgb
                            # vis render
                            render_mask_novel = self.vis_labels(render_mask_novel)
                            
                            gt_mask1 = self.mask_onehot(gt_mask) # 1 128 128 1[1,200] -> 1 128 128 3[100 010 001] 
                            gt_mask_vis =  self.vis_labels(gt_mask1)
                            # vis rgb render in mask camera 可视化mask相机的rgb  
                            data =self.pts2render_rgb(data, bg_color=self.bg_color)
                            ## !! 
                            next_render_mask = data['mask_view']['rgb_pred'].permute(0, 2, 3, 1)

                        if self.use_CEloss==2: # 未改
                            render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1)  # 1 128 128 3
                            render_mask_novel1 = render_mask_novel
                            print("render_mask_novel",render_mask_novel.shape) # [1 128 128 3]
                            render_mask_novel = self.generate_final_class_labels_ce1(render_mask_novel) # [1 128 128] 0/1
                            next_render_mask_right = self.vis_labels_ce1(render_mask_novel1)

                            # render_mask_novel = self.generate_final_class_labels(render_mask_novel1) # 1 128 128
                            render_mask_novel = render_mask_novel.unsqueeze(3).repeat(1, 1, 1, 3)   # 1 128 128 3
                            render_mask_gtrgb = gt_rgb * render_mask_novel
                            render_mask_novel = render_novel * render_mask_novel

                            # gt
                            gt_mask1 = self.mask_onehot(gt_mask)  # 1 128 128 3
                            # gt_mask1 =gt_mask1.permute(0, 3, 1, 2)
                            print("gt_mask1",gt_mask1.shape,gt_mask1) # 1 128 128 3 
                            next_render_rgb_right =  self.vis_labels(gt_mask1)
                        elif self.use_CEloss == 0: # l1
                            render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1) # 1 128 128 3
                            render_mask_novel = (render_mask_novel / 2 + 0.5)*(self.num_classes-1)  # [0 , 2]
                            ## !! 
                            next_render_mask_right = self.vis_labelsL1(render_mask_novel)           #
                            render_mask_novel = self.generate_final_class_labels_L1(render_mask_novel)

                            # vis gt left 
                            ## !! 
                            gt_mask_vis = self.vis_labelsL1(gt_mask_label)
                            ## 
                            next_render_rgb_right  = self.generate_final_class_labels_L1(gt_mask_label)
                            ## 
                            next_render_rgb_right = gt_rgb * next_render_rgb_right
                            
                            render_mask_gtrgb = gt_rgb * render_mask_novel
                            render_mask_novel = render_novel * render_mask_novel

                        elif self.use_CEloss == 4:
                            render_mask_novel = render_novel
                            render_mask_novel = render_mask_novel*(self.num_classes-1)
                            next_render_mask_right = self.vis_labelsL1(render_mask_novel) # debug 的时候用一下
                            # render_mask_novel = self.generate_final_class_labels(render_mask_novel)
                            # render_mask_novel = render_mask_novel.unsqueeze(3).repeat(1, 1, 1, 3)
                            render_mask_novel = self.generate_final_class_labels_L1(render_mask_novel)

                            # 可视化gt left 但是
                            next_render_mask = self.vis_labelsL1(gt_mask_label)
                            next_render_rgb_right  = self.generate_final_class_labels_L1(gt_mask_label)
                            next_render_rgb_right = gt_rgb * next_render_rgb_right
                            
                            render_mask_gtrgb = gt_rgb * render_mask_novel
                            print("render_mask_novel = ",render_mask_novel.shape,render_mask_novel)
                            render_mask_novel = render_novel * render_mask_novel                           
                        elif self.use_CEloss == 5:
                            data =self.pts2render1(data, bg_color=self.bg_color)
                            render_mask_novel = data['novel_view']['embed_pred'].permute(0, 2, 3, 1) 
                            print("1 render_mask_novel = ",render_mask_novel.shape, render_mask_novel)
                            render_mask_novel = render_mask_novel*(self.num_classes-1)
                            print("2 render_mask_novel = ",render_mask_novel.shape, render_mask_novel)
                            next_render_mask_right = self.vis_labelsL1(render_mask_novel) # debug 的时候用一下

                            render_mask_novel = self.generate_final_class_labels_L1(render_mask_novel)

                            # 可视化gt left 但是
                            next_render_mask = self.vis_labelsL1(gt_mask_label)
                            next_render_rgb_right  = self.generate_final_class_labels_L1(gt_mask_label)
                            next_render_rgb_right = gt_rgb * next_render_rgb_right
                            
                            render_mask_gtrgb = gt_rgb * render_mask_novel

                            render_mask_novel = render_novel * render_mask_novel    
                        elif self.use_CEloss == 6:
                            print("only rgb")                          

                        if self.use_dynamic_field:
                            # 2 next Left RGB     
                            if ('xyz_maps' in data['left_next']):
                                data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                                next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]
                            
                            # 3 next Left mask
                                # data['left_next'] =self.pts2render_mask(data['left_next'], bg_color=self.bg_mask)
                                # next_render_mask = data['left_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]
                                # next_render_mask = self.generate_final_class_labels(next_render_mask)
                                # next_render_mask = next_render_mask.unsqueeze(3).repeat(1, 1, 1, 3)
                                # next_render_mask = next_render_novel * next_render_mask
                            
                            data['left_next'] =self.pts2render_mask(data['left_next'], bg_color=self.bg_mask)
                            data['left_next'] =self.pts2render_mask_gen(data['left_next'], bg_color=self.bg_mask)
                            if self.use_CEloss==1 or self.use_CEloss == 7 or self.use_CEloss == 21:
                                next_render_mask = data['left_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1) 
                                next_render_mask = self.vis_labels(next_render_mask)
                                # gen exclude
                                next_left_mask_gen = data['left_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1)  # 1 128 128 3   
                                exclude_left_mask = self.generate_final_class_labels(next_left_mask_gen).unsqueeze(3).repeat(1, 1, 1, 3)
                                next_left_mask_gen = self.vis_labels(next_left_mask_gen)
                                next_gt_mask1 = self.mask_onehot(next_gt_mask) # 1 128 128 1[1,200] -> 1 128 128 3[100 010 001] 
                                next_gt_mask_vis =  self.vis_labels(next_gt_mask1)
                            elif self.use_CEloss == 0:     # Ll1
                                next_render_mask = data['left_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1) # 1 128 128 3(rgb label mean-> 0 1 2)
                                next_render_mask = (next_render_mask / 2 + 0.5  )*(self.num_classes-1)    
                                next_render_mask = self.vis_labelsL1(next_render_mask)
                                next_left_mask_gen = data['left_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1) # 1 128 128 3
                                next_left_mask_gen = (next_left_mask_gen / 2 + 0.5  )*(self.num_classes-1)           # 1 128 128 3 [0,2]
                                exclude_left_mask = self.generate_final_class_labels_L1(next_left_mask_gen)
                                next_left_mask_gen = self.vis_labelsL1(next_left_mask_gen)
                                next_gt_mask_vis = self.vis_labelsL1(next_gt_mask_label)
                            exclude_left_mask = exclude_left_mask * next_render_novel


                            # left gen mask
                                # data['left_next'] =self.pts2render_mask_gen(data['left_next'], bg_color=self.bg_mask)
                                # next_left_mask_gen = data['left_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1)
                                # exclude_left_mask = self.generate_final_class_labels(next_left_mask_gen).unsqueeze(3).repeat(1, 1, 1, 3) 
                                # next_left_mask_gen = self.vis_labels(next_left_mask_gen)

                            #  4 RGB loss_dyna_leader leader  利用前面得到的mask删去左臂
                            if ('xyz_maps' in data['right_next']):
                                # with torch.no_grad():  原来为了有的Loss没计算报无回传错误而写的
                                data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                                next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                                # 1
                            # 5 Mask loss_dyna_mask_next_right 右臂mask训练（和now一样 无用）
                                # next_render_mask_right = data['right_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1)
                            data['right_next'] =self.pts2render_mask(data['right_next'], bg_color=self.bg_mask)
                            next_render_mask_right = data['right_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1) 
                            if self.use_CEloss==1 or self.use_CEloss == 7 or self.use_CEloss == 21:
                                next_render_mask_right = self.vis_labels(next_render_mask_right)
                            elif self.use_CEloss == 0:     # Ll1
                                next_render_mask_right = (next_render_mask_right / 2 + 0.5  )*(self.num_classes-1) 
                                next_render_mask_right = self.vis_labelsL1(next_render_mask_right)

                            ## test
                                # next_render_mask_right = self.vis_labels(next_render_mask_right)
                            # next_render_mask_right = self.vis_labels(next_render_mask_right).unsqueeze(3).repeat(1, 1, 1, 3)
                            # next_render_mask_right = next_render_rgb_right * next_render_mask_right

                    elif self.mask_gen == 'None':
                        if self.use_dynamic_field :
                            # 2 GRB total(Follower)  先对left预测（最后的结果） loss_dyna_follower  左臂 RGB Loss   
                            # 也可以说是双手结果
                            if ('xyz_maps' in data['left_next']):
                                data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                                next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]

                            #  4 RGB loss_dyna_leader leader  利用前面得到的mask删去左臂
                            if ('xyz_maps' in data['right_next']):
                                # with torch.no_grad():  原来为了有的Loss没计算报无回传错误而写的
                                data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                                next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                    elif not self.use_nerf_picture: 
                        gt_rgb = gt_rgb.permute(0,2,3,1) # [1, 3, 256, 256] ->[1, 256, 256, 3]
                        gt_mask =gt_mask.permute(0,2,3,1)
                        gt_mask_label = self.mask_onehot(gt_mask)
                        data =self.pts2render_mask_gen(data, bg_color=self.bg_mask)
                        render_mask_novel = data['novel_view']['mask_gen'].permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]                           
                        # print("1",render_mask_novel.shape,render_mask_novel) # [1,256,256,3]
                        render_mask_novel = self.generate_final_class_labels(render_mask_novel)
                        # print("2",render_mask_novel.shape,render_mask_novel) # [1,3, 256,256]
                        render_mask_novel = render_mask_novel.unsqueeze(3).repeat(1, 1, 1, 3)
                        # render_mask_gtrgb = gt_rgb * render_mask_novel
                        render_mask_novel = render_novel * render_mask_novel # 反正背景都是False不用加
                        

                        if self.use_dynamic_field and (next_gt_rgb is not None and gt_mask is not None):
                            next_gt_rgb = next_gt_rgb.permute(0,2,3,1)
                            next_gt_mask = next_gt_mask.permute(0,2,3,1)
                            # 注意：这里的图片都是256*256的
                            next_gt_mask_label = self.mask_onehot(next_gt_mask)
                            # device = gt_mask.device  # 获取 next_gt_rgb 的设备
                            # gt_mask_label = gt_mask_label.to(device)
                            # next_gt_mask_label = next_gt_mask_label.to(device)
                            # 1 当前场景的mask 训练  loss_dyna_mask_novel

                            # 2 GRB total(Follower)  先对left预测（最后的结果） loss_dyna_follower  左臂 RGB Loss   
                            # 也可以说是双手结果
                            if ('xyz_maps' in data['left_next']):
                                data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                                next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]
                                # loss_dyna_follower = l2_loss(next_render_novel, next_gt_rgb) # 双臂结果预测

                            # 3 next mask train (pre - left(mask) ) next_loss_dyna_mask_left  左臂 Mask Loss
                            data['left_next'] =self.pts2render_mask_gen(data['left_next'], bg_color=self.bg_mask)
                            next_render_mask = data['left_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1) # [1,3，128, 128] -> [1,128, 128, 3]
                            exclude_left_mask = self.generate_final_class_labels(next_render_mask) # ?感觉有问题，用gt还是gen？ 用gt 因为这是对输出的计算
                            exclude_left_mask = exclude_left_mask.unsqueeze(3).repeat(1, 1, 1, 3)
                            result_right_image = next_render_novel * exclude_left_mask # 反正背景都是False不用加
                            next_render_mask = result_right_image

                            # gen mask and exclude
                            # data['left_next'] =self.pts2render_mask_gen(data['left_next'], bg_color=self.bg_color)
                            # next_left_mask_gen = data['left_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1)        

                            # exclude_left_mask = self.generate_final_class_labels(next_render_mask) # ?感觉有问题，用gt还是gen？
                            # print("next_gt_mask_label",next_gt_mask_label.shape,next_gt_mask_label)
                            exclude_left_mask = self.generate_final_class_labels(next_gt_mask_label) #[1 256 256 3] ?感觉有问题，用gt还是gen？ 用gt 因为这是对输出的计算
                            exclude_left_mask = exclude_left_mask.unsqueeze(3).repeat(1, 1, 1, 3)
                            # next_gt_rgb = next_gt_rgb.permute(0, 2, 3, 1)
                            device = next_gt_rgb.device  # 获取 next_gt_rgb 的设备
                            # 确保 exclude_left_mask 在同一个设备上
                            exclude_left_mask = exclude_left_mask.to(device)
                            # result_right_image = next_gt_rgb * exclude_left_mask # （gt）
                            next_left_mask_gen = next_gt_rgb * exclude_left_mask # （gt）

                            #  4 RGB loss_dyna_leader leader  利用前面得到的mask删去左臂
                            if ('xyz_maps' in data['right_next']):
                                data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                                next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                                next_render_novel_mask = next_render_rgb_right * exclude_left_mask  # 原来用错了...  next_gt_rgb -> next_render_rgb_right
                            
                            # 5 Mask loss_dyna_mask_next_right 右臂mask训练（和now一样 无用）
                            data['right_next'] =self.pts2render_mask_gen(data['right_next'], bg_color=self.bg_color)
                            next_render_mask_right = data['right_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1)
                            exclude_right_mask = self.generate_final_class_labels(next_render_mask_right) # ?感觉有问题，用gt还是gen？ 用gt 因为这是对输出的计算
                            exclude_right_mask = exclude_right_mask.unsqueeze(3).repeat(1, 1, 1, 3)
                            result_right_image = next_gt_rgb * exclude_right_mask # 反正背景都是False不用加
                            next_render_mask_right = result_right_image

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
                          render_mask_novel = render_mask_novel,           # now mask * render_rgb
                          render_mask_gtrgb = render_mask_gtrgb,           # now mask * gt_rgb
                        next_render_mask = next_render_mask,               # 左臂mask * next_render_novel
                        next_render_mask_right = next_render_mask_right,   # 无用 右臂mask 
                        next_render_rgb_right = next_render_rgb_right,            # 右臂next rgb
                        next_left_mask_gen = next_left_mask_gen, 
                        exclude_left_mask =exclude_left_mask,                      # 生成的左臂当时视角的mask
                        gt_mask_vis =gt_mask_vis,
                        next_gt_mask_vis =next_gt_mask_vis,
                        )             
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
        xyz_i = data['xyz_maps'][i, :, :].detach() 
        feature_i = data['sh_maps'][i, :, :, :].detach()  # [16384, 4, 3]
        rot_i = data['rot_maps'][i, :, :].detach() 
        scale_i = data['scale_maps'][i, :, :].detach() 
        opacity_i = data['opacity_maps'][i, :, :].detach() 
        precomputed_mask_i = data['mask_maps'][i, :, :] # mask  [1, 65536, 3]
        # precomputed_mask_i = precomputed_mask_i.reshape(1,256,256,3).permute(0,3,1,2) # [1, 65536, 3] -> [1, 3, 256, 256]
        # if self.mask_gen =='pre':    
            # precomputed_mask_i = F.interpolate(precomputed_mask_i, size=(128, 128), mode='bilinear', align_corners=False) # [1 3 256 256] -> [1 3 128 128]
        # precomputed_mask_i = precomputed_mask_i.squeeze(0).permute(1,2,0)
        feature_language_i = data['feature_maps'][i, :, :].detach()   # [B, N, 3]   [1, 65536, 3]  
        

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

    def pts2render_rgb(self, data: dict, bg_color=[0,0,0]):
        '''use render function in GSZ 在GSZ 中使用渲染功能(应该就是使用先前采集的数据重建场景)'''
        bs = data['intr'].shape[0]
        assert bs == 1, "batch size should be 1"
        # 公式2中 时刻i 的状态（θ 多了f 高级语义特征）
        i = 0
        xyz_i = data['xyz_maps'][i, :, :].detach() 
        feature_i = data['sh_maps'][i, :, :, :].detach()  # [16384, 4, 3]
        rot_i = data['rot_maps'][i, :, :].detach() 
        scale_i = data['scale_maps'][i, :, :].detach() 
        opacity_i = data['opacity_maps'][i, :, :].detach() 
        # precomputed_mask_i = data['mask_maps'][i, :, :] # mask  [1, 65536, 3]
        feature_language_i = data['feature_maps'][i, :, :].detach()   # [B, N, 3]   [1, 65536, 3]  
        

        # 渲染返回字典  render应该是用来渲染的  from agents.manigaussian_bc2.gaussian_renderer import render
        render_return_dict = render_rgb(
            data, i, xyz_i, rot_i, scale_i, opacity_i, 
            bg_color=bg_color, pts_rgb=None, features_color=feature_i, features_language=feature_language_i,
            # precomputed_mask = precomputed_mask_i,
            )

        # .unsqueeze(0): 这是PyTorch张量的一个操作，用于在张量的第0个维度（即最前面）增加一个维度。如果原始张量是一维的，这个操作会将其变成二维的，其中新加的维度大小为1。
        # data['novel_view']['img_pred']: 这是在 data 字典中的 'novel_view' 键下创建或更新一个子键 'img_pred'。这个子键被赋值为 render_return_dict['render'] 张量增加一个新维度后的结果。
        data['mask_view']['rgb_pred'] = render_return_dict['render'].unsqueeze(0)
        # data['novel_view']['embed_pred'] = render_return_dict['render_embed'].unsqueeze(0)
        return data



    def pts2render_mask_gen(self, data: dict, bg_color=[0,0,0]):
        '''use render function in GSZ 在GSZ 中使用渲染功能(应该就是使用先前采集的数据重建场景)'''
        bs = data['intr'].shape[0]
        assert bs == 1, "batch size should be 1"
        # 公式2中 时刻i 的状态（θ 多了f 高级语义特征）
        i = 0
        xyz_i = data['xyz_maps'][i, :, :].detach()       # [65536, 3]
        feature_i = data['sh_maps'][i, :, :, :].detach()  # [16384(现在应该是256 * 256), 4, 3]
        rot_i = data['rot_maps'][i, :, :].detach() 
        scale_i = data['scale_maps'][i, :, :].detach() 
        opacity_i = data['opacity_maps'][i, :, :].detach() 
        precomputed_mask_i = data['mask_maps'][i, :, :] # mask  [1, 65536, 3]
        # precomputed_mask_i = precomputed_mask_i.reshape(1,256,256,3).permute(0,3,1,2) # [1, 65536, 3] -> [1, 3, 256, 256]
        # if self.mask_gen =='pre':    
        #     precomputed_mask_i = F.interpolate(precomputed_mask_i, size=(128, 128), mode='bilinear', align_corners=False) # [1 3 256 256] -> [1 3 128 128]
        # precomputed_mask_i = precomputed_mask_i.squeeze(0).permute(1,2,0)
        
        feature_language_i = data['feature_maps'][i, :, :].detach()   # [B, N, 3]   [1, 65536, 3]  
        

        # 渲染返回字典  render应该是用来渲染的  from agents.manigaussian_bc2.gaussian_renderer import render
        render_return_dict = render_mask_gen(
            data, i, xyz_i, rot_i, scale_i, opacity_i, 
            bg_color=bg_color, pts_rgb=None, features_color=feature_i, features_language=feature_language_i,
            precomputed_mask = precomputed_mask_i,
            )

        # .unsqueeze(0): 这是PyTorch张量的一个操作，用于在张量的第0个维度（即最前面）增加一个维度。如果原始张量是一维的，这个操作会将其变成二维的，其中新加的维度大小为1。
        # data['novel_view']['img_pred']: 这是在 data 字典中的 'novel_view' 键下创建或更新一个子键 'img_pred'。这个子键被赋值为 render_return_dict['render'] 张量增加一个新维度后的结果。
        data['novel_view']['mask_gen'] = render_return_dict['mask'].unsqueeze(0)
        data['novel_view']['embed_pred'] = render_return_dict['render_embed'].unsqueeze(0)
        return data

    def createby_gt_mask(self, data: dict, gt_mask=None, gt_mask_camera_extrinsic=None, gt_mask_camera_intrinsic=None, next_gt_mask = None,
                gt_maskdepth=None,next_gt_maskdepth=None):
        # print("only for gen",gt_mask_camera_intrinsic)
        # assert bs == 1, "batch size should be 1" # 要检测吗？
        front_intrinsic = gt_mask_camera_intrinsic[0] # [tensor([[[-351.6771,    0.0000,  128.0000], 
        overhead_intrinsic = gt_mask_camera_intrinsic[1]
        # print("gt_mask_camera_intrinsic",gt_mask_camera_intrinsic)
        # print("gt_mask_camera_extrinsic",gt_mask_camera_extrinsic)
        # front_mask = gt_mask[0]
        # overhead_mask = gt_mask[1]
        # front_depth = gt_maskdepth[0]
        # overhead_depth = gt_maskdepth[1]
        # # .squeeze(0) or [0]
        # # # 三维映射到二维 但是depth的写的还是有问题
        # # newxyz_front = label_point_cloud(data['xyz'][0],front_depth,front_intrinsic,front_mask) # 应该有这个存着的吧
        # # newxyz_overhead = label_point_cloud(data['xyz'][0],overhead_depth,overhead_intrinsic,overhead_mask)
        # # print(newxyz_overhead) # []
        # # 左臂的点云
        # leftxyz_front = depth_mask_to_3d(front_depth,front_mask,front_intrinsic)
        # leftxyz_overhead = depth_mask_to_3d(overhead_depth,overhead_mask,overhead_intrinsic)
        # leftxyz = np.concatenate((leftxyz_front, leftxyz_overhead), axis=0) # 或者相加
        # leftxyz = torch.tensor(leftxyz)
        # if len(leftxyz) > 0:
        #     mask_3d = points_inside_convex_hull(data['xyz'][0].detach(), leftxyz)
        # # if len(leftxyz_overhead) > 0:
        #     # mask_3d_overhead = points_inside_convex_hull(data['xyz'][0].detach(), leftxyz_overhead)

        mask_3d = None
        # # --------上面部分是现在阶段的写法，耗时久，注释了---------------------------------------now -----------------------------------------------------

        # front_intrinsic = gt_mask_camera_intrinsic[0] # [tensor([[[-351.6771,    0.0000,  128.0000], 
        # overhead_intrinsic = gt_mask_camera_intrinsic[1]
        # time1 = time.perf_counter()
        # print('1 time', time1)
        next_front_mask = next_gt_mask[0]
        next_overhead_mask = next_gt_mask[1]
        next_front_depth = next_gt_maskdepth[0]
        next_overhead_depth = next_gt_maskdepth[1]
        # .squeeze(0) or [0]

        # 左臂的点云 前视角 0.15s
        next_leftxyz_front = depth_mask_to_3d(next_front_depth,next_front_mask,front_intrinsic)
        
        # time2 = time.perf_counter()
        # time_step1 = time2 - time1
        # print(f"time2 = {time2} step1 = {time_step1:.2f}s")    # 0.15s    
        
        # 左臂的点云 上视角
        next_leftxyz_overhead = depth_mask_to_3d(next_overhead_depth,next_overhead_mask,overhead_intrinsic)
        
        # time3 = time.perf_counter()
        # time_step2 = time3 - time2
        # print(f"time3 = {time3} step2 = {time_step2:.2f}s") # 0.12s

        # 相加    
        # # CPU
        # next_leftxyz = merge_arrays(next_leftxyz_front, next_leftxyz_overhead) # 或者相加
        # next_leftxyz = torch.tensor(next_leftxyz)
        # GPU
        next_leftxyz = merge_tensors(next_leftxyz_front, next_leftxyz_overhead).cpu()
    
        # time4 = time.perf_counter()
        # time_step3 = time4 - time3
        # print(f"time4 = {time4} step3 = {time_step3:.2f}s") # CPU 0.55s GPU:0s?

        if len(next_leftxyz) > 0:
            # next_mask_3d = points_inside_convex_hull( data['canon_xyz'][0].detach(), next_leftxyz)
            next_mask_3d = points_inside_convex_hull( data['xyz'][0].detach(), next_leftxyz)
        
        # time5 = time.perf_counter()
        # time_step4 = time5 - time4
        # print(f"time5 = {time5} step4 = {time_step4:.2f}s") # 0.55s

        return mask_3d, next_mask_3d

    def generate_final_class_labels(self,next_left_mask_gen):
        """
        排除左手区域。 1 128 128 3 ->
        """
        # 获取每个像素的最大类索引
        # exclude_left_mask = (next_left_mask_gen > 2.5) | (next_left_mask_gen < 1.5) # 原来 值的方式来判断
        # print("before",next_left_mask_gen.shape)
        class_indices = torch.argmax(next_left_mask_gen, dim=-1)
        # print("class_indices = ", class_indices.shape,class_indices)

        # 初始化最终的分类标签
        # final_class_labels = torch.zeros_like(class_indices, dtype=torch.long)
        # # 设置背景、右手和左手的标签
        # final_class_labels[class_indices == 0] = 0  # 背景
        # final_class_labels[class_indices == 1] = 1  # 右手
        # final_class_labels[class_indices == 2] = 2  # 左手

        # 生成排除左手的 mask
        # exclude_left_mask = final_class_labels != 2

        exclude_left_mask = class_indices != 2  # 不等于 2 的部分为 True
        # print("exclude_left_mask = ", exclude_left_mask.shape,exclude_left_mask)

        # return final_class_labels, exclude_left_mask
        return exclude_left_mask

    def generate_final_class_labels_ce1(self,next_left_mask_gen):
        """
        去除背景loss计算的(label-1) 排除左手区域。 1 128 128 3 -> 1 128 128 2 -> out: 1 128 128 [T:1 / F:0]
        """
        # 获取每个像素的最大类索引
        # exclude_left_mask = (next_left_mask_gen > 2.5) | (next_left_mask_gen < 1.5) # 原来 值的方式来判断
        # print("before",next_left_mask_gen.shape)
        next_left_mask_gen = next_left_mask_gen[:,:,:,[0,1]]
        class_indices = torch.argmax(next_left_mask_gen, dim=-1)
        print("class_indices = ", class_indices.shape,class_indices)

        exclude_left_mask = class_indices != 1  # 不等于 2 的部分为 True

        return exclude_left_mask


    def generate_final_class_labels_L1(self,next_left_mask_gen):
        """ [1 128 128 3] -> [1 128 128 3] [True False]       """
        class_indices = next_left_mask_gen.mean(dim=-1)
        print("class_indices = ", class_indices.shape,class_indices)
        next_left_mask_gen = next_left_mask_gen.squeeze(-1)
        exclude_left_mask = torch.ones_like(next_left_mask_gen, dtype=torch.float32)
        exclude_left_mask = (class_indices < 1.5) # 2是左手
        exclude_left_mask = exclude_left_mask.unsqueeze(-1).repeat(1,1,1,3)   
        print("exclude_left_mask",exclude_left_mask)
        # exclude_left_mask = class_indices != 2  # 不等于 2 的部分为 True
        return exclude_left_mask
    
    def mask_onehot(self,mask):
        """1 128 128 1 -> [1 128 128 3] 100 010 001"""
        right_min, right_max, left_min, left_max = 53, 73, 94, 114
        # print("gt_mask",mask,mask.shape)
        gt_mask1 = mask.squeeze(-1)
        # print("gt_mask1",gt_mask1,gt_mask1.shape) # [1,256,256]

        # 初始化独热编码张量 bg:[0,0,0]   right:[0,1,0]  left:[0,0,1]
        gt_mask_label = torch.zeros((*gt_mask1.shape, self.num_classes), dtype=torch.float32) 
        # print("gt_mask_label.shape",gt_mask_label.shape) # [1,256,256,3]
        # 将标签转换为独热编码
        bg_mask = (gt_mask1 < right_min) | ((gt_mask1 > right_max) & (gt_mask1 < left_min)) | (gt_mask1 > left_max)
        # print("bg_mask",bg_mask.shape)
        gt_mask_label[bg_mask] = torch.tensor([1, 0, 0], dtype=torch.float32)
       
        right_mask = (gt_mask1 > right_min - 1) & (gt_mask1 < right_max + 1)
        gt_mask_label[right_mask] = torch.tensor([0, 1, 0], dtype=torch.float32)  # 右手的独热编码                  
        left_mask = (gt_mask1 > left_min - 1) & (gt_mask1 < left_max + 1)
        gt_mask_label[left_mask] = torch.tensor([0, 0, 1], dtype=torch.float32) 


        """ # # Tensor写法（mask标签归类 0：bg    1:ritght    2:left）
        # # gt_mask_label = torch.zeros_like(gt_mask, dtype=torch.uint8)
        # gt_mask_label = torch.zeros_like(gt_mask, dtype=torch.long)
        # gt_mask_label[(gt_mask > right_min-1) & (gt_mask < right_max+1)] = 1
        # gt_mask_label[(gt_mask > left_min-1) & (gt_mask < left_max+1)] = 2
        # # next_gt_mask_label = torch.zeros_like(next_gt_mask, dtype=torch.uint8)
        # next_gt_mask_label = torch.zeros_like(next_gt_mask, dtype=torch.long)
        # next_gt_mask_label[(next_gt_mask > right_min-1) & (next_gt_mask < right_max+1)] = 1
        # next_gt_mask_label[(next_gt_mask > left_min-1) & (next_gt_mask < left_max+1)] = 2



        # gt_mask_label = torch.clamp(gt_mask_label, 0, 1)
                        # # gt_mask = gt_mask[indx].permute(0, 2, 3, 1).repeat(1, 1, 1, 3)  # 复制三次，得到 [1,128, 128, 3]
                        # gt_mask = gt_mask.repeat(1, 3, 1, 1)  #  [1,3,256,256]
                        # # gt_mask = F.interpolate(gt_mask, size=(128, 128), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                        # # next_gt_mask = next_gt_mask[indx].permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
                        # next_gt_mask = next_gt_mask.repeat(1, 3, 1, 1)  #  [1,3,256,256]
                        # # next_gt_mask = F.interpolate(next_gt_mask, size=(128, 128), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                        # gt_mask = gt_mask.repeat(1, 1, 1, self.num_classes)  #  [1,256,256,3]
                        # next_gt_mask = next_gt_mask.repeat(1, 1, 1, self.num_classes)
                        # print("gt_mask",gt_mask,gt_mask.shape)
                        # gt_mask1 = gt_mask.squeeze(-1)
                        # print("gt_mask1",gt_mask1,gt_mask1.shape) # [1,256,256]
                        # gt_mask = gt_mask.repeat(1, 1, 1, 3)
                        # # 初始化独热编码张量 bg:[0,0,0]   right:[0,1,0]  left:[0,0,1]
                        # gt_mask_label = torch.zeros((*gt_mask1.shape, self.num_classes), dtype=torch.float32) 
                        # print("gt_mask_label.shape",gt_mask_label.shape) # [1,256,256,3]
                        # # 将标签转换为独热编码
                        # bg_mask = (gt_mask1 < right_min) | ((gt_mask1 > right_max) & (gt_mask1 < left_min)) | (gt_mask1 > left_max)
                        # print("bg_mask",bg_mask.shape)
                        # # bg_mask_expanded = bg_mask.expand(-1, -1, -1, 3)
                        # # gt_mask_label[bg_mask_expanded] = torch.tensor([1, 0, 0], dtype=torch.float32)
                        # gt_mask_label[bg_mask] = torch.tensor([1, 0, 0], dtype=torch.float32)
                        # # gt_mask_label[(gt_mask1 < right_min) | ((gt_mask1 > right_max) & (gt_mask1 < left_min) | (gt_mask1 > left_max)), 0] = 1  # 背景
                        
                        # right_mask = (gt_mask1 > right_min - 1) & (gt_mask1 < right_max + 1)
                        # # right_mask_expanded = right_mask.expand(-1, -1, -1, 3)
                        # # gt_mask_label[right_mask_expanded] = torch.tensor([0, 1, 0], dtype=torch.float32)  # 右手的独热编码
                        # gt_mask_label[right_mask] = torch.tensor([0, 1, 0], dtype=torch.float32)  # 右手的独热编码
                        # # gt_mask_label[(gt_mask1 > right_min-1) & (gt_mask1 < right_max+1), 1] = 1  # 右手

                        # # gt_mask_label[(gt_mask1 > left_min-1) & (gt_mask1 < left_max+1), 2] = 1  # 左手                        
                        # left_mask = (gt_mask1 > left_min - 1) & (gt_mask1 < left_max + 1)
                        # # left_mask_expanded = left_mask.expand(-1, -1, -1, 3)
                        # # gt_mask_label[left_mask_expanded] = torch.tensor([0, 0, 1], dtype=torch.float32) 
                        # gt_mask_label[left_mask] = torch.tensor([0, 0, 1], dtype=torch.float32) 

                        # print("gt_mask_label",gt_mask_label, gt_mask_label.shape)

                        # # next_gt_mask_label 同理
                        # next_gt_mask_label = torch.zeros((*next_gt_mask.shape, num_classes), dtype=torch.float32)
                        # next_gt_mask_label[(next_gt_mask < right_min) | ((next_gt_mask > right_max) & (next_gt_mask < left_min) | (next_gt_mask > left_max)), 0] = 1  # 背景
                        # next_gt_mask_label[(next_gt_mask > right_min-1) & (next_gt_mask < right_max+1), 1] = 1  # 右手
                        # next_gt_mask_label[(next_gt_mask > left_min-1) & (next_gt_mask < left_max+1), 2] = 1  # 左手 """

        return gt_mask_label

    def mask_label_onehot(self,gt_mask):
        """[1 128 128 1] -> [1 128 128]  mask标签归类 0：bg    1:ritght    2:left"""
        right_min, right_max, left_min, left_max = 53, 73, 94, 114

        # Tensor写法（mask标签归类 0：bg    1:ritght    2:left）
        # gt_mask_label = torch.zeros_like(gt_mask, dtype=torch.uint8)
        # gt_mask = gt_mask.squeeze(-1)
        gt_mask_label = torch.zeros_like(gt_mask, dtype=torch.long)
        gt_mask_label[(gt_mask > right_min-1) & (gt_mask < right_max+1)] = 1
        gt_mask_label[(gt_mask > left_min-1) & (gt_mask < left_max+1)] = 2
        gt_mask_label = gt_mask_label.squeeze(-1)
        # # next_gt_mask_label = torch.zeros_like(next_gt_mask, dtype=torch.uint8)
        # next_gt_mask_label = torch.zeros_like(next_gt_mask, dtype=torch.long)
        # next_gt_mask_label[(next_gt_mask > right_min-1) & (next_gt_mask < right_max+1)] = 1
        # next_gt_mask_label[(next_gt_mask > left_min-1) & (next_gt_mask < left_max+1)] = 2
        
        return gt_mask_label

    def mask_label(self,gt_mask):
        """[1 128 128 1] -> [1 128 128 3] [000 / 111 / 222]"""
        right_min, right_max, left_min, left_max = 53, 73, 94, 114

        # Tensor写法（mask标签归类 0：bg    1:ritght    2:left）
        # gt_mask_label = torch.zeros_like(gt_mask, dtype=torch.uint8)
        gt_mask = gt_mask.squeeze(-1)
        gt_mask_label = torch.zeros_like(gt_mask, dtype=torch.float32)
        gt_mask_label[(gt_mask > right_min-1) & (gt_mask < right_max+1)] =1.0
        gt_mask_label[(gt_mask > left_min-1) & (gt_mask < left_max+1)] = 2.0
        gt_mask_label = gt_mask_label.unsqueeze(-1).repeat(1,1,1,3)         
        return gt_mask_label


    def vis_labels(self,next_left_mask_gen):
        """
        vis [1 128 128 3] - > rgb
        """
        # 获取每个像素的最大类索引
        class_indices = torch.argmax(next_left_mask_gen, dim=-1)
        print("class_indices = ", class_indices.shape,class_indices)

        # # 初始化最终的分类标签
        # final_class_labels = torch.zeros_like(class_indices, dtype=torch.long)
        # # # 设置背景、右手和左手的标签
        # final_class_labels[class_indices == 0] = 0  # 背景
        # final_class_labels[class_indices == 1] = 1  # 右手
        # final_class_labels[class_indices == 2] = 2  # 左手
        color_image = torch.zeros((*class_indices.shape, 3), dtype=torch.uint8)

        # 为每个类别设置颜色（RGB）
        color_image[class_indices == 0] = torch.tensor([0, 0, 0], dtype=torch.uint8)    # 背景 - 黑色
        color_image[class_indices == 1] = torch.tensor([255, 0, 0], dtype=torch.uint8)  # 右手 - 红色
        color_image[class_indices == 2] = torch.tensor([0, 0, 255], dtype=torch.uint8)  # 左手 - 蓝色

        # 生成排除左手的 mask
        # exclude_left_mask = final_class_labels != 2
        return color_image

    def vis_labels_ce1(self,next_left_mask_gen):
        """
        (减去bgloss计算) vis [1 128 128 3] -> [1 128 128 2] - > rgb
        """
        # 获取每个像素的最大类索引
        next_left_mask_gen = next_left_mask_gen[:,:,:,[0,1]]
        class_indices = torch.argmax(next_left_mask_gen, dim=-1)
        print("class_indices1 128 128 2 = ", class_indices.shape,class_indices)

        color_image = torch.zeros((*class_indices.shape, 3), dtype=torch.uint8)

        # 为每个类别设置颜色（RGB）
        # color_image[class_indices == 0] = torch.tensor([0, 0, 0], dtype=torch.uint8)    # 背景 - 黑色
        color_image[class_indices == 0] = torch.tensor([255, 0, 0], dtype=torch.uint8)  # 右手 - 红色
        color_image[class_indices == 1] = torch.tensor([0, 0, 255], dtype=torch.uint8)  # 左手 - 蓝色
        return color_image


    def vis_labelsL1(self,mask):
        """
        vis [1 128 128 3] - > rgb
        """
        mask_mean = mask.mean(dim=-1)
        print("vis_labelsL1 mask_mean =",mask_mean.shape,mask_mean)
        mask_rgb = torch.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 3), dtype=torch.uint8)
        # 设置颜色范围
        # mask_rgb[(mask_mean >= 0) & (mask_mean < 0.5)] = torch.tensor([0, 0, 0], dtype=torch.uint8)    # 黑色
        mask_rgb[(mask_mean >= 0.7) & (mask_mean < 1.3)] = torch.tensor([255, 0, 0], dtype=torch.uint8) # 红色
        mask_rgb[mask_mean >= 1.3] = torch.tensor([0, 0, 255], dtype=torch.uint8)                       # Green 
        # mask_rgb[(mask_mean >= 0.5) & (mask_mean < 1.5)] = torch.tensor([255, 0, 0], dtype=torch.uint8) # 红色
        # mask_rgb[mask_mean >= 1.5] = torch.tensor([0, 0, 255], dtype=torch.uint8)                       # Green
        return mask_rgb

    def one_hot_encode(self,mask, num_classes):
        "b h w -> b c h w c=(100 010 001)"
        # mask 的形状为 (batch_size, height, width)，数值为类别
        # 转换为 One-Hot 编码，结果形状为 (batch_size, num_classes, height, width)
        one_hot_mask = torch.nn.functional.one_hot(mask, num_classes=num_classes)
        one_hot_mask = one_hot_mask.permute(0, 3, 1, 2)  # 调整维度顺序
        return one_hot_mask.float()  # 转换为浮点数