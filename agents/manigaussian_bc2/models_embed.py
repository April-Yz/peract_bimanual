
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
import torch.autograd.profiler as profiler

import os
import os.path as osp
import warnings
from termcolor import colored, cprint

from agents.manigaussian_bc2.utils import PositionalEncoding, visualize_pcd
from agents.manigaussian_bc2.resnetfc import ResnetFC


from typing import List
import numpy as np
import visdom


class GSPointCloudRegresser(nn.Module):
    # 一个神经网络模块，用于回归点云数据的高斯参数
    def __init__(self, cfg, out_channels, bias, scale):
        '''
        for weight initialization
        '''
        super().__init__()
        self.out_channels = out_channels
        self.cfg = cfg
        self.activation = torch.nn.functional.softplus
        self.out = nn.Linear(
            in_features=sum(out_channels),
            out_features=sum(out_channels),
        )
    def forward(self, x):
        return self.out(self.activation(x, beta=100))

class GeneralizableGSEmbedNet(nn.Module):
    # 主要的网络类，用于嵌入和处理3D几何和语义信息
    def __init__(self, cfg, with_gs_render=True):
        super().__init__()
        self.cfg = cfg
        self.with_gs_render = with_gs_render

        # print(colored(f"[GeneralizableNeRFEmbedNet]的 with_gs_render参数: {with_gs_render}", "red"))
    
        # 坐标边界
        self.coordinate_bounds = cfg.coordinate_bounds # default: [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
        # print(colored(f"[GeneralizableNeRFEmbedNet] coordinate_bounds: {self.coordinate_bounds}", "red"))
    
        self.use_xyz = cfg.use_xyz
        d_in = 3 if self.use_xyz else 1

        self.use_code = cfg.use_code
        if self.use_code and d_in > 0:
            # Positional encoding for x,y,z
            self.code = PositionalEncoding.from_conf(cfg["code"], d_in=d_in)
            d_in = self.code.d_out  # 39

        self.d_in = d_in

        self.image_shape = (cfg.image_height, cfg.image_width)
        self.num_objs = 0
        self.num_views_per_obj = 1

        # 分割维度  比例初始化
        split_dimensions, scale_inits, bias_inits = self._get_splits_and_inits(cfg)

        # backbone
        self.d_latent = d_latent = cfg.d_latent # 128 (要不要改成256？)
        self.d_lang = d_lang = cfg.d_lang   # 128
        self.d_out = sum(split_dimensions)
        print(colored(f"self.d_in {self.d_in}", "red"))  # 39
        print(colored(f"self.d_out {self.d_out}", "red"))  # 26-> 29(mask) 如果变成29就对了

        self.encoder = ResnetFC(
                d_in=d_in, # xyz                    # 39
                d_latent=d_latent,  # volumetric representation 体积表示
                d_lang=d_lang, 
                d_out=self.d_out,                   # 26 + 3
                d_hidden=cfg.mlp.d_hidden,          #  512 每个隐藏层块中的隐藏单元（神经元）数量
                n_blocks=cfg.mlp.n_blocks,          # 5  隐藏层块数量
                combine_layer=cfg.mlp.combine_layer,    # 3
                beta=cfg.mlp.beta,  # 0.0
                use_spade=cfg.mlp.use_spade, # False
            )
        
        self.gs_parm_regresser = GSPointCloudRegresser(
            cfg,
            split_dimensions, # 分割尺寸
            scale=scale_inits, # scale_inits
            bias=bias_inits,
            )
        self.scaling_activation = torch.exp
        # self.scaling_activation = torch.nn.functional.softplus
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize    # [B, N, 4]
        self.max_sh_degree = cfg.mlp.max_sh_degree
        # self.mask_activation = torch.sigmoid

        # we move xyz, rot
        self.use_dynamic_field = cfg.use_dynamic_field
        self.mask_gen =cfg.mask_gen
        self.field_type = cfg.field_type
        self.use_nerf_picture = cfg.use_nerf_picture
        self.warm_up = cfg.next_mlp.warm_up
        self.use_action = cfg.next_mlp.use_action
        self.use_mask = ((self.field_type =='LF' and self.mask_gen== 'pre') or (not self.use_nerf_picture))
        cprint(f"[GeneralizableGSEmbedNet] Using dynamic field: {self.use_dynamic_field}", "red")
        if self.use_dynamic_field:
            cprint(f"[GeneralizableGSEmbedNet] field_type: {self.field_type}", "red")
            if self.field_type =='bimanual':
                self.use_semantic_feature = (cfg.foundation_model_name == 'diffusion')
                cprint(f"[GeneralizableGSEmbedNet] Using action input: {self.use_action}", "red")
                cprint(f"[GeneralizableGSEmbedNet] Using semantic feature: {self.use_semantic_feature}", "red")
                # if self.leader:
                next_d_in = self.d_out + self.d_in      # (26 + 3(mask) ) + 39 = (65 + 3) = 68      # cprint(f"65 next_d_in = self.d_out + self.d_in: {next_d_in}", "green") 
                next_d_in = next_d_in + 16 if self.use_action else next_d_in  # 73 -> 81 +3(mask) -> 84  action: 8->16 dim # new theta_right=8(还是16呢)      # cprint(f"73 next_d_in = next_d_in + 8 if self.use_action else next_d_in {next_d_in}", "green")
                next_d_in = next_d_in if self.use_semantic_feature else next_d_in - 3   # 70->78
                next_d_in = next_d_in if self.use_mask  else next_d_in - 3      # (-mask)如果双手一起预测   减3      # cprint(f"70 next_d_in = next_d_in if self.use_semantic_feature else next_d_in - 3 {next_d_in}", "green")
                self.gs_deformation_field = ResnetFC(
                        d_in=next_d_in, # all things despite volumetric representation (26 + 39 + 8 -3 = 70) 尽管有体积表示
                        d_latent=self.d_latent,
                        d_lang=self.d_lang,
                        d_out=3 + 4,    # xyz, rot
                        d_hidden=cfg.next_mlp.d_hidden, 
                        n_blocks=cfg.next_mlp.n_blocks, 
                        combine_layer=cfg.next_mlp.combine_layer,
                        beta=cfg.next_mlp.beta, use_spade=cfg.next_mlp.use_spade,
                    )

                # -------------------------------for leader - Follower ------------------------------
                # if self.field_type =='LF':
            else:
                self.use_semantic_feature = (cfg.foundation_model_name == 'diffusion')
                cprint(f"[GeneralizableGSEmbedNet] Using action input: {self.use_action}", "red")
                cprint(f"[GeneralizableGSEmbedNet] Using semantic feature: {self.use_semantic_feature}", "red")
                # if self.leader:
                next_d_in = self.d_out + self.d_in      # 26( +3 mask ) + 39 = 65 + 3 = 68   
                next_d_in = next_d_in + 8 if self.use_action else next_d_in  # 73 +3(mask) = 76 action: 8 dim # new theta_right=8(还是16呢)
                next_d_in = next_d_in if self.use_semantic_feature else next_d_in - 3   # 73-3 -> 70 +3(mask) = 73
                next_d_in = next_d_in if self.use_mask else next_d_in - 3      # mask    # cprint(f"70 next_d_in = next_d_in if self.use_semantic_feature else next_d_in - 3 {next_d_in}", "green")
                # 超显存
                # with torch.no_grad():
                # self.gs_deformation_field_leader = ResnetFC(
                #         d_in=next_d_in, # all things despite volumetric representation (26 + 39 + 8 -3 = 70) 尽管有体积表示
                #         d_latent=self.d_latent,
                #         d_lang=self.d_lang,
                #         d_out=3 + 4,    # xyz, rot
                #         d_hidden=cfg.next_mlp.d_hidden, 
                #         n_blocks=cfg.next_mlp.n_blocks, 
                #        combine_layer=cfg.next_mlp.combine_layer,
                #         beta=cfg.next_mlp.beta, use_spade=cfg.next_mlp.use_spade,
                #     )
                self.gs_deformation_field_leader_smaller = ResnetFC(
                        d_in=next_d_in, # all things despite volumetric representation (26 + 39 + 8 -3 = 70) 尽管有体积表示
                        d_latent=self.d_latent,
                        d_lang=self.d_lang,
                        d_out=3 + 4,    # xyz, rot
                        d_hidden=cfg.next_mlp_small.d_hidden, 
                        n_blocks=cfg.next_mlp_small.n_blocks, 
                       combine_layer=cfg.next_mlp_small.combine_layer,
                        beta=cfg.next_mlp_small.beta, use_spade=cfg.next_mlp_small.use_spade,
                    )
                # else:
                # follower
                next_d_in = self.d_out + self.d_in # 39+26=65
                next_d_in = next_d_in + 8 + 8 if self.use_action else next_d_in  # action: 8 再加上8 dim # new theta_right=8(还是16呢)
                next_d_in = next_d_in if self.use_semantic_feature else next_d_in - 3
                next_d_in = next_d_in if self.use_mask else next_d_in - 3
                self.gs_deformation_field_follower = ResnetFC(
                        d_in=next_d_in, # all things despite volumetric representation (26 + 39 + 8 -3 = 70) 尽管有体积表示
                        d_latent=self.d_latent,
                        d_lang=self.d_lang,
                        d_out=3 + 4,    # xyz, rot   # ？
                        d_hidden=cfg.next_mlp.d_hidden, 
                        n_blocks=cfg.next_mlp.n_blocks, 
                        combine_layer=cfg.next_mlp.combine_layer,
                        beta=cfg.next_mlp.beta, use_spade=cfg.next_mlp.use_spade,
                    )
            # -------------------------------for leader - Follower ------------------------------

    def _get_splits_and_inits(self, cfg):
        '''Gets channel split dimensions and last layer initialization
        获取通道分割尺寸和最后一层初始化值
        Credit: https://github.com/szymanowiczs/splatter-image/blob/main/scene/gaussian_predictor.py
        项目用于单图重建3DGS
        '''
        split_dimensions = []
        scale_inits = []
        bias_inits = []
        # split_dimensions = split_dimensions + [3, 1, 3, 4, 3, 3] # 17
        split_dimensions = split_dimensions + [3, 1, 3, 4, 3, 3, 3] # 17->20 最后一个3用于mask
        # print("1 split_dimensions: ", split_dimensions, len(split_dimensions))
        scale_inits = scale_inits + [              # 添加初始化缩放值
            cfg.mlp.xyz_scale,
            cfg.mlp.opacity_scale,
            cfg.mlp.scale_scale,
            1.0,    # rotation
            5.0,    # feature_dc
            1.0,    # feature
            1.0     # 最后一个1用于mask
            ]
        bias_inits = [                              # 添加偏置初始化值，主要用于每个输出通道的偏置
            cfg.mlp.xyz_bias, 
            cfg.mlp.opacity_bias,
            np.log(cfg.mlp.scale_bias),
            0.0,
            0.0,
            0.0,
            0.0 # 最后一个0用于mask
            ]
        if cfg.mlp.max_sh_degree != 0:    # default: 1
            sh_num = (self.cfg.mlp.max_sh_degree + 1) ** 2 - 1    # 3 # sh_num 计算的是球谐函数（SH，Spherical Harmonics）的参数数量
            sh_num_rgb = sh_num * 3                                   # sh_num_rgb 是它乘以 3，因为通常对于 RGB 图像，会分别计算每个颜色通道的 SH 参数。  
            split_dimensions.append(sh_num_rgb)                       # [3, 1, 3, 4, 3, 3, 3, 9] 20 + 9 = 29( 26 + 3(mask) ) 
            scale_inits.append(0.0)                                   # 将 SH 特征的维度、缩放和偏置都添加到相应的列表中，初始化为 0.0
            # print("2 split_dimensions: ", split_dimensions, len(split_dimensions))
            bias_inits.append(0.0)
        self.split_dimensions_with_offset = split_dimensions          # split_dimensions_with_offset 保存了计算出来的通道分割尺寸 
        return split_dimensions, scale_inits, bias_inits  # split_dimensions原来=23(无sem)26

    @torch.no_grad()
    def world_to_canonical(self, xyz):
        """
        :param xyz (B, N, 3) or (B, 3, N)                                      (批次大小 B, 点的数量 N, 3个坐标)   找到这个加上id
        :return (B, N, 3) or (B, 3, N)
        将世界坐标转换为具有边界框 [0, 1] 的规范坐标
        transform world coordinate to canonical coordinate with bounding box [0, 1]
        """
        xyz = xyz.clone()
        bb_min = self.coordinate_bounds[:3] # 前面手动记录的前三个是坐标系的最小值
        bb_max = self.coordinate_bounds[3:]
        bb_min = torch.tensor(bb_min, device=xyz.device).unsqueeze(0).unsqueeze(0) if xyz.shape[-1] == 3 \
            else torch.tensor(bb_min, device=xyz.device).unsqueeze(-1).unsqueeze(0)
        bb_max = torch.tensor(bb_max, device=xyz.device).unsqueeze(0).unsqueeze(0) if xyz.shape[-1] == 3 \
            else torch.tensor(bb_max, device=xyz.device).unsqueeze(-1).unsqueeze(0)
        xyz -= bb_min
        xyz /= (bb_max - bb_min)

        return xyz

    # 样本在规范体素中
    def sample_in_canonical_voxel(self, xyz, voxel_feat):   # USED
        """
        从体素特征中采样点云的特征
        :param xyz (B, 3)
        :param self.voxel_feat: [B, 128, 20, 20, 20] 每个体素的特征，128 是特征维度，20x20x20 是体素的空间尺寸
        :return (B, Feat)
        """
        xyz_voxel_space = xyz.clone()
        xyz_voxel_space = xyz_voxel_space * 2 - 1.0 # [0,1]->[-1,1]

        # unsqueeze the point cloud to also have 5 dim
        xyz_voxel_space = xyz_voxel_space.unsqueeze(1).unsqueeze(1)   
        # 在第二和第三维度上增加两个维度，使得 xyz_voxel_space 的形状变为 [B, 1, 1, N, 3]
        # xyz_voxel_space: [bs, 1, 1, N, 3]print("def sample_in_canonical_voxel(self, xyz, voxel_feat): xyz_voxel_space.shape=", xyz_voxel_space.shape) # [2,1,1,16384,3]sample in voxel space 体素空间中的样本print("def sample_in_canonical_voxel(self, xyz, voxel_feat): voxel_feat.shape=", voxel_feat.shape) # [2,1,100,100,100]
        point_feature = F.grid_sample(voxel_feat, xyz_voxel_space, align_corners=True, mode='bilinear')
        # print("point_feature.shape=", point_feature.shape)            # [2,1,1,1,16384] [bs, 128, 1, 1, N]
        # squeeze back to point cloud shape 
        point_feature = point_feature.squeeze(2).squeeze(2).permute(0, 2, 1)                                                            
        # 使用 squeeze 方法移除第二和第三维度（它们的大小为1），然后使用 permute 方法重新排列张量的形状，使其变为 [B, N, 128]。  [bs, N, 128]
        # print("def sample_in_canonical_voxel(self, xyz, voxel_feat): point_feature.shape=", point_feature.shape) # [2,16384,1]
        return point_feature

    def forward(self, data):
        """
        SB is batch size                批次大小（batch size）config/train                                                2
        N is batch of points            批次中的点的数量                                                错了? 128
        NS is number of input views     输入视图的数量                                           21?

        Predict gaussian parameter maps
        """
        SB, N, _ = data['xyz'].shape # 1 65536
        NS = self.num_views_per_obj # 1
        # print("SB=",SB,", N=",N,", NS=",NS)

        canon_xyz = self.world_to_canonical(data['xyz'])    # [1,N,3], （从数值变成比例）min:-2.28, max:1.39
        # print("first canon_xyz.shape=[1,65536,128]",canon_xyz.shape)     # real [1,65536,3]   [2,16384,3]
        data['canon_xyz'] = canon_xyz # gt mask临时用一下
        # volumetric sampling 体积采样 
        point_latent = self.sample_in_canonical_voxel(canon_xyz, data['dec_fts']) # [bs, N, 128]->[bs, 128, N] bs是批次大小，N是体素的数量，128是特征维度。  # [2,16384,1] bs=2 N=128 128  
        # print("point_latent.shape=[1,65536,128]-----------",point_latent.shape) # real [1,65536,3]               # [2,16384,1] 应该是 [1,16384,128]
        point_latent = point_latent.reshape(-1, self.d_latent)  # (SB * NS * B, latent)  [N, 128] [65536,128]        N=256 [256*128]  
        # print("point_latent.shape=[65536,128]---------point没问题256*128",point_latent.shape,self.d_latent)
        # print("canon_xyz.shape=[1,65536,3]",canon_xyz.shape)     # 输出z_feature张量的形状    [2,16384,3] N=16384
        if self.use_xyz:    # True
            z_feature = canon_xyz.reshape(-1, 3)  # (SB*B, 3)   [1*65536,3]    将canon_xyz重塑为形状(SB*B, 3)的张量，其中每个元素包含3个坐标值。
        # print("z_feature.shape= before code  = [65536,3]   ",z_feature.shape)     # 输出z_feature张量的形状
        if self.use_code:    # True
            # Positional encoding (no viewdirs) 位置编码（无 viewdirs）
            z_feature = self.code(z_feature)    # [N, 39] [65536,3]->[65536,39]
        
        # ----
        # print("point_latent.shape=[65536,128]",point_latent.shape,"z_feature.shape=[65536,39]",z_feature.shape)     # 输出z_feature张量的形状
        latent = torch.cat((point_latent, z_feature), dim=-1) # [N, 128+39] [65536,167]

        # Camera frustum culling stuff, currently disabled
        #  相机视椎体 剔除功能，目前禁用
        combine_index = None
        dim_size = None
        # backbone
        # 传入多个参数以计算潜在特征，返回的第一个值是编码后的潜在变量，第二个值被忽略（用下划线表示）
        latent, _ = self.encoder(
            latent, # [65536,167]  这是zx（latent） 需要等于 d_latent + d_in
            combine_inner_dims=(self.num_views_per_obj, N),
            combine_index=combine_index,
            dim_size=dim_size,
            language_embed=data['lang'],
            batch_size=SB,
            )   # 26

        # print("forward in embednet latent.shape before reshape=[1,65536,26]",latent.shape) 
        latent = latent.reshape(-1, N, self.d_out)  # [1, N, d_out] [1,65536,26] 
        # print("forward in embednet latent.shape=[1,65536,26]",latent.shape)  # 输出latent张量的形状

        ## regress gaussian parms 回归高斯参数
        split_network_outputs = self.gs_parm_regresser(latent) # [1, N, (3, 1, 3, 4, 3, 9)]  [3, 1, 3, 4, 3, 3, 3, 9] 20 + 9 = 29( 26 + 3(mask) )
        # print("split_network_outputs = self[1,65536,26]",split_network_outputs.shape)  # 输出split_network_outputs张量的形状
        # self.split_dimensions_with_offset 是一个列表，指定了每个输出部分的维度。 split_network_outputs 沿最后一个维度（dim=-1）分割
        split_network_outputs = split_network_outputs.split(self.split_dimensions_with_offset, dim=-1)
        # 元组tuple print("split_network_outputs = split_network_outputs.split(self.split_dimensions_with_offset, dim=-1)",split_network_outputs.shape)  # 输出split_network_outputs张量的形状
        # 将上面这些 分割后的数据   进行分配
        xyz_maps, opacity_maps, scale_maps, rot_maps, features_dc_maps, feature_maps, mask_maps = split_network_outputs[:7] # 6]
        # xyz_maps, opacity_maps, scale_maps, rot_maps, features_dc_maps, feature_maps,  precomputed_mask = split_network_outputs[:7]
        if self.max_sh_degree > 0:
            features_rest_maps = split_network_outputs[7] # [6]

        # spherical function head 球面函数头
        features_dc_maps = features_dc_maps.unsqueeze(2) #.transpose(2, 1).contiguous().unsqueeze(2) # [B, H*W, 1, 3] [1, 65536, 1, 3]
        # print("features_dc_maps = ",features_dc_maps.shape)  # torch.Size([1, 65536, 1, 3])  输出features_dc_maps张量的形状
        features_rest_maps = features_rest_maps.reshape(*features_rest_maps.shape[:2], -1, 3) # [B, H*W, 3, 3] [1, 65536, 3, 3]
        # print("features_rest_maps = ",features_rest_maps.shape)  # [1, 65536, 3, 3]  输出features_rest_maps张量的形状
        sh_out = torch.cat([features_dc_maps, features_rest_maps], dim=2)  # [B, H*W, 4, 3]   [1, 65536, 1, 3]+ [1, 65536, 3, 3] = [1, 65536, 4, 3]
        # print("sh_out = [1, 65536, 4, 3]",sh_out.shape)  # 输出sh_out张量的形状

        scale_maps = self.scaling_activation(scale_maps)    # exp   [1, 65536, 3]
        # print("scale_maps = ",scale_maps.shape)  # 输出scale_maps张量的形状 [1, 65536, 3]
        scale_maps = torch.clamp_max(scale_maps, 0.05) # [1, 65536, 3] # 将 scale_maps 中的所有值限制在最大值为 0.05
        # print("scale_maps = ",scale_maps.shape)  # 输出scale_maps张量的形状 [1, 65536, 3]

        data['xyz_maps'] = data['xyz'] + xyz_maps   # [B, N, 3]           [1, 65536, 3]
        data['sh_maps'] = sh_out    # [B, N, 4, 3]                        [1, 65536, 4, 3]        
        data['rot_maps'] = self.rotation_activation(rot_maps, dim=-1)   # [1, 65536, 4]
        data['scale_maps'] = scale_maps                                 # [1, 65536, 3]
        data['opacity_maps'] = self.opacity_activation(opacity_maps)    # [1, 65536, 1]    
        data['feature_maps'] = feature_maps # [B, N, 3]                   [1, 65536, 3]
        data['mask_maps'] = mask_maps #self.mask_activation(mask_maps) #10.24  [B, N, 3]  [1, 65536, 3]
        # print(data['xyz_maps'].shape ,data['sh_maps'].shape,data['rot_maps'].shape,data['scale_maps'].shape,data['opacity_maps'].shape,data['feature_maps'].shape)
        # torch.Size([1, 65536, 3]) torch.Size([1, 65536, 4, 3]) torch.Size([1, 65536, 4]) torch.Size([1, 65536, 3]) torch.Size([1, 65536, 1]) torch.Size([1, 65536, 3])
        # data['mask_maps'] = data['mask_maps'] *100
        # Dynamic Modeling: predict next gaussian maps
        # 动态建模：预测下一个高斯映射
        # print("self.field_type = ",self.field_type)
        if self.use_dynamic_field: #and data['step'] >= self.warm_up:

            if self.use_mask: # LF 中的pre 和nonerf(需要mask)
                if not self.use_semantic_feature :
                    dyna_input = torch.cat((
                        point_latent,   # [N, 128]
                        data['xyz_maps'].detach().reshape(N, 3), 
                        features_dc_maps.detach().reshape(N, 3),
                        features_rest_maps.detach().reshape(N, 9),
                        data['rot_maps'].detach().reshape(N, 4),
                        data['scale_maps'].detach().reshape(N, 3),
                        data['opacity_maps'].detach().reshape(N, 1),
                        data['mask_maps'].detach().reshape(N, 3),
                        # d_in:
                        z_feature,  
                    ), dim=-1) # no batch dim    
                else:
                    dyna_input = torch.cat((
                        point_latent,   # [N, 128]
                        data['xyz_maps'].detach().reshape(N, 3), 
                        features_dc_maps.detach().reshape(N, 3),
                        features_rest_maps.detach().reshape(N, 9),
                        data['rot_maps'].detach().reshape(N, 4),
                        data['scale_maps'].detach().reshape(N, 3),
                        data['opacity_maps'].detach().reshape(N, 1),
                        data['mask_maps'].detach().reshape(N, 3),
                        data['feature_maps'].detach().reshape(N, 3), # （加入语义特征后）多了特征图
                        # d_in:
                        z_feature,  
                    ), dim=-1) # no batch dim      
            # 不用mask
            elif self.field_type == 'bimanual' or (self.field_type =='LF' and self.mask_gen != 'pre'):
                if not self.use_semantic_feature :
                    # dyna_input: (d_latent, d_in)
                    dyna_input = torch.cat((
                        point_latent,   # [N, 128]
                        data['xyz_maps'].detach().reshape(N, 3), 
                        features_dc_maps.detach().reshape(N, 3),
                        features_rest_maps.detach().reshape(N, 9),
                        data['rot_maps'].detach().reshape(N, 4),
                        data['scale_maps'].detach().reshape(N, 3),
                        data['opacity_maps'].detach().reshape(N, 1),
                        # d_in:
                        z_feature,
                    ), dim=-1) # no batch dim
                else:
                    # 用语义特征
                    dyna_input = torch.cat((
                        point_latent,   # [N, 128]
                        data['xyz_maps'].detach().reshape(N, 3), 
                        features_dc_maps.detach().reshape(N, 3),
                        features_rest_maps.detach().reshape(N, 9),
                        data['rot_maps'].detach().reshape(N, 4),
                        data['scale_maps'].detach().reshape(N, 3),
                        data['opacity_maps'].detach().reshape(N, 1),
                        data['feature_maps'].detach().reshape(N, 3), # （加入语义特征后）多了特征图
                        # d_in:
                        z_feature,  
                    ), dim=-1) # no batch dim
          
            else:
                print("error in models_embed.py")

            # voxel embedding, stop gradient (gaussian xyz), (128+39)+3=170
            # 体素嵌入，停止梯度（高斯 XYZ），（128+39）+3=170
            if self.field_type =='bimanual':
                if self.use_action:
                    dyna_input = torch.cat((dyna_input, data['action'].repeat(N, 1)), dim=-1)   # action detach
                    # cprint(f"dyna_input.shape: {dyna_input.shape}", "red") 

                next_split_network_outputs, _ = self.gs_deformation_field(
                    dyna_input,
                    combine_inner_dims=(self.num_views_per_obj, N),
                    combine_index=combine_index,
                    dim_size=dim_size,
                    language_embed=data['lang'],
                    batch_size=SB,
                    )
                next_xyz_maps, next_rot_maps = next_split_network_outputs.split([3, 4], dim=-1)
                
                data['next']['xyz_maps'] = data['xyz_maps'].detach() + next_xyz_maps
                data['next']['sh_maps'] = data['sh_maps'].detach()
                data['next']['rot_maps'] = self.rotation_activation(data['rot_maps'].detach() + next_rot_maps, dim=-1)
                data['next']['scale_maps'] = data['scale_maps'].detach()
                data['next']['opacity_maps'] = data['opacity_maps'].detach()
                data['next']['feature_maps'] = data['feature_maps'].detach()
            else:
                # -------------------------------------------------------------------------------------------------
                if self.mask_gen == 'pre':
                    if self.use_action:
                        dyna_input = torch.cat((dyna_input, data['right_action'].repeat(N, 1)), dim=-1)   # action detach [65536, 206]
                        # cprint(f"dyna_input.shape: {dyna_input.shape}", "red") # [65536, 206]

                        # with torch.no_grad():
                        next_split_network_outputs_leader, _ = self.gs_deformation_field_leader_smaller(
                            dyna_input,
                            combine_inner_dims=(self.num_views_per_obj, N),
                            combine_index=combine_index,
                            dim_size=dim_size,
                            language_embed=data['lang'],
                            batch_size=SB,
                        )
                        # 这部分要不要缩进呢
                        next_xyz_maps, next_rot_maps = next_split_network_outputs_leader.split([3, 4], dim=-1)   
                        data['right_next']['xyz_maps'] = data['xyz_maps'].detach() + next_xyz_maps    # [1, 65536, 3]?
                        # print("data['right_next']['xyz_maps'].shape: ", data['right_next']['xyz_maps'].shape) # [1, 65536, 3]
                        data['right_next']['sh_maps'] = data['sh_maps'].detach()
                        data['right_next']['rot_maps'] = self.rotation_activation(data['rot_maps'].detach() + next_rot_maps, dim=-1)  # [1, 65536, 4]
                        # print("data['right_next']['rot_maps'].shape: ", data['right_next']['rot_maps'].shape,"next_rot_maps",next_rot_maps) # torch.Size([1, 65536, 4]) next_rot_maps tensor([[[ 3.3136e-01, -7.9674e-01,  1.2806e+00,  3.1690e-01],很长的一列
                        data['right_next']['scale_maps'] = data['scale_maps'].detach()
                        data['right_next']['opacity_maps'] = data['opacity_maps'].detach()
                        data['right_next']['feature_maps'] = data['feature_maps'].detach()
                        data['right_next']['mask_maps'] = data['mask_maps'].detach()
                
                        # if self.use_action:
                        dyna_input = torch.cat((dyna_input, data['left_action'].repeat(N, 1)), dim=-1)   # action detach
                        # cprint(f"dyna_input.shape: {dyna_input.shape}", "red")

                        next_split_network_outputs_follower, _ = self.gs_deformation_field_follower(
                            dyna_input,
                            combine_inner_dims=(self.num_views_per_obj, N),
                            combine_index=combine_index,
                            dim_size=dim_size,
                            language_embed=data['lang'],
                            batch_size=SB,
                            )
                        next_xyz_maps, next_rot_maps = next_split_network_outputs_follower.split([3, 4], dim=-1)   
                        data['left_next']['xyz_maps'] = data['xyz_maps'].detach() + next_xyz_maps
                        data['left_next']['sh_maps'] = data['sh_maps'].detach()
                        data['left_next']['rot_maps'] = self.rotation_activation(data['rot_maps'].detach() + next_rot_maps, dim=-1)
                        data['left_next']['scale_maps'] = data['scale_maps'].detach()
                        data['left_next']['opacity_maps'] = data['opacity_maps'].detach()
                        data['left_next']['feature_maps'] = data['feature_maps'].detach()
                        data['left_next']['mask_maps'] = data['mask_maps'].detach()
                elif self.mask_gen == 'nonerf':
                    if self.use_action:
                        dyna_input = torch.cat((dyna_input, data['right_action'].repeat(N, 1)), dim=-1)   # action detach [65536, 206]
                        # cprint(f"dyna_input.shape: {dyna_input.shape}", "red") # [65536, 206]

                        # with torch.no_grad():
                        next_split_network_outputs_leader, _ = self.gs_deformation_field_leader_smaller(
                            dyna_input,
                            combine_inner_dims=(self.num_views_per_obj, N),
                            combine_index=combine_index,
                            dim_size=dim_size,
                            language_embed=data['lang'],
                            batch_size=SB,
                        )
                        # 这部分要不要缩进呢
                        next_xyz_maps, next_rot_maps = next_split_network_outputs_leader.split([3, 4], dim=-1)   
                        data['right_next']['xyz_maps'] = data['xyz_maps'].detach() + next_xyz_maps    # [1, 65536, 3]?
                        # print("data['right_next']['xyz_maps'].shape: ", data['right_next']['xyz_maps'].shape) # [1, 65536, 3]
                        data['right_next']['sh_maps'] = data['sh_maps'].detach()
                        data['right_next']['rot_maps'] = self.rotation_activation(data['rot_maps'].detach() + next_rot_maps, dim=-1)  # [1, 65536, 4]
                        # print("data['right_next']['rot_maps'].shape: ", data['right_next']['rot_maps'].shape,"next_rot_maps",next_rot_maps) # torch.Size([1, 65536, 4]) next_rot_maps tensor([[[ 3.3136e-01, -7.9674e-01,  1.2806e+00,  3.1690e-01],很长的一列
                        data['right_next']['scale_maps'] = data['scale_maps'].detach()
                        data['right_next']['opacity_maps'] = data['opacity_maps'].detach()
                        data['right_next']['feature_maps'] = data['feature_maps'].detach()
                        data['right_next']['mask_maps'] = data['mask_maps'].detach()
                
                        # if self.use_action:
                        dyna_input = torch.cat((dyna_input, data['left_action'].repeat(N, 1)), dim=-1)   # action detach
                        # cprint(f"dyna_input.shape: {dyna_input.shape}", "red")

                        next_split_network_outputs_follower, _ = self.gs_deformation_field_follower(
                            dyna_input,
                            combine_inner_dims=(self.num_views_per_obj, N),
                            combine_index=combine_index,
                            dim_size=dim_size,
                            language_embed=data['lang'],
                            batch_size=SB,
                            )
                        next_xyz_maps, next_rot_maps = next_split_network_outputs_follower.split([3, 4], dim=-1)   
                        data['left_next']['xyz_maps'] = data['xyz_maps'].detach() + next_xyz_maps
                        data['left_next']['sh_maps'] = data['sh_maps'].detach()
                        data['left_next']['rot_maps'] = self.rotation_activation(data['rot_maps'].detach() + next_rot_maps, dim=-1)
                        data['left_next']['scale_maps'] = data['scale_maps'].detach()
                        data['left_next']['opacity_maps'] = data['opacity_maps'].detach()
                        data['left_next']['feature_maps'] = data['feature_maps'].detach()
                        data['left_next']['mask_maps'] = data['mask_maps'].detach()
                
                elif self.mask_gen == 'gt':
                    if self.use_action:
                        dyna_input = torch.cat((dyna_input, data['right_action'].repeat(N, 1)), dim=-1)   # action detach [65536, 206]
                        # cprint(f"dyna_input.shape: {dyna_input.shape}", "red") # [65536, 206]

                        # with torch.no_grad():
                        next_split_network_outputs_leader, _ = self.gs_deformation_field_leader_smaller(
                            dyna_input,
                            combine_inner_dims=(self.num_views_per_obj, N),
                            combine_index=combine_index,
                            dim_size=dim_size,
                            language_embed=data['lang'],
                            batch_size=SB,
                        )
                        # 这部分要不要缩进呢
                        next_xyz_maps, next_rot_maps = next_split_network_outputs_leader.split([3, 4], dim=-1)   
                        data['right_next']['xyz_maps'] = data['xyz_maps'].detach() + next_xyz_maps    # [1, 65536, 3]?
                        # print("data['right_next']['xyz_maps'].shape: ", data['right_next']['xyz_maps'].shape) # [1, 65536, 3]
                        data['right_next']['sh_maps'] = data['sh_maps'].detach()
                        data['right_next']['rot_maps'] = self.rotation_activation(data['rot_maps'].detach() + next_rot_maps, dim=-1)  # [1, 65536, 4]
                        # print("data['right_next']['rot_maps'].shape: ", data['right_next']['rot_maps'].shape,"next_rot_maps",next_rot_maps) # torch.Size([1, 65536, 4]) next_rot_maps tensor([[[ 3.3136e-01, -7.9674e-01,  1.2806e+00,  3.1690e-01],很长的一列
                        data['right_next']['scale_maps'] = data['scale_maps'].detach()
                        data['right_next']['opacity_maps'] = data['opacity_maps'].detach()
                        data['right_next']['feature_maps'] = data['feature_maps'].detach()
                        # data['right_next']['mask_maps'] = data['mask_maps'].detach()
                
                        # if self.use_action:
                        dyna_input = torch.cat((dyna_input, data['left_action'].repeat(N, 1)), dim=-1)   # action detach
                        # cprint(f"dyna_input.shape: {dyna_input.shape}", "red")

                        next_split_network_outputs_follower, _ = self.gs_deformation_field_follower(
                            dyna_input,
                            combine_inner_dims=(self.num_views_per_obj, N),
                            combine_index=combine_index,
                            dim_size=dim_size,
                            language_embed=data['lang'],
                            batch_size=SB,
                            )
                        next_xyz_maps, next_rot_maps = next_split_network_outputs_follower.split([3, 4], dim=-1)   
                        data['left_next']['xyz_maps'] = data['xyz_maps'].detach() + next_xyz_maps
                        data['left_next']['sh_maps'] = data['sh_maps'].detach()
                        data['left_next']['rot_maps'] = self.rotation_activation(data['rot_maps'].detach() + next_rot_maps, dim=-1)
                        data['left_next']['scale_maps'] = data['scale_maps'].detach()
                        data['left_next']['opacity_maps'] = data['opacity_maps'].detach()
                        data['left_next']['feature_maps'] = data['feature_maps'].detach()
                        # data['left_next']['mask_maps'] = data['mask_maps'].detach()

                # -------------------------------------------------------------------------------------------------
        return data
    
