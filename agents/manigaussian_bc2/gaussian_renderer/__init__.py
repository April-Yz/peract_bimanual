#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import numpy as np

# def mask_to_rgb(mask):
#     """Convert a single channel mask to an RGB image."""
#     # mask is a single channel encoded as R + G * 256 + B * 256 * 256
#     r = mask % 256
#     g = (mask // 256) % 256
#     b = (mask // (256 * 256)) % 256
#     print("mask",mask)
#     print("mask.shape",mask.shape) # [655366, 3]
#     print(r,g,b)
#     # Stack R, G, B channels to get RGB image
#     rgb_image = np.stack([r, g, b], axis=-1)
#     return rgb_image

def render(data, idx, pts_xyz, rotations, scales, opacity, bg_color, pts_rgb=None, features_color=None, features_language=None):
    """
    Render the scene.     
    Background tensor (bg_color) must be on GPU!
    features_language: the newly-added feature (Nxd)
    背景张量 (bg_color) 必须使用 GPU!
    features_language:新添加的功能(Nxd)
    """
    device = pts_xyz.device
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建零张量。我们将用它来使 pytorch 返回二维（屏幕空间）手段的梯度
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # 光栅化设置：根据给定的视角数据计算光栅化所需的参数
    tanfovx = math.tan(data['novel_view']['FovX'][idx] * 0.5)
    tanfovy = math.tan(data['novel_view']['FovY'][idx] * 0.5)

    # 用于存储高斯渲染器的设置参数。
    # 包括图像的尺寸、焦距、背景张量、缩放修正因子、视图矩阵、投影矩阵、球谐函数阶数、相机位置以及调试模式
    # 设置光栅化的配置，包括图像的大小、视场的 tan 值、背景颜色、视图矩阵、投影矩阵等。
    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['novel_view']['height'][idx]),
        image_width=int(data['novel_view']['width'][idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=data['novel_view']['world_view_transform'][idx],  # 视图矩阵  (用X*X的单位矩阵作为视图矩阵)
        projmatrix=data['novel_view']['full_proj_transform'][idx],   # 投影矩阵
        sh_degree=3 if features_color is None else 1,                # SH（球谐函数）的阶数 （1）
        campos=data['novel_view']['camera_center'][idx],             # 相机位置
        prefiltered=False,                                           # 是否进行预过滤
        debug=False,
        include_feature=(features_language is not None),
    )

    # 用于实现高斯渲染器的功能：这个类的主要作用是封装了高斯渲染器的逻辑，包括可见性标记和前向渲染。
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # If precomputed colors are provided, use them. Otherwise, SH -> RGB conversion will be done by rasterizer.
    # 如果提供了预计算颜色，则使用它们。否则，光栅将进行 SH -> RGB 转换。
    shs = None
    colors_precomp = None
    # shs后续有用 颜色特征？（不应该是旋转和移动有用）
    if features_color is not None:  # default: None, use SHs
        shs = features_color
        # print("shs.shape",shs.shape) # [65536, 4, 3]
    else:
        assert pts_rgb is not None
        colors_precomp = pts_rgb
        # print("pts_rgb.shape",pts_rgb.shape)

    # 语义特征    
    if features_language is not None:
        MIN_DENOMINATOR = 1e-12
        language_feature_precomp = features_language    # [N, 3]
        language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + MIN_DENOMINATOR)
    else:
        # FIXME: other dimension choices may cause "illegal memory access"
        language_feature_precomp = torch.zeros((opacity.shape[0], 3), dtype=opacity.dtype, device=opacity.device)   

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # 将可见高斯光栅化到图像上，获取它们的半径（在屏幕上）。 在这里处理了？
    rendered_image, language_feature_image, radii = rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        language_feature_precomp=language_feature_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # 那些被剔除或半径为 0 的高斯不可见。
    # 它们将被排除在分割标准中使用的值更新之外。
    return {
        "render": rendered_image,   # cuda
        "render_embed": language_feature_image,
        "viewspace_points": screenspace_points,
        "radii": radii,
        }
def render1(data, idx, pts_xyz, rotations, scales, opacity, bg_color, pts_rgb=None, features_color=None, features_language=None):
    """
    Render the scene.     
    Background tensor (bg_color) must be on GPU!
    features_language: the newly-added feature (Nxd)
    背景张量 (bg_color) 必须使用 GPU!
    features_language:新添加的功能(Nxd)
    """
    device = pts_xyz.device
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建零张量。我们将用它来使 pytorch 返回二维（屏幕空间）手段的梯度
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # 光栅化设置：根据给定的视角数据计算光栅化所需的参数
    tanfovx = math.tan(data['mask_view']['FovX'][idx] * 0.5)
    tanfovy = math.tan(data['mask_view']['FovY'][idx] * 0.5)

    # 用于存储高斯渲染器的设置参数。
    # 包括图像的尺寸、焦距、背景张量、缩放修正因子、视图矩阵、投影矩阵、球谐函数阶数、相机位置以及调试模式
    # 设置光栅化的配置，包括图像的大小、视场的 tan 值、背景颜色、视图矩阵、投影矩阵等。
    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['mask_view']['height'][idx]),
        image_width=int(data['mask_view']['width'][idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=data['mask_view']['world_view_transform'][idx],  # 视图矩阵  (用X*X的单位矩阵作为视图矩阵)
        projmatrix=data['mask_view']['full_proj_transform'][idx],   # 投影矩阵
        sh_degree=3 if features_color is None else 1,                # SH（球谐函数）的阶数 （1）
        campos=data['mask_view']['camera_center'][idx],             # 相机位置
        prefiltered=False,                                           # 是否进行预过滤
        debug=False,
        include_feature=(features_language is not None),
    )

    # 用于实现高斯渲染器的功能：这个类的主要作用是封装了高斯渲染器的逻辑，包括可见性标记和前向渲染。
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # If precomputed colors are provided, use them. Otherwise, SH -> RGB conversion will be done by rasterizer.
    # 如果提供了预计算颜色，则使用它们。否则，光栅将进行 SH -> RGB 转换。
    shs = None
    colors_precomp = None
    # shs后续有用 颜色特征？（不应该是旋转和移动有用）
    if features_color is not None:  # default: None, use SHs
        shs = features_color
        # print("shs.shape",shs.shape) # [65536, 4, 3]
    else:
        assert pts_rgb is not None
        colors_precomp = pts_rgb
        # print("pts_rgb.shape",pts_rgb.shape)

    # 语义特征    
    if features_language is not None:
        MIN_DENOMINATOR = 1e-12
        language_feature_precomp = features_language    # [N, 3]
        language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + MIN_DENOMINATOR)
    else:
        # FIXME: other dimension choices may cause "illegal memory access"
        language_feature_precomp = torch.zeros((opacity.shape[0], 3), dtype=opacity.dtype, device=opacity.device)   

    # language_feature_precomp = language_feature_precomp/2 + 0.5
    print("language_feature_precomp = ",language_feature_precomp.shape,language_feature_precomp)
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # 将可见高斯光栅化到图像上，获取它们的半径（在屏幕上）。 在这里处理了？
    rendered_image, language_feature_image, radii = rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        language_feature_precomp=language_feature_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # 那些被剔除或半径为 0 的高斯不可见。
    # 它们将被排除在分割标准中使用的值更新之外。
    return {
        "render": rendered_image,   # cuda
        "render_embed": language_feature_image,
        "viewspace_points": screenspace_points,
        "radii": radii,
        }



# new from mask 新增 precomputed_mask = None
def render_mask(data, idx, pts_xyz, rotations, scales, opacity, bg_color, pts_rgb=None, features_color=None, features_language=None,precomputed_mask = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    features_language: the newly-added feature (Nxd)
    new add mask 
    渲染场景。
    背景张量 (bg_color) 必须使用 GPU!
    features_language:新添加的功能(Nxd)
    """
    device = pts_xyz.device
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建零张量。我们将用它来使 pytorch 返回二维（屏幕空间）手段的梯度
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device=device) + 0
    # screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.long, requires_grad=True, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # 光栅化设置：根据给定的视角数据计算光栅化所需的参数
    # 计算相机的水平方向和垂直方向的视场角（FOV）的切线，用于设置透视投影矩阵。
    tanfovx = math.tan(data['mask_view']['FovX'][idx] * 0.5)
    tanfovy = math.tan(data['mask_view']['FovY'][idx] * 0.5)

    # 用于存储高斯渲染器的设置参数。包括图像的尺寸、焦距、背景张量、缩放修正因子、视图矩阵、投影矩阵、球谐函数阶数、相机位置以及调试模式设置光栅化的配置，包括图像的大小、视场的 tan 值、背景颜色、视图矩阵、投影矩阵等。相当于render得到的图片配置
    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['mask_view']['height'][idx]),
        image_width=int(data['mask_view']['width'][idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0, #? 这边虽然也是，但是下面会有预计算的协方差 scaling_modifier =1.0
        viewmatrix=data['mask_view']['world_view_transform'][idx],  # 视图矩阵  (用X*X的单位矩阵作为视图矩阵)
        projmatrix=data['mask_view']['full_proj_transform'][idx],   # 投影矩阵
        sh_degree=3 if features_color is None else 1,              #?  # SH（球谐函数）的阶数
        campos=data['mask_view']['camera_center'][idx],             # 相机位置
        prefiltered=False,                                           # 是否进行预过滤
        debug=False,                                                 #?  debug=pipe.debug（不清楚传递的是什么）  
        include_feature=(features_language is not None),             # ?SAGA中没有   
    )

    # 用于实现高斯渲染器的功能：这个类的主要作用是封装了高斯渲染器的逻辑，包括可见性标记和前向渲染。
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # ############################for mask################################
    # 获取高斯模型的遮罩数据。如果传入了 precomputed_mask，则直接使用预计算的遮罩。对于一维或单通道遮罩，进行维度扩展，使其成为三通道（RGB）。
    mask = precomputed_mask # [65536, 3]
    # mask = pc.get_mask if precomputed_mask is None else precomputed_mask

    # 原来的方法print("0 mask.shape",mask.shape) # [N=256 *256 , 3]if len(mask.shape) == 1 or mask.shape[-1] == 1: # [W, H, 1] -> [W, H, 3] repeat([1,3])最后一维复制三  # mask = mask.squeeze().unsqueeze(-1).repeat([1,3]).cuda()预想的方法，但是好像不对，格式 [65536,3] print("in render mask.shape",mask.shape)
    # target：[h,w,3]
    mask = torch.tensor(mask).cuda()
    # if len(mask.shape) == 2:  
    mask = mask.reshape(256,256,3)
        # mask =mask.unsqueeze(1)
    # print("in gen mask.shape",mask.shape)
        # mask = mask.reshape(128,128,3)# elif (len(mask.shape) == 3 and mask.shape[-1] == 1):    # mask = mask.repeat([1,3])    # mask = mask.squeeze()  # 移除多余的维度    # mask = mask_to_rgb(mask)  # 使用之前定义的函数转换为RGB
    # print("1 mask.shape",mask.shape)mask = mask.reshape(256,256,3)print("2 mask.shape",mask.shape)将 mask 移动到 GPU 上mask = torch.tensor(mask).cuda()
    # ############################for mask################################
    # If precomputed colors are provided, use them. Otherwise, SH -> RGB conversion will be done by rasterizer.
    # 如果提供了预计算颜色，则使用它们。否则，光栅将进行 SH -> RGB 转换。
    shs = None
    colors_precomp = None
    # shs后续有用 颜色特征？（不应该是旋转和移动有用）
    # default: use mask ？？！！ 好像也不对，传入的时候好像有数据，丸辣
    if mask is not None:
        colors_precomp = mask  # / 2 + 0.5
        # shs = mask# elif features_color is not None:  # default: None, use SHs 选择相信这个default 来更改代码#     shs = features_color# else:#     assert pts_rgb is not None#     colors_precomp = pts_rgb

    # 语义特征    
    if features_language is not None:
        MIN_DENOMINATOR = 1e-12
        language_feature_precomp = features_language    # [N, 3]
        language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + MIN_DENOMINATOR)
    else:
        # FIXME: other dimension choices may cause "illegal memory access"
        language_feature_precomp = torch.zeros((opacity.shape[0], 3), dtype=opacity.dtype, device=opacity.device)   

    # # 概率性重新缩放 #  from openGaussian
    # prob = torch.rand(1)
    # rescale_factor = torch.tensor(1.0, dtype=torch.float32).cuda()
    # # if prob > 0.5: #  and rescale:(default true 在stage2中采用)
    #     # rescale_factor = torch.rand(1).cuda()  # 随机缩放因子

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # 将可见高斯光栅化到图像上，获取它们的半径（在屏幕上）。 在这里处理了？
    # rendered_image, language_feature_image, radii = rasterizer(
    rendered_mask, language_feature_image, radii = rasterizer(
        means3D=pts_xyz,                    # 3D点的坐标
        means2D=screenspace_points,         # 2D点的坐标
        shs=None, # shs,
        colors_precomp=colors_precomp,   #?
        language_feature_precomp=language_feature_precomp,
        opacities=opacity, 
        scales=scales, # * rescale_factor,
        rotations=rotations,
        cov3D_precomp=None,
        )
    # print("3 rendered_mask.shape",rendered_mask.shape) # [3, 128, 128]
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.  They will be excluded from value updates used in the splitting criteria. 那些被剔除或半径为 0 的高斯不可见。它们将被排除在分割标准中使用的值更新之外。
    return {
        "mask": rendered_mask,   # cuda
        "render_embed": language_feature_image,
        "viewspace_points": screenspace_points,
        "radii": radii,
        }

# new from mask 新增 precomputed_mask = None(专门为了左臂时的相机姿态设定)
def render_mask_gen(data, idx, pts_xyz, rotations, scales, opacity, bg_color, pts_rgb=None, features_color=None, features_language=None,precomputed_mask = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    features_language: the newly-added feature (Nxd)
    new add mask 
    渲染场景。
    背景张量 (bg_color) 必须使用 GPU!
    features_language:新添加的功能(Nxd)
    """
    device = pts_xyz.device
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建零张量。我们将用它来使 pytorch 返回二维（屏幕空间）手段的梯度
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device=device) + 0
    # screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.long, requires_grad=True, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # 光栅化设置：根据给定的视角数据计算光栅化所需的参数
    # 计算相机的水平方向和垂直方向的视场角（FOV）的切线，用于设置透视投影矩阵。
    tanfovx = math.tan(data['novel_view']['FovX'][idx] * 0.5)
    tanfovy = math.tan(data['novel_view']['FovY'][idx] * 0.5)

    # 用于存储高斯渲染器的设置参数。
    # 包括图像的尺寸、焦距、背景张量、缩放修正因子、视图矩阵、投影矩阵、球谐函数阶数、相机位置以及调试模式
    # 设置光栅化的配置，包括图像的大小、视场的 tan 值、背景颜色、视图矩阵、投影矩阵等。
    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['novel_view']['height'][idx]),
        image_width=int(data['novel_view']['width'][idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0, #? 这边虽然也是，但是下面会有预计算的协方差 scaling_modifier =1.0
        viewmatrix=data['novel_view']['world_view_transform'][idx],  # 视图矩阵  (用X*X的单位矩阵作为视图矩阵)
        projmatrix=data['novel_view']['full_proj_transform'][idx],   # 投影矩阵
        sh_degree=3 if features_color is None else 1,              #?  # SH（球谐函数）的阶数
        campos=data['novel_view']['camera_center'][idx],             # 相机位置
        prefiltered=False,                                           # 是否进行预过滤
        debug=False,                                                 #?  debug=pipe.debug（不清楚传递的是什么）  
        include_feature=(features_language is not None),             # ?SAGA中没有   
    )

    # 用于实现高斯渲染器的功能：这个类的主要作用是封装了高斯渲染器的逻辑，包括可见性标记和前向渲染。
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # ############################for mask################################

    # 获取高斯模型的遮罩数据。如果传入了 precomputed_mask，则直接使用预计算的遮罩。对于一维或单通道遮罩，进行维度扩展，使其成为三通道（RGB）。
    mask = precomputed_mask
    # mask = pc.get_mask if precomputed_mask is None else precomputed_mask

    # 原来的方法
    # if len(mask.shape) == 1 or mask.shape[-1] == 1: # [W, H, 1] -> [W, H, 3] repeat([1,3])最后一维复制三次
        # mask = mask.squeeze().unsqueeze(-1).repeat([1,3]).cuda()
    # 预想的方法，但是好像不对，格式 [65536,3] 
    # if len(mask.shape) == 2 or (len(mask.shape) == 3 and mask.shape[-1] == 1):
    #     mask = mask.squeeze()  # 移除多余的维度
    #     mask = mask_to_rgb(mask)  # 使用之前定义的函数转换为RGB

    mask = mask.reshape(256,256,3)
    # mask =mask.unsqueeze(1)
    # print("in gen mask.shape",mask.shape)
    # mask = mask.reshape(128,128,3)
    # print("in gen mask.shape",mask.shape)

    # 将 mask 移动到 GPU 上
    # mask = torch.tensor(mask).cuda()
    mask = mask.clone().detach().cuda()
    # ############################for mask################################

    # If precomputed colors are provided, use them. Otherwise, SH -> RGB conversion will be done by rasterizer.
    # 如果提供了预计算颜色，则使用它们。否则，光栅将进行 SH -> RGB 转换。
    shs = None
    colors_precomp = None
    # shs后续有用 颜色特征？（不应该是旋转和移动有用）
    # default: use mask ？？！！ 好像也不对，传入的时候好像有数据，丸辣
    if mask is not None:
        colors_precomp = mask
    elif features_color is not None:  # default: None, use SHs 选择相信这个default 来更改代码
        shs = features_color
    else:
        assert pts_rgb is not None
        colors_precomp = pts_rgb

    # 语义特征    
    if features_language is not None:
        MIN_DENOMINATOR = 1e-12
        language_feature_precomp = features_language    # [N, 3]
        language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + MIN_DENOMINATOR)
    else:
        # FIXME: other dimension choices may cause "illegal memory access"
        language_feature_precomp = torch.zeros((opacity.shape[0], 3), dtype=opacity.dtype, device=opacity.device)   

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # 将可见高斯光栅化到图像上，获取它们的半径（在屏幕上）。 在这里处理了？
    # rendered_image, language_feature_image, radii = rasterizer(
    rendered_mask, language_feature_image, radii = rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,   #?
        language_feature_precomp=language_feature_precomp,
        opacities=opacity, 
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # 那些被剔除或半径为 0 的高斯不可见。
    # 它们将被排除在分割标准中使用的值更新之外。
    return {
        "mask": rendered_mask,   # cuda
        "render_embed": language_feature_image,
        "viewspace_points": screenspace_points,
        "radii": radii,
        }

def render_rgb(data, idx, pts_xyz, rotations, scales, opacity, bg_color, pts_rgb=None, features_color=None, features_language=None):
    """
    Render the scene.     
    Background tensor (bg_color) must be on GPU!
    features_language: the newly-added feature (Nxd)
    背景张量 (bg_color) 必须使用 GPU!
    features_language:新添加的功能(Nxd)
    """
    device = pts_xyz.device
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建零张量。我们将用它来使 pytorch 返回二维（屏幕空间）手段的梯度
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # 光栅化设置：根据给定的视角数据计算光栅化所需的参数
    tanfovx = math.tan(data['mask_view']['FovX'][idx] * 0.5)
    tanfovy = math.tan(data['mask_view']['FovY'][idx] * 0.5)

    # 用于存储高斯渲染器的设置参数。
    # 包括图像的尺寸、焦距、背景张量、缩放修正因子、视图矩阵、投影矩阵、球谐函数阶数、相机位置以及调试模式
    # 设置光栅化的配置，包括图像的大小、视场的 tan 值、背景颜色、视图矩阵、投影矩阵等。
    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['mask_view']['height'][idx]),
        image_width=int(data['mask_view']['width'][idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=data['mask_view']['world_view_transform'][idx],  # 视图矩阵  (用X*X的单位矩阵作为视图矩阵)
        projmatrix=data['mask_view']['full_proj_transform'][idx],   # 投影矩阵
        sh_degree=3 if features_color is None else 1,                # SH（球谐函数）的阶数 （1）
        campos=data['mask_view']['camera_center'][idx],             # 相机位置
        prefiltered=False,                                           # 是否进行预过滤
        debug=False,
        include_feature=(features_language is not None),
    )

    # 用于实现高斯渲染器的功能：这个类的主要作用是封装了高斯渲染器的逻辑，包括可见性标记和前向渲染。
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # If precomputed colors are provided, use them. Otherwise, SH -> RGB conversion will be done by rasterizer.
    # 如果提供了预计算颜色，则使用它们。否则，光栅将进行 SH -> RGB 转换。
    shs = None
    colors_precomp = None
    # shs后续有用 颜色特征？（不应该是旋转和移动有用）
    if features_color is not None:  # default: None, use SHs
        shs = features_color
        # print("shs.shape",shs.shape) # [65536, 4, 3]
    else:
        assert pts_rgb is not None
        colors_precomp = pts_rgb
        # print("pts_rgb.shape",pts_rgb.shape)

    # 语义特征    
    if features_language is not None:
        MIN_DENOMINATOR = 1e-12
        language_feature_precomp = features_language    # [N, 3]
        language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + MIN_DENOMINATOR)
    else:
        # FIXME: other dimension choices may cause "illegal memory access"
        language_feature_precomp = torch.zeros((opacity.shape[0], 3), dtype=opacity.dtype, device=opacity.device)   

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # 将可见高斯光栅化到图像上，获取它们的半径（在屏幕上）。 在这里处理了？
    rendered_image, language_feature_image, radii = rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        language_feature_precomp=language_feature_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # 那些被剔除或半径为 0 的高斯不可见。
    # 它们将被排除在分割标准中使用的值更新之外。
    return {
        "render": rendered_image,   # cuda
        "render_embed": language_feature_image,
        "viewspace_points": screenspace_points,
        "radii": radii,
        }