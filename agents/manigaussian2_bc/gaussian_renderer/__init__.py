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


def render(data, idx, pts_xyz, rotations, scales, opacity, bg_color, pts_rgb=None, features_color=None, features_language=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    features_language: the newly-added feature (Nxd)
    渲染场景。
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
    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['novel_view']['height'][idx]),
        image_width=int(data['novel_view']['width'][idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=data['novel_view']['world_view_transform'][idx],  # 视图矩阵  (用X*X的单位矩阵作为视图矩阵)
        projmatrix=data['novel_view']['full_proj_transform'][idx],   # 投影矩阵
        sh_degree=3 if features_color is None else 1,                # SH（球谐函数）的阶数
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
