import numpy as np
import cv2
from scipy.spatial import ConvexHull, Delaunay
import torch

def project_points_3d_to_2d(points_3d, intrinsic_matrix):
    """
    将 3D 点投影到 2D 图像平面上
    :param points_3d: (N, 3) numpy 数组, 3D 点云 (x, y, z)
    :param intrinsic_matrix: 3x3 相机内参矩阵
    :return: (N, 2) numpy 数组, 2D 图像坐标 (u, v)
    """
    # 提取 x, y, z
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    
    # 利用内参进行投影
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    u = (fx * x / z + cx).astype(np.int32)
    v = (fy * y / z + cy).astype(np.int32)
    
    return np.stack((u, v), axis=-1)

def mark_points_in_mask(points_3d, mask, intrinsic_matrix, depth_map):
    """
    no use
    在 mask 范围内标记 3D 点
    :param points_3d: (N, 3) numpy 数组, 3D 点云
    :param mask: (H, W) numpy 数组, 二值化的 mask
    :param intrinsic_matrix: 3x3 相机内参矩阵
    :param depth_map: (H, W) numpy 数组, 深度图
    :return: 标记后的图像
    """
    # 将 3D 点投影到 2D 图像上
    points_2d = project_points_3d_to_2d(points_3d, intrinsic_matrix)
    
    # 遍历每个点，判断是否在 mask 范围内
    for u, v in points_2d:
        if u >= 0 and u < mask.shape[1] and v >= 0 and v < mask.shape[0]:
            if mask[v, u] > 0:
                # 如果点在 mask 范围内，进行标记
                cv2.circle(mask, (u, v), 3, (255, 0, 0), -1)  # 用红色标记
    return mask

# 输出3d在2d上的映射
# marked_mask = mark_points_in_mask(points_3d, mask, intrinsic_matrix, depth_map)
# cv2.imshow('Marked Mask', marked_mask)
# cv2.waitKey(0)



def label_point_cloud(points, D, K, mask):
    """
    2D->3D
    根据mask标记三维点云中的点。

    参数:
    points : np.ndarray
        三维点云，形状为 (N, 3)，包含 (X, Y, Z)
    D : np.ndarray
        深度图，形状为 (高度, 宽度)
    K : np.ndarray
        相机内参矩阵，形状为 (1, 3, 3)(1无用)
    mask : np.ndarray
        二维mask图像,形状为 (高度, 宽度)

    返回:
    labeled_points : np.ndarray
        每个点的三维坐标 (X, Y, Z) 以及对应的label,形状为 (M, 4)
    """
    # print(f"K shape: {K.shape}    {K}")
    fx, fy = K[0, 0, 0].item(), K[0, 1, 1].item()
    cx, cy = K[0, 0, 2].item(), K[0, 1, 2].item()
    # print(K)

    labeled_points = []

    # print(f"points shape: {points.shape}")  # [1, 65536, 3] 改成了 [65536, 3]
    # for point in points:
    for X, Y, Z in points:
        # print(f"point shape: {point.shape}")    # [65536, 3]
        # X, Y, Z = point
        # 将三维点投影回二维图像坐标

        u = int((fx * X / Z) + cx)
        v = int((fy * Y / Z) + cy)

        # X, Y, Z: -0.9378253221511841, -2.7748615741729736, 1.1342734098434448, 418, 988
        print(f"X, Y, Z: {X}, {Y}, {Z}, {u}, {v}")
        # print(f"D[v, u] =  {D[v, u]}, mask[v, u] = {mask[v, u]}") u,v都超过界限了
        if 0 <= u < D.shape[1] and 0 <= v < D.shape[0]:
            depth = D[v, u]
            print("v,u = ",v,u)
            print(depth,mask[v,u])
            print(mask.shape)
            if depth > 0 and mask[v, u] > 0:  # 检查深度和mask
                label = mask[v, u]
                labeled_points.append((X, Y, Z, label))

    return np.array(labeled_points)  # 转换为numpy数组

# 调用方法
# points = np.array(...)  # 三维点云 (N, 3)
# D = np.array(...)       # 深度图
# K = np.array(...)       # 相机内参矩阵
# mask = np.array(...)    # 二维mask图像
# labeled_points = label_point_cloud(points, D, K, mask)


def project_3d_to_2d(points, K):
    """
    将三维点投影到二维平面。
    参数:
    points : np.ndarray  三维点，形状为 (N, 3->4)。 XYZ + Bool:是否在
    K : np.ndarray       相机内参矩阵，形状为 (3, 3)。
    返回:
    projected_points : np.ndarray   投影后的二维点，形状为 (N, 2)。
    """
    projected_points = []
    # for point in points:
    #     X, Y, Z = point
    # print(f"points shape: {points}{points.shape}") 
    for X, Y, Z, B in points:
        if B:
            u = (K[0, 0, 0] * X / Z) + K[0, 0, 2]  # x 坐标
            v = (K[0, 1, 1] * Y / Z) + K[0, 1, 2]  # y 坐标
            projected_points.append((u, v))
    
    return np.array(projected_points)


def depth_mask_to_3d(D, mask, K):
    """
    遍历深度图和mask图,将满足条件的二维点映射到三维空间。
    参数:
    D : np.ndarray
        深度图，形状为 (高度, 宽度)
    mask : np.ndarray
        二维mask图像,形状为 (高度, 宽度)，用于过滤点
    K : np.ndarray
        相机内参矩阵，形状为 (3, 3)
    返回:
    labeled_points : np.ndarray
        每个点的三维坐标 (X, Y, Z) 以及对应的mask标签, 形状为 (M, 4)
    """
    # 从相机内参矩阵中提取参数
    fx, fy = K[0, 0, 0].item(), K[0, 1, 1].item()
    cx, cy = K[0, 0, 2].item(), K[0, 1, 2].item()

    # 存储三维点及其对应的mask标签
    labeled_points = []

    # 遍历每个像素坐标
    # print(D,D.shape,D[0][0].shape) # [1,1,256,256]
    D = D[0][0]                     #[256,256]
    mask = mask[0][0]
    # 如果 D 和 mask 是张量，转为 NumPy 数组
    if isinstance(D, torch.Tensor):
        D = D.cpu().numpy()  # 转为 NumPy 数组
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()  # 转为 NumPy 数组

    height, width = D.shape
    for v in range(height):
        for u in range(width):
            depth = D[v, u]
            label = mask[v, u]          # 右53-73 左94-114

            # 如果mask中该点的值大于0，并且深度有效
            if label >= 94 and label <= 114 and depth > 0:
                # 通过公式将像素坐标 (u, v) 映射到三维空间
                Z = depth
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy

                # 将三维坐标及mask标签添加到结果中
                labeled_points.append((X, Y, Z)) #label就不用加了 , 1))
    labeled_points = np.array(labeled_points)
    return labeled_points  # 转换为numpy数组

# # 示例使用
# D = np.array([[1.5, 2.0], [3.0, 4.0]])  # 假设的深度图，形状为(2, 2)
# mask = np.array([[1, 0], [0, 1]])  # 假设的mask图，形状为(2, 2)
# K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])  # 相机内参矩阵

# labeled_points = depth_mask_to_3d(D, mask, K)

# print("三维坐标及标签:\n", labeled_points)



def points_inside_convex_hull(point_cloud, masked_points, remove_outliers=False, outlier_factor=1.0):
    # 原来应该是True但是格式不对，直接False试试
    """
    Given a point cloud and a mask indicating a subset of points, this function computes the convex hull of the
    subset of points and then identifies all points from the original point cloud that are inside this convex hull.
    给定一个点云和一个表示点子集的掩码，该函数会计算点子集的凸壳，然后从原始点云中识别出凸壳内的所有点。
    子集的凸壳，然后从原始点云中找出位于该凸壳内的所有点。
    Parameters:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the point cloud. 
    (N, 3) 的张量，表示 N 个 3D 点的点云数据
    - mask (torch.Tensor): A tensor of shape (N,) indicating the subset of points to be used for constructing the convex hull. 
    遮罩(torch.Tensor): 形状为 (N,) 的张量，表示用于构建凸壳的点的子集。
      (N,) 的张量，表示子集的 N 个 3D 点(通过某个掩码选择的点云子集)，用于计算凸包。?  
    - remove_outliers (bool): Whether to remove outliers from the masked points before computing the convex hull. Default is True.
       表示是否从子集点中移除离群点(outliers)。默认为 True,表示要移除离群点。
    - outlier_factor (float): The factor used to determine outliers based on the IQR method. Larger values will classify more points as outliers.
       用于确定离群点的因子，基于四分位距(IQR)方法。较大的值会将更多的点分类为离群点。     
    
    Returns:
    - inside_hull_tensor_mask (torch.Tensor): A mask of shape (N,) with values set to True for the points inside the convex hull
                                            and False otherwise.
     形状为 (N,) 的掩码，其中凸壳内部的点的值设置为 True,否则为 False。
    """

    # Remove outliers if the option is selected 如果选择该选项，则删除异常值
    if remove_outliers:                                                 # Default is True
        # 通过四分位距（IQR）方法删除异常值
        # 第25百分位数（Q1）表示有25%的数据小于或等于这个值，第50百分位数（Q2或中位数）表示有50%的数据小于或等于这个值
        Q1 = np.percentile(masked_points, 0, axis=0)
        Q3 = np.percentile(masked_points, 80, axis=0)                   # 计算子集点的第 0 和第 80 百分位数（Q1 和 Q3），这些代表四分位距的范围。
        IQR = Q3 - Q1
        # Q1 = torch.tensor(Q1).to(device)
        # Q3 = torch.tensor(Q3).to(device)
        # IQR = torch.tensor(IQR).to(device)
        print("masked_points", masked_points, Q1)
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR)) # outlier_factor=1 应该原来0-80%的值，去搞到 -80%的值 160%以外的值去检测异常值
        # mask去除的方式
        # filtered_masked_points：应用掩码，移除所有被标记为异常值的点。np.any(outlier_mask, axis=1) 返回一个布尔数组，表示在每个点的行中是否存在异常值。
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points

    # Compute the Delaunay triangulation of the filtered masked points
    #  计算过滤后的遮罩点的 Delaunay 三角剖分
    # 通过构建三角形网格，连接一组点形成凸包（Convex Hull）
    delaunay = Delaunay(filtered_masked_points)

    # Determine which points from the original point cloud are inside the convex hull
    #  确定原始点云中哪些点位于凸壳内 find_simplex 返回一个整数数组，表示每个点所在的简单单元的索引。如果某个点不在凸包内，返回的索引为 -1，因此 >= 0 表示在凸包内的点。
    points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0
    # points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) <= 0 # 小于0这样子说明排除了左手点
    points_inside_hull_mask = torch.cat([point_cloud, torch.tensor(points_inside_hull_mask, device=point_cloud.device).unsqueeze(1)], dim=1)
    # Convert the numpy mask back to a torch tensor and return
    # 将 points_inside_hull_mask（一个 numpy 数组）转换为 torch 张量，并将其放到 GPU 上。
    inside_hull_tensor_mask = torch.tensor(points_inside_hull_mask, device="cuda")

    return inside_hull_tensor_mask


def create_2d_mask_from_convex_hull(points_2d, shape):
    """
    创建二维掩码。
    
    参数:
    points_2d : np.ndarray
        投影后的二维点，形状为 (M, 2)。
    shape : tuple
        掩码的形状 (高度, 宽度)。
    
    返回:
    mask : np.ndarray
        二维掩码，形状为 (高度, 宽度)。
    """
    # mask = np.zeros(shape, dtype=np.uint8)
    mask = torch.ones(shape, dtype=torch.uint8)
    # 创建 Delaunay 三角剖分
    print("points_2d",points_2d,points_2d.shape)
    if points_2d.shape[0] ==0:
        return mask
    else:
        delaunay = Delaunay(points_2d)
        
        # 填充掩码
        for x in range(shape[1]):
            for y in range(shape[0]):
                if delaunay.find_simplex((x, y)) >= 0:
                    mask[y, x] = 0  # 在掩码中标记
        
        return mask

