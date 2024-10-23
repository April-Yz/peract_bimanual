import os
import torch

import torch.nn.functional as F
import torchvision.transforms.functional as func


from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import groundingdino.datasets.transforms as T
from torchvision.ops import box_convert
from PIL import Image

def image_transform(image) -> torch.Tensor:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(image, None)
    return image_transformed


def grounding_dino_prompt(image, text):
    
    image_tensor = image_transform(Image.fromarray(image))
    model_root = '/data1/zjyang/program/peract_bimanual/third_part/Grounded-Segment-Anything/'
    # /data1/zjyang/program/test/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
    # /data1/zjyang/program/peract_bimanual/third_part/GroundingDINO/weights/groundingdino_swint_ogc.pth
    # 但是这里的map location是cpu
    model = load_model(os.path.join(model_root, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"), \
                       os.path.join(model_root, "weights/groundingdino_swint_ogc.pth"))
    
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    
    h, w, _ = image.shape
    print("boxes device", boxes.device)
    boxes = boxes * torch.Tensor([w, h, w, h]).to(boxes.device)
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    print(xyxy)
    return xyxy

def porject_to_2d(viewpoint_camera, points3D):
    full_matrix = viewpoint_camera.full_proj_transform  # w2c @ K 
    # project to image plane
    points3D = F.pad(input=points3D, pad=(0, 1), mode='constant', value=1)
    p_hom = (points3D @ full_matrix).transpose(0, 1)  # N, 4 -> 4, N   -1 ~ 1
    p_w = 1.0 / (p_hom[-1, :] + 0.0000001)
    p_proj = p_hom[:3, :] * p_w

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width

    point_image = 0.5 * ((p_proj[:2] + 1) * torch.tensor([w, h]).unsqueeze(-1).to(p_proj.device) - 1) # image plane
    point_image = point_image.detach().clone()
    point_image = torch.round(point_image.transpose(0, 1))

    return point_image

## assume obtain 2d convariance matrx: N, 2, 2
def compute_ratios(conv_2d, points_xy, indices_mask, sam_mask, h, w):
    means = points_xy[indices_mask]
    # 计算特征值和特征向量
    eigvals, eigvecs = torch.linalg.eigh(conv_2d)
    # 判断长轴
    max_eigval, max_idx = torch.max(eigvals, dim=1)
    max_eigvec = torch.gather(eigvecs, dim=1, 
                        index=max_idx.unsqueeze(1).unsqueeze(2).repeat(1,1,2)) # (N, 1, 2)最大特征向量
    # 3 sigma，计算两个顶点的坐标
    long_axis = torch.sqrt(max_eigval) * 3
    max_eigvec = max_eigvec.squeeze(1)
    max_eigvec = max_eigvec / torch.norm(max_eigvec, dim=1).unsqueeze(-1)
    vertex1 = means + 0.5 * long_axis.unsqueeze(1) * max_eigvec
    vertex2 = means - 0.5 * long_axis.unsqueeze(1) * max_eigvec
    vertex1 = torch.clip(vertex1, torch.tensor([0, 0]).to(points_xy.device), torch.tensor([w-1, h-1]).to(points_xy.device))
    vertex2 = torch.clip(vertex2, torch.tensor([0, 0]).to(points_xy.device), torch.tensor([w-1, h-1]).to(points_xy.device))
    # 得到每个gaussian顶点的label
    vertex1_xy = torch.round(vertex1).long()
    vertex2_xy = torch.round(vertex2).long()
    vertex1_label = sam_mask[vertex1_xy[:, 1], vertex1_xy[:, 0]]
    vertex2_label = sam_mask[vertex2_xy[:, 1], vertex2_xy[:, 0]]
    # 得到需要调整gaussian的索引  还有一种情况 中心在mask内，但是两个端点在mask以外
    index = torch.nonzero(vertex1_label ^ vertex2_label, as_tuple=True)[0]
    # special_index = (vertex1_label == 0) & (vertex2_label == 0)
    # index = torch.cat((index, special_index), dim=0)
    selected_vertex1_xy = vertex1_xy[index]
    selected_vertex2_xy = vertex2_xy[index]
    # 找到2D 需要平移的方向, 用一个符号函数，1表示沿着特征向量方向，-1表示相反
    sign_direction = vertex1_label[index] - vertex2_label[index]
    direction_vector = max_eigvec[index] * sign_direction.unsqueeze(-1)

    # 两个顶点连线上的像素点
    ratios = []
    update_index = []
    for k in range(len(index)):
        x1, y1 = selected_vertex1_xy[k]
        x2, y2 = selected_vertex2_xy[k]
        # print(k, x1, x2)
        if x1 < x2:
            x_point = torch.arange(x1, x2+1).to(points_xy.device)
            y_point = y1 + (y2- y1) / (x2- x1) * (x_point - x1)
        elif x1 < x2:
            x_point = torch.arange(x2, x1+1).to(points_xy.device)
            y_point = y1 + (y2- y1) / (x2- x1) * (x_point - x1)
        else:
            if y1 < y2:
                y_point = torch.arange(y1, y2+1).to(points_xy.device)
                x_point = torch.ones_like(y_point) * x1
            else:
                y_point = torch.arange(y2, y1+1).to(points_xy.device)
                x_point = torch.ones_like(y_point) * x1
        
        x_point = torch.round(torch.clip(x_point, 0, w-1)).long()
        y_point = torch.round(torch.clip(y_point, 0, h-1)).long()
        # print(x_point.max(), y_point.max())
        # 判断连线上的像素是否在sam mask之内, 计算所占比例
        in_mask = sam_mask[y_point, x_point]
        ratios.append(sum(in_mask) / len(in_mask))

    ratios = torch.tensor(ratios)
    # 在3D Gaussian中对这些gaussians做调整，xyz和scaling
    index_in_all = indices_mask[index]

    return index_in_all, ratios, direction_vector

import math

def compute_conv3d(conv3d):
    complete_conv3d = torch.zeros((conv3d.shape[0], 3, 3))
    complete_conv3d[:, 0, 0] = conv3d[:, 0]
    complete_conv3d[:, 1, 0] = conv3d[:, 1]
    complete_conv3d[:, 0, 1] = conv3d[:, 1]
    complete_conv3d[:, 2, 0] = conv3d[:, 2]
    complete_conv3d[:, 0, 2] = conv3d[:, 2]
    complete_conv3d[:, 1, 1] = conv3d[:, 3]
    complete_conv3d[:, 2, 1] = conv3d[:, 4]
    complete_conv3d[:, 1, 2] = conv3d[:, 4]
    complete_conv3d[:, 2, 2] = conv3d[:, 5]

    return complete_conv3d

def conv2d_matrix(gaussians, viewpoint_camera, indices_mask, device):
    # 3d convariance matrix
    conv3d = gaussians.get_covariance(scaling_modifier=1)[indices_mask]
    conv3d_matrix = compute_conv3d(conv3d).to(device)

    w2c = viewpoint_camera.world_view_transform
    mask_xyz = gaussians.get_xyz[indices_mask]
    pad_mask_xyz = F.pad(input=mask_xyz, pad=(0, 1), mode='constant', value=1)
    t = pad_mask_xyz @ w2c[:, :3]   # N, 3
    height = viewpoint_camera.image_height
    width = viewpoint_camera.image_width
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_x = width / (2.0 * tanfovx)
    focal_y = height / (2.0 * tanfovy)
    lim_xy = torch.tensor([1.3 * tanfovx, 1.3 * tanfovy]).to(device)
    t[:, :2] = torch.clip(t[:, :2] / t[:, 2, None], -1. * lim_xy, lim_xy) * t[:, 2, None]
    J_matrix = torch.zeros((mask_xyz.shape[0], 3, 3)).to(device)
    J_matrix[:, 0, 0] = focal_x / t[:, 2]
    J_matrix[:, 0, 2] = -1 * (focal_x * t[:, 0]) / (t[:, 2] * t[:, 2])
    J_matrix[:, 1, 1] = focal_y / t[:, 2]
    J_matrix[:, 1, 2] = -1 * (focal_y * t[:, 1]) / (t[:, 2] * t[:, 2])
    W_matrix = w2c[:3, :3]  # 3,3
    T_matrix = (W_matrix @ J_matrix.permute(1, 2, 0)).permute(2, 0, 1) # N,3,3

    conv2d_matrix = torch.bmm(T_matrix.permute(0, 2, 1), torch.bmm(conv3d_matrix, T_matrix))[:, :2, :2]

    return conv2d_matrix


def update(gaussians, view, selected_index, ratios, dir_vector):

    ratios = ratios.unsqueeze(-1).to("cuda")
    selected_xyz = gaussians.get_xyz[selected_index]
    selected_scaling = gaussians.get_scaling[selected_index]
    conv3d = gaussians.get_covariance(scaling_modifier=1)[selected_index]
    conv3d_matrix = compute_conv3d(conv3d).to("cuda")

    # 计算特征值和特征向量
    eigvals, eigvecs = torch.linalg.eigh(conv3d_matrix)
    # 判断长轴
    max_eigval, max_idx = torch.max(eigvals, dim=1)
    max_eigvec = torch.gather(eigvecs, dim=1, 
                        index=max_idx.unsqueeze(1).unsqueeze(2).repeat(1,1,3)) # (N, 1, 3)最大特征向量
    long_axis = torch.sqrt(max_eigval) * 3
    max_eigvec = max_eigvec.squeeze(1)
    max_eigvec = max_eigvec / torch.norm(max_eigvec, dim=1).unsqueeze(-1)
    new_scaling = selected_scaling * ratios * 0.8
    # new_scaling = selected_scaling
    
    # 更新原gaussians里面相应的点，有两个方向，需要判断哪个方向: 
    # 把3d特征向量投影到2d，与2d的平移方向计算内积，大于0表示正方向，小于0表示负方向
    max_eigvec_2d = porject_to_2d(view, max_eigvec)
    sign_direction = torch.sum(max_eigvec_2d * dir_vector, dim=1).unsqueeze(-1)
    sign_direction = torch.where(sign_direction > 0, 1, -1)
    new_xyz = selected_xyz + 0.5 * (1 - ratios) * long_axis.unsqueeze(1) * max_eigvec * sign_direction

    gaussians._xyz = gaussians._xyz.detach().clone().requires_grad_(False)
    gaussians._scaling =  gaussians._scaling.detach().clone().requires_grad_(False)
    gaussians._xyz[selected_index] = new_xyz
    gaussians._scaling[selected_index] = gaussians.scaling_inverse_activation(new_scaling)

    return gaussians



import numpy as np
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
from seg_utils import grounding_dino_prompt

# 定义了使用的SAM模型的架构，这里是vit_h（Vision Transformer H模型）
SAM_ARCH = 'vit_h'
# 模型的预训练权重路径
SAM_CKPT_PATH = '/data1/zjyang/program/peract_bimanual/third_part/Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth'

model_type = SAM_ARCH
# 进行预测的SAM预测器
sam = sam_model_registry[model_type](checkpoint=SAM_CKPT_PATH).to('cuda')
predictor = SamPredictor(sam)

# text guided 文本提示分割函数
def text_prompting_ID(image, text, id):
		# 图像 提示文本 多掩码输出的掩码索引
		# 使用Grounding DINO生成基于文本的目标物体的边界框input_boxes
    input_boxes = grounding_dino_prompt(image, text)

		# 将返回的框转换为 Torch 张量，并传递给 CUDA
    boxes = torch.tensor(input_boxes)[0:1].cuda()

    # 在预测之前，调用 predictor.set_image(image)，将图像传递给 SAM 模型
    predictor.set_image(image)  # 需要在预测前设置图像

    # 将框转换为适合当前图像尺寸的格式 
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
    # 基于这些框生成掩码，不需要点提示（point_coords=None）
    masks,  _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True,
    )

    masks = masks[0].cpu().numpy()
    # 将选定的掩码（通过 id 索引）缩放到 [0, 255] 范围，并返回一个标准化的掩码（值范围为 [0, 1]）。
    return_mask = (masks[id, :, :, None]*255).astype(np.uint8)
    return return_mask / 255

# text guided 文本提示分割函数
def text_prompting(image, text, ID):
		# 图像 提示文本 多掩码输出的掩码索引
	# 使用Grounding DINO生成基于文本的目标物体的边界框input_boxes
    input_boxes = grounding_dino_prompt(image, text)
	# 将返回的框转换为 Torch 张量，并传递给 CUDA
    boxes = torch.tensor(input_boxes)[ID:1].cuda()
    # 在预测之前，调用 predictor.set_image(image)，将图像传递给 SAM 模型
    predictor.set_image(image)  # 需要在预测前设置图像
    # 将框转换为适合当前图像尺寸的格式 
    print("boxes",boxes.shape)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
    # 基于这些框生成掩码，不需要点提示（point_coords=None）
    masks,  _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True,
    )
    # mask_images = []
    # # 遍历每个生成的掩码
    # for i, mask in enumerate(masks):
    #     # 将掩码缩放到 [0, 255] 范围，并转换为 uint8 类型
    #     return_mask = (mask.cpu().numpy() * 255).astype(np.uint8)
    #     return_mask = return_mask[:, :, None]  # 扩展维度以适应保存为图像
    #     # 将掩码添加到掩码图像列表中
    #     mask_images.append(return_mask)

    return masks # mask_images


# point guided 点引导的分割
def self_prompt(point_prompts, sam_feature, id):
		# 用户提供的点提示，用于引导分割。通过将其转换为Numpy数组进行处理。
    input_point = point_prompts.detach().cpu().numpy()
    
    # input_label：为每个提示点创建一个全为1的标签数组（表示所有点为正样本）
    # input_point = input_point[::-1]
    input_label = np.ones(len(input_point))

		# 将输入的 sam_feature 赋值给预测器的 features，这是预计算的 SAM 特征。
    predictor.features = sam_feature
    
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    # 返回选定的掩码，并将其标准化为 [0, 1]
    # return_mask = (masks[ :, :, 0]*255).astype(np.uint8)
    return_mask = (masks[id, :, :, None]*255).astype(np.uint8)

    return return_mask / 255


# 定义 main 函数
def main():
    # 加载图像
    # task_name = #'bimanual_sweep_to_dustpan' #'bimanual_straighten_rope' # 'bimanual_pick_plate' # 'bimanual_pick_laptop'
    # task_name = #'coordinated_put_item_in_drawer' #'coordinated_put_bottle_in_fridge' # 'coordinated_push_box' #'coordinated_lift_tray'
    task_name='handover_item_easy' # 'handover_item' #'dual_push_buttons' #'coordinated_take_tray_out_of_oven'
    rgb_id='0000'
    eposide_id='episode0'
    camera_id='overhead_rgb' # 'overhead_mask' #'over_shoulder_left_mask' #'front_mask'
    # mask_path="/data1/zjyang/program/peract_bimanual/data2/train_data/${task_name}/all_variations/episodes/${eposide_id}/${camera_id}/rgb_${rgb_id}.png"
    mask_path = f"/data1/zjyang/program/peract_bimanual/data2/train_data/{task_name}/all_variations/episodes/{eposide_id}/{camera_id}/rgb_{rgb_id}.png"
    output_name = f"{task_name}_{eposide_id}_{camera_id}_{rgb_id}.png"
    # output_dir = "/data1/zjyang/program/peract_bimanual/scripts/test_demo/${output_name}"
    output_dir = f"/data1/zjyang/program/peract_bimanual/scripts/test_demo/3/" # {output_name}

    image_path = mask_path  # 替换为实际图像路径
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
        return

    # 转换为 RGB 格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 文本提示，输入的提示文本和对应的掩码索引
    text = "object"  # 替换为你的实际文本提示
    mask_id = 0  # 选择返回的掩码索引

    # 使用 text_prompting_ID 函数
    mask_from_text = text_prompting_ID(image, text, mask_id)

    # 保存基于文本的掩码结果
    output_path_text = os.path.join(output_dir, f"mask_id_{mask_id}.png")  # 替换为实际保存路径
    cv2.imwrite(output_path_text, (mask_from_text * 255).astype(np.uint8))
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print(f"Mask from text prompting saved to {output_path_text}")
    print(f"Mask from text prompting saved to {output_path_text} at {current_time}")

    mask_id = 0
    mask_list = text_prompting(image, text, mask_id)
    # print(mask_images.shape)
    value = 0  # 0 for background
    import matplotlib.pyplot as plt
    for idx, mask in enumerate(mask_list):
        mask_img = torch.zeros(mask.shape[-2:])
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        # 绘制并保存掩码图像
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy(), cmap='gray')  # 使用灰度图显示
        plt.axis('off')  # 不显示坐标轴
        plt.savefig(os.path.join(output_dir, f'mask_{value + idx + 1}.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
        plt.close() 
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")   
    print(f"all images saved at {current_time}")


    # # 点引导分割的输入提示点
    # point_prompts = torch.tensor([[100, 10], [10, 10]]).cuda()  # 替换为实际的提示点
    # # 假设已经有预先计算的SAM特征
    # sam_feature = predictor.features  # 在真实代码中需要调用 predictor.set_image 或通过前面提到的管道获取 features
    # # 使用 self_prompt 函数
    # mask_from_points = self_prompt(point_prompts, sam_feature, mask_id)
    # # 保存基于点引导的掩码结果
    # file_number = 2
    # output_path_points = os.path.join(output_dir, f"{file_number}.png")  # 替换为实际保存路径
    # cv2.imwrite(output_path_points, (mask_from_points * 255).astype(np.uint8))
    # print(f"point guide Mask from point prompting saved to {output_path_points}")

if __name__ == "__main__":
    main()