from PIL import Image
import numpy as np

# 读取 mask.png
# task_name = #'bimanual_sweep_to_dustpan' #'bimanual_straighten_rope' # 'bimanual_pick_plate' # 'bimanual_pick_laptop'
# task_name = #'coordinated_put_item_in_drawer' #'coordinated_put_bottle_in_fridge' # 'coordinated_push_box' #'coordinated_lift_tray'
task_name = 'handover_item_easy' # 'handover_item' #'dual_push_buttons' #'coordinated_take_tray_out_of_oven'
mask_id = '0000'
eposide_id = 'episode0'
camera_id =  'overhead_mask' # 'overhead_mask' #'over_shoulder_left_mask' #'front_mask'
mask_path = f'/data1/zjyang/program/peract_bimanual/data2/train_data/{task_name}/all_variations/episodes/{eposide_id}/{camera_id}/mask_{mask_id}.png'
output_name = f'{task_name}_{eposide_id}_{camera_id}_{mask_id}'
output_dir = f'/data1/zjyang/program/peract_bimanual/scripts/test_demo/{output_name}'

import numpy as np
import cv2  # 如果需要保存为图片

def exclude_label_range(mask, min_val=94, max_val=114):
    # 生成掩码，排除min_val到max_val范围的值
    exclusion_mask = np.ones_like(mask, dtype=np.uint8)
    np.savetxt('/data1/zjyang/program/peract_bimanual/scripts/test_demo/excluded_mask1.txt', exclusion_mask, fmt='%d') 
    # 找到在min_val到max_val范围内的像素
    exclusion_mask[(mask >= 10) & (mask <= 30)] = 0
    
    return exclusion_mask
def process_mask_image(mask_path):
    # 读取 mask 图片
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    print("mask = ", mask, mask.shape) # (256,256,3)
    tensor_max = np.max(mask, axis=-1, keepdims=True)
    print("tensor_max = ",tensor_max)
    output_path = "/data1/zjyang/program/peract_bimanual/scripts/test_demo/mask_label1.png"
    cv2.imwrite(output_path, tensor_max)
    # maks1 = mask.permute(0, 2, 3, 1).repeat(1, 1, 1, 3) # 论文中的变换
    # print(mask1)
    # maks1 = maks1.view(maks1.shape[0], maks1.shape[1], maks1.shape[2], 1)
    # maks1.permute(0, 3, 1, 2)
    unique_labels = np.unique(mask)

    # for label in unique_labels:
        # if label == 0:  # 跳过背景
            # continue
        # 创建一个与 mask 同样大小的空白图像
    label_mask = np.zeros_like(mask)
        # 设置当前标签区域
    # label_mask[94<mask & mask<114] = 1
    label_mask[(94 < mask) & (mask < 114)] = 255 # [255,255,255]


        # 保存当前标签的 mask 图像
    output_path = "/data1/zjyang/program/peract_bimanual/scripts/test_demo/mask_label0.png"
    cv2.imwrite(output_path, label_mask)

# 假设有一个mask，大小为(256, 256)
# mask = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
# mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
np.savetxt('/data1/zjyang/program/peract_bimanual/scripts/test_demo/excluded_mask_0.txt', mask, fmt='%d') 
# 生成排除label范围的掩码
excluded_mask = exclude_label_range(mask, 94, 114)
process_mask_image(mask_path)
# 保存掩码为图片（可选）
cv2.imwrite('/data1/zjyang/program/peract_bimanual/scripts/test_demo/excluded_mask.png', excluded_mask * 255)  # 将掩码保存为图片
print(excluded_mask)

output_txt_path = '/data1/zjyang/program/peract_bimanual/scripts/test_demo/excluded_mask.txt'
np.savetxt(output_txt_path, excluded_mask, fmt='%d')  # 保存为整型格式

# right_min = 53
# right_max = 73
# left_min = 94
# left_max = 114
# left_mask = (next_render_mask_novel > left_min) & (next_render_mask_novel < left_max) # 保留左臂标签      [128,128]
# # exclude_right_mask = (render_mask_novel_next < right_min) | (render_mask_novel_next > right_max)
# exclude_left_mask = (next_render_mask_novel < left_min) | (next_render_mask_novel > left_max) # 排除左臂标签  

# mask_image = Image.open(mask_path)
# unique_labels = np.unique(mask_image)
# print(unique_labels)
# # for label in unique_labels:
# # print(f'Label: {label}, Count: {np.sum(mask == label)}')
# # 转换为 NumPy 数组
# mask_array = np.array(mask_image)

# # print(mask_array)  # 输出 mask 的数字形式
# print(mask_array.shape)  # 输出形状
# with open('/data1/zjyang/program/peract_bimanual/scripts/test_demo/mask_output.txt', 'w') as f:
#     # 保存形状
#     f.write(f'Shape: {mask_array.shape}\n{unique_labels}\n')
    
#     # 保存数组内容
#     np.savetxt(f, mask_array.reshape(-1, mask_array.shape[-1]) if len(mask_array.shape) == 3 else mask_array, fmt='%d')

# print("输出已保存到 mask_output.txt")