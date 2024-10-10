import cv2
import numpy as np
import os

def process_mask_image(mask_path, output_dir):
    # 读取 mask 图片
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    # 获取唯一的标签（假设标签是整数值）
    unique_labels = np.unique(mask)

    for label in unique_labels:
        if label == 0:  # 跳过背景
            continue
        
        # 创建一个与 mask 同样大小的空白图像
        label_mask = np.zeros_like(mask)

        # 设置当前标签区域
        label_mask[mask == label] = label

        # 保存当前标签的 mask 图像
        output_path = f"{output_dir}/mask_label_{label}.png"
        cv2.imwrite(output_path, label_mask)

# 使用示例
# task_name = #'bimanual_sweep_to_dustpan' #'bimanual_straighten_rope' # 'bimanual_pick_plate' # 'bimanual_pick_laptop'
# task_name = #'coordinated_put_item_in_drawer' #'coordinated_put_bottle_in_fridge' # 'coordinated_push_box' #'coordinated_lift_tray'
task_name = 'handover_item_easy' # 'handover_item' #'dual_push_buttons' #'coordinated_take_tray_out_of_oven'
mask_id = '0000'
eposide_id = 'episode0'
camera_id =  'overhead_mask' # 'overhead_mask' #'over_shoulder_left_mask' #'front_mask'
mask_path = f'/data1/zjyang/program/peract_bimanual/data2/train_data/{task_name}/all_variations/episodes/{eposide_id}/{camera_id}/mask_{mask_id}.png'
output_name = f'{task_name}_{eposide_id}_{camera_id}_{mask_id}'
output_dir = f'/data1/zjyang/program/peract_bimanual/scripts/test_demo/{output_name}'
os.makedirs(output_dir, exist_ok=True)
process_mask_image(mask_path, output_dir)
# process_mask_image('/data1/zjyang/program/peract_bimanual/data2/train_data/bimanual_pick_laptop/all_variations/episodes/episode0/front_mask/mask_0000.png', '/data1/zjyang/program/peract_bimanual/scripts/test_demo')
