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

# 假设有一个mask，大小为(256, 256)
# mask = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
# mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
np.savetxt('/data1/zjyang/program/peract_bimanual/scripts/test_demo/excluded_mask_0.txt', mask, fmt='%d') 
# 生成排除label范围的掩码
excluded_mask = exclude_label_range(mask, 94, 114)

# 保存掩码为图片（可选）
cv2.imwrite('/data1/zjyang/program/peract_bimanual/scripts/test_demo/excluded_mask.png', excluded_mask * 255)  # 将掩码保存为图片
print(excluded_mask)

output_txt_path = '/data1/zjyang/program/peract_bimanual/scripts/test_demo/excluded_mask.txt'
np.savetxt(output_txt_path, excluded_mask, fmt='%d')  # 保存为整型格式