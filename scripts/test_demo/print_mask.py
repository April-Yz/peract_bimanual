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


# right_min = 53
# right_max = 73
# left_min = 94
# left_max = 114
# left_mask = (next_render_mask_novel > left_min) & (next_render_mask_novel < left_max) # 保留左臂标签      [128,128]
# # exclude_right_mask = (render_mask_novel_next < right_min) | (render_mask_novel_next > right_max)
# exclude_left_mask = (next_render_mask_novel < left_min) | (next_render_mask_novel > left_max) # 排除左臂标签  

mask_image = Image.open(mask_path)
print(mask_image.size)
print(mask_image)
# np.savetxt('/data1/zjyang/program/peract_bimanual/scripts/test_demo/excluded_mask4.txt', mask_image, fmt='%d') 
unique_labels = np.unique(mask_image)
print(unique_labels)
# for label in unique_labels:
# print(f'Label: {label}, Count: {np.sum(mask == label)}')
# 转换为 NumPy 数组
mask_array = np.array(mask_image)

# print(mask_array)  # 输出 mask 的数字形式
print(mask_array.shape)  # 输出形状
with open('/data1/zjyang/program/peract_bimanual/scripts/test_demo/mask_output.txt', 'w') as f:
    # 保存形状
    f.write(f'Shape: {mask_array.shape}\n{unique_labels}\n')
    
    # 保存数组内容
    np.savetxt(f, mask_array.reshape(-1, mask_array.shape[-1]) if len(mask_array.shape) == 3 else mask_array, fmt='%d')

print("输出已保存到 mask_output.txt")