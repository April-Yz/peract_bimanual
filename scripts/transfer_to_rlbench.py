import os
import pickle
from rlbench.demo import Demo

# 定义文件夹路径模板
# base_dir = "/mnt/disk_1/tengbo/bimanual_data/keypoint/press/all_variations/episodes/"
base_dir = "/data1/zjyang/program/peract_bimanual/data_ntu/zips/press_keyframe/all_variations/episodes/"
# lift_keyframe handover_keyframe pick_in_one_keyframe pick_in_two_keyframe press_keyframe
# 遍历 episode0 到 episode29 的文件夹
for episode_num in range(47):
    # 生成每个文件夹的路径和 pkl 文件路径
    episode_folder = os.path.join(base_dir, f"episode{episode_num}")
    pkl_file_path = os.path.join(episode_folder, "low_dim_obs.pkl")
    
    # 检查 pkl 文件是否存在
    if not os.path.isfile(pkl_file_path):
        print(f"File not found: {pkl_file_path}")
        continue

    # 读取 pkl 文件内容
    with open(pkl_file_path, 'rb') as pkl_file:
        obs = pickle.load(pkl_file)

    # 将 obs 转换为 Demo 对象
    demo_obj = Demo(observations=obs)

    # 覆盖保存新的 Demo 对象到原文件
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(demo_obj, f)

    print(f"Converted and saved Demo to {pkl_file_path}")