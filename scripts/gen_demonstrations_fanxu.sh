# this script generate demonstrations for a given task, for both training and evaluation.
# example:
#       bash scripts/gen_demonstrations.sh open_drawer
# 该脚本可为给定任务生成演示，用于培训和评估。
# 示例
# bash scripts/gen_demonstrations.sh open_drawer
task=${1}
printf 'task = %s\n' "$task"

cd third_part/RLBench/tools

# # 开始爆错，实在不行加一个is_nerf,在修改的地方加if 但是文件好像没问题..
# xvfb-run -a python dataset_generator_bimanual.py --tasks=${task} \
#                             --save_path="../../../data2/test_data"  \
#                             --image_size=256x256 \
#                             --episodes_per_task=23 #\
#                             # --all_variations=True   # default is True
#                             # --processes=1 \
#                             # --renderer=opengl \

xvfb-run -a python nerf_dataset_generator_bimanual.py --tasks=${task} \
                            --save_path="../../../data2/train_data" \
                            --image_size=256x256 \
                            --episodes_per_task=100  #\
                            # --all_variations=True
                            # --processes=1 \      
                            # --renderer=opengl \                     


cd ..