# this script generate demonstrations for a given task, for both training and evaluation.
# example:
#       bash scripts/gen_demonstrations.sh open_drawer
# 该脚本可为给定任务生成演示，用于培训和评估。
# 示例
# bash scripts/gen_demonstrations.sh open_drawer
task=${1}

cd third_part/RLBench/tools
# xvfb-run -a python dataset_generator.py --tasks=${task} \
#                             --save_path="../../../data1/test_data" \
#                             --image_size=128,128 \
#                             --renderer=opengl \
#                             --episodes_per_task=25 \
#                             --processes=1 \
#                             --all_variations=True

xvfb-run -a python nerf_dataset_generator_bimanual.py --tasks=${task} \
                            --save_path="../../../data/train_data" \
                            --image_size=128x128 \
                            # --renderer=opengl \
                            --episodes_per_task=10 \    #20
                            # --processes=1 \
                            --all_variations=True

# xvfb-run -a python dataset_generator_bimanual.py --tasks=${task} \
#                             --save_path="../../../data/test_data"  \
#                             --image_size=128x128 \
#                             # --renderer=opengl \
#                             --episodes_per_task=10 \   # 25 \
#                             # --processes=1 \
#                             --all_variations=True

cd ..