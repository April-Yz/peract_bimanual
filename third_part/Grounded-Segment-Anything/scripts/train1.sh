eval_gpu=${1:-"0"}

# task_name = #'bimanual_sweep_to_dustpan' #'bimanual_straighten_rope' # 'bimanual_pick_plate' # 'bimanual_pick_laptop'
# task_name = #'coordinated_put_item_in_drawer' #'coordinated_put_bottle_in_fridge' # 'coordinated_push_box' #'coordinated_lift_tray'
task_name='handover_item_easy' # 'handover_item' #'dual_push_buttons' #'coordinated_take_tray_out_of_oven'
rgb_id='0000'
eposide_id='episode0'
camera_id='overhead_rgb' # 'overhead_mask' #'over_shoulder_left_mask' #'front_mask'
# mask_path="/data1/zjyang/program/peract_bimanual/data2/train_data/${task_name}/all_variations/episodes/${eposide_id}/${camera_id}/rgb_${rgb_id}.png"
mask_path="/data1/zjyang/program/peract_bimanual/scripts/test_demo/real_1.png"
# ="/data1/zjyang/program/peract_bimanual/data2/train_data/${task_name}/all_variations\
# /episodes/${eposide_id}/nerf_data/0/images/10.png"

output_name="${task_name}_${eposide_id}_${camera_id}_${rgb_id}"
# output_dir="/data1/zjyang/program/peract_bimanual/scripts/test_demo/${output_name}"
output_dir="/data1/zjyang/program/peract_bimanual/scripts/test_demo/4/10"


# CUDA_VISIBLE_DEVICES=${eval_gpu} python demo/inference_on_a_image.py \
# -c groundingdino/config/GroundingDINO_SwinT_OGC.py \
# -p weights/groundingdino_swint_ogc.pth \
# -i ${mask_path} \
# -o "/data1/zjyang/program/test/GroundingDINO/output/2" \
# -t "object"
# #  [--cpu-only] # open it for cpu mode
# # "/data1/zjyang/program/peract_bimanual/data2/train_data/bimanual_pick_laptop/all_variations/episodes/episode0/overhead_rgb/rgb_0001.png" \

 CUDA_VISIBLE_DEVICES=${eval_gpu} python grounded_sam_demo.py \
  --config /data1/zjyang/program/peract_bimanual/third_part/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint /data1/zjyang/program/peract_bimanual/third_part/Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth \
  --sam_checkpoint /data1/zjyang/program/peract_bimanual/third_part/Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth \
  --input_image ${mask_path} \
  --output_dir "/data1/zjyang/program/peract_bimanual/scripts/test_demo/4/10" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "object" \
  --device "cuda"