eval_gpu=${1:-"0"}

# task_name = #'bimanual_sweep_to_dustpan' #'bimanual_straighten_rope' # 'bimanual_pick_plate' # 'bimanual_pick_laptop'
# task_name = #'coordinated_put_item_in_drawer' #'coordinated_put_bottle_in_fridge' # 'coordinated_push_box' #'coordinated_lift_tray'
task_name='handover_item_easy' # 'handover_item' #'dual_push_buttons' #'coordinated_take_tray_out_of_oven'
mask_id='0000'
eposide_id='episode0'
camera_id='overhead_rgb' # 'overhead_mask' #'over_shoulder_left_mask' #'front_mask'
mask_path="/data1/zjyang/program/peract_bimanual/data2/train_data/${task_name}/all_variations/episodes/${eposide_id}/${camera_id}/rgb_${mask_id}.png"

CUDA_VISIBLE_DEVICES=${eval_gpu} python demo/inference_on_a_image.py \
-c groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p weights/groundingdino_swint_ogc.pth \
-i ${mask_path} \
-o "/data1/zjyang/program/test/GroundingDINO/output/2" \
-t "object"
#  [--cpu-only] # open it for cpu mode
# "/data1/zjyang/program/peract_bimanual/data2/train_data/bimanual_pick_laptop/all_variations/episodes/episode0/overhead_rgb/rgb_0001.png" \