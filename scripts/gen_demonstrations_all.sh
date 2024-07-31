# this script generates demonstrations for multiple tasks sequentially.
# 该脚本会依次生成多个任务的演示。
# 示例：
# bash scripts/gen_demonstrations_all.sh
# example:
#       bash scripts/gen_demonstrations_all.sh

# The recommended 10 tasks
ALL_TASK="bimanual_pick_laptop coordinated_lift_tray handover_item_medium bimanual_pick_plate"
tasks=($ALL_TASK)

for task in "${tasks[@]}"; do
    echo "###Generating demonstrations for task: $task"
    bash scripts/gen_demonstrations.sh "$task"
done


# ALL_TASK="close_jar open_drawer sweep_to_dustpan_of_size meat_off_grill turn_tap slide_block_to_color_target put_item_in_drawer reach_and_drag push_buttons stack_blocks"
# bimanual_pick_laptop  coordinated_lift_tray handover_item_medium bimanual_pick_plate   
# coordinated_push_box bimanual_push_single_button   coordinated_put_bottle_in_fridge  
# bimanual_set_the_table  coordinated_put_item_in_drawer left_open_drawer 
# bimanual_straighten_rope coordinated_put_item_in_drawer_right  
# bimanual_sweep_to_dustpan  coordinated_take_shoes_out_of_box right_open_drawer 
# coordinated_close_jar coordinated_take_tray_out_of_oven unimanual_push_single_button_left 
# coordinated_lift_ball dual_push_buttons coordinated_lift_stick  handover_item_easy 

# bimanual_pick_laptop  coordinated_lift_tray handover_item_medium bimanual_pick_plate coordinated_push_box bimanual_push_single_button   coordinated_put_bottle_in_fridge  bimanual_set_the_table  coordinated_put_item_in_drawer left_open_drawer bimanual_straighten_rope coordinated_put_item_in_drawer_right  bimanual_sweep_to_dustpan  coordinated_take_shoes_out_of_box right_open_drawer coordinated_close_jar coordinated_take_tray_out_of_oven unimanual_push_single_button_left coordinated_lift_ball dual_push_buttons coordinated_lift_stick  handover_item_easy 