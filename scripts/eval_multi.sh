# this script is for evaluating a given checkpoint.
#       bash scripts/eval.sh PERACT_BC  0 ${exp_name}
# bash scripts/eval.sh PERACT_BC 0 PERACT_BC
# bash scripts/eval.sh PERACT_BC 0 multi
# bash scripts/eval.sh BIMANUAL_PERACT 0 test

# some params specified by user
method_name=$1 # ManiGaussian_BC2
# set the seed number
seed="0"
# set the gpu id for evaluation. we use one gpu for parallel evaluation.
eval_gpu=${2:-"0"}

# test_demo_path="/home/zjyang/download/peract/squashfs-root-test"
# "/mnt/disk_1/tengbo/bimanual_data/test"
test_demo_path="/data1/zjyang/program/peract_bimanual/data2/test_data_formal"

addition_info="$(date +%Y%m%d)"
exp_name=${3:-"${method}_${addition_info}"}
# tasks=${4:-"None"}
tasks=[bimanual_pick_laptop,bimanual_straighten_rope,coordinated_lift_tray,coordinated_push_box,coordinated_put_bottle_in_fridge,dual_push_buttons,handover_item,bimanual_sweep_to_dustpan,coordinated_take_tray_out_of_oven,handover_item_easy]
starttime=`date +'%Y-%m-%d %H:%M:%S'`
# printf 'exp_name = %s\n' "$exp_name"
# printf '%s\n' "$starttime"
# tasks=[bimanual_pick_laptop,bimanual_push_single_button,coordinated_lift_tray,coordinated_push_box,coordinated_put_bottle_in_fridge,handover_item_medium]

eval_type=95000 # [50000,22000,24000,28000] #[20000,30000,40000,50000,60000,70000]  # [32000,37000,39000,43000,60000,70000,65000] #[5000,15000,25000,35000,51000,59000,61000,82000,89000] # [58000,62000,15000,25000,35000,52000,65000,75000,85000] # 90000    # "[20000,65000,49000,51000]" # 80000 #'last' # or 'best', 'missing', or 'last' or 'all'
camera=False
eval_episodes=25 #25 #eval每个task的轮数
# camera=False # 是否录制视频
gripper_mode='BimanualDiscrete'
arm_action_mode='BimanualEndEffectorPoseViaPlanning'
action_mode='BimanualMoveArmThenGripper'
logdir="/data1/zjyang/program/peract_bimanual/log-mani/${exp_name}"
camera_resolution="[256,256]"
# printf "logdir = %s\n" "$logdir"
# printf "${logdir}"

CUDA_VISIBLE_DEVICES=${eval_gpu} xvfb-run -a python eval.py \
    rlbench.task_name=${exp_name} \
    rlbench.demo_path=${test_demo_path} \
    framework.start_seed=${seed} \
    framework.eval_type=${eval_type} \
    framework.eval_episodes=${eval_episodes} \
    cinematic_recorder.enabled=${camera} \
    rlbench.gripper_mode=${gripper_mode} \
    rlbench.arm_action_mode=${arm_action_mode} \
    rlbench.action_mode=${action_mode} \
    framework.logdir=${logdir} \
    rlbench.tasks=${tasks} \
    rlbench.camera_resolution=${camera_resolution}

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "eclipsed time "$((end_seconds-start_seconds))"s"