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
test_demo_path="/home/zjyang/program/peract_bimanual/data1/test_data"

addition_info="$(date +%Y%m%d)"
exp_name=${3:-"${method}_${addition_info}"}

starttime=`date +'%Y-%m-%d %H:%M:%S'`
printf 'exp_name = %s\n' "$exp_name"
printf '%s\n' "$starttime"

# camera=True
camera=False
gripper_mode='BimanualDiscrete'
arm_action_mode='BimanualEndEffectorPoseViaPlanning'
action_mode='BimanualMoveArmThenGripper'

CUDA_VISIBLE_DEVICES=${eval_gpu} xvfb-run -a python eval.py \
    rlbench.task_name=${exp_name} \
    rlbench.demo_path=${test_demo_path} \
    framework.start_seed=${seed} \
    cinematic_recorder.enabled=${camera} \
    rlbench.gripper_mode=${gripper_mode} \
    rlbench.arm_action_mode=${arm_action_mode} \
    rlbench.action_mode=${action_mode}


endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "eclipsed time "$((end_seconds-start_seconds))"s"