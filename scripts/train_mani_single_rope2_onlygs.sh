# example to run:
#       bash train.sh PERACT_BC 0,1 12345 ${exp_name}
#       bash train.sh PERACT_BC 0,1,2,3 12345 abc
#       bash scripts/train.sh PERACT_BC 0,1,2,3 12345 abcd
#       bash scripts/train.sh ManiGaussian_BC2 0,1,2,3 12345 mani
# 
# set the method name
method=${1} # PERACT_BC / BIMANUAL_PERACT  / ManiGaussian_BC2

# set the seed number
seed="0"
train_gpu=${2:-"0,1"}
train_gpu_list=(${train_gpu//,/ })

# set the port for ddp training.
port=${3:-"12345"}
# you could enable/disable wandb by this.
use_wandb=True

train_demo_path="/data1/zjyang/program/peract_bimanual/data2/train_data"

# we set experiment name as method+date. you could specify it as you like.
addition_info="$(date +%Y%m%d)"
exp_name=${4:-"${method}_${addition_info}"}
logdir="/data1/zjyang/program/peract_bimanual/log-mani/${exp_name}"

# create a tmux window for training
echo "I am going to kill the session ${exp_name}, are you sure? (5s)"
sleep 5s
tmux kill-session -t ${exp_name}
sleep 3s
echo "start new tmux session: ${exp_name}, running main.py"
tmux new-session -d -s ${exp_name}
batch_size=1 # 1 #4 # 2

tasks=[bimanual_straighten_rope]
replay_path="/data1/zjyang/program/peract_bimanual/replay/"
# for debug
demo=100 # 100
episode_length=25 #25 # 20 # 4
save_freq=1000
camera_resolution="[256,256]"
training_iterations=100001
field_type='bimanual' # 'bimanual' 'LF'
lambda_dyna=0.1
lambda_reg=0.0
render_freq=1000 #2000

tmux select-pane -t 0 
# peract rlbench
tmux send-keys "conda activate rlbench; 
CUDA_VISIBLE_DEVICES=${train_gpu}  QT_AUTO_SCREEN_SCALE_FACTOR=0 python train.py method=$method \
        rlbench.task_name=${exp_name} \
        framework.logdir=${logdir} \
        rlbench.demo_path=${train_demo_path} \
        framework.save_freq=${save_freq} \
        framework.start_seed=${seed} \
        framework.use_wandb=${use_wandb} \
        framework.wandb_group=${exp_name} \
        framework.wandb_name=${exp_name} \
        framework.training_iterations=${training_iterations} \
        ddp.num_devices=${#train_gpu_list[@]} \
        replay.batch_size=${batch_size} \
        ddp.master_port=${port} \
        rlbench.tasks=${tasks} \
        rlbench.demos=${demo} \
        replay.path=${replay_path} \
        rlbench.episode_length=${episode_length} \
        rlbench.camera_resolution=${camera_resolution} \
        method.neural_renderer.render_freq=${render_freq} \
        method.neural_renderer.field_type=${field_type} \
        method.neural_renderer.use_dynamic_field=False
"
        # method.neural_renderer.lambda_embed=0.0 \
        # method.neural_renderer.lambda_dyna=${lambda_dyna} \
        # method.neural_renderer.lambda_reg=${lambda_reg} \
        # method.neural_renderer.foundation_model_name=null \
        #  \
# remove 0.ckpt
# rm -rf logs/${exp_name}/seed${seed}/weights/0
rm -rf log-mani/${exp_name}/${exp_name}/${method}/seed${seed}/weights/0

tmux -2 attach-session -t ${exp_name}