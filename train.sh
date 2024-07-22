# example to run:
#       bash train.sh PERACT_BC 0,1 12345 ${exp_name}


# set the method name
method=${1} # PERACT_BC or BIMANUAL_BC

# set the seed number
seed="0"
# set the gpu id for training. we use two gpus for training. you could also use one gpu.
train_gpu=${2:-"0,1"}
train_gpu_list=(${train_gpu//,/ })

# set the port for ddp training.
port=${3:-"12345"}
# you could enable/disable wandb by this.
use_wandb=True

# cur_dir=$(pwd)
train_demo_path="/mnt/disk_1/tengbo/bimanual_data/train"


# we set experiment name as method+date. you could specify it as you like.
tasks=[coordinated_push_box]
addition_info="$(date +%Y%m%d)"
exp_name=${4:-"${method}_${addition_info}"}
logdir="/mnt/disk_1/tengbo/peract_bimanual/log"

# create a tmux window for training
echo "I am going to kill the session ${exp_name}, are you sure? (5s)"
sleep 5s
tmux kill-session -t ${exp_name}
sleep 3s
echo "start new tmux session: ${exp_name}, running main.py"
tmux new-session -d -s ${exp_name}

#######
# override hyper-params in config.yaml
#######
batch_size=2
# task_name=${"multi_${addition_info}"}
demo=100
episode_length=25
# for debug
# demo=1
# episode_length=4

tmux select-pane -t 0 
tmux send-keys "conda activate per2; 

CUDA_VISIBLE_DEVICES=${train_gpu} python train.py method=$method \
        rlbench.task_name=${exp_name} \
        framework.logdir=${logdir} \
        rlbench.demo_path=${train_demo_path} \
        framework.start_seed=${seed} \
        framework.use_wandb=${use_wandb} \
        framework.use_wandb=${use_wandb} \
        framework.wandb_group=${exp_name} \
        framework.wandb_name=${exp_name} \
        ddp.num_devices=${#train_gpu_list[@]} \
        replay.batch_size=${batch_size} \
        ddp.master_port=${port} \
        rlbench.tasks=${tasks} \
        rlbench.demos=${demo} \
        rlbench.episode_length=${episode_length}

"
# remove 0.ckpt
rm -rf logs/${exp_name}/seed${seed}/weights/0

tmux -2 attach-session -t ${exp_name}