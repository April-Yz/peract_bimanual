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
tasks=[bimanual_pick_laptop,bimanual_straighten_rope,coordinated_lift_tray,coordinated_push_box,coordinated_put_bottle_in_fridge,dual_push_buttons,handover_item,bimanual_sweep_to_dustpan,coordinated_take_tray_out_of_oven,handover_item_easy]
# coordinated_lift_ball, bimanual_pick_plate,

# tasks=[dual_push_buttons]
# replay_path="/data1/zjyang/program/peract_bimanual/replay/depthmeters/"
replay_path="/data1/zjyang/program/peract_bimanual/replay/withoutnerf/"
# for debug
demo=100 # 100
episode_length=25 #25 # 20 # 4
save_freq=2500 #1000
camera_resolution="[256,256]"
training_iterations=100001 #100001
field_type='LF' # 'bimanual' 'LF'
lambda_dyna=0.1
lambda_reg=0.0
render_freq=500 # 1000 #2000
lambda_nerf=0.001 # 0.01 # 0.01

mask_gt_rgb=True        
lambda_dyna_leader=0.5 # 0.2 # 0.5 # 0.3  # 0.2  # V4 0.3  # （rgb dyn中左右的权重比例）
lambda_mask=0.5         # V4 0.2         # 2:rgb8mask的权重（相对于dyn总）    
lambda_mask_right=0.4 # mask中 右臂的权重(无用，单纯去掉会报Loss算少了 错)
mask_type='exclude' # 'include' # 无用 直接删除next中左臂和右臂比较
lambda_next_loss_mask=0.6

mask_gen='nonerf' #'gt' # 'pre' 'nonerf'  'None'# 是否用凸包围成的mask来确定物体
use_nerf_picture=False
image_width=256
image_height=256

tmux select-pane -t 0 
# peract rlbench
tmux send-keys "conda activate rlbench; 
CUDA_VISIBLE_DEVICES=${train_gpu}  QT_AUTO_SCREEN_SCALE_FACTOR=0 python train.py method=$method \
        rlbench.task_name=${exp_name} \
        framework.logdir=${logdir} \
        rlbench.demo_path=${train_demo_path} \
        method.neural_renderer.use_nerf_picture=${use_nerf_picture} \
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
        method.neural_renderer.lambda_nerf=${lambda_nerf} \
        method.neural_renderer.render_freq=${render_freq} \
        method.neural_renderer.image_width=${image_width} \
        method.neural_renderer.image_height=${image_height} \
        method.neural_renderer.lambda_embed=0.0 \
        method.neural_renderer.lambda_dyna=${lambda_dyna} \
        method.neural_renderer.lambda_reg=${lambda_reg} \
        method.neural_renderer.foundation_model_name=null \
        method.neural_renderer.field_type=${field_type} \
        method.neural_renderer.mask_gen=${mask_gen} \
        method.neural_renderer.dataset.mask_gt_rgb=${mask_gt_rgb} \
        method.neural_renderer.lambda_dyna_leader=${lambda_dyna_leader} \
        method.neural_renderer.lambda_mask=${lambda_mask} \
        method.neural_renderer.lambda_mask_right=${lambda_mask_right} \
        method.neural_renderer.mask_type=${mask_type} \
        method.neural_renderer.lambda_next_loss_mask=${lambda_next_loss_mask} \
        method.neural_renderer.use_dynamic_field=True 
"

        # 
        # 
        #  \
# remove 0.ckpt
# rm -rf logs/${exp_name}/seed${seed}/weights/0
rm -rf log-mani/${exp_name}/${exp_name}/${method}/seed${seed}/weights/0

tmux -2 attach-session -t ${exp_name}