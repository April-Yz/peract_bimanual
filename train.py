# import sys

# # 接受三个参数：type（异常类型），value（异常值），和tb（追踪栈，即traceback）
# def info(type, value, tb):
#     if hasattr(sys, 'ps1') or not sys.stderr.isatty():
#     # we are in interactive mode or we don't have a tty-like
#     # 我们处于交互模式，或者没有类似 tty 的 # 设备。
#     # 设备，所以我们调用默认钩子
#     # device, so we call the default hook
#         sys.__excepthook__(type, value, tb)
#     else:
#         import traceback, pdb
#         # 我们不在交互模式下，打印异常...
#         # we are NOT in interactive mode, print the exception...
#         traceback.print_exception(type, value, tb)
#         print
#         # ...then start the debugger in post-mortem mode.
#         # pdb.pm() # deprecated
#         pdb.post_mortem(tb) # more "modern"

# sys.excepthook = info
# 上面--------------new---------------
from typing import List
import logging
import os
import sys
from datetime import datetime

import peract_config

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig

import run_seed_fn
from helpers.observation_utils import create_obs_config

import torch.multiprocessing as mp
import wandb
# --------------------------Mani------------------------
import numpy as np
import lightning as L
import torch
#---------------------------Mani-------------------------

@hydra.main(config_name="config", config_path="conf")
def main(cfg: DictConfig) -> None:

    # print("main start")
    cfg_yaml = OmegaConf.to_yaml(cfg)
    logging.info("\n" + cfg_yaml)

    peract_config.on_config(cfg)
    # new-------------------------------------------------------
    os.environ['MASTER_ADDR'] = cfg.ddp.master_addr
    os.environ['MASTER_PORT'] = str(cfg.ddp.master_port)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    print("device available: ", torch.cuda.device_count())
    # new-------------------------------------------------------

    cfg.rlbench.cameras = (
        cfg.rlbench.cameras
        if isinstance(cfg.rlbench.cameras, ListConfig)
        else [cfg.rlbench.cameras]
    )

    # sanity check if rgb is not used as camera name   
    for camera_name in cfg.rlbench.cameras:
        assert("rgb" not in camera_name)

    print("nerf_multi_view = cfg.method.neural_renderer.use_nerf_picture",cfg.method.neural_renderer.use_nerf_picture)
    # if cfg.method.name.startswith("ManiGaussian_BC2") #搞错了，use
    obs_config = create_obs_config(
        cfg.rlbench.cameras, cfg.rlbench.camera_resolution, cfg.method.name,
        use_depth=cfg.method.use_depth, # !!! Mani新增的depth
        nerf_multi_view = cfg.method.neural_renderer.use_nerf_picture
    )
    multi_task = len(cfg.rlbench.tasks) > 1 # Mani新增的multi_task

    cwd = os.getcwd()
    logging.info("CWD:" + os.getcwd())

    if cfg.framework.start_seed >= 0:
        # seed specified
        start_seed = cfg.framework.start_seed
    elif (
        cfg.framework.start_seed == -1
        and len(list(filter(lambda x: "seed" in x, os.listdir(cwd)))) > 0
    ):
        # unspecified seed; use largest existing seed plus one
        largest_seed = max(
            [
                int(n.replace("seed", ""))
                for n in list(filter(lambda x: "seed" in x, os.listdir(cwd)))
            ]
        )
        start_seed = largest_seed + 1
    else:
        # start with seed 0
        start_seed = 0

    seed_folder = os.path.join(os.getcwd(), "seed%d" % start_seed)
    os.makedirs(seed_folder, exist_ok=True)

    start_time = datetime.now()
    with open(os.path.join(seed_folder, "config.yaml"), "w") as f:
        f.write(cfg_yaml)


    # check if previous checkpoints already exceed the number of desired training iterations
    # if so, exit the script
    latest_weight = 0
    weights_folder = os.path.join(seed_folder, "weights")
    # ----------------------------Mani---------------------------
    # 列出weight文件夹文件>0
    if os.path.isdir(weights_folder) and len(os.listdir(weights_folder)) > 0:
        weights = os.listdir(weights_folder)
        latest_weight = sorted(map(int, weights))[-1]
        if latest_weight >= cfg.framework.training_iterations:
            logging.info(
                "Agent was already trained for %d iterations. Exiting." % latest_weight
            )
            sys.exit(0)
    # ----------------------------Mani---------------------------

    with open(os.path.join(seed_folder, "training.log"), "a") as f:

        f.write(f"# Starting training from weights: {latest_weight} to {cfg.framework.training_iterations}")
        f.write(f"# Training started on: {start_time.isoformat()}")
        f.write(os.linesep)

    

    # run train jobs with multiple seeds (sequentially)
    for seed in range(start_seed, start_seed + cfg.framework.seeds):
        
        logging.info("Starting seed %d." % seed)

        world_size = cfg.ddp.num_devices
        # ----------------------------Mani---------------------------
        if cfg.method.use_fabric:
            # we use fabric DDP 我们使用织物 DDP
            fabric = L.Fabric(devices=world_size, strategy='ddp', precision = 32)
            fabric.launch()
            # print("we  are using ddp ----------问题出在run_seed-----------------")
            run_seed_fn.run_seed(
                                0,  # 多rank, will be overwrited by fabric
                                cfg,
                                obs_config,
                                cfg.rlbench.cameras,    #多
                                multi_task, # T/F
                                seed,
                                world_size,
                                fabric, 
                                )
            # print("we  are using ddp ------------end---------------")
        else:
            # use pytorch DDP 
            # "DDP"指的是"Distributed Data Parallel"，即分布式数据并行
            import torch.multiprocessing as mp
            mp.set_sharing_strategy('file_system')
            from torch.multiprocessing import set_start_method, get_start_method

            try:
                if get_start_method() != 'spawn':
                    set_start_method('spawn', force=True)
            except RuntimeError:
                # 无法将启动方法设置为生成
                print("Could not set start method to spawn")
                pass
            mp.spawn(run_seed_fn.run_seed,
                    args=(cfg,
                        obs_config,
                        cfg.rlbench.cameras,
                        multi_task,
                        seed,
                        world_size,
                        None,   # fabric
                        ),
                    nprocs=world_size,
                    join=True)
        # ----------------------------Mani---------------------------
        
        # ----------------------------bimanual---------------------------
        # mp.spawn(
        #     run_seed_fn.run_seed,
        #     args=(
        #         # 0, rank
        #         cfg,
        #         obs_config,
        #         # y7.26新增 cfg.rlbench.cameras,
        #         seed,
        #         # y7.26新增 fabric, 
        #         world_size,
        #     ),
        #     nprocs=world_size,
        #     join=True,
        # )
        # ----------------------------bimanual---------------------------
    end_time = datetime.now()
    duration = (end_time - start_time)
    with open(os.path.join(seed_folder, "training.log"), "a") as f:
        f.write(f"# Training finished on: {end_time.isoformat()}")
        f.write(f"# Took {duration.total_seconds()}")
        f.write(os.linesep)
        f.write(os.linesep)

if __name__ == "__main__":
    peract_config.on_init()
    main()
