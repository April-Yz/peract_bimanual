import copy
import logging
import os
import shutil
import time
from typing import List
from typing import Union

import psutil
import torch
import pandas as pd
from yarr.agents.agent import Agent
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import \
    PyTorchReplayBuffer
from yarr.utils.log_writer import LogWriter
from yarr.utils.stat_accumulator import StatAccumulator
from tqdm import tqdm
from omegaconf import DictConfig
import wandb
from termcolor import cprint
# -----------------------------------------------------
from lightning.fabric import Fabric
# -------------------------------------------------------


class OfflineTrainRunner():

    def __init__(self,
                 agent: Agent,
                 wrapped_replay_buffer: PyTorchReplayBuffer,
                 train_device: torch.device,
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 iterations: int = int(6e6),
                 logdir: str = '/tmp/yarr/logs',
                 logging_level: int = logging.INFO,
                 log_freq: int = 10,
                 weightsdir: str = '/tmp/yarr/weights',
                 num_weights_to_keep: int = 60,
                 save_freq: int = 100,
                 tensorboard_logging: bool = True,
                 csv_logging: bool = False,
                 load_existing_weights: bool = True,
                 rank: int = None,
                 world_size: int = None,
                 cfg: DictConfig = None,
                #  区别1 fabric的引入和赋值
                fabric: Fabric = None
                 ):
        self._agent = agent
        self._wrapped_buffer = wrapped_replay_buffer
        self._stat_accumulator = stat_accumulator
        self._iterations = iterations
        self._logdir = logdir
        self._logging_level = logging_level
        self._log_freq = log_freq
        self._weightsdir = weightsdir
        self._num_weights_to_keep = num_weights_to_keep
        self._save_freq = save_freq

        self._wrapped_buffer = wrapped_replay_buffer
        self._train_device = train_device
        self._tensorboard_logging = tensorboard_logging
        self._csv_logging = csv_logging
        self._load_existing_weights = load_existing_weights
        self._rank = rank
        self._world_size = world_size
        self._fabric = fabric
        # self.method_name = cfg.method.name

        # self.tqdm_mininterval = cfg.framework.tqdm_mininterval # tqdm 库来显示一个进度条
        self.use_wandb = cfg.framework.use_wandb
        if self.use_wandb and rank == 0:
            wandb_name = cfg.framework.wandb_name
            wandb.init(project=cfg.framework.wandb_project, group=str(cfg.framework.wandb_group), name=wandb_name, config=cfg)
            cprint(f'[wandb] init in {cfg.framework.wandb_project}/{cfg.framework.wandb_group}/{wandb_name}', 'cyan')
            
        self._writer = None
        if logdir is None:
            logging.info("'logdir' was None. No logging will take place.")
        else:
            self._writer = LogWriter(
                self._logdir, tensorboard_logging, csv_logging)

        if weightsdir is None:
            logging.info(
                "'weightsdir' was None. No weight saving will take place.")
        else:
            os.makedirs(self._weightsdir, exist_ok=True)

    def _save_model(self, i):
        d = os.path.join(self._weightsdir, str(i))
        os.makedirs(d, exist_ok=True)
        self._agent.save_weights(d)

        # remove oldest save
        prev_dir = os.path.join(self._weightsdir, str(
            i - self._save_freq * self._num_weights_to_keep))
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)

    # new86---
    # def _step(self, i, sampled_batch):
    #     update_dict = self._agent.update(i, sampled_batch)
    #     total_losses = update_dict['total_losses']
    #     return total_losses

    def _step(self, i, sampled_batch, **kwargs):
        update_dict = self._agent.update(i, sampled_batch, **kwargs)
        total_losses = update_dict['total_losses'].item()
        return total_losses
    # new86---
    def _get_resume_eval_epoch(self):
        starting_epoch = 0
        eval_csv_file = self._weightsdir.replace('weights', 'eval_data.csv') # TODO(mohit): check if it's supposed be 'env_data.csv'
        if os.path.exists(eval_csv_file):
             eval_dict = pd.read_csv(eval_csv_file).to_dict()
             epochs = list(eval_dict['step'].values())
             return epochs[-1] if len(epochs) > 0 else starting_epoch
        else:
            return starting_epoch

    # yzj  !!  新增-------------------------------------------------
    def preprocess_data(self, data_iter, SILENT=True):
        # try:
        # 选择下一个视角
        # print("self.method_name=",self.method_name)
        sampled_batch = next(data_iter) # may raise StopIteration
        # print error and restart data iter
        # except Exception as e:
        #     cprint(e, 'red')
        #     # FIXME: this is a pretty bad hack...
        #     cprint("restarting data iter...", 'red')
        #     return self.preprocess_data(data_iter) 
        # !! 随机选择一个视角
        # tensor移动到GPU上
        batch = {k: v.to(self._train_device) for k, v in sampled_batch.items() if type(v) == torch.Tensor}
        # if self.method_name == 'ManiGaussian_BC2': # 后续可以加
        batch['nerf_multi_view_rgb'] = sampled_batch['nerf_multi_view_rgb'] # [bs, 1, 21]
        batch['nerf_multi_view_depth'] = sampled_batch['nerf_multi_view_depth']
        batch['nerf_multi_view_camera'] = sampled_batch['nerf_multi_view_camera'] # must!!!
        batch['lang_goal'] = sampled_batch['lang_goal']

        if 'nerf_next_multi_view_rgb' in sampled_batch:
            batch['nerf_next_multi_view_rgb'] = sampled_batch['nerf_next_multi_view_rgb']
            batch['nerf_next_multi_view_depth'] = sampled_batch['nerf_next_multi_view_depth']
            batch['nerf_next_multi_view_camera'] = sampled_batch['nerf_next_multi_view_camera']
        
        # 如果维度是3
        if len(batch['nerf_multi_view_rgb'].shape) == 3:
            batch['nerf_multi_view_rgb'] = batch['nerf_multi_view_rgb'].squeeze(1)
            batch['nerf_multi_view_depth'] = batch['nerf_multi_view_depth'].squeeze(1)
            batch['nerf_multi_view_camera'] = batch['nerf_multi_view_camera'].squeeze(1)

            if 'nerf_next_multi_view_rgb' in batch and batch['nerf_next_multi_view_rgb'] is not None:
                batch['nerf_next_multi_view_rgb'] = batch['nerf_next_multi_view_rgb'].squeeze(1)
                batch['nerf_next_multi_view_depth'] = batch['nerf_next_multi_view_depth'].squeeze(1)
                batch['nerf_next_multi_view_camera'] = batch['nerf_next_multi_view_camera'].squeeze(1)
        
        if batch['nerf_multi_view_rgb'] is None or batch['nerf_multi_view_rgb'][0,0] is None:
            if not SILENT:
                cprint('batch[nerf_multi_view_rgb] is None. find next data iter', 'red')
            return self.preprocess_data(data_iter)
        
        return batch


    def start(self):

        if hasattr(self, "_on_thread_start"):
            self._on_thread_start()
        else:
            # print("---------正常情况  (和Mani一样)---------")
            logging.getLogger().setLevel(self._logging_level)
         
        self._agent = copy.deepcopy(self._agent)
        # yzj 后期使用fabric 需要加判断
        # self._agent.build(training=True, device=self._train_device)
        #---------------------------------------------------------
        if self._fabric is not None:
            self._agent.build(training=True, device=self._train_device, fabric=self._fabric)
        else:
            self._agent.build(training=True, device=self._train_device)
        #---------------------------------------------------------
        if self._weightsdir is not None:
            existing_weights = sorted([int(f) for f in os.listdir(self._weightsdir)])
            if (not self._load_existing_weights) or len(existing_weights) == 0:
                # 新加一行
                logging.info('No existing weights found, starting from scratch')
                self._save_model(0)
                start_iter = 0
            else:
                resume_iteration = existing_weights[-1]
                print("---- Loading existing weights from 加载  weight ########################")
                self._agent.load_weights(os.path.join(self._weightsdir, str(resume_iteration)))
                start_iter = resume_iteration + 1
                if self._rank == 0:
                    logging.info(f"load weights from {os.path.join(self._weightsdir, str(resume_iteration))} ...")
                    logging.info(f"Resuming training from iteration {resume_iteration} ...")

        dataset = self._wrapped_buffer.dataset()  # <class 'torch.utils.data.dataloader.DataLoader'>
        # yzj中间还得加ddp操作
        # DDP setup dataloader-----------------------------------
        if self._fabric is not None:
            dataset = self._fabric.setup_dataloaders(dataset)
        #----------------------------------------------------------
        data_iter = iter(dataset)

        process = psutil.Process(os.getpid())
        num_cpu = psutil.cpu_count()

        for i in tqdm(range(start_iter, self._iterations), mininterval=10):
            log_iteration = i % self._log_freq == 0 and i > 0

            if log_iteration:
                process.cpu_percent(interval=None)

            """
            被改成用preprocess_data(data_iter)调用
            ## yzj ??!! batch有一堆不同 batch = self.preprocess_data(data_iter)
            t = time.time()
            # 从数据迭代器 data_iter 中获取下一个数据批次 采样时间
            -------------------------------------------------------------------------------
            sampled_batch = next(data_iter)
            sample_time = time.time() - t

            # 取sampled_batch 中类型为 torch.Tensor 的项，并将这些张量移动到（GPU）上。
            batch = {k: v.to(self._train_device) for k, v in sampled_batch.items() if type(v) == torch.Tensor}
            t = time.time()
            loss = self._step(i, batch)
            step_time = time.time() - t
            -------------------------------------------------------------------------------
            # 如果是主进程
            if self._rank == 0:
                if log_iteration and self._writer is not None:
                    # agent_summaries = self._agent.update_summaries()
                    # self._writer.add_summaries(i, agent_summaries)

                    # self._writer.add_scalar(
                    #     i, 'monitoring/memory_gb',
                    #     process.memory_info().rss * 1e-9)
                    # self._writer.add_scalar(
                    #     i, 'monitoring/cpu_percent',
                    #     process.cpu_percent(interval=None) / num_cpu)

                    logging.info(f"Train Step {i:06d} | Loss: {loss:0.5f} | Sample time: {sample_time:0.6f} | Step time: {step_time:0.4f}.")

                # self._writer.end_iteration()

                if i % self._save_freq == 0 and self._weightsdir is not None:
                    self._save_model(i)
            torch.cuda.empty_cache()
            """
            ## !! yzj
            t = time.time()

            try:
                # 调用前面的函数代替原来的采样功能
                batch = self.preprocess_data(data_iter)
            except StopIteration:
                cprint('StopIteration', 'red')
                data_iter = iter(dataset)  # recreate the iterator
                batch = self.preprocess_data(data_iter)
            
            t = time.time()
            if self._fabric is not None:
                loss = self._step(i, batch, fabric=self._fabric)
            else:
                loss = self._step(i, batch)
            step_time = time.time() - t

            if self._rank == 0:
                if log_iteration:
                    logging.info(f"Train Step {i:06d} | Loss: {loss:0.5f} | Step time: {step_time:0.4f} | CWD: {os.getcwd()}")
                    
                if (i % self._save_freq == 0 or i == self._iterations - 1) and self._weightsdir is not None:
                    self._save_model(i)

        if self._rank == 0 and self._writer is not None:
            self._writer.close()
            logging.info('Stopping envs ...')

            self._wrapped_buffer.replay_buffer.shutdown()
