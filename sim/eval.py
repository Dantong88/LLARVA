import gc
import logging
import os
import sys
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, ListConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from yarr.runners.independent_env_runner import IndependentEnvRunner
from yarr.utils.stat_accumulator import SimpleAccumulator
from agents import llarva_bc
from helpers import utils

from yarr.utils.rollout_generator import RolloutGenerator
from torch.multiprocessing import Process, Manager

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def eval_seed(eval_cfg,
              logdir,
              env_device,
              multi_task,
              env_config) -> None:

    tasks = eval_cfg.rlbench.tasks
    rg = RolloutGenerator()

    agent = llarva_bc.launch_utils.create_agent(eval_cfg)
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    env_runner = IndependentEnvRunner(
        train_env=None,
        agent=agent,
        train_replay_buffer=None,
        num_train_envs=0,
        num_eval_envs=eval_cfg.framework.eval_envs,
        rollout_episodes=99999,
        eval_episodes=eval_cfg.framework.eval_episodes,
        training_iterations=100000,
        eval_from_eps_number=eval_cfg.framework.eval_from_eps_number,
        episode_length=eval_cfg.rlbench.episode_length,
        stat_accumulator=stat_accum,
        weightsdir=eval_cfg.method.ckpt,
        logdir=logdir,
        env_device=env_device,
        rollout_generator=rg,
        num_eval_runs=len(tasks),
        multi_task=multi_task)

    manager = Manager()
    save_load_lock = manager.Lock()
    writer_lock = manager.Lock()

    weight = eval_cfg.method.ckpt
    env_runner.start(weight, save_load_lock, writer_lock, env_config, 0, eval_cfg.framework.eval_save_metrics, eval_cfg.cinematic_recorder)

    del env_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()


@hydra.main(config_name='eval', config_path='conf')
def main(eval_cfg: DictConfig) -> None:
    logging.info('\n' + OmegaConf.to_yaml(eval_cfg))

    start_seed = eval_cfg.framework.start_seed
    logdir = os.path.join(eval_cfg.framework.logdir,
                                eval_cfg.rlbench.task_name,
                                eval_cfg.method.name,
                        'seed%d' % start_seed)

    train_cfg = eval_cfg

    env_device = utils.get_device(eval_cfg.framework.gpu)
    logging.info('Using env device %s.' % str(env_device))

    gripper_mode = Discrete()
    # arm_action_mode = EndEffectorPoseViaPlanning()
    arm_action_mode = JointVelocity()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [t.replace('.py', '') for t in os.listdir(rlbench_task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]
    eval_cfg.rlbench.cameras = eval_cfg.rlbench.cameras if isinstance(
        eval_cfg.rlbench.cameras, ListConfig) else [eval_cfg.rlbench.cameras]
    obs_config = utils.create_obs_config(eval_cfg.rlbench.cameras,
                                         eval_cfg.rlbench.camera_resolution,)

    if eval_cfg.cinematic_recorder.enabled:
        obs_config.record_gripper_closing = True

    # single-task
    task = eval_cfg.rlbench.tasks[0]
    multi_task = False

    if task not in task_files:
        raise ValueError('Task %s not recognised!.' % task)
    task_class = task_file_to_task_class(task)

    env_config = (task_class,
                  obs_config,
                  action_mode,
                  eval_cfg.rlbench.demo_path,
                  eval_cfg.rlbench.episode_length,
                  eval_cfg.rlbench.headless,
                  train_cfg.rlbench.include_lang_goal_in_obs,
                  eval_cfg.rlbench.time_in_state,
                  eval_cfg.framework.record_every_n)

    logging.info('Evaluating seed %d.' % start_seed)
    eval_seed(eval_cfg,
              logdir,
              env_device,
              multi_task,
              env_config)

if __name__ == '__main__':
    main()
