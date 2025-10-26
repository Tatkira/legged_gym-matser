"""
MPC-RL混合控制训练脚本

与标准train.py完全兼容的MPC训练脚本
"""

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    # 注册MPC环境（如果需要）
    if args.task == "a1mpc" or args.task == "A1MPC":
        from legged_gym.envs.a1.a1_mpc import register_a1_mpc_env
        register_a1_mpc_env()
        # 统一使用小写
        args.task = "a1mpc"
        
        print(f"使用MPC增强环境: {args.task}")
        print(f"观测维度将包含MPC增强信息")
    
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    # 显示MPC环境信息
    if hasattr(env, 'num_obs') and env.num_obs > 100:
        print(f"MPC环境信息:")
        print(f"  观测维度: {env.num_obs} (包含MPC增强)")
        print(f"  动作维度: {env.num_actions}")
        print(f"  环境数量: {env.num_envs}")
    
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)