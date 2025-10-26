#!/usr/bin/env python3
"""
MPC训练监控脚本
监控MPC-RL训练过程中的关键指标
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import isaacgym
import torch
from legged_gym.envs.a1.a1_mpc import register_a1_mpc_env
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import get_args

def monitor_training():
    """监控MPC训练过程"""
    print("=== MPC训练监控 ===")
    
    # 注册环境
    register_a1_mpc_env()
    
    # 创建环境
    args = get_args()
    args.task = "a1mpc"
    args.headless = True
    args.num_envs = 16  # 使用较少环境进行监控
    
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    print(f"环境配置:")
    print(f"  观测维度: {env.num_obs}")
    print(f"  动作维度: {env.num_actions}")
    print(f"  环境数量: {env.num_envs}")
    
    # 重置环境
    obs = env.reset()
    
    print(f"\n开始监控训练过程...")
    print(f"按Ctrl+C停止监控")
    
    try:
        step = 0
        while True:
            # 随机动作
            actions = torch.randn(env.num_envs, env.num_actions, device=env.device) * 0.1
            
            # 执行步骤
            result = env.step(actions)
            if len(result) == 4:
                obs, rewards, dones, infos = result
            elif len(result) == 5:
                obs, rewards, dones, infos, _ = result
            
            step += 1
            
            # 每50步输出一次统计信息
            if step % 50 == 0:
                print(f"\n步骤 {step}:")
                print(f"  平均奖励: {rewards.mean().item():.4f}")
                print(f"  奖励范围: [{rewards.min().item():.4f}, {rewards.max().item():.4f}]")
                print(f"  完成episode数: {dones.sum().item()}")
                
                # 检查观测范围
                print(f"  观测统计:")
                print(f"    均值: {obs.mean().item():.4f}")
                print(f"    标准差: {obs.std().item():.4f}")
                print(f"    范围: [{obs.min().item():.4f}, {obs.max().item():.4f}]")
                
                # 检查前几个观测值（基础观测）
                print(f"  基础观测 (前10维):")
                for i in range(min(10, obs.shape[1])):
                    print(f"    维度{i}: 均值={obs[:, i].mean().item():.4f}, 范围=[{obs[:, i].min().item():.4f}, {obs[:, i].max().item():.4f}]")
                
                # 检查MPC增强观测（后10维）
                if obs.shape[1] > 100:
                    print(f"  MPC增强观测 (最后10维):")
                    for i in range(max(0, obs.shape[1]-10), obs.shape[1]):
                        print(f"    维度{i}: 均值={obs[:, i].mean().item():.4f}, 范围=[{obs[:, i].min().item():.4f}, {obs[:, i].max().item():.4f}]")
    
    except KeyboardInterrupt:
        print(f"\n监控停止，总共运行了 {step} 步")
    
    print("监控结束")

if __name__ == '__main__':
    monitor_training()