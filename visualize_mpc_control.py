#!/usr/bin/env python3
"""
MPC可视化测试脚本
用于检查MPC-RL混合控制的效果
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import isaacgym
import torch
from legged_gym.envs.a1.a1_mpc import register_a1_mpc_env
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import get_args

def visualize_mpc_control():
    """可视化MPC控制效果"""
    print("=== MPC控制可视化测试 ===")
    
    # 注册环境
    register_a1_mpc_env()
    
    # 创建环境（不使用headless以显示图形）
    args = get_args()
    args.task = "a1mpc"
    args.headless = False  # 显示图形窗口
    args.num_envs = 4      # 使用较少环境以便观察
    
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    print(f"环境创建成功:")
    print(f"  观测维度: {env.num_obs}")
    print(f"  动作维度: {env.num_actions}")
    print(f"  环境数量: {env.num_envs}")
    print(f"\n控制说明:")
    print(f"  - MPC权重: {env.mpc_wrapper.config.mpc_weight:.2f}")
    print(f"  - RL权重: {env.mpc_wrapper.config.rl_weight:.2f}")
    print(f"  - MPC更新间隔: {env.mpc_wrapper.config.mpc_update_interval} 步")
    print(f"  - 最大足端力: {env.mpc_wrapper.mpc_adapter.mpc_controllers[0].config.f_max:.1f} N")
    print(f"\n按Ctrl+C停止测试")
    
    # 重置环境
    obs = env.reset()
    
    step = 0
    try:
        while True:
            # 使用较小的随机动作
            actions = torch.randn(env.num_envs, env.num_actions, device=env.device) * 0.05
            
            # 执行步骤
            result = env.step(actions)
            if len(result) == 4:
                obs, rewards, dones, infos = result
            elif len(result) == 5:
                obs, rewards, dones, infos, _ = result
            
            step += 1
            
            # 每100步输出一次统计
            if step % 100 == 0:
                print(f"\n步骤 {step}:")
                print(f"  平均奖励: {rewards.mean().item():.4f}")
                print(f"  动作范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
                
                # 检查动作是否合理
                action_norm = torch.norm(actions, dim=1)
                print(f"  动作范数: 均值={action_norm.mean().item():.3f}, 最大={action_norm.max().item():.3f}")
                
                # 检查观测中的基础部分
                base_obs = obs[:, :12]  # 关节位置和速度
                print(f"  关节位置范围: [{base_obs[:, :6].min().item():.3f}, {base_obs[:, :6].max().item():.3f}]")
                print(f"  关节速度范围: [{base_obs[:, 6:].min().item():.3f}, {base_obs[:, 6:].max().item():.3f}]")
                
                # 检查是否有异常大的值
                if torch.abs(obs).max() > 100:
                    print(f"  警告：观测中有异常大值: {obs.abs().max().item():.3f}")
                if torch.abs(actions).max() > 1.0:
                    print(f"  警告：动作中有异常大值: {actions.abs().max().item():.3f}")
    
    except KeyboardInterrupt:
        print(f"\n测试停止，总共运行了 {step} 步")
    
    env.close()
    print("可视化测试结束")

if __name__ == '__main__':
    visualize_mpc_control()
