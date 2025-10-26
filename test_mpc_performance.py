#!/usr/bin/env python3
"""
MPC性能测试脚本
用于测试和监控MPC求解器的性能
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import isaacgym
import torch
from legged_gym.envs.a1.a1_mpc import register_a1_mpc_env, A1MPCRoughCfg
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import get_args
import time

def test_mpc_performance():
    """测试MPC控制器性能"""
    print("=== MPC性能测试 ===")
    
    # 注册环境
    register_a1_mpc_env()
    
    # 创建环境
    args = get_args()
    args.task = "a1mpc"
    args.headless = True
    
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    print(f"环境创建成功:")
    print(f"  观测维度: {env.num_obs}")
    print(f"  动作维度: {env.num_actions}")
    print(f"  环境数量: {env.num_envs}")
    
    # 测试MPC求解性能
    print("\n=== MPC求解性能测试 ===")
    
    # 重置环境
    obs = env.reset()
    
    # 统计变量
    total_steps = 0
    mpc_success_count = 0
    mpc_fail_count = 0
    mpc_inaccurate_count = 0
    
    start_time = time.time()
    
    # 运行测试
    for step in range(100):
        # 随机动作
        actions = torch.randn(env.num_envs, env.num_actions, device=env.device) * 0.1
        
        # 执行步骤
        try:
            result = env.step(actions)
            if len(result) == 4:
                obs, rewards, dones, infos = result
            elif len(result) == 5:
                obs, rewards, dones, infos, _ = result
            else:
                print(f"意外的返回值数量: {len(result)}")
                break
            total_steps += env.num_envs
            
            # 检查MPC状态（如果有相关信息）
            if hasattr(env, 'mpc_wrapper'):
                # 这里可以添加MPC状态检查
                pass
                
            if step % 20 == 0:
                print(f"步骤 {step}: 平均奖励 = {rewards.mean().item():.4f}")
                
        except Exception as e:
            print(f"步骤 {step} 出错: {e}")
            break
    
    end_time = time.time()
    
    # 输出统计信息
    print(f"\n=== 测试完成 ===")
    print(f"总时间: {end_time - start_time:.2f}s")
    print(f"总步数: {total_steps}")
    print(f"平均步频: {total_steps / (end_time - start_time):.1f} steps/s")
    
    if hasattr(env, 'mpc_wrapper'):
        print(f"MPC统计:")
        print(f"  成功求解: {mpc_success_count}")
        print(f"  近似求解: {mpc_inaccurate_count}")
        print(f"  求解失败: {mpc_fail_count}")
        total_mpc = mpc_success_count + mpc_fail_count
        if total_mpc > 0:
            print(f"  成功率: {mpc_success_count / total_mpc * 100:.1f}%")
        else:
            print(f"  无MPC统计数据")

if __name__ == '__main__':
    test_mpc_performance()