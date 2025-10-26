"""
MPC-RL混合控制包装器

实现MPC上层规划与RL下层执行的分层控制架构，
支持训练和推理两种模式
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass

from .mpc_legged_gym_adapter import MPCLeggedGymAdapter, create_a1_mpc_adapter


@dataclass
class HybridConfig:
    """混合控制配置"""
    mpc_weight: float = 0.3        # MPC输出权重 (降低，让RL主导)
    rl_weight: float = 0.7          # RL输出权重 (增加，让RL主导)
    mpc_frequency: int = 20         # MPC运行频率 (Hz) - 降低调用频率
    rl_frequency: int = 50          # RL运行频率 (Hz)
    adaptation_rate: float = 0.1    # RL适应MPC参考的速率
    safety_threshold: float = 0.8   # 安全阈值
    mpc_update_interval: int = 5    # MPC更新间隔 (仿真步数)


class MPCHybridWrapper:
    """
    MPC-RL混合控制包装器
    
    提供：
    1. MPC上层规划功能
    2. RL策略增强观测
    3. 混合控制输出融合
    4. 训练和推理模式切换
    """
    
    def __init__(self, 
                 mpc_adapter: MPCLeggedGymAdapter,
                 config: HybridConfig,
                 num_envs: int = 1):
        """
        初始化混合控制包装器
        
        Args:
            mpc_adapter: MPC控制器适配器
            config: 混合控制配置
            num_envs: 并行环境数量
        """
        self.mpc_adapter = mpc_adapter
        self.config = config
        self.num_envs = num_envs
        
        # 状态管理
        self.step_count = 0
        self.mpc_update_counter = 0
        self.last_mpc_forces = np.zeros((num_envs, 12))
        self.last_rl_actions = np.zeros((num_envs, 12))
        
        # 训练模式标志
        self.training_mode = True
        
        # 安全监控
        self.safety_scores = np.ones(num_envs)
        
    def reset(self, env_ids: Optional[np.ndarray] = None):
        """
        重置混合控制器
        
        Args:
            env_ids: 需要重置的环境ID
        """
        self.mpc_adapter.reset(env_ids)
        self.step_count = 0
        self.mpc_update_counter = 0
        
        if env_ids is None:
            self.last_mpc_forces[:] = 0
            self.last_rl_actions[:] = 0
            self.safety_scores[:] = 1.0
        else:
            # 确保env_ids是numpy数组
            if isinstance(env_ids, torch.Tensor):
                env_ids = env_ids.cpu().numpy()
            self.last_mpc_forces[env_ids] = 0
            self.last_rl_actions[env_ids] = 0
            self.safety_scores[env_ids] = 1.0
    
    def compute_mpc_reference(self, 
                            obs: Dict[str, np.ndarray],
                            gait: np.ndarray) -> np.ndarray:
        """
        计算MPC参考输出
        
        Args:
            obs: 观测数据
            gait: 步态矩阵
            
        Returns:
            mpc_forces: MPC足端力参考 [num_envs x 12]
        """
        # 检查是否需要更新MPC
        update_mpc = (self.step_count % (self.config.rl_frequency // self.config.mpc_frequency)) == 0
        
        if update_mpc:
            self.last_mpc_forces = self.mpc_adapter.compute_action(obs, gait)
            self.mpc_update_counter += 1
        
        return self.last_mpc_forces
    
    def enhance_observations(self, 
                           obs: Dict[str, np.ndarray],
                           mpc_forces: np.ndarray) -> Dict[str, np.ndarray]:
        """
        为RL策略增强观测信息
        
        Args:
            obs: 原始观测
            mpc_forces: MPC足端力
            
        Returns:
            enhanced_obs: 增强观测
        """
        enhanced_obs = dict(obs)
        
        # 添加MPC参考信息
        enhanced_obs['mpc_forces'] = mpc_forces.copy()
        enhanced_obs['mpc_weight'] = np.full(self.num_envs, self.config.mpc_weight)
        
        # 添加混合控制状态
        enhanced_obs['hybrid_step'] = np.full(self.num_envs, self.step_count)
        enhanced_obs['safety_score'] = self.safety_scores.copy()
        
        # 添加MPC预测信息（可选）
        mpc_obs = self.mpc_adapter.get_mpc_observations(obs)
        enhanced_obs['mpc_predictions'] = mpc_obs.get('mpc_forces', np.zeros((self.num_envs, 120)))
        
        return enhanced_obs
    
    def blend_actions(self, 
                     mpc_forces: np.ndarray,
                     rl_actions: np.ndarray,
                     safety_scores: Optional[np.ndarray] = None) -> np.ndarray:
        """
        融合MPC和RL控制输出
        
        Args:
            mpc_forces: MPC足端力 [num_envs x 12]
            rl_actions: RL动作 [num_envs x 12] 
            safety_scores: 安全评分 [num_envs]
            
        Returns:
            blended_actions: 融合后动作 [num_envs x 12]
        """
        if safety_scores is None:
            safety_scores = self.safety_scores
        
        # 动态调整权重
        mpc_weight = self.config.mpc_weight * safety_scores
        rl_weight = self.config.rl_weight
        
        # 归一化权重
        total_weight = mpc_weight + rl_weight
        mpc_weight = mpc_weight / total_weight
        rl_weight = rl_weight / total_weight
        
        # 融合动作
        blended_actions = (mpc_weight[:, np.newaxis] * mpc_forces + 
                          rl_weight[:, np.newaxis] * rl_actions)
        
        return blended_actions
    
    def compute_safety_scores(self, 
                            obs: Dict[str, np.ndarray],
                            mpc_forces: np.ndarray) -> np.ndarray:
        """
        计算安全评分
        
        Args:
            obs: 观测数据
            mpc_forces: MPC足端力
            
        Returns:
            safety_scores: 安全评分 [num_envs]
        """
        safety_scores = np.ones(self.num_envs)
        
        # 基于姿态稳定性
        if 'obs' in obs:
            obs_data = obs['obs']
            if len(obs_data.shape) == 1:
                obs_data = obs_data[np.newaxis, :]
            
            # 检查投影重力向量 (3-5维)
            projected_gravity = obs_data[:, 3:6]
            gravity_magnitude = np.linalg.norm(projected_gravity, axis=1)
            gravity_stability = 1.0 - np.abs(gravity_magnitude - 1.0) * 0.5
            
            # 检查角速度 (0-2维)
            angular_velocity = obs_data[:, 0:3]
            angular_magnitude = np.linalg.norm(angular_velocity, axis=1)
            angular_stability = np.maximum(0, 1.0 - angular_magnitude * 0.1)
            
            # 综合安全评分
            safety_scores = gravity_stability * angular_stability
        
        # 基于MPC输出合理性
        force_magnitudes = np.linalg.norm(mpc_forces.reshape(self.num_envs, 4, 3), axis=2)
        max_forces = np.max(force_magnitudes, axis=1)
        force_safety = np.maximum(0, 1.0 - (max_forces - 50.0) * 0.01)
        
        safety_scores = np.minimum(safety_scores, force_safety)
        
        # 平滑更新
        self.safety_scores = (0.9 * self.safety_scores + 0.1 * safety_scores)
        self.safety_scores = np.clip(self.safety_scores, 0.1, 1.0)
        
        return self.safety_scores
    
    def step(self, 
            obs: Dict[str, np.ndarray],
            rl_actions: np.ndarray,
            gait: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        执行一步混合控制
        
        Args:
            obs: 观测数据
            rl_actions: RL策略动作
            gait: 步态矩阵
            
        Returns:
            blended_actions: 融合后动作
            enhanced_obs: 增强观测
        """
        # 更新状态
        self.step_count += 1
        self.mpc_update_counter += 1
        
        # 控制MPC调用频率
        if self.mpc_update_counter >= self.config.mpc_update_interval:
            # 计算MPC参考
            mpc_forces = self.compute_mpc_reference(obs, gait)
            self.last_mpc_forces = mpc_forces
            self.mpc_update_counter = 0
        else:
            # 使用上次MPC结果
            mpc_forces = self.last_mpc_forces
        
        # 计算安全评分
        safety_scores = self.compute_safety_scores(obs, mpc_forces)
        
        # 融合控制输出
        blended_actions = self.blend_actions(mpc_forces, rl_actions, safety_scores)
        
        # 增强观测
        enhanced_obs = self.enhance_observations(obs, mpc_forces)
        self.last_rl_actions = rl_actions.copy()
        
        return blended_actions, enhanced_obs
    
    def get_training_data(self) -> Dict[str, np.ndarray]:
        """
        获取训练相关数据
        
        Returns:
            training_data: 训练数据字典
        """
        return {
            'mpc_forces': self.last_mpc_forces.copy(),
            'rl_actions': self.last_rl_actions.copy(),
            'safety_scores': self.safety_scores.copy(),
            'step_count': self.step_count,
            'mpc_update_count': self.mpc_update_counter
        }
    
    def set_training_mode(self, training: bool):
        """
        设置训练模式
        
        Args:
            training: 是否为训练模式
        """
        self.training_mode = training
        
        # 训练模式下增加探索性
        if training:
            self.config.rl_weight = min(0.5, self.config.rl_weight * 1.2)
            self.config.mpc_weight = 1.0 - self.config.rl_weight
        else:
            # 推理模式下更保守
            self.config.mpc_weight = 0.8
            self.config.rl_weight = 0.2


def create_hybrid_wrapper(num_envs: int = 1, 
                         robot_type: str = 'a1') -> MPCHybridWrapper:
    """
    创建混合控制包装器
    
    Args:
        num_envs: 并行环境数量
        robot_type: 机器人类型 ('a1', 'anymal')
        
    Returns:
        混合控制包装器实例
    """
    # 创建MPC适配器
    if robot_type == 'a1':
        mpc_adapter = create_a1_mpc_adapter(num_envs)
    elif robot_type == 'anymal':
        mpc_adapter = create_anymal_mpc_adapter(num_envs)
    else:
        raise ValueError(f"不支持的机器人类型: {robot_type}")
    
    # 混合控制配置
    config = HybridConfig(
        mpc_weight=0.3,        # 让RL主导
        rl_weight=0.7,         # 让RL主导
        mpc_frequency=20,      # 降低MPC频率
        rl_frequency=50,
        adaptation_rate=0.1,
        safety_threshold=0.8,
        mpc_update_interval=5  # 每5步更新一次MPC
    )
    
    return MPCHybridWrapper(mpc_adapter, config, num_envs)


def test_hybrid_wrapper():
    """测试混合控制包装器"""
    print("测试MPC-RL混合控制包装器...")
    
    # 创建混合包装器
    wrapper = create_hybrid_wrapper(num_envs=2, robot_type='a1')
    
    # 模拟观测数据
    obs = {
        'obs': np.random.randn(2, 45)
    }
    
    # 模拟RL动作
    rl_actions = np.random.randn(2, 12)
    
    # 模拟步态
    gait = np.array([[[1, 0, 1, 0]] * 10, [[1, 0, 1, 0]] * 10])
    
    # 执行控制步骤
    for step in range(10):
        blended_actions, enhanced_obs = wrapper.step(obs, rl_actions, gait)
        
        print(f"步骤 {step}:")
        print(f"  融合动作形状: {blended_actions.shape}")
        print(f"  增强观测键: {list(enhanced_obs.keys())}")
        print(f"  安全评分: {wrapper.safety_scores}")
    
    # 获取训练数据
    training_data = wrapper.get_training_data()
    print(f"训练数据键: {list(training_data.keys())}")
    
    print("混合控制包装器测试完成!")


if __name__ == "__main__":
    test_hybrid_wrapper()