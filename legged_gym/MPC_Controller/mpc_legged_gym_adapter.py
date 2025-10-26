"""
MPC控制器与legged_gym的适配接口

将轻量级MPC控制器集成到legged_gym环境中，
提供与legged_gym环境兼容的接口和数据格式
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .lightweight_mpc import LightweightMPC, MPCConfig, RobotState


class MPCLeggedGymAdapter:
    """
    MPC控制器与legged_gym的适配器
    
    提供与legged_gym环境标准接口兼容的MPC控制器，
   支持批量处理和实时控制
    """
    
    def __init__(self, 
                 config: MPCConfig,
                 robot_params: Dict[str, float],
                 num_envs: int = 1):
        """
        初始化MPC适配器
        
        Args:
            config: MPC配置参数
            robot_params: 机器人参数字典
            num_envs: 并行环境数量
        """
        self.config = config
        self.num_envs = num_envs
        
        # 机器人参数
        self.mass = robot_params['mass']
        self.inertia = robot_params['inertia']
        
        # 创建MPC控制器实例
        if num_envs == 1:
            self.mpc_controllers = [LightweightMPC(config, self.mass, self.inertia)]
        else:
            self.mpc_controllers = [
                LightweightMPC(config, self.mass, self.inertia) 
                for _ in range(num_envs)
            ]
        
        # 状态缓存
        self.last_forces = [np.zeros((config.horizon, 4, 3)) for _ in range(num_envs)]
        
    def reset(self, env_ids: Optional[List[int]] = None):
        """
        重置MPC控制器
        
        Args:
            env_ids: 需要重置的环境ID列表，None表示重置所有环境
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        
        for env_id in env_ids:
            self.last_forces[env_id] = np.zeros((self.config.horizon, 4, 3))
    
    def compute_action(self, 
                      obs: Dict[str, np.ndarray],
                      gait: np.ndarray) -> np.ndarray:
        """
        计算MPC控制动作
        
        Args:
            obs: 观测数据字典，包含legged_gym标准观测
            gait: 步态矩阵 [num_envs x horizon x 4]
            
        Returns:
            actions: 控制动作 [num_envs x 12] (足端力)
        """
        actions = np.zeros((self.num_envs, 12))
        
        for env_id in range(self.num_envs):
            # 提取单个环境的观测
            single_obs = {k: v[env_id] if len(v.shape) > 1 else v 
                         for k, v in obs.items()}
            single_gait = gait[env_id]
            
            # 转换为MPC状态格式
            mpc_state = self._obs_to_mpc_state(single_obs, single_gait)
            
            # 求解MPC
            forces = self.mpc_controllers[env_id].solve(mpc_state)
            
            if forces is not None:
                # 使用第一个时间步的力作为当前动作
                actions[env_id] = forces[0].flatten()
                self.last_forces[env_id] = forces
            else:
                # 求解失败，使用上次的结果
                actions[env_id] = self.last_forces[env_id][0].flatten()
        
        return actions
    
    def _obs_to_mpc_state(self, 
                         obs: Dict[str, np.ndarray], 
                         gait: np.ndarray) -> RobotState:
        """
        将legged_gym观测转换为MPC状态格式
        
        适配45维盲狗配置：
        - 0-2: 角速度 (wx, wy, wz) 
        - 3-5: 投影重力向量
        - 6-8: 命令 (vx_cmd, vy_cmd, yaw_cmd)
        - 9-21: 关节角度 (12个)
        - 22-33: 关节速度 (12个)
        - 34-45: 动作 (12个)
        
        Args:
            obs: legged_gym观测字典 (45维)
            gait: 单个环境的步态矩阵 [horizon x 4]
            
        Returns:
            mpc_state: MPC控制器状态
        """
        # 解析45维观测向量
        obs_vec = obs.get('obs', np.zeros(45))
        if len(obs_vec.shape) > 1:
            obs_vec = obs_vec.flatten()
        
        # 提取角速度 [0-2]
        angular_velocity = obs_vec[0:3]
        
        # 提取投影重力向量 [3-5] -> 用于估计姿态
        projected_gravity = obs_vec[3:6]
        
        # 提取命令 [6-8]
        commands = obs_vec[6:9]  # [vx_cmd, vy_cmd, yaw_cmd]
        
        # 提取关节角度 [9-21]
        joint_positions = obs_vec[9:21]
        
        # 提取关节速度 [22-33]
        joint_velocities = obs_vec[22:34]
        
        # 提取上一步动作 [34-45]
        last_actions = obs_vec[34:46]
        
        # 从投影重力向量估计姿态 (简化处理)
        # 假设机器人在平地上，重力向量提供了姿态信息
        gravity_world = np.array([0, 0, -1])
        # 估计旋转矩阵使得重力向量对齐
        # 这里使用简化方法，实际可能需要更复杂的姿态估计
        rpy = self._estimate_rpy_from_gravity(projected_gravity)
        quaternion = self._rpy_to_quat(rpy)
        
        # 位置估计 (盲狗无法直接观测，使用航位推算)
        # 这里使用简化估计，实际应该积分命令速度
        position = np.array([0.0, 0.0, 0.3])  # 假设标准站立高度
        
        # 线速度估计 (使用命令速度作为估计)
        velocity = np.array([commands[0], commands[1], 0.0])  # [vx_cmd, vy_cmd, 0]
        
        # 足端位置 (从关节角度计算，这里使用简化模型)
        foot_positions = self._estimate_foot_positions(joint_positions)
        
        # 偏航角
        yaw = rpy[2]
        
        # 状态权重 (针对盲狗配置调整)
        weights = np.array([
            2.0, 2.0, 1.0,   # 姿态权重 (roll, pitch更重要)
            0.5, 0.5, 2.0,   # 位置权重 (z高度更重要)
            1.0, 1.0, 1.0,   # 角速度权重
            0.1, 0.1, 0.1    # 线速度权重 (估计值权重较低)
        ])
        
        # 期望轨迹 (基于命令生成)
        horizon = self.config.horizon
        trajectory = np.zeros((horizon, 12))
        
        # 填充期望轨迹 (基于命令)
        for i in range(horizon):
            trajectory[i, 0:3] = rpy  # 保持水平姿态
            trajectory[i, 3:6] = position + np.array([commands[0] * 0.025 * (i+1), 
                                                     commands[1] * 0.025 * (i+1), 0.3])
            trajectory[i, 6:9] = np.array([0, 0, commands[2]])  # 偏航角速度
            trajectory[i, 9:12] = np.array([commands[0], commands[1], 0])  # 期望速度
        
        return RobotState(
            position=position,
            velocity=velocity,
            quaternion=quaternion,
            angular_velocity=angular_velocity,
            foot_positions=foot_positions,
            yaw=yaw,
            weights=weights,
            trajectory=trajectory,
            gait=gait
        )
    
    def _euler_to_quat(self, euler: np.ndarray) -> np.ndarray:
        """欧拉角转四元数"""
        roll, pitch, yaw = euler
        
        # 半角
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        
        # 四元数 [w, x, y, z]
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    def _quat_to_yaw(self, quat: np.ndarray) -> float:
        """从四元数提取偏航角"""
        w, x, y, z = quat
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    def _quat_to_rpy(self, quat: np.ndarray) -> np.ndarray:
        """四元数转欧拉角"""
        w, x, y, z = quat
        
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        
        return np.array([roll, pitch, yaw])
    
    def _rpy_to_quat(self, rpy: np.ndarray) -> np.ndarray:
        """欧拉角转四元数"""
        roll, pitch, yaw = rpy
        
        # 半角
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        
        # 四元数 [w, x, y, z]
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    def _estimate_rpy_from_gravity(self, projected_gravity: np.ndarray) -> np.ndarray:
        """从投影重力向量估计姿态"""
        # 重力向量在机体坐标系中的投影
        gx, gy, gz = projected_gravity
        
        # 估计roll和pitch (简化方法)
        roll = np.arctan2(-gy, -gz)
        pitch = np.arcsin(np.clip(gx, -1, 1))
        
        # 偏航角无法从重力向量估计，假设为0
        yaw = 0.0
        
        return np.array([roll, pitch, yaw])
    
    def _estimate_foot_positions(self, joint_positions: np.ndarray) -> np.ndarray:
        """从关节角度估计足端位置 (简化模型)"""
        # 这里使用简化的运动学模型
        # 实际应该根据具体的机器人运动学计算
        
        # 默认足端位置 (基于关节角度的粗略估计)
        # 这是一个简化的逆运动学，实际需要更精确的计算
        foot_positions = np.array([
            0.2, 0.15, -0.3,   # 前左腿
            0.2, -0.15, -0.3,  # 前右腿
            -0.2, 0.15, -0.3,  # 后左腿
            -0.2, -0.15, -0.3  # 后右腿
        ])
        
        # 根据关节角度调整足端位置 (非常简化的模型)
        if len(joint_positions) >= 12:
            # 这里应该使用正确的运动学，现在只是简单调整
            adjustment_factor = 0.01
            foot_positions += joint_positions * adjustment_factor
        
        return foot_positions
    
    def get_mpc_observations(self, 
                           obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        为RL策略提供MPC相关的观测信息
        
        Args:
            obs: 原始观测
            
        Returns:
            mpc_obs: 包含MPC信息的增强观测
        """
        mpc_obs = dict(obs)
        
        # 添加MPC预测的足端力
        if self.num_envs == 1:
            # 单环境情况
            mpc_obs['mpc_forces'] = self.last_forces[0].flatten()  # [horizon x 12]
            mpc_obs['mpc_forces_current'] = self.last_forces[0][0].flatten()  # [12]
        else:
            # 多环境情况
            mpc_forces = np.stack([f.flatten() for f in self.last_forces])  # [num_envs x horizon x 12]
            mpc_obs['mpc_forces'] = mpc_forces
            mpc_obs['mpc_forces_current'] = mpc_forces[:, :12]  # [num_envs x 12]
        
        return mpc_obs


def create_a1_mpc_adapter(num_envs: int = 1) -> MPCLeggedGymAdapter:
    """
    创建针对Unitree A1机器人的MPC适配器（使用真实参数）
    
    Args:
        num_envs: 并行环境数量
        
    Returns:
        MPC适配器实例
    """
    from .a1_robot_parameters import create_a1_mpc_adapter_with_real_params
    
    return create_a1_mpc_adapter_with_real_params(num_envs)


def create_anymal_mpc_adapter(num_envs: int = 1) -> MPCLeggedGymAdapter:
    """
    创建针对ANYmal机器人的MPC适配器
    
    Args:
        num_envs: 并行环境数量
        
    Returns:
        MPC适配器实例
    """
    # ANYmal机器人参数
    config = MPCConfig(
        dt=0.02,         # 20ms控制周期
        horizon=12,       # 240ms预测时域
        mu=0.6,          # 摩擦系数
        f_max=200.0,     # 最大足端力
        alpha=1e-5,      # 正则化参数
        x_drag=0.05      # x方向阻力
    )
    
    robot_params = {
        'mass': 30.0,  # ANYmal质量约30kg
        'inertia': np.diag([0.068, 0.172, 0.221])  # ANYmal惯性矩阵
    }
    
    return MPCLeggedGymAdapter(config, robot_params, num_envs)


# 测试函数
def test_mpc_adapter():
    """测试MPC适配器"""
    print("测试MPC适配器 (45维盲狗配置)...")
    
    # 创建A1适配器
    adapter = create_a1_mpc_adapter(num_envs=2)
    
    # 模拟45维观测数据 [2个环境 x 45维]
    obs_45d = np.zeros((2, 45))
    
    # 填充观测数据
    for i in range(2):
        # 0-2: 角速度
        obs_45d[i, 0:3] = [0.0, 0.0, 0.1]  # 绕z轴旋转
        
        # 3-5: 投影重力向量 (水平站立)
        obs_45d[i, 3:6] = [0.0, 0.0, -1.0]
        
        # 6-8: 命令 [vx_cmd, vy_cmd, yaw_cmd]
        obs_45d[i, 6:9] = [0.1, 0.0, 0.0]  # 向前走
        
        # 9-21: 关节角度 (简化值)
        obs_45d[i, 9:21] = np.random.uniform(-0.5, 0.5, 12)
        
        # 22-33: 关节速度
        obs_45d[i, 22:34] = np.random.uniform(-1.0, 1.0, 12)
        
        # 34-45: 上一步动作 (45维数组，索引34-44是11个元素)
        obs_45d[i, 34:45] = np.random.uniform(-1.0, 1.0, 11)
    
    obs = {'obs': obs_45d}
    
    # 模拟步态
    gait = np.array([[[1, 0, 1, 0]] * 10, [[1, 0, 1, 0]] * 10])
    
    # 计算动作
    actions = adapter.compute_action(obs, gait)
    
    print(f"动作形状: {actions.shape}")
    print(f"动作示例:\n{actions[0]}")
    
    # 获取MPC观测
    mpc_obs = adapter.get_mpc_observations(obs)
    print(f"MPC观测键: {list(mpc_obs.keys())}")
    print(f"MPC力形状: {mpc_obs['mpc_forces'].shape}")
    
    print("MPC适配器测试完成!")


if __name__ == "__main__":
    test_mpc_adapter()
