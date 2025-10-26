"""
轻量级MPC控制器 - 基于Cheetah-Software架构的Python实现

参考Cheetah-Software的convexMPC实现，使用Python+NumPy+OSQP构建轻量级MPC接口
专门为legged_gym项目优化，支持四足机器人的模型预测控制
"""

import numpy as np
import osqp
from scipy.linalg import expm
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MPCConfig:
    """MPC配置参数 - 对应Cheetah-Software的problem_setup结构"""
    dt: float = 0.025          # 时间步长 (s)
    horizon: int = 10          # 预测时域长度
    mu: float = 0.8            # 摩擦系数 (使用较高值，让RL处理摩擦变化)
    f_max: float = 100.0       # 最大足端力 (根据机械工程师建议)
    alpha: float = 1e-4        # 正则化参数
    x_drag: float = 0.1        # x方向阻力系数


@dataclass
class RobotState:
    """机器人状态 - 对应Cheetah-Software的update_data_t结构"""
    # 位置和速度
    position: np.ndarray       # [3] 位置 (x, y, z)
    velocity: np.ndarray       # [3] 速度 (vx, vy, vz)
    
    # 姿态和角速度
    quaternion: np.ndarray     # [4] 四元数 (w, x, y, z)
    angular_velocity: np.ndarray  # [3] 角速度 (wx, wy, wz)
    
    # 足端位置
    foot_positions: np.ndarray # [12] 足端位置 (4足 x 3轴)
    
    # 其他参数
    yaw: float                 # 偏航角
    weights: np.ndarray        # [12] 状态权重
    trajectory: np.ndarray     # [horizon x 12] 期望轨迹
    gait: np.ndarray          # [horizon x 4] 步态模式


class LightweightMPC:
    """
    轻量级MPC控制器
    
    基于Cheetah-Software的数学模型，使用Python实现：
    - 13维状态空间：[rpy, xyz, rpy_dot, xyz_dot, -g]
    - 12维控制输入：[4足 x 3轴足端力]
    - 20维约束：[4足 x 5维摩擦锥约束]
    """
    
    def __init__(self, config: MPCConfig, mass: float, inertia: np.ndarray):
        """
        初始化MPC控制器
        
        Args:
            config: MPC配置参数
            mass: 机器人质量 (kg)
            inertia: 惯性矩阵 [3x3]
        """
        self.config = config
        self.mass = mass
        self.inertia = inertia
        self.inv_inertia = np.linalg.inv(inertia)
        
        # 状态和控制维度
        self.state_dim = 13      # [rpy, xyz, rpy_dot, xyz_dot, -g]
        self.action_dim = 12     # [4足 x 3轴力]
        self.constraint_dim = 20 # [4足 x 5维约束]
        
        # 初始化矩阵
        self._init_matrices()
        
    def _init_matrices(self):
        """初始化MPC矩阵"""
        h = self.config.horizon
        
        # 状态空间矩阵
        self.A_ct = np.zeros((self.state_dim, self.state_dim))
        self.B_ct = np.zeros((self.state_dim, self.action_dim))
        
        # 离散化矩阵
        self.A_dt = np.zeros((self.state_dim, self.state_dim))
        self.B_dt = np.zeros((self.state_dim, self.action_dim))
        
        # QP矩阵
        self.A_qp = np.zeros((h * self.state_dim, self.state_dim))
        self.B_qp = np.zeros((h * self.state_dim, h * self.action_dim))
        
        # 目标函数矩阵
        self.P = np.zeros((h * self.action_dim, h * self.action_dim))
        self.q = np.zeros(h * self.action_dim)
        
        # 约束矩阵
        self.A_constraint = np.zeros((h * self.constraint_dim, h * self.action_dim))
        self.lb = np.zeros(h * self.constraint_dim)
        self.ub = np.zeros(h * self.constraint_dim)
        
        # OSQP求解器
        self.solver = None
        
    def _compute_continuous_matrices(self, state: RobotState) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算连续时间状态空间矩阵
        
        对应Cheetah-Software的ct_ss_mats函数
        
        Args:
            state: 当前机器人状态
            
        Returns:
            A_ct: 连续时间状态矩阵 [13x13]
            B_ct: 连续时间控制矩阵 [13x12]
        """
        # 重置矩阵
        A_ct = np.zeros((self.state_dim, self.state_dim))
        B_ct = np.zeros((self.state_dim, self.action_dim))
        
        # 状态矩阵 A_ct
        # 位置导数 = 速度
        A_ct[3, 9] = 1.0    # dx/dt = vx
        A_ct[4, 10] = 1.0   # dy/dt = vy  
        A_ct[5, 11] = 1.0   # dz/dt = vz
        
        # 速度阻力
        A_ct[11, 9] = self.config.x_drag  # vx阻力
        
        # 姿态导数 = 角速度 (简化模型)
        A_ct[0:3, 6:9] = self._rpy_matrix(state.yaw).T
        
        # 控制矩阵 B_ct
        # 重构足端位置矩阵 [3x4]
        foot_pos_matrix = state.foot_positions.reshape(4, 3).T
        
        for i in range(4):
            # 角速度控制项
            r_cross = self._cross_matrix(self.inv_inertia, foot_pos_matrix[:, i])
            B_ct[6:9, i*3:(i+1)*3] = r_cross
            
            # 线速度控制项
            B_ct[9:12, i*3:(i+1)*3] = np.eye(3) / self.mass
            
        return A_ct, B_ct
    
    def _discretize_matrices(self, A_ct: np.ndarray, B_ct: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        离散化状态空间矩阵
        
        对应Cheetah-Software的c2qp函数
        
        Args:
            A_ct: 连续时间状态矩阵
            B_ct: 连续时间控制矩阵
            
        Returns:
            A_dt: 离散时间状态矩阵
            B_dt: 离散时间控制矩阵
        """
        # 构造增广矩阵 [A, B; 0, I]
        AB = np.zeros((self.state_dim + self.action_dim, self.state_dim + self.action_dim))
        AB[:self.state_dim, :self.state_dim] = A_ct * self.config.dt
        AB[:self.state_dim, self.state_dim:] = B_ct * self.config.dt
        AB[self.state_dim:, self.state_dim:] = np.eye(self.action_dim)
        
        # 计算矩阵指数
        AB_exp = expm(AB)
        
        # 提取离散化矩阵
        A_dt = AB_exp[:self.state_dim, :self.state_dim]
        B_dt = AB_exp[:self.state_dim, self.state_dim:]
        
        return A_dt, B_dt
    
    def _compute_qp_matrices(self, A_dt: np.ndarray, B_dt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算QP问题矩阵
        
        Args:
            A_dt: 离散时间状态矩阵
            B_dt: 离散时间控制矩阵
            
        Returns:
            A_qp: QP状态矩阵
            B_qp: QP控制矩阵
        """
        h = self.config.horizon
        
        # 初始化
        A_qp = np.zeros((h * self.state_dim, self.state_dim))
        B_qp = np.zeros((h * self.state_dim, h * self.action_dim))
        
        # 构建预测时域的动态方程
        A_power = np.eye(self.state_dim)
        
        for i in range(h):
            A_qp[i*self.state_dim:(i+1)*self.state_dim, :] = A_power
            
            B_block = np.zeros((self.state_dim, h * self.action_dim))
            for j in range(i+1):
                B_block[:, j*self.action_dim:(j+1)*self.action_dim] = np.linalg.matrix_power(A_dt, i-j) @ B_dt
            
            B_qp[i*self.state_dim:(i+1)*self.state_dim, :] = B_block
            A_power = A_power @ A_dt
            
        return A_qp, B_qp
    
    def _build_qp_problem(self, state: RobotState, A_qp: np.ndarray, B_qp: np.ndarray) -> bool:
        """
        构建QP优化问题
        
        Args:
            state: 机器人状态
            A_qp: QP状态矩阵
            B_qp: QP控制矩阵
            
        Returns:
            bool: 是否成功构建QP问题
        """
        try:
            h = self.config.horizon
            
            # 当前状态 [13]
            x0 = self._state_to_vector(state)
            
            # 期望状态 [horizon x 12] -> 扩展到 [horizon x 13]
            if state.trajectory.shape[1] == 12:
                # 扩展12维轨迹到13维状态空间
                x_des = np.zeros(h * self.state_dim)
                for i in range(h):
                    x_des[i*self.state_dim:(i+1)*self.state_dim-1] = state.trajectory[i]
                    x_des[(i+1)*self.state_dim-1] = -9.81  # 重力
            else:
                x_des = state.trajectory.reshape(h * self.state_dim)
            
            # 状态权重矩阵 - 扩展到horizon x 13维状态空间
            W = np.zeros((h * self.state_dim, h * self.state_dim))
            for i in range(h):
                W[i*self.state_dim:(i+1)*self.state_dim-1, i*self.state_dim:(i+1)*self.state_dim-1] = np.diag(state.weights)
                W[(i+1)*self.state_dim-1, (i+1)*self.state_dim-1] = 1.0  # 重力权重
            
            # 目标函数: min ||x - x_des||^2_W + alpha||u||^2
            # = u^T (B^T W B + alpha I) u - 2 x^T A^T W B u + const
            
            # 计算目标函数矩阵
            self.P = 2 * (B_qp.T @ W @ B_qp + self.config.alpha * np.eye(h * self.action_dim))
            
            # 计算目标函数向量
            state_error = A_qp @ x0 - x_des
            self.q = 2 * B_qp.T @ W @ state_error
            
            # 构建约束矩阵 (摩擦锥约束)
            self._build_friction_constraints(state.gait)
            
            return True
            
        except Exception as e:
            print(f"构建QP问题失败: {e}")
            return False
    
    def _build_friction_constraints(self, gait: np.ndarray):
        """
        构建摩擦锥约束
        
        Args:
            gait: 步态矩阵 [horizon x 4]
        """
        h = self.config.horizon
        
        for i in range(h):
            for j in range(4):
                # 约束索引
                constraint_idx = i * self.constraint_dim + j * 5
                action_idx = i * self.action_dim + j * 3
                
                if gait[i, j] == 1:  # 足端接触
                    # 摩擦锥约束: |fx| <= mu * fz, |fy| <= mu * fz, fz >= 0, fz <= f_max
                    self.A_constraint[constraint_idx, action_idx] = 1      # fx
                    self.A_constraint[constraint_idx, action_idx + 2] = -self.config.mu  # -mu*fz
                    self.ub[constraint_idx] = 0
                    self.lb[constraint_idx] = -np.inf
                    
                    self.A_constraint[constraint_idx + 1, action_idx + 1] = 1  # fy
                    self.A_constraint[constraint_idx + 1, action_idx + 2] = -self.config.mu  # -mu*fz
                    self.ub[constraint_idx + 1] = 0
                    self.lb[constraint_idx + 1] = -np.inf
                    
                    self.A_constraint[constraint_idx + 2, action_idx] = -1     # -fx
                    self.A_constraint[constraint_idx + 2, action_idx + 2] = -self.config.mu  # -mu*fz
                    self.ub[constraint_idx + 2] = 0
                    self.lb[constraint_idx + 2] = -np.inf
                    
                    self.A_constraint[constraint_idx + 3, action_idx + 1] = -1  # -fy
                    self.A_constraint[constraint_idx + 3, action_idx + 2] = -self.config.mu  # -mu*fz
                    self.ub[constraint_idx + 3] = 0
                    self.lb[constraint_idx + 3] = -np.inf
                    
                    self.A_constraint[constraint_idx + 4, action_idx + 2] = 1  # fz
                    self.ub[constraint_idx + 4] = self.config.f_max
                    self.lb[constraint_idx + 4] = 0
                else:  # 足端悬空
                    # 力必须为0
                    self.A_constraint[constraint_idx:constraint_idx + 3, action_idx:action_idx + 3] = np.eye(3)
                    self.ub[constraint_idx:constraint_idx + 3] = 0
                    self.lb[constraint_idx:constraint_idx + 3] = 0
                    
                    # 冗余约束设为0
                    self.ub[constraint_idx + 3:constraint_idx + 5] = 0
                    self.lb[constraint_idx + 3:constraint_idx + 5] = 0
    
    def solve(self, state: RobotState) -> Optional[np.ndarray]:
        """
        求解MPC问题
        
        Args:
            state: 当前机器人状态
            
        Returns:
            优化的足端力 [horizon x 12] 或 None (求解失败)
        """
        try:
            # 计算连续时间矩阵
            A_ct, B_ct = self._compute_continuous_matrices(state)
            
            # 离散化
            A_dt, B_dt = self._discretize_matrices(A_ct, B_ct)
            
            # 计算QP矩阵
            A_qp, B_qp = self._compute_qp_matrices(A_dt, B_dt)
            
            # 构建QP问题
            if not self._build_qp_problem(state, A_qp, B_qp):
                return None
            
            # 设置OSQP问题
            m = osqp.OSQP()
            
            # 转换为稀疏矩阵
            import scipy.sparse as sp
            P_sparse = sp.csc_matrix(self.P)
            A_sparse = sp.csc_matrix(self.A_constraint)
            
            # 设置问题数据 - 调整参数以提高求解成功率
            m.setup(P=P_sparse, q=self.q, A=A_sparse, l=self.lb, u=self.ub, 
                   verbose=False, polish=True, max_iter=4000, eps_abs=1e-6, eps_rel=1e-6,
                   eps_prim_inf=1e-6, eps_dual_inf=1e-6, alpha=1.6)
            
            # 求解
            results = m.solve()
            
            # 检查求解状态
            if results.info.status == 'solved':
                # 理想情况：精确求解
                return results.x.reshape(self.config.horizon, 4, 3)
            elif results.info.status == 'solved inaccurate':
                # 可接受：近似求解，但仍可使用
                return results.x.reshape(self.config.horizon, 4, 3)
            elif results.info.status == 'max iter reached':
                # 达到最大迭代次数，但仍可能有可用解
                if hasattr(results, 'x') and results.x is not None:
                    return results.x.reshape(self.config.horizon, 4, 3)
                else:
                    print(f"OSQP求解失败: {results.info.status}")
                    return None
            else:
                print(f"OSQP求解失败: {results.info.status}")
                return None
            
            # 返回足端力序列
            forces = results.x.reshape(self.config.horizon, 4, 3)
            return forces
            
        except Exception as e:
            print(f"MPC求解失败: {e}")
            return None
    
    def _state_to_vector(self, state: RobotState) -> np.ndarray:
        """将机器人状态转换为状态向量"""
        # 四元数到欧拉角
        rpy = self._quat_to_rpy(state.quaternion)
        
        # 构建状态向量 [rpy, xyz, rpy_dot, xyz_dot, -g]
        x = np.zeros(self.state_dim)
        x[0:3] = rpy
        x[3:6] = state.position
        x[6:9] = state.angular_velocity
        x[9:12] = state.velocity
        x[12] = -9.81  # 重力加速度
        
        return x
    
    def _quat_to_rpy(self, q: np.ndarray) -> np.ndarray:
        """四元数转欧拉角"""
        w, x, y, z = q
        
        # 欧拉角
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        
        return np.array([roll, pitch, yaw])
    
    def _rpy_matrix(self, yaw: float) -> np.ndarray:
        """偏航角旋转矩阵"""
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    def _cross_matrix(self, I_inv: np.ndarray, r: np.ndarray) -> np.ndarray:
        """叉乘矩阵 I_inv * [r]x"""
        rx, ry, rz = r
        r_cross = np.array([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]])
        return I_inv @ r_cross


def create_default_config() -> MPCConfig:
    """创建默认MPC配置"""
    return MPCConfig(
        dt=0.025,
        horizon=10,
        mu=0.7,
        f_max=120.0,
        alpha=1e-5,
        x_drag=0.1
    )


def create_test_state() -> RobotState:
    """创建测试机器人状态"""
    return RobotState(
        position=np.array([0.0, 0.0, 0.3]),
        velocity=np.array([0.1, 0.0, 0.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        foot_positions=np.array([
            0.2, 0.15, -0.3,   # 前左腿
            0.2, -0.15, -0.3,  # 前右腿
            -0.2, 0.15, -0.3,  # 后左腿
            -0.2, -0.15, -0.3  # 后右腿
        ]),
        yaw=0.0,
        weights=np.ones(12),
        trajectory=np.zeros((10, 12)),
        gait=np.array([[1, 0, 1, 0]] * 10)  # 交替步态
    )


if __name__ == "__main__":
    # 测试轻量级MPC
    print("测试轻量级MPC控制器...")
    
    # 创建配置
    config = create_default_config()
    
    # 机器人参数
    mass = 10.0  # kg
    inertia = np.diag([0.1, 0.1, 0.1])  # kg*m^2
    
    # 创建MPC控制器
    mpc = LightweightMPC(config, mass, inertia)
    
    # 创建测试状态
    state = create_test_state()
    
    # 求解MPC
    forces = mpc.solve(state)
    
    if forces is not None:
        print("MPC求解成功!")
        print(f"足端力形状: {forces.shape}")
        print(f"第一时刻足端力:\n{forces[0]}")
    else:
        print("MPC求解失败!")