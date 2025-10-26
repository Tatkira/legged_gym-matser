"""
A1机器人真实参数配置

基于URDF文件提取的精确物理参数
用于MPC控制器和混合控制策略
"""

import numpy as np


class A1RobotParameters:
    """A1机器人真实物理参数"""
    
    # 基础物理参数
    BODY_MASS = 12.454  # kg - 机器人总质量
    
    # 身体惯性矩阵 (kg*m^2)
    BODY_INERTIA = np.array([
        [0.01683993, 8.3902e-05, 0.000597679],
        [8.3902e-05, 0.056579028, 2.5134e-05],
        [0.000597679, 2.5134e-05, 0.064713601]
    ])
    
    # 身体尺寸 (m)
    BODY_LENGTH = 0.267  # 长
    BODY_WIDTH = 0.194   # 宽
    BODY_HEIGHT = 0.114  # 高
    
    # 腿部连杆长度 (m)
    ABAD_LINK_LENGTH = 0.08505  # 髋关节长度
    HIP_LINK_LENGTH = 0.2       # 大腿连杆长度
    KNEE_LINK_LENGTH = 0.2      # 小腿连杆长度
    MAX_LEG_LENGTH = 0.359       # 最大伸展长度
    
    # 关节减速比
    GEAR_RATIO = 9.0
    
    # 电机参数
    MOTOR_TAU_MAX = {
        'abad': 20.0,   # N·m - 髋关节最大扭矩
        'hip': 55.0,    # N·m - 大腿关节最大扭矩
        'knee': 55.0    # N·m - 小腿关节最大扭矩
    }
    
    BATTERY_VOLTAGE = 24.0       # V - 电池电压
    MOTOR_KT = 0.9287           # N·m/A - 电机转矩常数
    
    # 各连杆惯性矩阵 (kg*m^2)
    ABAD_INERTIA = {
        'FR': np.array([[0.000469246, 9.409e-06, -3.42e-07],
                       [9.409e-06, 0.00080749, 4.66e-07],
                       [-3.42e-07, 4.66e-07, 0.000552929]]),
        'FL': np.array([[0.000469246, -9.409e-06, -3.42e-07],
                       [-9.409e-06, 0.00080749, -4.66e-07],
                       [-3.42e-07, -4.66e-07, 0.000552929]]),
        'RR': np.array([[0.000469246, -9.409e-06, 3.42e-07],
                       [-9.409e-06, 0.00080749, 4.66e-07],
                       [3.42e-07, 4.66e-07, 0.000552929]]),
        'RL': np.array([[0.000469246, 9.409e-06, 3.42e-07],
                       [9.409e-06, 0.00080749, -4.66e-07],
                       [3.42e-07, -4.66e-07, 0.000552929]])
    }
    
    HIP_INERTIA = {
        'FR_RR': np.array([[0.005529065, -4.825e-06, 0.000343869],
                          [-4.825e-06, 0.005139339, -2.2448e-05],
                          [0.000343869, -2.2448e-05, 0.001367788]]),
        'FL_RL': np.array([[0.005529065, 4.825e-06, 0.000343869],
                          [4.825e-06, 0.005139339, 2.2448e-05],
                          [0.000343869, 2.2448e-05, 0.001367788]])
    }
    
    KNEE_INERTIA = np.array([
        [0.002997972, 0.0, -0.000141163],
        [0.0, 0.003014022, 0.0],
        [-0.000141163, 0.0, 3.2426e-05]
    ])
    
    # 关节相对位置 (m)
    ABAD_LOCATION = {
        'FR': np.array([0.183, -0.047, 0]),
        'FL': np.array([0.183, 0.047, 0]),
        'RR': np.array([-0.183, -0.047, 0]),
        'RL': np.array([-0.183, 0.047, 0])
    }
    
    HIP_LOCATION = {
        'right': np.array([0, -0.08505, 0]),
        'left': np.array([0, 0.08505, 0])
    }
    
    KNEE_LOCATION = np.array([0, 0, -0.2])  # 所有小腿关节相同
    
    # MPC特定参数 (需要实验确定)
    # TODO: 需要机械同事计算最大足端力
    MAX_FOOT_FORCE = 120.0  # N - 临时值，需要更新
    
    # 摩擦系数 (交给RL处理，MPC中使用保守估计)
    FRICTION_COEFFICIENT = 0.6  # 保守估计
    
    # 默认站立姿态下的足端位置 (m)
    DEFAULT_FOOT_POSITIONS = np.array([
        [0.183, -0.047, -0.25],  # FR
        [0.183, 0.047, -0.25],   # FL
        [-0.183, -0.047, -0.25], # RR
        [-0.183, 0.047, -0.25]   # RL
    ])
    
    @classmethod
    def get_total_inertia(cls):
        """计算总惯性矩阵 (身体 + 所有腿部)"""
        total_inertia = cls.BODY_INERTIA.copy()
        
        # 添加所有腿部连杆的惯性贡献
        # 这里简化处理，实际应该考虑平行轴定理
        for leg_inertia in cls.ABAD_INERTIA.values():
            total_inertia += leg_inertia
            
        for leg_inertia in cls.HIP_INERTIA.values():
            total_inertia += leg_inertia
            
        total_inertia += 4 * cls.KNEE_INERTIA  # 4个膝关节
        
        return total_inertia
    
    @classmethod
    def get_mpc_parameters(cls):
        """获取MPC控制器所需参数"""
        return {
            'mass': cls.BODY_MASS,
            'inertia': cls.BODY_INERTIA,  # 使用身体惯性，腿部通过足端力体现
            'max_foot_force': cls.MAX_FOOT_FORCE,
            'friction_coefficient': cls.FRICTION_COEFFICIENT,
            'default_foot_positions': cls.DEFAULT_FOOT_POSITIONS
        }


def create_a1_mpc_adapter_with_real_params(num_envs: int = 1):
    """使用真实参数创建A1 MPC适配器"""
    from .mpc_legged_gym_adapter import MPCLeggedGymAdapter, MPCConfig
    
    # MPC配置
    config = MPCConfig(
        dt=0.025,        # 25ms控制周期
        horizon=10,       # 250ms预测时域
        mu=A1RobotParameters.FRICTION_COEFFICIENT,
        f_max=A1RobotParameters.MAX_FOOT_FORCE,
        alpha=1e-5,      # 正则化参数
        x_drag=0.1       # x方向阻力
    )
    
    # 机器人参数
    robot_params = A1RobotParameters.get_mpc_parameters()
    
    return MPCLeggedGymAdapter(config, robot_params, num_envs)


if __name__ == "__main__":
    # 测试参数
    print("A1机器人参数验证:")
    print(f"总质量: {A1RobotParameters.BODY_MASS} kg")
    print(f"身体惯性矩阵:\n{A1RobotParameters.BODY_INERTIA}")
    print(f"总惯性矩阵:\n{A1RobotParameters.get_total_inertia()}")
    print(f"MPC参数: {A1RobotParameters.get_mpc_parameters()}")