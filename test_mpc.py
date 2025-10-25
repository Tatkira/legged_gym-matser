#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the MPC_Controller directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'legged_gym', 'MPC_Controller', 'convex_MPC'))

try:
    from mpc_osqp import ConvexMpc, QPSolverName
    print("✅ MPC模块导入成功")
except ImportError as e:
    print(f"❌ MPC模块导入失败: {e}")
    sys.exit(1)

def test_mpc_initialization():
    """测试MPC控制器初始化"""
    try:
        # 基本参数设置
        mass = 10.0  # kg
        inertia = [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1]  # 3x3惯性矩阵展开
        num_legs = 4
        planning_horizon = 10
        timestep = 0.025  # 25ms
        
        # 创建MPC控制器
        mpc = ConvexMpc(
            mass=mass,
            inertia=inertia,
            num_legs=num_legs,
            planning_horizon=planning_horizon,
            timestep=timestep,
            alpha=1e-5,
            qp_solver_name=QPSolverName.OSQP
        )
        
        print("✅ MPC控制器初始化成功")
        return mpc
        
    except Exception as e:
        print(f"❌ MPC控制器初始化失败: {e}")
        return None

def test_mpc_computation(mpc):
    """测试MPC计算功能"""
    try:
        # 测试输入数据
        qp_weights = [1.0] * 13  # 状态权重
        com_position = [0.0, 0.0, 0.3]  # 质心位置
        com_velocity = [0.0, 0.0, 0.0]  # 质心速度
        com_roll_pitch_yaw = [0.0, 0.0, 0.0]  # 姿态角
        ground_normal_vec = [0.0, 0.0, 1.0]  # 地面法向量
        com_angular_velocity = [0.0, 0.0, 0.0]  # 角速度
        
        # 足端接触状态 (4条腿，10个时间步)
        foot_contact_states = []
        for i in range(10):
            foot_contact_states.extend([1, 0, 1, 0])  # 交替步态
        foot_contact_states_flattened = foot_contact_states
        
        # 足端位置 (机体坐标系)
        foot_positions_body_frame = [
            0.2, 0.15, -0.3,   # 前左腿
            0.2, -0.15, -0.3,  # 前右腿
            -0.2, 0.15, -0.3,  # 后左腿
            -0.2, -0.15, -0.3  # 后右腿
        ]
        
        # 摩擦系数
        foot_friction_coeffs = [0.7, 0.7, 0.7, 0.7]
        
        # 期望状态
        desired_com_position = [0.0, 0.0, 0.3]
        desired_com_velocity = [0.1, 0.0, 0.0]  # 向前走
        desired_com_roll_pitch_yaw = [0.0, 0.0, 0.0]
        desired_com_angular_velocity = [0.0, 0.0, 0.0]
        
        # 执行MPC计算
        contact_forces = mpc.compute_contact_forces(
            qp_weights=qp_weights,
            com_position=com_position,
            com_velocity=com_velocity,
            com_roll_pitch_yaw=com_roll_pitch_yaw,
            ground_normal_vec=ground_normal_vec,
            com_angular_velocity=com_angular_velocity,
            foot_contact_states_flattened=foot_contact_states_flattened,
            foot_positions_body_frame=foot_positions_body_frame,
            foot_friction_coeffs=foot_friction_coeffs,
            desired_com_position=desired_com_position,
            desired_com_velocity=desired_com_velocity,
            desired_com_roll_pitch_yaw=desired_com_roll_pitch_yaw,
            desired_com_angular_velocity=desired_com_angular_velocity
        )
        
        # 验证输出
        expected_size = 4 * 3 * 10  # 4条腿 * 3个方向 * 10个时间步
        if len(contact_forces) == expected_size:
            print(f"✅ MPC计算成功，输出尺寸正确: {len(contact_forces)}")
            
            # 打印一些示例输出
            forces = np.array(contact_forces).reshape(10, 4, 3)
            print(f"   示例输出 (第一个时间步):")
            for leg in range(4):
                print(f"   腿{leg+1}力: [{forces[0, leg, 0]:.2f}, {forces[0, leg, 1]:.2f}, {forces[0, leg, 2]:.2f}]")
            
            return True
        else:
            print(f"❌ MPC输出尺寸错误: 期望{expected_size}, 实际{len(contact_forces)}")
            return False
            
    except Exception as e:
        print(f"❌ MPC计算失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== MPC控制器独立运行测试 ===\n")
    
    # 测试初始化
    mpc = test_mpc_initialization()
    if mpc is None:
        print("\n❌ 测试失败: MPC控制器初始化失败")
        return False
    
    # 测试计算
    if not test_mpc_computation(mpc):
        print("\n❌ 测试失败: MPC控制器计算失败")
        return False
    
    print("\n✅ 所有测试通过! MPC控制器可以独立运行")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
