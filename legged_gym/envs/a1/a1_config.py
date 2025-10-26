# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.base.legged_robot import LeggedRobot
import torch


# 盲狗奖励函数实现
def _reward_command_consistency(self):
    """
    奖励与命令方向一致的关节运动
    原理：分析命令方向与关节运动模式的一致性
    """
    # 获取运动命令 (vx_cmd, vy_cmd, yaw_cmd)
    forward_cmd = self.commands[:, 0]  # 前进命令
    lateral_cmd = self.commands[:, 1]  # 侧向命令
    yaw_cmd = self.commands[:, 2]      # 转向命令
    
    # 前进运动：大腿关节应该前后摆动
    forward_motion = torch.abs(self.dof_vel[:, [0, 3, 6, 9]]).sum(dim=1)  # 髋关节前后运动
    forward_reward = forward_motion * torch.abs(forward_cmd)
    
    # 侧向运动：髋关节应该有侧向摆动
    lateral_motion = torch.abs(self.dof_vel[:, [1, 4, 7, 10]]).sum(dim=1)  # 髋关节侧向运动
    lateral_reward = lateral_motion * torch.abs(lateral_cmd)
    
    # 转向运动：左右腿应该有差异运动
    left_legs = self.dof_vel[:, [0, 1, 2, 6, 7, 8]]  # 左腿关节
    right_legs = self.dof_vel[:, [3, 4, 5, 9, 10, 11]]  # 右腿关节
    turn_asymmetry = torch.abs(left_legs.sum(dim=1) - right_legs.sum(dim=1))
    turn_reward = turn_asymmetry * torch.abs(yaw_cmd)
    
    return forward_reward + lateral_reward + turn_reward

def _reward_gait_coordination(self):
    """
    奖励协调的步态模式
    原理：四足动物应该有协调的腿部运动模式
    """
    # 获取腿部关节速度
    fl_leg = self.dof_vel[:, 0:3]   # 前左腿
    fr_leg = self.dof_vel[:, 3:6]   # 前右腿  
    rl_leg = self.dof_vel[:, 6:9]   # 后左腿
    rr_leg = self.dof_vel[:, 9:12]  # 后右腿
    
    # 计算腿部运动幅度
    fl_motion = torch.norm(fl_leg, dim=1)
    fr_motion = torch.norm(fr_leg, dim=1)
    rl_motion = torch.norm(rl_leg, dim=1)
    rr_motion = torch.norm(rr_leg, dim=1)
    
    # 对角步态协调：前左-后右，前右-后左应该相位相近
    diagonal1_coord = 1.0 - torch.abs(fl_motion - rr_motion) / (fl_motion + rr_motion + 0.1)
    diagonal2_coord = 1.0 - torch.abs(fr_motion - rl_motion) / (fr_motion + rl_motion + 0.1)
    
    return (diagonal1_coord + diagonal2_coord) * 0.5

def _reward_gait_frequency(self):
    """
    奖励合理的步态频率
    原理：避免过快或过慢的步态
    """
    # 计算腿部运动的频率特征
    leg_velocities = self.dof_vel[:, [0, 3, 6, 9]]  # 主要运动关节
    motion_magnitude = torch.norm(leg_velocities, dim=1)
    
    # 理想频率范围（通过运动幅度估算）
    target_frequency = 2.0  # 约2Hz的步态频率
    frequency_error = torch.abs(motion_magnitude - target_frequency)
    
    return torch.exp(-frequency_error)

def _reward_diagonal_pairing(self):
    """
    奖励对角腿协调运动
    原理：典型的四足步态中，对角腿应该同步运动
    """
    # 对角腿关节位置相似性
    fl_pos = self.dof_pos[:, 0:3]   # 前左腿
    fr_pos = self.dof_pos[:, 3:6]   # 前右腿
    rl_pos = self.dof_pos[:, 6:9]   # 后左腿
    rr_pos = self.dof_pos[:, 9:12]  # 后右腿
    
    # 对角腿位置差异
    diagonal1_diff = torch.norm(fl_pos - rr_pos, dim=1)
    diagonal2_diff = torch.norm(fr_pos - rl_pos, dim=1)
    
    # 转换为奖励（差异越小奖励越高）
    diagonal1_reward = torch.exp(-diagonal1_diff)
    diagonal2_reward = torch.exp(-diagonal2_diff)
    
    return (diagonal1_reward + diagonal2_reward) * 0.5

def _reward_joint_motion(self):
    """
    奖励合理的关节运动，避免完全静止
    原理：适度的关节运动表明机器人在尝试移动
    """
    # 计算所有关节的运动幅度
    joint_motion = torch.norm(self.dof_vel, dim=1)
    
    # 适度运动奖励（太小或太大都不好）
    optimal_motion = 0.5  # 最优运动幅度
    motion_error = torch.abs(joint_motion - optimal_motion)
    
    return torch.exp(-motion_error * 2.0)  # 高斯型奖励


# 动态添加盲狗奖励函数到LeggedRobot基类
LeggedRobot._reward_command_consistency = _reward_command_consistency
LeggedRobot._reward_gait_coordination = _reward_gait_coordination
LeggedRobot._reward_gait_frequency = _reward_gait_frequency
LeggedRobot._reward_diagonal_pairing = _reward_diagonal_pairing
LeggedRobot._reward_joint_motion = _reward_joint_motion


class A1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.0,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.0,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.30
        class scales( LeggedRobotCfg.rewards.scales ):
            # === 基于命令一致性的奖励（方案2）===
            command_consistency = 1.2       # 奖励与命令方向一致的关节运动
            gait_coordination = 0.8         # 奖励协调的步态模式
            
            # === 基于步态周期的奖励（方案3）===
            feet_air_time = 1.5             # 抬腿时间奖励（动态步态）
            gait_frequency = 0.5            # 奖励合理的步态频率
            diagonal_pairing = 0.6          # 奖励对角腿协调
            
            # === 姿态稳定性 ===
            base_height = -20.0              # 身体高度惩罚（适度）
            orientation = -1.5              # 身体姿态惩罚（保持水平）
            ang_vel_xy = -0.3               # 横滚/俯仰角速度惩罚
            
            # === 能量效率 ===
            torques = -1.0                 # 力矩惩罚（轻微）
            dof_acc = -2.5e-7                # 关节加速度惩罚
            action_rate = -0.08             # 动作平滑奖励
            
            # === 安全约束 ===
            collision = -6.0                # 碰撞惩罚（适度）
            dof_pos_limits = -10.0           # 关节限制惩罚
            termination = -10.0             # 终止惩罚
            
            # === 避免完全静止 ===
            joint_motion = 0.3              # 奖励合理的关节运动
            stand_still = -1.0              # 站立不动惩罚

class A1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_a1'
        max_iterations = 1500