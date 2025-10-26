"""
A1机器人MPC-RL混合控制环境

基于原始A1环境，集成MPC控制器实现分层混合控制架构
支持45维盲狗配置和MPC增强观测
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from isaacgym import gymapi

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from legged_gym.utils.helpers import class_to_dict

from legged_gym.MPC_Controller.mpc_hybrid_wrapper import create_hybrid_wrapper, MPCHybridWrapper


class A1MPCRobot(LeggedRobot):
    """
    A1机器人MPC-RL混合控制环境
    
    扩展原始A1环境，添加：
    1. MPC控制器集成
    2. 混合控制架构
    3. 增强观测空间
    4. 安全监控机制
    """
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """初始化A1 MPC环境"""
        # 调用父类初始化
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # 确保max_episode_length_s是int
        self.max_episode_length_s = int(self.max_episode_length_s)
        
        # 初始化MPC混合控制器
        self._init_mpc_controller()
        
        # 扩展观测空间维度
        # self.num_obs = 179  # 在_init_buffers中设置
        # self.num_privileged_obs = 179
        
        # MPC相关状态
        self.mpc_step_counter = 0
        self.gait_scheduler = self._create_gait_scheduler()
        
    def _init_mpc_controller(self):
        """初始化MPC混合控制器"""
        # 创建混合控制包装器
        self.mpc_wrapper = create_hybrid_wrapper(
            num_envs=self.num_envs, 
            robot_type='a1'
        )
        
        # 重置MPC控制器
        self.mpc_wrapper.reset()
        
    def _init_buffers(self):
        """重写初始化缓冲区以支持179维观测"""
        # 调用父类初始化
        super()._init_buffers()
        
        # 重新分配观测缓冲区为179维（45盲狗 + 134 MPC增强）
        self.num_obs = 179
        self.num_privileged_obs = 179
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        
        # 重新计算噪声尺度向量
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        
    def _create_gait_scheduler(self):
        """创建步态调度器"""
        # 简单的交替步态
        horizon = 10
        gait = np.zeros((self.num_envs, horizon, 4))
        
        # 初始化步态模式
        for i in range(self.num_envs):
            for t in range(horizon):
                if t % 2 == 0:
                    gait[i, t] = [1, 0, 1, 0]  # 左腿支撑
                else:
                    gait[i, t] = [0, 1, 0, 1]  # 右腿支撑
        
        return torch.from_numpy(gait).float().to(self.device)
    
    def step(self, actions):
        """执行一步仿真，集成MPC混合控制"""
        # 转换actions为numpy格式
        actions_np = actions.detach().cpu().numpy()
        
        # 获取当前观测
        obs_dict = self._get_obs_dict()
        
        # 获取当前步态
        current_gait = self._get_current_gait()
        
        # 执行MPC混合控制
        blended_actions, enhanced_obs = self.mpc_wrapper.step(
            obs=obs_dict,
            rl_actions=actions_np,
            gait=current_gait
        )
        
        # 转换回torch张量
        blended_actions = torch.from_numpy(blended_actions).float().to(self.device)
        
        # 使用融合后的动作执行仿真
        obs, privileged_obs, rewards, dones, infos = super().step(blended_actions)
        return obs, privileged_obs, rewards, dones, infos
    
    def _get_obs_dict(self) -> Dict[str, np.ndarray]:
        """获取观测字典格式"""
        # 提取原始观测
        obs_np = self.obs_buf.detach().cpu().numpy()
        
        # 截取原始45维观测（盲狗配置，去掉MPC增强部分）
        original_obs_dim = 45
        original_obs = obs_np[:, :original_obs_dim]
        
        return {
            'obs': original_obs
        }
    
    def _get_current_gait(self) -> np.ndarray:
        """获取当前步态矩阵"""
        # 更新步态调度器
        self.mpc_step_counter += 1
        
        # 循环步态
        phase = self.mpc_step_counter % 20
        
        # 生成当前步态
        current_gait = np.zeros((self.num_envs, 10, 4))
        for i in range(self.num_envs):
            for t in range(10):
                step_phase = (phase + t) % 20
                if step_phase < 10:
                    current_gait[i, t] = [1, 0, 1, 0]
                else:
                    current_gait[i, t] = [0, 1, 0, 1]
        
        return current_gait
    
    def reset(self):
        """重置环境"""
        # 调用父类重置
        obs, privileged_obs = super().reset()
        
        # 重置MPC控制器
        self.mpc_wrapper.reset()
        self.mpc_step_counter = 0
        
        # 重置步态调度器
        self.gait_scheduler = self._create_gait_scheduler()
        
        return obs, privileged_obs
        
        return self.obs_buf
    
    def reset_idx(self, env_ids):
        """重置指定环境"""
        # 调用父类重置
        super().reset_idx(env_ids)
        
        # 重置MPC控制器
        self.mpc_wrapper.reset(env_ids)
    
    def _get_noise_scale_vec(self, cfg):
        """重写噪声尺度向量以匹配179维观测"""
        # 先创建45维基础噪声向量（盲狗配置）
        noise_vec = torch.zeros(self.num_envs, 179, device=self.device)
        
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        # 盲狗观测的噪声（45维）
        # noise_vec[:, 0:3] = 0  # 线速度（盲狗无此观测）
        noise_vec[:, 3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[:, 6:9] = noise_scales.gravity * noise_level
        noise_vec[:, 9:12] = 0.  # 命令
        noise_vec[:, 12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[:, 24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[:, 36:45] = 0.  # 上次动作
        
        # MPC增强观测的噪声（134维）
        # mpc_forces (12维) - 较小噪声
        noise_vec[:, 45:57] = 0.01
        # mpc_weight (1维) - 无噪声
        noise_vec[:, 57] = 0.
        # safety_score (1维) - 无噪声  
        noise_vec[:, 58] = 0.
        # mpc_predictions (120维) - 较小噪声
        noise_vec[:, 59:179] = 0.01
        
        return noise_vec
    
    def compute_observations(self):
        """计算观测，包含MPC增强信息"""
        # 直接计算45维基础观测
        base_obs = torch.cat((  
            self.base_ang_vel  * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
            ),dim=-1)
        
        # 复制到观测缓冲区的前45维
        self.obs_buf[:, :45] = base_obs
        
        # 添加噪声（只对前45维）
        if self.add_noise:
            noise = (2 * torch.rand_like(base_obs) - 1) * self.noise_scale_vec[:, :45]
            self.obs_buf[:, :45] += noise
        
        # 更新MPC增强观测
        self._update_mpc_observations()
    
    def compute_reward(self):
        """计算MPC增强奖励"""
        # 先计算基础奖励（45维观测）
        base_obs = self.obs_buf[:, :45]
        
        # 计算基础奖励函数
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            # 检查奖励函数是否需要额外参数
            import inspect
            sig = inspect.signature(self.reward_functions[i])
            if len(sig.parameters) > 1:  # 需要额外参数
                rew = self.reward_functions[i](base_obs) * self.reward_scales[name]
            else:  # 只需要self参数
                rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        # 计算MPC相关奖励
        if hasattr(self.cfg, 'mpc') and self.cfg.mpc.enable_mpc:
            mpc_rewards = self._compute_mpc_rewards()
            self.rew_buf += mpc_rewards
            if 'mpc_rewards' not in self.episode_sums:
                self.episode_sums['mpc_rewards'] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.episode_sums['mpc_rewards'] += mpc_rewards
        
        # 只对奖励进行正值限制（不包含终止奖励）
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        
        # 添加终止奖励
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def _compute_mpc_rewards(self):
        """计算MPC相关奖励"""
        mpc_rewards = torch.zeros(self.num_envs, device=self.device)
        
        # 1. MPC足端力跟踪奖励
        if hasattr(self, 'mpc_wrapper') and hasattr(self.mpc_wrapper, 'last_mpc_forces'):
            mpc_forces = torch.from_numpy(self.mpc_wrapper.last_mpc_forces).float().to(self.device)
            # 计算与期望MPC力的偏差
            expected_mpc_forces = torch.zeros_like(mpc_forces)
            force_error = torch.norm(mpc_forces - expected_mpc_forces, dim=-1)
            mpc_rewards += -self.cfg.rewards.scales.mpc_force_tracking * force_error
        
        # 2. MPC权重遵循奖励
        if hasattr(self, 'mpc_wrapper') and hasattr(self.mpc_wrapper, 'config'):
            mpc_weight = self.mpc_wrapper.config.mpc_weight
            target_weight = 0.7  # 期望权重
            weight_error = abs(mpc_weight - target_weight)  # 使用Python内置abs
            # 转换为与mpc_rewards相同形状的张量
            weight_error_tensor = torch.full((self.num_envs,), weight_error, device=self.device)
            mpc_rewards += -self.cfg.rewards.scales.mpc_weight_adherence * weight_error_tensor
        
        # 3. 安全评分奖励
        if hasattr(self, 'mpc_wrapper') and hasattr(self.mpc_wrapper, 'last_mpc_forces'):
            safety_scores = torch.from_numpy(self.mpc_wrapper.safety_scores).float().to(self.device)
            mpc_rewards += self.cfg.rewards.scales.safety_score_reward * safety_scores
        
        # 4. MPC能量效率奖励（简化计算）
        mpc_forces = torch.from_numpy(self.mpc_wrapper.last_mpc_forces).float().to(self.device)
        force_magnitudes = torch.norm(mpc_forces.reshape(self.num_envs, 4, 3), dim=-1)
        mpc_rewards += -self.cfg.rewards.scales.mpc_energy_efficiency * force_magnitudes.sum(dim=-1)
        
        # 5. MPC稳定性奖励
        # 简化实现：基于MPC求解成功率
        mpc_success = torch.ones(self.num_envs, device=self.device)  # 假设MPC总是成功
        mpc_rewards += self.cfg.rewards.scales.mpc_stability_bonus * mpc_success
        
        return mpc_rewards
    
    def _update_mpc_observations(self):
        """更新MPC增强观测"""
        # 盲狗配置：45维原始观测
        original_obs_dim = 45
        
        # 获取原始观测
        obs_dict = self._get_obs_dict()
        
        # 获取MPC增强观测
        enhanced_obs = self.mpc_wrapper.enhance_observations(
            obs_dict, 
            self.mpc_wrapper.last_mpc_forces
        )
        
        # 添加MPC增强观测
        mpc_forces = torch.from_numpy(enhanced_obs['mpc_forces']).float().to(self.device)
        mpc_weight = torch.from_numpy(enhanced_obs['mpc_weight']).float().to(self.device)
        safety_score = torch.from_numpy(enhanced_obs['safety_score']).float().to(self.device)
        mpc_predictions = torch.from_numpy(enhanced_obs['mpc_predictions']).float().to(self.device)
        
        # 填充增强观测部分（从第45维开始）
        self.obs_buf[:, original_obs_dim:original_obs_dim+12] = mpc_forces
        self.obs_buf[:, original_obs_dim+12] = mpc_weight
        self.obs_buf[:, original_obs_dim+13] = safety_score
        self.obs_buf[:, original_obs_dim+14:original_obs_dim+134] = mpc_predictions
    
    def get_mpc_training_data(self) -> Dict[str, torch.Tensor]:
        """获取MPC训练数据"""
        training_data = self.mpc_wrapper.get_training_data()
        
        # 转换为torch张量
        torch_data = {}
        for key, value in training_data.items():
            if isinstance(value, np.ndarray):
                torch_data[key] = torch.from_numpy(value).float().to(self.device)
            else:
                torch_data[key] = value
        
        return torch_data
    
    def set_mpc_training_mode(self, training: bool):
        """设置MPC训练模式"""
        self.mpc_wrapper.set_training_mode(training)
    
    # MPC相关奖励函数实现
    def _reward_mpc_force_tracking(self):
        """奖励跟踪MPC足端力"""
        if not hasattr(self, 'mpc_wrapper') or self.mpc_wrapper is None:
            return torch.zeros(self.num_envs, device=self.device)
        
        # 获取MPC目标足端力
        mpc_forces = self.obs_buf[:, 45:57]  # MPC足端力观测
        
        # 估算实际足端力（简化）
        actual_forces = torch.abs(self.torques) * 10.0  # 简化的力估算
        
        # 计算跟踪误差
        force_error = torch.norm(mpc_forces - actual_forces, dim=1)
        
        # 转换为奖励（误差越小奖励越高）
        return torch.exp(-force_error)
    
    def _reward_mpc_weight_adherence(self):
        """奖励遵循MPC权重分配"""
        if not hasattr(self, 'mpc_wrapper') or self.mpc_wrapper is None:
            return torch.zeros(self.num_envs, device=self.device)
        
        # 获取MPC权重
        mpc_weight = self.obs_buf[:, 57]  # MPC权重观测
        
        # 检查RL动作是否与MPC权重一致
        # 这里简化处理，实际应该检查动作分布
        return mpc_weight  # 直接使用权重作为奖励
    
    def _reward_mpc_prediction_accuracy(self):
        """奖励MPC预测准确性"""
        if not hasattr(self, 'mpc_wrapper') or self.mpc_wrapper is None:
            return torch.zeros(self.num_envs, device=self.device)
        
        # 获取MPC预测信息
        mpc_predictions = self.obs_buf[:, 58:178]  # MPC预测观测
        
        # 简化的预测准确性评估
        # 实际应该比较预测与实际状态
        prediction_variance = torch.var(mpc_predictions, dim=1)
        
        # 预测越稳定（方差越小）奖励越高
        return torch.exp(-prediction_variance)
    
    def _reward_safety_score_reward(self):
        """奖励安全评分"""
        if not hasattr(self, 'mpc_wrapper') or self.mpc_wrapper is None:
            return torch.zeros(self.num_envs, device=self.device)
        
        # 获取安全评分
        safety_score = self.obs_buf[:, 58]  # 安全评分观测
        
        # 安全评分越高奖励越高
        return safety_score
    
    def _reward_mpc_energy_efficiency(self):
        """奖励MPC能量效率"""
        if not hasattr(self, 'mpc_wrapper') or self.mpc_wrapper is None:
            return torch.zeros(self.num_envs, device=self.device)
        
        # 计算能量消耗
        energy_consumption = torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)
        
        # 能量消耗越低奖励越高
        return torch.exp(-energy_consumption * 0.1)
    
    def _reward_mpc_stability_bonus(self):
        """奖励MPC稳定性"""
        if not hasattr(self, 'mpc_wrapper') or self.mpc_wrapper is None:
            return torch.zeros(self.num_envs, device=self.device)
        
        # 计算稳定性指标
        base_height = self.root_states[:, 2]  # 身体高度
        height_error = torch.abs(base_height - 0.42)  # 目标高度0.42m
        
        # 姿态稳定性
        orientation_error = torch.norm(self.base_quat[:, 0:3], dim=1)  # 四元数前三个分量
        
        # 稳定性越好（误差越小）奖励越高
        stability_score = torch.exp(-(height_error + orientation_error))
        
        return stability_score


class A1MPCRoughCfg(A1RoughCfg):
    """A1 MPC环境配置"""
    
    class env(A1RoughCfg.env):
        # 环境特定配置
        # num_envs = 64  # 使用与基础配置相同的环境数量
        num_observations = 45 + 134  # 盲狗45维 + MPC增强134维
        num_privileged_obs = 45 + 134  # 包含MPC信息
        num_actions = 12
        episode_length_s = 20  # 每个episode20秒
        
        # 观测缩放
        obs_scales = {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "height": 0.5,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "clip_observations": 100.0,
        }
        
    class safety:
        """安全配置"""
        enable_safety_monitor = True
        max_base_height = 0.8
        min_base_height = 0.2
        max_roll_pitch = 0.5
        
    class mpc:
        """MPC特定配置"""
        enable_mpc = True
        mpc_weight = 0.7
        rl_weight = 0.3
        mpc_frequency = 40
        rl_frequency = 50
    
    class rewards(A1RoughCfg.rewards):
        """MPC增强奖励函数"""
        
        soft_dof_pos_limit = 0.9
        base_height_target = 0.30
        
        class scales(A1RoughCfg.rewards.scales):
            # MPC相关奖励
            mpc_force_tracking = 2.0      # 跟踪MPC足端力
            mpc_weight_adherence = 1.0    # 遵循MPC权重分配
            mpc_prediction_accuracy = 0.5   # MPC预测准确性
            safety_score_reward = 1.5      # 安全评分奖励
            mpc_energy_efficiency = 0.3   # MPC能量效率
            mpc_stability_bonus = 0.8     # MPC稳定性奖励


class A1MPCRoughCfgPPO(A1RoughCfgPPO):
    """A1 MPC训练配置"""
    
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 2000  # 增加训练迭代
        
        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'mpc_a1'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
        
    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.015  # 增加探索性
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-4
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]  # 增加网络容量以处理增强观测
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'


# 注册环境
def register_a1_mpc_env():
    """注册A1 MPC环境到任务注册表"""
    from legged_gym.utils import task_registry
    
    task_registry.register(
        name="a1mpc",
        task_class=A1MPCRobot,
        env_cfg=A1MPCRoughCfg,
        train_cfg=A1MPCRoughCfgPPO
    )


if __name__ == "__main__":
    # 测试环境
    print("测试A1 MPC环境...")
    
    # 注册环境
    register_a1_mpc_env()
    
    # 创建环境实例
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import get_args
    
    args = get_args()
    args.task = "A1MPC"
    
    # 创建环境
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    print(f"环境创建成功!")
    print(f"观测维度: {env.num_obs}")
    print(f"动作维度: {env.num_actions}")
    print(f"环境数量: {env.num_envs}")
    
    # 测试步骤
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    
    for i in range(5):
        obs, rewards, dones, infos = env.step(actions)
        print(f"步骤 {i}: 观测形状={obs.shape}, 奖励形状={rewards.shape}")
    
    print("A1 MPC环境测试完成!")