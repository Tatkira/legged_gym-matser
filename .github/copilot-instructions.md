```instructions
# 给 AI 代理的快速入门（精简版）

目的：让自动化编码代理在本仓库中能立刻做出可运行、更改并验证的改动。

1) 架构要点（快速）：
   - 训练平台：基于 NVIDIA Isaac Gym（核心仿真） + `rsl_rl`（PPO runner）。
   - 代码三大区域：环境（`legged_gym/envs/`）、脚本入口（`legged_gym/scripts/`）、资源/模型（`resources/`, `onnx/`, `runs/`）。
   - 配置驱动：行为由 `*_config.py`（例如 `anymal_config.py`）控制，改动优先在配置中完成。

2) 关键文件（立即打开）：
   - `legged_gym/scripts/train.py`：训练入口，使用 `task_registry` 创建 env/runner。
   - `legged_gym/scripts/play.py`：加载并可视化训练/ONNX 模型。
   - `legged_gym/envs/__init__.py`：任务注册点（`task_registry.register`）。
   - `legged_gym/envs/*_config.py`：每个 env 的配置信息（`<Name>Cfg` / `<Name>CfgPpo`）。
   - `legged_gym/sim2sim.py`：MuJoCo ↔ ONNX 桥接示例（含可选 ROS2 依赖）。

3) 立即可用的最小命令（在项目根目录运行）：
```bash
# 安装包与依赖（示例）
cd legged_gym && pip install -e .
# 安装外部 runner
cd ../rsl_rl && git checkout v1.0.2 && pip install -e .
# 训练（小规模 smoke run）
python legged_gym/scripts/train.py --task=anymal_c_flat --num_envs=1 --headless
# 播放/可视化
python legged_gym/scripts/play.py --task=anymal_c_flat --headless
```

4) 项目特有约定（必须遵守）：
   - 奖励系统：`*_config.py` 中的 reward scale 字段名必须与 env 中的 reward 函数名对应（查看 env 的 add_reward/compute_reward）。
   - 配置优先：不要把超参硬编码到 env 类里；把它放到对应的 `<Env>_config.py`。
   - 继承复用：新增 env 通常继承已有实现并覆盖少量方法（查看 `legged_gym/envs/anymal*` 示例）。

5) 集成点与外部依赖（可操作说明）：
   - Isaac Gym: 必须事先安装且与系统 CUDA/PyTorch 匹配（见仓库 README）。
   - rsl_rl: 提供 PPO runner；仓库通过 `task_registry.make_alg_runner` 使用它。
   - ONNX: `onnx/legged.onnx` 为示例策略；`sim2sim.py` 展示如何使用 ONNX 策略在 MuJoCo 中运行。
   - ROS2（可选）: `sim2sim.py` 中有 `rclpy` 订阅逻辑；训练通常不需要 ROS。

6) 调试快速技巧（常见故障与验证）：
   - 若遇 libpython/库加载错误：检查 `LD_LIBRARY_PATH` 或安装系统 libpython（如 `libpython3.8`）。
   - 修改 env 后出现 NaN/崩溃：用 `--num_envs=1 --max_iterations=10 --headless` 复现并观察日志。
   - 渲染开销：训练时尽量使用 `--headless` 或在渲染时按 `v` 关闭可视化提升性能。

7) 变更/PR 要点给代理：
   - 当修改 reward 名称或 scale：同时更新对应 `env` 文件中实际实现函数以及 `*_config.py`。
   - 新增 env：在 `legged_gym/envs/__init__.py` 注册任务（参照已有 register 调用）。
   - 变更模型格式（ONNX/PyTorch）：确保 `onnx/`、`runs/` 路径和加载代码（`play.py`/`sim2sim.py`）一致。

更多细节可按子目录细化（例如把 `legged_gym/envs/anymal*` 的 3 个重要文件片段内嵌到本说明）。
``` 