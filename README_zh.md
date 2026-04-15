# Global Humanoid Robot Challenge 2026 Baseline

**English Version:** [README.md](README.md)

[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

全球人形机器人挑战赛 2026（Global Humanoid Robot Challenge 2026）官方基线代码仓库，基于 [LeRobot](https://github.com/huggingface/lerobot) 框架构建，覆盖仿真环境、数据采集、模型训练到部署的完整工作流。

---

## 项目概述

本仓库面向 **全球人形机器人挑战赛 2026** 参赛者与研发团队，提供统一的基线实现：

- 基于 **NVIDIA Isaac Sim** 搭建高保真机器人仿真环境
- 通过遥操作完成数据录制，输出标准化 **LeRobotDataset V2.1**
- 基于模仿学习算法进行模型训练与微调
- 使用官方预训练权重快速复现与对比实验

---

## 核心能力

| 能力                   | 说明                                                            |
| ---------------------- | --------------------------------------------------------------- |
| 🤖**仿真环境**   | 基于 NVIDIA Isaac Sim 的高保真机器人仿真                        |
| 📊**数据采集**   | 支持键盘、Pico 等遥操作；输出**LeRobotDataset V2.1** 格式 |
| 🧠**模型训练**   | 支持**ACT**、**Pi0** 等模仿学习算法                 |

---

## 资源说明

本项目部分大文件托管于 Hugging Face，首次使用前请先完成下载：

| 资源类别                | 本地目录                    | 远程地址                                                                                                    |
| ----------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 🤖 仿真环境与机器人资产 | `assets/`（Git 子模块）   | [UBTECH-Robotics/challenge2026_assets](https://huggingface.co/UBTECH-Robotics/challenge2026_assets)            |
| 📊 训练数据集           | `datasets/`               | [UBTECH-Robotics/challenge2026_dataset](https://huggingface.co/datasets/UBTECH-Robotics/challenge2026_dataset) |

### 快速下载

```bash
# 安装 huggingface-cli
pip install huggingface-hub

# 获取仿真资产（推荐使用 Git 子模块）
git submodule update --init --recursive

# 或手动下载到 ./assets
# huggingface-cli download UBTECH-Robotics/challenge2026_assets --local-dir ./assets --repo-type model

# 下载训练数据集
huggingface-cli download UBTECH-Robotics/challenge2026_dataset --local-dir ./datasets --repo-type dataset

```

---

## 系统要求

### 基础环境

- NVIDIA GPU
- Docker
- NVIDIA Container Toolkit
- CUDA 12.8+
- Python 3.11+

### 推荐配置

* NVIDIA Isaac Sim 5.1.0+
* RTX 4090 或更高等级 GPU

---

## 快速开始

### 1. 启动运行环境

本项目使用 Docker 容器化部署，从项目根目录启动：

```bash
chmod +x run.sh
sudo ./run.sh
```

#### 可自定义环境变量

| 环境变量                | 说明                 | 默认值                                                   |
| ----------------------- | -------------------- | -------------------------------------------------------- |
| `IMAGE_NAME`          | Docker 镜像名称      | `isaacsim5.1_lerobot5.1:v0`                            |
| `CONTAINER_NAME`      | 容器名称             | `isaac_sim_lerobot`                                    |
| `HOST_WORKSPACE`      | 主机项目目录路径     | `run.sh` 所在目录                                      |
| `CONTAINER_WORKSPACE` | 容器内工作目录路径   | `/workspace/GlobalHumanoidRobotChallenge2026_Baseline` |
| `SHM_SIZE`            | 共享内存大小         | `8g`                                                   |
| `ISAAC_CACHE_ROOT`    | Isaac Sim 缓存目录   | `${HOME}/.cache/isaac_sim_container`                   |
| `HF_CACHE`            | HuggingFace 缓存目录 | `${HOME}/.cache/huggingface`                           |
| `HEADLESS`            | 是否启用无头模式     | `0`（否）                                              |

#### 使用示例

```bash
# 基本启动
./run.sh

# 无头模式（远程服务器）
./run.sh --headless

# 自定义镜像名
IMAGE_NAME=my_custom_image:v1 ./run.sh

# 自定义挂载路径
HOST_WORKSPACE=/my/project/path ./run.sh
```

> ⚠️ 首次运行前请确保已完成 Docker、NVIDIA Container Toolkit 安装。

### 2. 遥操作

在容器内执行（默认工作目录：`/workspace/GlobalHumanoidRobotChallenge2026_Baseline`）：

```bash
/isaac-sim/python.sh lerobot/scripts/control_robot.py \
    --robot.type=walker_s2_sim \
    --control.type=teleoperate \
    --control.task=Packing_Box \
    --control.fps=30 \
    --control.display_cameras=true \
    --control.teleop_time_s=100000000
```

| 参数                        | 说明                           | 默认值            |
| --------------------------- | ------------------------------ | ----------------- |
| `robot.type`              | 机器人类型                     | `walker_s2_sim` |
| `control.type`            | 控制模式                       | `teleoperate`   |
| `control.task`            | 任务名称，用于加载对应场景配置 | `Packing_Box`   |
| `control.fps`             | 控制频率                       | `30`            |
| `control.teleop_time_s`   | 遥操作时长（秒）               | `100000000`     |
| `control.display_cameras` | 是否显示摄像头画面             | `true`          |

#### 键盘映射

**末端执行器位移（按住持续移动）**

| 按键  | 动作                   |
| ----- | ---------------------- |
| `1` | 末端 +X 方向移动       |
| `3` | 末端 -X 方向移动       |
| `4` | 末端 +Y 方向移动       |
| `6` | 末端 -Y 方向移动       |
| `7` | 末端 +Z 方向移动（上） |
| `9` | 末端 -Z 方向移动（下） |

**末端执行器旋转（按住持续旋转）**

| 按键  | 动作            |
| ----- | --------------- |
| `y` | 绕 X 轴正向旋转 |
| `u` | 绕 X 轴负向旋转 |
| `v` | 绕 Y 轴正向旋转 |
| `b` | 绕 Y 轴负向旋转 |
| `n` | 绕 Z 轴正向旋转 |
| `m` | 绕 Z 轴负向旋转 |

**夹爪控制**

| 按键  | 动作     |
| ----- | -------- |
| `k` | 夹爪张开 |
| `l` | 夹爪关闭 |

**系统控制**

| 按键          | 动作                                     |
| ------------- | ---------------------------------------- |
| `o`         | 切换控制的机械臂（左 ↔ 右）             |
| `0`         | 切换单臂 / 双臂同步控制模式              |
| `2`         | 切换夹爪控制模式（位置控制 ↔ 力矩控制） |
| `+` / `=` | 提升移动速度等级                         |
| `-`         | 降低移动速度等级                         |
| `q`         | 退出遥操作                               |

### 3. 数据采集

```bash
/isaac-sim/python.sh lerobot/scripts/control_robot.py \
    --robot.type=walker_s2_sim \
    --control.root=/workspace/GlobalHumanoidRobotChallenge2026_Baseline/datasets/Packing_Box/v1 \
    --control.type=record \
    --control.task=Packing_Box \
    --control.fps=30 \
    --control.single_task="Packing_Box" \
    --control.repo_id=your_org/your_dataset \
    --control.num_episodes=50 \
    --control.warmup_time_s=0 \
    --control.episode_time_s=10000 \
    --control.reset_time_s=10 \
    --control.push_to_hub=false \
    --control.video=true \
    --control.num_image_writer_threads_per_camera=4 \
    --control.display_cameras=true \
    --control.play_sounds=false \
    --control.resume=false
```

| 参数                       | 说明                    | 默认值          |
| -------------------------- | ----------------------- | --------------- |
| `control.root`           | 数据保存路径            | 必填            |
| `control.task`           | 任务名称                | `Packing_Box` |
| `control.fps`            | 采集帧率                | `30`          |
| `control.single_task`    | 任务描述文本            | 可选            |
| `control.repo_id`        | 数据集 ID（HF）         | 必填            |
| `control.num_episodes`   | 采集回合数              | `50`          |
| `control.episode_time_s` | 每回合时长（秒）        | `10000`       |
| `control.reset_time_s`   | 回合重置间隔（秒）      | `10`          |
| `control.push_to_hub`    | 是否上传到 Hugging Face | `false`       |
| `control.video`          | 是否录制视频            | `true`        |
| `control.resume`         | 是否断点续录            | `false`       |

### 4. 回放（Replay）

回放已采集的某一回合数据，用于验证数据质量或调试场景：

```bash
/isaac-sim/python.sh lerobot/scripts/control_robot.py \
    --robot.type=walker_s2_sim \
    --control.type=replay \
    --control.task=Packing_Box \
    --control.root=./challenge2026_dataset/Packing_Box/v1 \
    --control.repo_id=your_org/Packing_Box \
    --control.episode=10 \
    --control.fps=30 \
    --control.play_sounds=false
```

| 参数                    | 说明                           | 默认值 / 备注 |
| ----------------------- | ------------------------------ | ------------- |
| `control.task`        | 任务名称，用于加载对应场景配置 | `Packing_Box` |
| `control.root`        | 数据集本地根路径               | 必填          |
| `control.repo_id`     | 数据集 ID                      | 必填          |
| `control.episode`     | 要回放的回合编号（从 0 开始）  | 必填          |
| `control.fps`         | 回放帧率                       | `30`        |
| `control.play_sounds` | 是否播放提示音                 | `false`     |

### 5. 模型训练

以下是针对 Task 4 使用 **ACT** 策略、附带完整超参数的训练示例：

```bash
/isaac-sim/python.sh lerobot/scripts/train.py \
  --dataset.repo_id=your_org/Packing_Box_dataset \
  --dataset.root=./challenge2026_dataset/Packing_Box/packing_box_episode_50 \
  --policy.type=act \
  --policy.n_obs_steps=1 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.dim_model=256 \
  --policy.n_heads=4 \
  --policy.dim_feedforward=1024 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=1 \
  --policy.use_vae=true \
  --policy.latent_dim=32 \
  --policy.n_vae_encoder_layers=4 \
  --policy.dropout=0.1 \
  --policy.kl_weight=10.0 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_weight_decay=1e-4 \
  --policy.optimizer_lr_backbone=1e-5 \
  --policy.device=cuda \
  --policy.use_amp=true \
  --output_dir=challenge2026_baseline/Packing_Box/act_001/ \
  --job_name=task4_act \
  --resume=false \
  --seed=1000 \
  --num_workers=8 \
  --batch_size=8 \
  --steps=100000 \
  --eval_freq=0 \
  --log_freq=200 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --wandb.entity=your_wandb_entity
```

> 请将 `your_org/Packing_Box_dataset` 替换为您自己的数据集 repo ID，将 `your_wandb_entity` 替换为您的 WandB 用户名或团队名。若不使用 WandB，可删除 `--wandb.entity` 参数。

**数据集与输出参数**

| 参数                | 说明                                   | 默认值 / 备注 |
| ------------------- | -------------------------------------- | ------------- |
| `dataset.repo_id` | 数据集 ID（Hugging Face 或本地组织名） | 必填          |
| `dataset.root`    | 数据集本地根路径                       | 必填          |
| `output_dir`      | 检查点与日志保存目录                   | 必填          |
| `job_name`        | 任务标识（显示在日志 / WandB 中）      | 可选          |
| `resume`          | 是否从上次检查点续训                   | `false`     |
| `seed`            | 全局随机种子                           | `1000`      |

**训练循环参数**

| 参数                | 说明                       | 默认值 / 备注 |
| ------------------- | -------------------------- | ------------- |
| `steps`           | 总训练步数                 | `100000`    |
| `batch_size`      | 每步样本数                 | `8`         |
| `num_workers`     | DataLoader 工作进程数      | `8`         |
| `eval_freq`       | 评估间隔步数（0 表示禁用） | `0`         |
| `log_freq`        | 日志打印间隔步数           | `200`       |
| `save_checkpoint` | 是否保存检查点             | `true`      |
| `save_freq`       | 检查点保存间隔步数         | `5000`      |

**ACT 策略参数**

| 参数                                   | 说明                   | 默认值 / 备注                      |
| -------------------------------------- | ---------------------- | ---------------------------------- |
| `policy.type`                        | 策略算法类型           | `act` / `pi0`                  |
| `policy.device`                      | 运行设备               | `cuda` / `cpu`                 |
| `policy.use_amp`                     | 是否开启混合精度训练   | `true`                           |
| `policy.n_obs_steps`                 | 观测步数               | `1`                              |
| `policy.chunk_size`                  | 动作块长度             | `50`                             |
| `policy.n_action_steps`              | 每次推理执行的动作步数 | `50`                             |
| `policy.vision_backbone`             | 视觉编码器架构         | `resnet18`                       |
| `policy.pretrained_backbone_weights` | 主干网络预训练权重     | `ResNet18_Weights.IMAGENET1K_V1` |
| `policy.dim_model`                   | Transformer 模型维度   | `256`                            |
| `policy.n_heads`                     | 注意力头数             | `4`                              |
| `policy.dim_feedforward`             | 前馈网络维度           | `1024`                           |
| `policy.n_encoder_layers`            | 编码器层数             | `4`                              |
| `policy.n_decoder_layers`            | 解码器层数             | `1`                              |
| `policy.use_vae`                     | 是否启用 VAE 隐空间    | `true`                           |
| `policy.latent_dim`                  | VAE 隐变量维度         | `32`                             |
| `policy.n_vae_encoder_layers`        | VAE 编码器层数         | `4`                              |
| `policy.dropout`                     | Dropout 比率           | `0.1`                            |
| `policy.kl_weight`                   | KL 散度损失权重        | `10.0`                           |
| `policy.optimizer_lr`                | 主网络学习率           | `1e-5`                           |
| `policy.optimizer_weight_decay`      | 权重衰减               | `1e-4`                           |
| `policy.optimizer_lr_backbone`       | 主干网络学习率         | `1e-5`                           |

**WandB 参数**

| 参数             | 说明                 | 默认值 / 备注              |
| ---------------- | -------------------- | -------------------------- |
| `wandb.enable` | 是否启用 WandB 日志  | `true` / `false`       |
| `wandb.entity` | WandB 用户名或团队名 | 例如 `your_wandb_entity` |

### 6. 推理（Inference）

使用训练好的策略模型在仿真环境中执行推理并自动录制结果：

```bash
/isaac-sim/python.sh lerobot/scripts/control_robot.py \
    --robot.type=walker_s2_sim \
    --control.task=Packing_Box \
    --control.root=./datasets/Packing_Box/inference \
    --control.type=record \
    --control.policy.path=./challenge2026_baseline/Packing_Box/act/checkpoints/last/pretrained_model \
    --control.fps=30 \
    --control.single_task="Packing_Box" \
    --control.repo_id=your_org/eval_task4_act \
    --control.num_episodes=10 \
    --control.warmup_time_s=5 \
    --control.episode_time_s=2000 \
    --control.reset_time_s=10 \
    --control.push_to_hub=false \
    --control.video=true \
    --control.num_image_writer_threads_per_camera=4 \
    --control.display_cameras=true \
    --control.play_sounds=false \
    --control.resume=false
```

| 参数                       | 说明                            | 默认值 / 备注   |
| -------------------------- | ------------------------------- | --------------- |
| `control.task`           | 任务名称，用于加载对应场景配置  | `Packing_Box` |
| `control.root`           | 推理结果保存路径                | 必填            |
| `control.type`           | 控制模式，推理时设为 `record` | `record`      |
| `control.policy.path`    | 策略模型检查点路径              | 必填            |
| `control.fps`            | 控制频率                        | `30`          |
| `control.single_task`    | 任务描述文本                    | 可选            |
| `control.repo_id`        | 推理结果数据集 ID               | 必填            |
| `control.num_episodes`   | 推理回合数                      | `10`          |
| `control.warmup_time_s`  | 预热时长（秒）                  | `5`           |
| `control.episode_time_s` | 每回合最长时长（秒）            | `2000`        |
| `control.reset_time_s`   | 回合重置间隔（秒）              | `10`          |
| `control.push_to_hub`    | 是否上传到 Hugging Face         | `false`       |
| `control.video`          | 是否录制视频                    | `true`        |
| `control.resume`         | 是否断点续录                    | `false`       |

---

## 任务定义

### Task1：抓取 - 放置（**Part_Sorting**）

**任务目标：** 从工作台上抓取零件，并放入指定料盒。

**评分规则：**

| 指标     | 分值 | 说明                                      |
| -------- | ---: | ----------------------------------------- |
| 抬升评分 |   40 | 零件 Z 高度达到阈值，每个 10 分，满分 40  |
| 入箱评分 |   40 | 零件成功进入正确箱体，每个 10 分，满分 40 |
| 时间评分 |   20 | 40 秒内完成满分，超时每 10 秒扣 5 分      |
| 总分     |  100 | 总分达到 100 判定成功                     |

**配置文件：** `Ubtech_sim/config/Part_Sorting.yaml`

### Task2：工件传送带分拣（**Conveyor_Sorting**）

**任务目标：** 对运行中的传送带进行实时感知，识别舵机组装件（B 零件）与正交减速器（A 零件），并分别放入传送带左右两侧指定料箱，实现自动化分拣。

**场景元素：** 桌子、传送带、零件、料箱。

**传送带参数：**

| 参数         | 数值                                 |
| ------------ | ------------------------------------ |
| 速度         | 0.02 m/s                             |
| 运行方向     | 沿 X 轴，行程 1000 mm                |
| 尺寸         | 长 1500 mm × 宽 300 mm，高 1000 mm  |
| 零件出现间隔 | 每 5–10 s 随机出现一个，总时长 80 s |
| 出现位置     | 传送带起点中轴，出现姿态随机         |

**零件清单：**

| 编号   | 名称       | 尺寸               | 颜色 |
| ------ | ---------- | ------------------ | ---- |
| A 零件 | 正交减速器 | 最大边长 40–60 mm | 蓝色 |
| B 零件 | 舵机组装件 | 80 × 50 × 45 mm  | 原色 |

**评分规则：** 每轮 10 个工件（5×A + 5×B），共 10 轮，满分 1000 分。

| 评分项             |       分值 | 说明                                 |
| ------------------ | ---------: | ------------------------------------ |
| 分拣成功           |   10 分/个 | 工件完全脱离夹爪并静止在正确料箱内部 |
| 分拣失败           |       0 分 | 工件掉落或放入错误料箱               |
| 抓取成功但分拣错误 | 仅得抓取分 | 抓取计分，分拣不计分                 |
| 单轮总分           |      80 分 | —                                   |

**配置文件：** `Ubtech_sim/config/Conveyor_Sorting.yaml`

### Task3：工件嵌装（**Foam_Inlaying**）

**任务目标：** 将料箱中指定数量、指定类别的工件全部正确嵌装至航空箱泡棉的对应槽位，确保数量、位置与姿态均满足要求。

**场景元素：** 桌子、带槽泡棉、料箱。

**时间限制：** ≤ 2 分钟；每超时 30 秒扣 5 分（扣完为止）。

**零件清单：**

| 编号   | 名称              | 尺寸              | 颜色     |
| ------ | ----------------- | ----------------- | -------- |
| A 零件 | 28 步进电机（小） | 50 × 20 × 25 mm | 两种颜色 |
| B 零件 | 舵机组装件（大）  | 80 × 50 × 45 mm | 两种颜色 |

**泡棉参数：** 600 × 400 × 100 mm，居中放置于桌面；6 个槽位（每类 3 个），槽深 60 mm。任务开始前泡棉内无工件，工件随机分布在左侧料箱中（大 3 + 小 3）。

**评分规则：** 每轮 100 分，共 10 轮，满分 1000 分。

| 评分项     |                  分值 | 说明                                                          |
| ---------- | --------------------: | ------------------------------------------------------------- |
| 嵌装成功率 | 15 分/个 × 6 = 90 分 | 工件完全脱离夹爪并放入对应类别槽位内                          |
| 嵌装效率   |                 10 分 | 2 分钟内完成全部嵌装；每超 30 s 扣 5 分；未完成全部类别不得分 |

**完成判定标准：**

1. 数量完整性：6 个工件全部放置。
2. 位置正确性：每个工件放入与其类别对应的专用槽位，不允许错槽。
3. 稳定性：工件不悬空、不明显倾斜、不与其他工件发生干涉。
4. 姿态要求：两类电机均无强制方向约束，完整嵌入槽位即可。

**配置文件：** `Ubtech_sim/config/Foam_Inlaying.yaml`

### Task4：装箱（**Packing_Box**）

**任务目标：** 控制折叠箱四个关节完成装箱动作。

**评分规则：**

| 指标     | 分值 | 说明                                  |
| -------- | ---: | ------------------------------------- |
| 短边闭合 |   30 | 两个短边关节达到目标，每个 15 分      |
| 长边闭合 |   30 | 两个长边关节达到目标，每个 15 分      |
| 时间评分 |   40 | 120 秒内完成满分，超时每 10 秒扣 5 分 |
| 协同系数 |   — | 单臂 ×0.7，双臂协同 ×1.0            |
| 总分     |  100 | 连续 10 步稳定判定成功                |

**关键参数：**

- 短边目标关节：`[-3.3219733, -3.3213105]`（关节 2、3）
- 长边目标关节：`[-3.4906585, -3.4906585]`（关节 0、1）
- 关节阈值：`0.2 rad`

**配置文件：** `Ubtech_sim/config/Packing_Box.yaml`

---

## 项目结构

```text
.
├── datasets/                       # 本地训练数据集目录
│   ├── Part_Sorting/
│   ├── Conveyor_Sorting/
│   ├── Foam_Inlaying/
│   └── Packing_Box/
├── assets/                         # 仿真资产（Git 子模块）
│   └── resources/                  # USD、URDF 等资源文件
├── Ubtech_sim/                     # 仿真环境与任务逻辑
│   ├── config/                     # task1.yaml … task4.yaml
│   ├── source/                     # SceneBuilder、IK 等
│   └── main.py
├── lerobot/                        # LeRobot 核心代码
│   └── scripts/
│       ├── control_robot.py        # 遥操作与数据采集
│       ├── train.py                # 模型训练
│       └── eval.py                 # 模型评估
├── Dockerfile
├── run.sh                          # 容器启动脚本
└── pyproject.toml
```

---

## 仿真架构说明

### Assets

`assets/` 通过 Git 子模块管理，对应 Hugging Face 仓库 `challenge2026_assets`，主要存放：

- USD 场景资源
- URDF 机器人模型
- 箱体、零件、任务场景文件

核心目录：`assets/resources/`

### Ubtech_sim

`Ubtech_sim/` 负责场景配置、任务逻辑、机器人控制与仿真流程封装：

- 任务 YAML：`Ubtech_sim/config/`
- 仿真源码：`Ubtech_sim/source/`
- 独立运行入口：`Ubtech_sim/main.py`

### LeRobot 集成方式

LeRobot 的 `walker_s2_sim` 通过 `WalkerS2SimRobotConfig` 读取 `task_cfg_path`，再解析其中的 `root_path` 指向 `assets/resources/`。

示例 `Ubtech_sim/config/Part_Sorting.yaml`：

```yaml
root_path: "../assets/resources/"
scene_usd: "Collected_Task1_PartA_ori_color/scene.usd"
```

---

## Ubtech_sim 模块说明

### 模块结构

| 目录/文件     | 说明                                                  |
| ------------- | ----------------------------------------------------- |
| `source/`   | Python 源码（SceneBuilder、IK、RobotArticulation 等） |
| `config/`   | 任务配置：task1.yaml ~ task4.yaml                     |
| `main.py`   | 独立 Isaac Sim 运行入口                               |
| `README.md` | 模块说明文档                                          |

### 核心组件

| 模块                  | 说明                                               |
| --------------------- | -------------------------------------------------- |
| `config_loader`     | 加载 YAML 并解析绝对路径                           |
| `SceneBuilder`      | 构建仿真场景，加载桌子、箱子、零件、机器人等       |
| `RobotArticulation` | WalkerS2 控制接口，封装双臂 IK、力传感器与相机数据 |
| `DualArmIK`         | 基于 Pinocchio 的双臂逆运动学                      |
| `GraspPlanner`      | 抓取目标规划、选臂与姿态跟踪                       |
| `DataLogger`        | 记录 CSV 位姿数据与 HDF5 图像数据                  |

### 使用方式

**方式一：独立运行**

```bash
python Ubtech_sim/main.py
```

**方式二：通过 LeRobot 调用**

在 `lerobot/common/robot_devices/robots/configs.py` 中配置 `WalkerS2SimRobotConfig`：

```python
root_path: str = "Ubtech_sim"
task_cfg_path: str = "Ubtech_sim/config/Packing_Box.yaml"
```

### 常见目录映射

| 路径                              | 用途                 |
| --------------------------------- | -------------------- |
| `Ubtech_sim/config/*.yaml`      | 仿真任务配置         |
| `Ubtech_sim/source/*.py`        | 仿真逻辑实现         |
| `assets/resources/`             | USD、URDF 等资源文件 |
| `assets/resources/s2.urdf`      | WalkerS2 URDF        |
| `assets/resources/Collected_*/` | 场景文件             |

---

## 数据集说明

`datasets/` 目录存放各任务训练数据，采用 **LeRobotDataset V2.1** 标准：

```text
datasets/
├── Part_Sorting/
├── Conveyor_Sorting/
├── Foam_Inlaying/
└── Packing_Box/
```

| 类型 | 格式说明                        |
| ---- | ------------------------------- |
| 视频 | MP4，按相机与 episode 组织      |
| 状态 | 关节位置、速度、力矩（Parquet） |
| 动作 | 目标关节或末端位姿              |
| 频率 | 默认 `30 Hz`                  |
| 格式 | LeRobotDataset V2.1             |

---

## Git 子模块管理

### 初始化项目

```bash
git clone --recursive https://github.com/UBTECH-Robotics/GlobalHumanoidRobotChallenge2026_Baseline.git
cd GlobalHumanoidRobotChallenge2026_Baseline

# 若已单独克隆：
git submodule update --init --recursive
```

### 更新子模块

```bash
git pull
git submodule update --remote --merge

# 仅更新 assets：
cd assets
git pull origin main
cd ..
```

### 子模块关系说明

- 主项目保存对子模块的提交引用
- 子模块具备独立 Git 历史
- 更新子模块后需在主项目中执行 `git add assets` 并提交指针变化

---

## 开发指南

### 代码规范

- Python 3.11+
- 遵循 PEP 8
- 推荐使用 `pre-commit` 执行提交前检查

### 安装开发依赖

```bash
pip install -e ".[dev]"
pre-commit install
```

---

## 相关资源

| 仓库                                                                                          | 说明               |
| --------------------------------------------------------------------------------------------- | ------------------ |
| 🤗[LeRobot](https://github.com/huggingface/lerobot)                                              | 机器人学习底层框架 |
| 🤗[challenge2026_assets](https://huggingface.co/UBTECH-Robotics/challenge2026_assets)            | 仿真资产           |
| 🤗[challenge2026_dataset](https://huggingface.co/datasets/UBTECH-Robotics/challenge2026_dataset) | 训练数据集         |

---

## 许可证

本项目采用 [Apache 2.0](LICENSE) 许可证。

## 致谢

| 项目                                                        | 说明               |
| ----------------------------------------------------------- | ------------------ |
| [Hugging Face LeRobot](https://github.com/huggingface/lerobot) | 开源机器人学习框架 |
| [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)     | 机器人仿真平台     |

---

**Global Humanoid Robot Challenge 2026** | UBTECH Robotics
