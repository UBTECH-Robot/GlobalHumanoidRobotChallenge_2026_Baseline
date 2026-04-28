# Global Humanoid Robot Challenge 2026 Baseline

> **[🇨🇳 中文版](README_zh.md)**

[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

Official baseline repository for the **Global Humanoid Robot Challenge 2026 (GHRC 2026)**, built on the [LeRobot](https://github.com/huggingface/lerobot) framework. Covers the full pipeline from simulation, data collection, and model training to deployment.

---

## Overview

This repository provides a unified baseline implementation for GHRC 2026 participants:

- High-fidelity robot simulation with **NVIDIA Isaac Sim**
- Teleoperation-based data recording in **LeRobotDataset V2.1** format
- Imitation-learning model training and fine-tuning

---

## Features

| Feature                        | Description                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------------ |
| 🤖**Simulation**         | High-fidelity robot simulation based on NVIDIA Isaac Sim                             |
| 📊**Data Collection**    | Keyboard; recordings in standard**LeRobotDataset V2.1** format |
| 🧠**Model Training**     | Imitation-learning policies including**ACT**, **Pi0**, and others        |

---

## Resources

Large artifacts are hosted on Hugging Face. Download them before use:

| Resource Type                       | Local Path                  | Hugging Face URL                                                                                            |
| :---------------------------------- | :-------------------------- | :---------------------------------------------------------------------------------------------------------- |
| 🤖 Simulation Assets & Robot Models | `assets/` (Git submodule) | [UBTECH-Robotics/challenge2026_assets](https://huggingface.co/UBTECH-Robotics/challenge2026_assets)            |
| 📊 Training Datasets                | `datasets/`               | [UBTECH-Robotics/challenge2026_dataset](https://huggingface.co/datasets/UBTECH-Robotics/challenge2026_dataset) |

### Quick download commands

```bash
# Install Hugging Face CLI
pip install huggingface-hub

# Simulation assets
# Method 1: Using Git submodules
git submodule update --init --recursive

# Method 2: Download directly into ./assets
hf download UBTECH-Robotics/challenge2026_assets --local-dir ./assets

# Training datasets
hf download UBTECH-Robotics/challenge2026_dataset --local-dir ./datasets --repo-type dataset

```

---

## System Requirements

### Basic

- NVIDIA GPU
- Docker
- NVIDIA Container Toolkit
- CUDA 12.8+
- Python 3.11+

### Recommended

- NVIDIA Isaac Sim 5.1.0+
- RTX 4090 or better GPU

---

## Quick Start

### 1. Start the container

The project uses Docker. 

#### Docker Image Load

Build the image from Dockefile.

```bash
# Navigate to the Baseline directory
cd GlobalHumanoidRobotChallenge_2026_Baseline/

# Build the container
docker build --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ -t ghrc_2026:v0 -f Dockerfile .

# If encountered network connectivity issues, try:
docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ -t ghrc_2026:v0 -f Dockerfile .
```

Start from the repository root:

```bash
chmod +x run.sh
sudo ./run.sh
```

#### Configurable environment variables

| Variable                | Description                 | Default                                                  |
| ----------------------- | --------------------------- | -------------------------------------------------------- |
| `IMAGE_NAME`          | Docker image name           | `ghrc_2026:v0`                            |
| `CONTAINER_NAME`      | Container name              | `isaac_sim_ubt`                                    |
| `HOST_WORKSPACE`      | Host project path           | directory of `run.sh`                                  |
| `CONTAINER_WORKSPACE` | Container workspace path    | `/workspace/GlobalHumanoidRobotChallenge_2026_Baseline` |
| `SHM_SIZE`            | Shared memory size          | `8g`                                                   |
| `ISAAC_CACHE_ROOT`    | Isaac Sim cache directory   | `${HOME}/.cache/isaac_sim_container`                   |
| `HF_CACHE`            | HuggingFace cache directory | `${HOME}/.cache/huggingface`                           |
| `HEADLESS`            | Headless mode               | `0` (disabled)                                         |

#### Examples

```bash
# Basic
./run.sh

# Headless mode (remote server)
./run.sh --headless

# Custom image
IMAGE_NAME=my_custom_image:v1 ./run.sh

# Custom mount path
HOST_WORKSPACE=/my/project/path ./run.sh
```

> ⚠️ Make sure Docker and NVIDIA Container Toolkit are installed before first run.

### 2. Teleoperation

Inside the container (default workspace: `/workspace/GlobalHumanoidRobotChallenge_2026_Baseline`):

```bash
/isaac-sim/python.sh lerobot/scripts/control_robot.py \
    --robot.type=walker_s2_sim \
    --control.type=teleoperate \
    --control.task=Packing_Box \
    --control.fps=30 \
    --control.display_cameras=true \
    --control.teleop_time_s=100000000
```

| Flag                        | Description                                          | Default           |
| --------------------------- | ---------------------------------------------------- | ----------------- |
| `robot.type`              | Robot type                                           | `walker_s2_sim` |
| `control.type`            | Control mode                                         | `teleoperate`   |
| `control.task`            | Task name for loading the corresponding scene config | `Packing_Box`   |
| `control.fps`             | Control rate                                         | `30`            |
| `control.teleop_time_s`   | Teleop duration (seconds)                            | `100000000`     |
| `control.display_cameras` | Show camera views                                    | `true`          |

| Task Name（`control.task`）            | config file   |
| -------------------------- | --------------- |
| Part_Sorting  | `Ubtech_sim/config/Part_Sorting.yaml` |
| Conveyor_Sorting  | `Ubtech_sim/config/Conveyor_Sorting.yaml` |
| Foam_Inlaying  | `Ubtech_sim/config/Part_Sorting.yaml` |
| Packing_Box  | `Ubtech_sim/config/Part_Sorting.yaml` |

#### Keyboard Controls

**End-effector translation (hold to move continuously)**

| Key   | Action                                   |
| ----- | ---------------------------------------- |
| `1` | Move end-effector in +X direction        |
| `3` | Move end-effector in -X direction        |
| `4` | Move end-effector in +Y direction        |
| `6` | Move end-effector in -Y direction        |
| `7` | Move end-effector in +Z direction (up)   |
| `9` | Move end-effector in -Z direction (down) |

**End-effector rotation (hold to rotate continuously)**

| Key   | Action                          |
| ----- | ------------------------------- |
| `y` | Rotate around X-axis (positive) |
| `u` | Rotate around X-axis (negative) |
| `v` | Rotate around Y-axis (positive) |
| `b` | Rotate around Y-axis (negative) |
| `n` | Rotate around Z-axis (positive) |
| `m` | Rotate around Z-axis (negative) |

**Gripper control**

| Key   | Action        |
| ----- | ------------- |
| `k` | Open gripper  |
| `l` | Close gripper |

**System controls**

| Key           | Action                                                   |
| ------------- | -------------------------------------------------------- |
| `o`         | Switch active arm (left ↔ right)                        |
| `0`         | Toggle single-arm / dual-arm sync mode                   |
| `2`         | Toggle gripper mode (position control ↔ torque control) |
| `+` / `=` | Increase movement speed level                            |
| `-`         | Decrease movement speed level                            |
| `q`         | Quit teleoperation                                       |

### 3. Data collection

```bash
/isaac-sim/python.sh lerobot/scripts/control_robot.py \
    --robot.type=walker_s2_sim \
    --control.root=/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/datasets/Packing_Box/v1 \
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

| Flag                       | Description                  | Default         |
| -------------------------- | ---------------------------- | --------------- |
| `control.root`           | Output directory             | required        |
| `control.task`           | Task name                    | `Packing_Box` |
| `control.fps`            | Recording frame rate         | `30`          |
| `control.single_task`    | Task description string      | optional        |
| `control.repo_id`        | Dataset ID                   | required        |
| `control.num_episodes`   | Number of episodes           | `50`          |
| `control.episode_time_s` | Episode length (seconds)     | `10000`       |
| `control.reset_time_s`   | Reset interval (seconds)     | `10`          |
| `control.push_to_hub`    | Push to Hugging Face Hub     | `false`       |
| `control.video`          | Record video                 | `true`        |
| `control.resume`         | Resume interrupted recording | `false`       |

### 4. Replay

Replay a previously recorded episode to verify data quality or debug the scene:

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

| Flag                    | Description                                          | Default / Notes |
| ----------------------- | ---------------------------------------------------- | --------------- |
| `control.task`        | Task name for loading the corresponding scene config | `Packing_Box` |
| `control.root`        | Local dataset root path                              | required        |
| `control.repo_id`     | Dataset ID                                           | required        |
| `control.episode`     | Episode index to replay (0-based)                    | required        |
| `control.fps`         | Replay frame rate                                    | `30`          |
| `control.play_sounds` | Play audio cues                                      | `false`       |

### 5. Training

Below is a complete example for training an **ACT** policy on Task 4 with all recommended hyperparameters:

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

> Replace `your_org/Packing_Box_dataset` with your own dataset repo ID and `your_wandb_entity` with your WandB username or team name. Remove `--wandb.entity` if you do not use WandB.

**Dataset & output flags**

| Flag                | Description                                 | Default / Notes |
| ------------------- | ------------------------------------------- | --------------- |
| `dataset.repo_id` | Dataset ID (Hugging Face or local org name) | required        |
| `dataset.root`    | Local dataset root path                     | required        |
| `output_dir`      | Directory to save checkpoints and logs      | required        |
| `job_name`        | Job identifier shown in logs / WandB        | optional        |
| `resume`          | Resume training from last checkpoint        | `false`       |
| `seed`            | Global random seed                          | `1000`        |

**Training loop flags**

| Flag                | Description                        | Default / Notes |
| ------------------- | ---------------------------------- | --------------- |
| `steps`           | Total training steps               | `100000`      |
| `batch_size`      | Samples per step                   | `8`           |
| `num_workers`     | DataLoader worker count            | `8`           |
| `eval_freq`       | Evaluation interval (0 = disabled) | `0`           |
| `log_freq`        | Logging interval (steps)           | `200`         |
| `save_checkpoint` | Enable checkpoint saving           | `true`        |
| `save_freq`       | Checkpoint save interval (steps)   | `5000`        |

**ACT policy flags**

| Flag                                   | Description                         | Default / Notes                    |
| -------------------------------------- | ----------------------------------- | ---------------------------------- |
| `policy.type`                        | Policy algorithm                    | `act` / `pi0`                  |
| `policy.device`                      | Compute device                      | `cuda` / `cpu`                 |
| `policy.use_amp`                     | Mixed-precision training            | `true`                           |
| `policy.n_obs_steps`                 | Number of observation steps         | `1`                              |
| `policy.chunk_size`                  | Action chunk length                 | `50`                             |
| `policy.n_action_steps`              | Action steps executed per inference | `50`                             |
| `policy.vision_backbone`             | Visual encoder architecture         | `resnet18`                       |
| `policy.pretrained_backbone_weights` | Backbone pre-trained weights        | `ResNet18_Weights.IMAGENET1K_V1` |
| `policy.dim_model`                   | Transformer model dimension         | `256`                            |
| `policy.n_heads`                     | Attention heads                     | `4`                              |
| `policy.dim_feedforward`             | Feedforward dimension               | `1024`                           |
| `policy.n_encoder_layers`            | Encoder layer count                 | `4`                              |
| `policy.n_decoder_layers`            | Decoder layer count                 | `1`                              |
| `policy.use_vae`                     | Enable VAE latent space             | `true`                           |
| `policy.latent_dim`                  | VAE latent dimension                | `32`                             |
| `policy.n_vae_encoder_layers`        | VAE encoder layer count             | `4`                              |
| `policy.dropout`                     | Dropout rate                        | `0.1`                            |
| `policy.kl_weight`                   | KL divergence loss weight           | `10.0`                           |
| `policy.optimizer_lr`                | Learning rate (main network)        | `1e-5`                           |
| `policy.optimizer_weight_decay`      | Weight decay                        | `1e-4`                           |
| `policy.optimizer_lr_backbone`       | Learning rate (backbone)            | `1e-5`                           |

**WandB flags**

| Flag             | Description                 | Default / Notes           |
| ---------------- | --------------------------- | ------------------------- |
| `wandb.enable` | Enable WandB logging        | `true` / `false`      |
| `wandb.entity` | WandB username or team name | e.g.`your_wandb_entity` |

### 6. Inference

Run a trained policy in the simulation environment and automatically record results:

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

| Flag                       | Description                                          | Default / Notes |
| -------------------------- | ---------------------------------------------------- | --------------- |
| `control.task`           | Task name for loading the corresponding scene config | `Packing_Box` |
| `control.root`           | Path to save inference results                       | required        |
| `control.type`           | Control mode; set to `record` for inference        | `record`      |
| `control.policy.path`    | Path to the trained model checkpoint                 | required        |
| `control.fps`            | Control rate                                         | `30`          |
| `control.single_task`    | Task description text                                | optional        |
| `control.repo_id`        | Dataset ID for inference results                     | required        |
| `control.num_episodes`   | Number of inference episodes                         | `10`          |
| `control.warmup_time_s`  | Warm-up duration (seconds)                           | `5`           |
| `control.episode_time_s` | Max duration per episode (seconds)                   | `2000`        |
| `control.reset_time_s`   | Reset interval between episodes (seconds)            | `10`          |
| `control.push_to_hub`    | Upload results to Hugging Face                       | `false`       |
| `control.video`          | Record video                                         | `true`        |
| `control.resume`         | Resume interrupted recording                         | `false`       |

---

## Repository Layout

```
.
├── datasets/                       # Training data (download from HF)
│   ├── Part_Sorting/
│   ├── Conveyor_Sorting/
│   ├── Foam_Inlaying/
│   └── Packing_Box/
├── assets/                         # Simulation assets (Git submodule)
│   └── resources/                  # USD, URDF, etc.
├── Ubtech_sim/                     # Scene building & task logic
│   ├── config/                     # task1.yaml … task4.yaml
│   ├── source/                     # SceneBuilder, IK, etc.
│   └── main.py
├── lerobot/                        # LeRobot core
│   └── scripts/
│       ├── control_robot.py        # Teleoperation & data recording
│       ├── train.py                # Model training
│       └── eval.py                 # Model evaluation
├── Dockerfile
├── run.sh                          # Container launcher
└── pyproject.toml
```

---

## Simulation Stack

Simulation is split into two parts:

### Assets

`assets/` is a Git submodule aligned with [challenge2026_assets](https://huggingface.co/UBTECH-Robotics/challenge2026_assets). It stores:

- Scene resources
- Robot models
- Boxes, parts, and task scene files

Core directory: `assets/resources/`

### Ubtech_sim

`Ubtech_sim/` handles scene configuration, task logic, robot control, and simulation orchestration:

- Task YAMLs: `Ubtech_sim/config/`
- Source code: `Ubtech_sim/source/`
- Standalone entry: `Ubtech_sim/main.py`

### LeRobot Integration

LeRobot's `walker_s2_sim` reads `task_cfg_path` from `WalkerS2SimRobotConfig`, then resolves `root_path` to `assets/resources/`.

Example `Ubtech_sim/config/Part_Sorting.yaml`:

```yaml
root_path: "../assets/resources/"
scene_usd: "Collected_Task1_PartA_ori_color/scene.usd"
```

---

## Ubtech_sim Module

### Layout

| Directory/File | Description                                                    |
| -------------- | -------------------------------------------------------------- |
| `source/`    | Python source code (SceneBuilder, IK, RobotArticulation, etc.) |
| `config/`    | Task configurations: task1.yaml ~ task4.yaml                   |
| `main.py`    | Standalone Isaac Sim entry point                               |
| `README.md`  | Module documentation                                           |

### Core Components

| Module                | Description                                                                   |
| --------------------- | ----------------------------------------------------------------------------- |
| `config_loader`     | Load YAML and resolve paths; converts `root_path` to absolute paths         |
| `SceneBuilder`      | Build and initialize simulation scenes; load USD from `assets/resources/`   |
| `RobotArticulation` | WalkerS2 control interface; dual-arm IK; force sensors and camera data        |
| `DualArmIK`         | Dual-arm inverse kinematics using weighted damped least squares (DLS)         |
| `GraspPlanner`      | Grasp planning; automatic arm selection, pose calculation, real-time tracking |
| `DataLogger`        | Record CSV (poses) and HDF5 (images)                                          |

### Usage

**Option A — Standalone**

```bash
python Ubtech_sim/main.py
```

**Option B — Via LeRobot**

Configure `WalkerS2SimRobotConfig` in `lerobot/common/robot_devices/robots/configs.py`:

```python
root_path: str = "Ubtech_sim"
task_cfg_path: str = "Ubtech_sim/config/Packing_Box.yaml"
```

### Path Reference

| Kind   | Path                              | Notes          |
| ------ | --------------------------------- | -------------- |
| Config | `Ubtech_sim/config/*.yaml`      | Per-task YAML  |
| Code   | `Ubtech_sim/source/*.py`        | Implementation |
| Assets | `assets/resources/`             | USD, URDF, …  |
| URDF   | `assets/resources/s2.urdf`      | Walker S2      |
| Scenes | `assets/resources/Collected_*/` | Scene USD      |

---

## Datasets

`datasets/` holds per-task data in **LeRobotDataset V2.1** format:

```
datasets/
├── Part_Sorting/
├── Conveyor_Sorting/
├── Foam_Inlaying/
└── Packing_Box/
```

| Type              | Format/Description                             |
| ----------------- | ---------------------------------------------- |
| **Video**   | MP4, organized by camera and episode           |
| **State**   | Joint positions, velocities, torques (Parquet) |
| **Actions** | Target joints or end-effector poses            |
| **Rate**    | 30 Hz by default                               |
| **Schema**  | LeRobotDataset V2.1                            |

---

## Git Submodules

### Clone with submodules

```bash
git clone --recursive https://github.com/UBTECH-Robotics/GlobalHumanoidRobotChallenge_2026_Baseline.git
cd GlobalHumanoidRobotChallenge_2026_Baseline

# If already cloned without submodules:
git submodule update --init --recursive
```

### Update submodules

```bash
git pull
git submodule update --remote --merge

# Update only assets:
cd assets
git pull origin main
cd ..
```

### Notes

- The main repo stores a **pinned commit** for each submodule.
- Each submodule has its own Git history.
- After updating a submodule, run `git add assets` in the main repo and commit.

---

## Development

### Code Style

- Python 3.11+
- Follow PEP 8
- Use `pre-commit` hooks

### Dev install

```bash
pip install -e ".[dev]"
pre-commit install
```

---

## Related Repositories

| Repository                                                                                    | Description          |
| --------------------------------------------------------------------------------------------- | -------------------- |
| 🤗[LeRobot](https://github.com/huggingface/lerobot)                                              | Underlying framework |
| 🤗[challenge2026_assets](https://huggingface.co/UBTECH-Robotics/challenge2026_assets)            | Simulation assets    |
| 🤗[challenge2026_dataset](https://huggingface.co/datasets/UBTECH-Robotics/challenge2026_dataset) | Training datasets    |

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgements

| Project                                                     | Description                          |
| ----------------------------------------------------------- | ------------------------------------ |
| [Hugging Face LeRobot](https://github.com/huggingface/lerobot) | Open-source robot learning framework |
| [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)     | Robot simulation platform            |

---

**Global Humanoid Robot Challenge 2026** | UBTECH Robotics
