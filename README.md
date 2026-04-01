# AutoDrive-R2: Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving

[![arXiv Paper](https://img.shields.io/badge/arXiv-2509.01944-b31b1b.svg)](https://arxiv.org/pdf/2509.01944)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

|[[📖 Paper](https://arxiv.org/pdf/2509.01944)] [[🤗 AutoDrive-R2-7B](https://huggingface.co/ZhenlongYuan/AutoDrive-R2-7B)] [[🤗 AutoDrive-R2-7B-COT-SFT](https://huggingface.co/ZhenlongYuan/AutoDrive-R2-7B-COT-SFT)] [[📊 All Data](https://huggingface.co/datasets/ZhenlongYuan/AutoDrive-R2-all-data/tree/main)]

<p align="center">
    <img src="./images/1.png" width="90%" height="90%">
</p>

## 👀 About AutoDrive-R2

AutoDrive-R2 is a specialized Vision-Language-Action (VLA) model designed for **autonomous driving trajectory prediction**, which elicits reasoning and self-reflection capacities through rule-based Reinforcement Learning (RL).

Given a front-view camera image and historical vehicle status (position, velocity, acceleration, steering angle), AutoDrive-R2 predicts **future waypoints** at 0.5s intervals for the next 3 seconds.

---

## 🎯 Core Task

**Input:**
- Front-view camera image
- Historical vehicle status (last 2.0-3.0 seconds at 0.5s intervals)

**Output:**
- 6 future waypoints: [x(t+0.5s), x(t+1.0s), ..., x(t+3.0s)]
- Each waypoint: [x_coordinate, y_coordinate] in meters

**Reasoning Process:**
1. **Visual Analysis**: Analyze road conditions, obstacles, and traffic signals
2. **Motion Modeling**: Apply kinematic equations for trajectory prediction
3. **Logical Deductions**: Safety checks and path planning
4. **Self-Reflection**: Validate predicted trajectory feasibility

---

## 🏗️ Architecture

<p align="center">
    <img src="./images/2.png" width="90%" height="90%">
</p>

<p align="left">
    <b>Pipeline of our method.</b> We adopt a two-stage training process. The first stage introduces an innovative CoT dataset named nuScenesR²-6K for SFT. The nuScenesR²-6K adopts a four-step logical chain with self-reflection to generate valuable chain-of-thought data.
    The second stage proposes an novel physics-grounded reward framework for RL optimization, which incorporates spatial alignment, vehicle dynamic, and temporal smoothness for reliable trajectory planning.
</p>

---

## 📍 Features

- **Qwen2.5-VL Base Model**: Leverages state-of-the-art vision-language capabilities
- **Chain-of-Thought Reasoning**: Explicit reasoning steps for interpretable predictions
- **Rule-based RL Training**: GRPO with physics-grounded rewards
- **Multi-dataset Support**: Trained and evaluated on nuScenes and Waymo

---

## 🔍 Dataset

### Training Data

| Dataset | Samples | Description |
|---------|---------|-------------|
| `sft.json` | ~60K | SFT training data (raw) |
| `sft_cot.json` | ~60K | SFT training data (with CoT reasoning) |
| `rl.json` | ~60K | RL training data |

### Evaluation Data

| Dataset | Samples | Description |
|---------|---------|-------------|
| `nuscenes_test.json` | ~54K | nuScenes benchmark |
| `waymo_test.json` | ~12K | Waymo benchmark |

### Input Format

Each sample contains historical vehicle status at 0.5s intervals:
- Position: `[x, y]` coordinates (x=forward, y=leftward)
- Velocity: `v` in m/s
- Acceleration: `a_x, a_y` in m/s²
- Steering angle: `θ` (positive=left turn, negative=right turn)

### Download Datasets

#### nuScenes Dataset

Download from [nuScenes Official Website](https://www.nuscenes.org/download):

```
nuscenes/
├── samples/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   └── CAM_FRONT_RIGHT/
└── sweeps/
    ├── CAM_BACK/
    ├── CAM_BACK_LEFT/
    ├── CAM_BACK_RIGHT/
    ├── CAM_FRONT/
    ├── CAM_FRONT_LEFT/
    └── CAM_FRONT_RIGHT/
```

#### Waymo Dataset

Download from Baidu Netdisk: [Link Coming Soon]

---

## 📐 Set up

```bash
# Clone the repository
git clone https://github.com/your-repo/AutoDrive-R2
cd AutoDrive-R2

# Create conda environment
conda create -n autodrive-r2 python=3.11
conda activate autodrive-r2

# Install dependencies
pip install -r requirements.txt

# Install flash-attention
pip install flash-attn --no-build-isolation
```

---

## 🚀 Training

### Step 1: Generate CoT Data

Generate Chain-of-Thought reasoning for SFT training:

```bash
python AScripts/cot.py
```

### Step 2: Supervised Fine-Tuning (SFT)

Perform SFT training on CoT-annotated data:

```bash
bash src/scripts/run_sft_video_7B_6k.sh
```

### Step 3: Reinforcement Learning (GRPO)

Fine-tune with GRPO using physics-grounded rewards:

```bash
bash src/scripts/run_grpo_video_7B_6k.sh
```

---

## 🔮 Inference & Evaluation

### Evaluate on nuScenes

```bash
python AScripts/eval_nuscene.py
```

### Evaluate on Waymo

```bash
python AScripts/eval_waymo.py
```

---

## 📜 Citation

If you find our work helpful for your research, please consider citing:

```bibtex
@article{yuan2025autodrive,
  title={AutoDrive-R2: Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving},
  author={Yuan, Zhenlong and Qian, Chengxuan and Tang, Jing and Chen, Rui and Song, Zijian and Sun, Lei and Chu, Xiangxiang and Cai, Yujun and Zhang, Dapeng and Li, Shuo},
  journal={arXiv preprint arXiv:2509.01944},
  year={2025}
}
```

---

## 🤝 Acknowledgements

We sincerely appreciate the contributions of the open-source community:
- [Video-R1](https://github.com/tulerfeng/Video-R1)
- [lmm-r1](https://github.com/TideDra/lmm-r1)
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [nuScenes](https://www.nuscenes.org/)
- [Waymo Open Dataset](https://waymo.com/open/)
