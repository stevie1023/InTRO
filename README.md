# OpenRLHF Math Reasoning Implementation

This directory contains the implementation for training and evaluating math reasoning models using OpenRLHF with InTRO.


## Directory Structure

```
code/
├── data/                    # Data processing scripts
│   └── data_preprocessing.py
├── examples/                # Training scripts
│   └── a_train_InTRO_math_ray.sh
├── eval/                    # Evaluation scripts
│   └── Qwen2.5-Math/
│       └── evaluation/
│           └── sh/
│               └── run_eval_qwen2_math.sh
└── README.md               # This file
```

## Prerequisites

Before running the scripts, ensure you have:

**OpenRLHF installed**: 
cd OpenRLHF
pip install -e .
Follow the main [OpenRLHF installation guide](../README.md). Please note that we use an earlier version of 0.6.2 compared to the current version in the official OpenRLHF repo(0.0.1), which have difference in implementation details, so please 


## Hardware Requirements

- **Training**: 5+ GPUs (A100 80GB recommended)
- **Evaluation**: 1+ GPUs for model inference
- **Memory**: At least 64GB system RAM
- **Storage**: Sufficient space for model checkpoints and datasets

## Quick Start

### 1. Data Processing

First, prepare the math dataset by running the preprocessing script:

```bash
cd code
python data/data_preprocessing.py
```

This script:
- Loads the Hendrycks Math dataset with multiple configurations (algebra, geometry, etc.)
- Filters problems to include only Level 3+ difficulty
- Extracts answers from solution text
- Saves processed data to JSON format

**Note**: Update the `save_path` variables in the script to specify your desired output locations.

### 2. Training

Navigate to the examples directory and run the training script:

```bash
cd examples
bash a_train_InTRO_math_ray.sh
```

#### Key Training Parameters

The training script includes several important configurations:

- **Model**: Qwen2.5-1.5B as the base model
- **Hardware**: 5 GPUs (3,4,5,6,7) with Ray distributed computing
- **Batch sizes**: 
  - Micro train batch: 2
  - Train batch: 128
  - Rollout batch: 1024
- **Learning rates**: 
  - Actor: 5e-7
- **Optimization**: DeepSpeed ZeRO Stage 3, FlashAttention, gradient checkpointing

#### Customizing Training

To modify training parameters, edit the following in `a_train_InTRO_math_ray.sh`:

- **GPU allocation**: Change `CUDA_VISIBLE_DEVICES` and Ray GPU counts
- **Model size**: Update `--pretrain` parameter for different model sizes
- **Batch sizes**: Adjust `--micro_train_batch_size`, `--train_batch_size`, etc.
- **Learning rates**: Modify `--actor_learning_rate` and `--critic_learning_rate`
- **Data path**: Update `--prompt_data` to point to your processed dataset

### 3. Evaluation

After training, evaluate your model on the math dataset:

```bash
cd eval/Qwen2.5-Math/evaluation
bash sh/run_eval_qwen2_math.sh
```

The evaluation script:
- Tests multiple model sizes (1.5B, 7B, 72B)
- Uses different prompt templates
- Generates comprehensive evaluation metrics

