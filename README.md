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

## Installation

Before running the scripts, ensure you have the environment installed: 
```
git clone https://github.com/stevie1023/InTRO.git
cd InTRO
pip install -e .
```
For more details, please refer to the [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) installation guide. Note that this codebase targets OpenRLHF v0.6.2, which differs from the current version in the official OpenRLHF repository (v0.9.1) and includes implementation changes. If you use a newer OpenRLHF release, you may need to update the code accordingly. Please also ensure that your dependencies match the specific OpenRLHF version/repository you are using.

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
bash examples/a_train_InTRO_math_ray.sh
```

#### Customizing Training

To modify training parameters, edit the following in `a_train_InTRO_math_ray.sh`:

- **GPU allocation**: Change `CUDA_VISIBLE_DEVICES` and Ray GPU counts
- **Model size**: Update `--pretrain` parameter for different model sizes
- **Batch sizes**: Adjust `--micro_train_batch_size`, `--train_batch_size`, etc.
- **Learning rates**: Modify `--actor_learning_rate` and `--critic_learning_rate`
- **Data path**: Update `--prompt_data` to point to your processed dataset

  
Please note that we **do not apply any prompt templates** during either training or evaluation, as this yields better performance in our setting. If you choose to use a specific template, you may need to modify how the answer is concatenated with the query when estimating the posterior during experience generation.


### 3. Evaluation

After training, evaluate your model on the math dataset:

```bash
cd eval/Qwen2.5-Math/evaluation
bash sh/run_eval_qwen2_math.sh
```

