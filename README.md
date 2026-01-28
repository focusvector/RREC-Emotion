# RREC-Emotion: Emotion Classification with Reasoning

A GRPO (Group Relative Policy Optimization) training framework for emotion classification using large language models with chain-of-thought reasoning. Built on top of R2ec (Towards Large Recommender Models with Reasoning).

## Overview

This project implements reinforcement learning-based training for emotion classification on the ED_easy_4 subset of Empathetic Dialogues, supporting 4 emotion classes. The system uses vLLM for efficient inference and incorporates reasoning capabilities before classification.

**Key Features:**
- GRPO training with vLLM inference engine
- Chain-of-thought reasoning before emotion classification
- Support for Gemma-2-2b-it and Qwen2.5-3B-Instruct models
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- 4-class emotion taxonomy (ED_easy_4)
- Distributed training support with DeepSpeed

**Original Paper:** [R2ec: Towards Large Recommender Models with Reasoning](https://arxiv.org/abs/2505.16994)

## Installation

### Environment Setup

The project requires Python 3.11 and uses conda for environment management. Create the environment using the provided configuration:

```bash
# Create conda environment from environment.yaml
conda env create -f environment.yaml
conda activate RREC
```

### Environment Configuration (environment.yaml)

Key dependencies:
- **PyTorch**: 2.6.0+cu124 (CUDA 12.4)
- **vLLM**: 0.7.3 (inference engine)
- **Transformers**: 4.57.6
- **Accelerate**: 1.6.0 (distributed training)
- **PEFT**: 0.15.0 (LoRA support)
- **TRL**: 0.16.1 (reinforcement learning)
- **Datasets**: 3.4.1
- **Flash Attention**: flashinfer-python 0.5.3
- **xformers**: 0.0.28.post3
- **Triton**: 3.2.0

Full environment specification:
```yaml
name: RREC
channels:
  - defaults
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.11
  - pip
  - pip:
    # Core ML/DL Frameworks
    - torch==2.6.0+cu124
    - torchaudio==2.6.0+cu124
    - torchvision==0.21.0+cu124
    - triton==3.2.0
    
    # Transformers & HuggingFace
    - transformers==4.57.6
    - tokenizers==0.22.2
    - huggingface-hub==0.36.0
    - safetensors==0.7.0
    - datasets==3.4.1
    - accelerate==1.6.0
    - peft==0.15.0
    - trl==0.16.1
    
    # Inference & Optimization
    - vllm==0.7.3
    - bitsandbytes==0.49.1
    - flashinfer-python==0.5.3
    - compressed-tensors==0.9.1
    - xformers==0.0.28.post3
    
    # CUDA Python
    - cuda-python==13.1.1
    - cupy-cuda12x==13.6.0
    
    # Data Processing
    - pandas==2.2.3
    - numpy==1.26.4
    - pyarrow==22.0.0
    - scipy==1.17.0
    
    # Utilities
    - fire==0.7.0
    - rich==13.9.4
    - tqdm==4.67.1
    - pyyaml==6.0.3
    - einops==0.8.1
```

### Manual Installation (Alternative)

If you prefer manual installation:

```bash
pip install -r requirements.txt
```

**Note**: The requirements.txt provides a minimal installation. For the full tested environment, use environment.yaml.

### Path Configuration

Before running the code, you need to configure the model paths in `paths.py`. The file contains paths to the pretrained models:

```python
model_names = {
    "Gemma-2-2b-it": "/path/to/your/gemma-2-2b-it",
    "Qwen2.5-3B-Instruct": "/path/to/your/Qwen2.5-3B-Instruct",
}
```

You need to:
1. Download the pretrained models:
   - Gemma-2-2b-it: [Download from Hugging Face](https://huggingface.co/google/gemma-2-2b-it)
   - Qwen2.5-3B-Instruct: [Download from Hugging Face](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

2. Update the paths in `paths.py` to point to your local model directories:
   - Replace `/path/to/your/gemma-2-2b-it` with the actual path to your Gemma model
   - Replace `/path/to/your/Qwen2.5-3B-Instruct` with the actual path to your Qwen model

Make sure the paths are absolute paths and the directories contain the complete model files.

### Data Preparation

The project uses the ED_easy_4 dataset stored under `data/ED_easy_4/`. To prepare the data:

1. Run the preprocessing script:
```bash
# Preprocess ED_easy_4 from local CSVs
python preprocess.py --dataset_name=ED_easy_4 --download=False --output_name=ED_easy_4_processed
```

The script will:
- Read ED_easy_4 local CSVs (train.csv, valid.csv, test.csv, label_tree.tsv)
- Clean and normalize text data (HTML unescape, remove tags, normalize whitespace)
- Map the 4-class label IDs to emotions
- Process data into HuggingFace datasets format
- Save the processed dataset to `data/ED_easy_4_processed/`

**Dataset Statistics (ED_easy_4):**
- Train: 2,386 samples
- Valid: 358 samples
- Test: 328 samples
- Total: 3,072 dialogues with emotion labels

## Emotion Classification

### Supported Emotions (4 classes)

The system classifies text into 4 emotion categories:

sad, joyful, angry, afraid

### Prompt Format

The model uses a reasoning-based prompt format:
dialogue and identify the emotion expressed. Think through your reasoning step-by-step before providing your answer.

Dialogue: [dialoguelowing review and identify the emotion expressed. Think through your reasoning step-by-step before providing your answer.

Review: [review text]

Provide your analysis in the following format:
1. First, explain your reasoning about the emotion
2. Then, provide your final answer in this exact format: <answer>emotion_name</answer>
```

### Output Format

Expected model output:
```
[Chain-of-thought reasoning explaining the emotion analysis]

<answer>joy</answer>
```

## Performance Metrics

Training metrics tracked:
- `output_len`: Average output length (should be ~200+ tokens with reasoning)
- `reward_mean`: Average reward signal from GRPO
- `grad_norm`: Gradient norm for training stability
- `loss`: Training loss

## System Requirements

**Hardware:**
- GPU: NVIDIA GPU with 48GB+ VRAM (tested on RTX 6000 Ada)
- RAM: 64GB+ recommended
- Storage: 100GB+ for models and data

**Software:**
- Ubuntu 20.04+ (tested on Ubuntu via WSL)
- CUDA 12.4
- Python 3.11
- Miniconda/Anaconda

## Project Structure

```
r2ec/
├── train.py                 # Main training script
├── inference.py             # Inference script
├── preprocess.py           # Data preprocessing
├── environment.yaml        # Conda environment configuration
├── requirements.txt        # Minimal pip requirements
├── paths.py               # Model path configuration
├── models/
│   ├── abstract_models.py  # Base model classes
│   ├── gemma_models.py     # Gemma model implementation
│   └── qwen_models.py      # Qwen model implementation
├── trainers/
│   ├── GRecTrainer.py      # GRPO trainer
│   ├── RecPOTrainer.py     # RecPO trainer
│   └── utils.py            # Training utilities
├── prompters/
│   ├── abstract_prompter.py # Base prompter class
│   ├── prompts.py          # Prompt templates
│   └── rrec_prompter.py    # RRec prompter implementation
├── data_collators/
│   └── data_collator.py    # Data collation utilities
└── accelerates/
    └── deepspeed_config.yaml # DeepSpeed configuration
```

## Known Issues

**GPU Memory:**
- vLLM + training model + optimizer requires ~35-44GB on 48GB GPU
- Use `vllm_gpu_memory_utilization=0.25` to prevent OOM errors
- Batch sizes are reduced for memory efficiency

**Training Stability:**
- Model may generate invalid emotions not in the 4-class list
- Gradient instability possible (monitor grad_norm)

**PyTorch Version:**
- PyTorch 2.6.0 required for checkpoint resume with safetensors
- vLLM 0.7.3 officially requires PyTorch 2.5.1 (compatibility warning expected)

## Troubleshooting

**OOM Errors:**
```bash
# Reduce batch sizes and GPU memory utilization
python train.py --vllm_gpu_memory_utilization=0.2 --train_batch_size=1
```

**Too Many Open Files:**
```bash
# Increase file descriptor limit
ulimit -n 65536
```

**Checkpoint Resume Issues:**
```bash
# Ensure PyTorch 2.6.0+ is installed
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```



## Training

### Basic Training

To train the model:

```bash
# Single GPU training with vLLM
python train.py \
    --use_vllm=True \
    --num_train_epochs=3 \
    --run_name=emotion-rl-v1 \
    --model=gemma \
    --vllm_gpu_memory_utilization=0.25 \
    --eval_on_start=False \
    --max_new_tokens=300
```

### Training Parameters

Key parameters:
- `--use_vllm`: Enable vLLM for efficient inference (default: True)
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--run_name`: Experiment name for checkpoints and logs
- `--model`: Base model to use ("gemma" or "qwen")
- `--vllm_gpu_memory_utilization`: GPU memory fraction for vLLM (0.2-0.3 recommended)
- `--eval_on_start`: Run evaluation before training (default: False)
- `--max_new_tokens`: Maximum tokens to generate (300 for reasoning)
- `--resume_from_checkpoint`: Resume from last checkpoint (default: False)

### Training Hyperparameters (defaults)

```python
train_batch_size: 4
eval_batch_size: 32
warmup_steps: 32
num_train_epochs: 3
group_size: 4  # GRPO group size
learning_rate: 5e-7
max_new_tokens: 300  # Increased for reasoning
```

### Checkpoint Management

Checkpoints are saved to `../checkpoints/[run_name]/`:
- Model saved every 20 steps
- Optimizer state included for exact resume
- Uses safetensors format (requires PyTorch 2.6.0+)

Resume training:
```bash
python train.py \
    --use_vllm=True \
    --num_train_epochs=1 \
    --run_name=emotion-rl-v1 \
    --model=gemma \
    --vllm_gpu_memory_utilization=0.25 \
    --resume_from_checkpoint=True
```

## Citation

If you use this code or find it helpful, please cite the original R2ec paper:

```bibtex
@misc{you2025r2ec,
    title={R$^2$ec: Towards Large Recommender Models with Reasoning},
    author={Runyang You and Yongqi Li and Xinyu Lin and Xin Zhang and Wenjie Wang and Wenjie Li and Liqiang Nie},
    year={2025},
    eprint={2505.16994},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
}
```

## License

See LICENSE file for details.

## Acknowledgments

This project builds upon the R2ec framework for large recommender models with reasoning capabilities, adapting it for emotion classification tasks with reinforcement learning.
