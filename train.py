"""
Emotion Classification Training Script
Uses RL-based training with similarity-based rewards (GRPO-style policy gradient).
"""
import os
import resource
import datasets
import rich
import torch

# Increase file descriptor limit to prevent "Too many open files" errors
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
except Exception:
    pass  # Ignore if we can't set the limit

# Memory optimizations to prevent OOM
# Enable expandable segments to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from transformers import EarlyStoppingCallback, GenerationConfig
from data_collators.data_collator import RRecDataCollator as DataCollator
from paths import model_names
from trainers.utils import get_compute_metrics, get_tokenizer, MetricUpdater
from trainers.RecPOTrainer import RecPOTrainer, RecPOTrainingArguments


def train(
        output_dir="../checkpoints",
        run_name: str = "emotion-v1",
        
        # Batch sizes (reduced for memory efficiency)
        train_batch_size: int = 2,
        eval_batch_size: int = 20,
        train_generation_batch_size=10,
        test_generation_batch_size=20,
        item_emb_batch_size: int = 80,
        mini_batch_size: int = 2,
        
        # Training settings
        warmup_steps: int = 32,
        eval_freq=8,
        early_stopping_patience=8,
        eval_on_start: bool = True,
        gradient_accumulation_steps: int = 2,
        num_train_epochs: int = 1,
        learning_rate: float = 5e-6,
        item_emb_refresh_steps: int = 0,
        
        # RL settings
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.28,
        reward_softmax_weight: float = 1.0,
        advantage_type: str = "gaussian",
        
        # Dataset
        dataset_dir="data/ED_hard_a_processed",
        
        # Model settings
        use_lora=True,
        seed=42,
        model='qwen',
        resume_from_checkpoint: bool = False,
        gather_negs_across_processes=True,
        lr_scheduler_type='constant',
        use_vllm=True,
        vllm_sync_freq: int = 1,
        direct_item_embeddings: bool = False,
        eval_use_vllm: bool = True,
        
        # Generation settings
        max_new_tokens=64,  # Increased for reasoning + answer format
        group_size=4,
        gen_top_k=100,
        gen_temperature=0.3,
        gen_top_p=0.9,
        
        cleanup_previous_checkpoints=False,
        **kwargs,
):
    trainer_extra_kwargs = {}
    lora_kwargs = {}
    # Known trainer args that shouldn't go to LoRA
    trainer_args = {'vllm_sync_freq', 'vllm_gpu_memory_utilization', 'vllm_dtype', 'vllm_device'}
    for k in kwargs:
        if k.startswith('trainer') or k in trainer_args:
            key = k.replace('trainer_', '') if k.startswith('trainer') else k
            trainer_extra_kwargs[key] = kwargs[k]
        else:
            lora_kwargs[k] = kwargs[k]

    datasets.disable_progress_bars()
    
    # Model selection
    if model == 'gemma':
        model_name = model_names["Gemma-2-2b-it"]
        from models.gemma_models import (Gemma2RRecCasualLM as ModelClass,
                                         Gemma2RRecConfig as ConfigClass)
    elif model == 'qwen':
        model_name = model_names["Qwen2.5-3B-Instruct"]
        from models.qwen_models import (Qwen2RRecCasualLM as ModelClass,
                                        Qwen2RRecConfig as ConfigClass)
    else:
        raise NotImplementedError(f"Model {model} not supported")
    
    output_dir = os.path.join(output_dir, run_name)
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        rich.print(accelerator.deepspeed_plugin)

    ################## Load Dataset ##################
    script_dir = os.path.dirname(os.path.abspath(__file__))
    normalized_dataset_dir = dataset_dir.replace("\\", "/")
    if not os.path.isabs(normalized_dataset_dir):
        normalized_dataset_dir = os.path.join(script_dir, normalized_dataset_dir)
    
    dset = datasets.load_from_disk(normalized_dataset_dir)
    num_emotions = len(dset['item_info'])
    
    if accelerator.is_main_process:
        rich.print(f"\n{'='*60}")
        rich.print(f"Task: Emotion Classification")
        rich.print(f"Dataset: {dataset_dir}")
        rich.print(f"Number of emotions: {num_emotions}")
        rich.print(f"Train size: {len(dset['train'])}")
        rich.print(f"Valid size: {len(dset['valid'])}")
        rich.print(f"Test size: {len(dset['test'])}")
        rich.print(f"{'='*60}\n")
        rich.print("Arguments:", locals())

    ################## Setup Model ##################
    tokenizer = get_tokenizer(model_name)
    emb_token = '<answer>'
    emb_end_token = '</answer>'

    config = ConfigClass.from_pretrained(model_name)
    config.use_cache = False
    config.pad_token_id = tokenizer.pad_token_id
    tokenizer.save_pretrained(output_dir)

    base_model = ModelClass.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.process_index},
        config=config
    )

    ################## Setup Generation ##################
    gen_config = GenerationConfig.from_pretrained(model_name)
    gen_config.max_new_tokens = max_new_tokens
    gen_config.num_return_sequences = group_size
    gen_config.top_k = gen_top_k
    gen_config.top_p = gen_top_p
    gen_config.temperature = gen_temperature

    ################## Setup LoRA ##################
    peft_config_dict = {
        "inference_mode": False,
        "target_modules": ['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    peft_config_dict.update(lora_kwargs)

    if use_lora:
        lora_cfg = {"r": 4, "lora_alpha": 128}
        lora_cfg.update(peft_config_dict)
        peft_config = LoraConfig(**lora_cfg)
        if accelerator.is_main_process:
            rich.print(peft_config)
        base_model = get_peft_model(base_model, peft_config)
    else:
        if accelerator.is_main_process:
            rich.print("No PEFT applied, training the base model")

    ################## Setup Trainer ##################
    eval_steps = len(dset['train']) / (train_batch_size * gradient_accumulation_steps * 3)
    eval_steps = eval_steps // eval_freq

    training_args = RecPOTrainingArguments(
        seed=seed,
        item_emb_batch_size=item_emb_batch_size,
        per_device_train_batch_size=train_batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        bf16=True,
        
        # RL settings
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        reward_softmax_weight=reward_softmax_weight,
        advantage_type=advantage_type,
        item_emb_refresh_steps=item_emb_refresh_steps,
        
        save_strategy="steps",
        save_steps=eval_steps,
        save_only_model=False,
        save_total_limit=5,
        load_best_model_at_end=True,
        
        eval_strategy="steps",
        eval_steps=eval_steps,
        bf16_full_eval=True,
        per_device_eval_batch_size=eval_batch_size,
        metric_for_best_model='eval_valid_accuracy@1',
        eval_on_start=eval_on_start,
        batch_eval_metrics=True,
        
        logging_steps=1,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        report_to='none',
        run_name=run_name,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        
        gather_negs_across_processes=gather_negs_across_processes,
        generation_config=gen_config,
        train_generation_batch_size=train_generation_batch_size,
        test_generation_batch_size=test_generation_batch_size,
        
        emb_token=emb_token,
        emb_end_token=emb_end_token,
        use_vllm=use_vllm,
        eval_use_vllm=eval_use_vllm,
        vllm_sync_freq=vllm_sync_freq,
        direct_item_embeddings=direct_item_embeddings,
        **trainer_extra_kwargs,
    )
    
    metric_updater = MetricUpdater(ks=[1, 3, 5], num_emotions=num_emotions)

    trainer = RecPOTrainer(
        model=base_model,
        compute_metrics=get_compute_metrics(metric_updater),
        data_collator=DataCollator(tokenizer=tokenizer, return_tensors="pt"),
        full_dataset=dset,
        callbacks=[],
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if cleanup_previous_checkpoints:
        os.system(f"rm -rf {output_dir}/checkpoint-*")
        print(f"Removed previous checkpoints in {output_dir}")

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(output_dir)


if __name__ == "__main__":
    import fire
    fire.Fire(train)
