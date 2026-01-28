"""
Inference script for R2ec emotion model.
Evaluates on a dataset split and reports accuracy and weighted F1.
"""
import argparse
import json
import os
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from transformers import GenerationConfig
from tqdm import tqdm
from peft import PeftModel

from trainers.utils import get_tokenizer, Similarity
from prompters.rrec_prompter import UserGenPrompter, UserPrompter, ItemPrompter


def _get_model_classes(model_name: str):
    if model_name == "gemma":
        from models.gemma_models import (Gemma2RRecCasualLM as ModelClass,
                                         Gemma2RRecConfig as ConfigClass)
    elif model_name == "qwen":
        from models.qwen_models import (Qwen2RRecCasualLM as ModelClass,
                                        Qwen2RRecConfig as ConfigClass)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return ModelClass, ConfigClass


@dataclass
class SimilarityConfig:
    similarity_type: str = "dot"
    similarity_temperature: float = 0.02
    similarity_normalization: bool = True


def _weighted_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    preds = preds.view(-1)
    labels = labels.view(-1)
    total = labels.numel()
    if total == 0:
        return 0.0

    f1_sum = 0.0
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        support = (labels == c).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_sum += f1 * support

    return f1_sum / total


def _batch_generate(model, tokenizer, input_ids_list, attention_mask_list, gen_config, device, batch_size):
    outputs = []
    for i in tqdm(range(0, len(input_ids_list), batch_size), desc="Generating reasoning"):
        batch_ids = input_ids_list[i:i + batch_size]
        batch_mask = attention_mask_list[i:i + batch_size]
        input_ids = pad_sequence([torch.tensor(x) for x in batch_ids],
                                 batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        attn_mask = pad_sequence([torch.tensor(x) for x in batch_mask],
                                 batch_first=True, padding_value=0).to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                generation_config=gen_config,
            )
        for b in range(input_ids.size(0)):
            gen_ids = out[b, input_ids.size(1):]
            outputs.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return outputs


def _batch_embed(model, input_ids_list, attention_mask_list, device, batch_size, pad_token_id):
    embs = []
    for i in tqdm(range(0, len(input_ids_list), batch_size), desc="Embedding"):
        batch_ids = input_ids_list[i:i + batch_size]
        batch_mask = attention_mask_list[i:i + batch_size]
        input_ids = pad_sequence([torch.tensor(x) for x in batch_ids],
                                 batch_first=True, padding_value=pad_token_id).to(device)
        attn_mask = pad_sequence([torch.tensor(x) for x in batch_mask],
                                 batch_first=True, padding_value=0).to(device)
        with torch.no_grad():
            batch_emb = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                return_with_last_hidden_states=True,
                return_causal_output=False,
            )
        embs.append(batch_emb.cpu())
    return torch.cat(embs, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="../checkpoints/emotion-v1/checkpoint-0")
    parser.add_argument("--dataset_dir", type=str, default="data/ED_easy_4_processed")
    parser.add_argument("--model", type=str, default="qwen", choices=["qwen", "gemma"])
    parser.add_argument("--use_base_model", action="store_true",
                        help="Evaluate the base model without trained weights.")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--max_eval_samples", type=int, default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, args.dataset_dir)
    dataset = load_from_disk(dataset_path)

    if args.use_base_model:
        load_path = None
    else:
        load_path = os.path.join(script_dir, args.checkpoint_path)

    # If a LoRA adapter is provided, infer base model type from adapter config
    if load_path is not None:
        adapter_config_path = os.path.join(load_path, "adapter_config.json")
        if os.path.isfile(adapter_config_path):
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_cfg = json.load(f)
            base_model_ref = str(adapter_cfg.get("base_model_name_or_path", "")).lower()
            inferred = None
            if "gemma" in base_model_ref:
                inferred = "gemma"
            elif "qwen" in base_model_ref:
                inferred = "qwen"
            if inferred and inferred != args.model:
                print(f"[warn] Overriding --model {args.model} -> {inferred} based on adapter_config.json")
                args.model = inferred

    ModelClass, ConfigClass = _get_model_classes(args.model)

    from paths import model_names
    base_model_path = model_names["Qwen2.5-3B-Instruct"] if args.model == "qwen" else model_names["Gemma-2-2b-it"]

    if args.use_base_model:
        load_path = base_model_path

    # Use local checkpoint tokenizer/config only if they exist; otherwise fall back to base model
    has_local_tokenizer = (
        os.path.isdir(load_path)
        and (
            os.path.isfile(os.path.join(load_path, "tokenizer.json"))
            or os.path.isfile(os.path.join(load_path, "tokenizer_config.json"))
        )
    )
    has_local_config = os.path.isdir(load_path) and os.path.isfile(os.path.join(load_path, "config.json"))

    tokenizer_path = load_path if has_local_tokenizer else base_model_path
    config_path = load_path if has_local_config else base_model_path

    tokenizer = get_tokenizer(tokenizer_path)
    config = ConfigClass.from_pretrained(config_path)
    config.use_cache = False
    if config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id

    # If the checkpoint is a LoRA adapter, load base model then apply adapter
    adapter_config_path = os.path.join(load_path, "adapter_config.json")
    if os.path.isfile(adapter_config_path) and not args.use_base_model:
        base_model = ModelClass.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            config=config,
        )
        model = PeftModel.from_pretrained(base_model, load_path)
    else:
        model = ModelClass.from_pretrained(
            load_path if os.path.isdir(load_path) else base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            config=config,
        )
    device = next(model.parameters()).device

    gen_config_source = load_path if os.path.isfile(os.path.join(load_path, "generation_config.json")) else base_model_path
    gen_config = GenerationConfig.from_pretrained(gen_config_source)
    gen_config.max_new_tokens = args.max_new_tokens
    gen_config.do_sample = False

    emb_token = "<answer>"
    emb_end_token = "</answer>"

    user_gen_prompter = UserGenPrompter(
        dset=dataset,
        tokenizer=tokenizer,
        emb_token=emb_token,
        emb_end_token=emb_end_token,
    )
    user_prompter = UserPrompter(
        dset=dataset,
        tokenizer=tokenizer,
        input_ids_max_length=2048,
        emb_token=emb_token,
        emb_end_token=emb_end_token,
    )
    item_prompter = ItemPrompter(
        dset=dataset,
        tokenizer=tokenizer,
        input_ids_max_length=768,
        emb_token=emb_token,
        emb_end_token=emb_end_token,
    )

    eval_dataset = dataset[args.eval_split]
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    # Item embeddings
    item_dataset = item_prompter.convert_dataset(dset=dataset["item_info"])
    item_embs = _batch_embed(
        model,
        item_dataset["item_input_ids"],
        item_dataset["item_attention_mask"],
        device,
        args.batch_size,
        tokenizer.pad_token_id,
    )

    # Generate reasoning profiles
    eval_dataset = user_gen_prompter.convert_dataset(dset=eval_dataset)
    profiles = _batch_generate(
        model,
        tokenizer,
        eval_dataset["user_gen_input_ids"],
        eval_dataset["user_gen_attention_mask"],
        gen_config,
        device,
        args.batch_size,
    )
    eval_dataset = eval_dataset.add_column("profile", profiles)

    # User embeddings
    eval_dataset = user_prompter.convert_dataset(dset=eval_dataset)
    user_embs = _batch_embed(
        model,
        eval_dataset["user_input_ids"],
        eval_dataset["user_attention_mask"],
        device,
        args.batch_size,
        tokenizer.pad_token_id,
    )

    sim = Similarity(SimilarityConfig())
    sim_out = sim(user_embs.to(item_embs.device), item_embs.to(user_embs.device))
    logits = sim_out["softmax_sim"]
    preds = logits.argmax(dim=1).cpu()

    if "seq_labels" in eval_dataset.column_names:
        labels = eval_dataset["seq_labels"]
    else:
        labels = eval_dataset["label"]
    labels = torch.tensor(labels, dtype=torch.long)

    acc = (preds == labels).float().mean().item()
    f1 = _weighted_f1(preds, labels, num_classes=logits.size(1))

    mode = "base" if args.use_base_model else "trained"
    print(f"Eval split: {args.eval_split}")
    print(f"Model: {args.model} ({mode})")
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1:.4f}")


if __name__ == "__main__":
    main()
