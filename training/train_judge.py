#!/usr/bin/env python3
"""
Day 5 — Tenacious-Bench ORPO Judge Training Script
Trains Qwen2.5-1.5B-Instruct with LoRA using ORPO (reference-free preference optimization).
Run on Colab T4 or locally with sufficient VRAM.
All hyperparameters are in hyperparams.json and replicated here for auditability.

Usage:
    python train_judge.py [--data-path PATH] [--output-dir DIR]
"""

import os
import sys
import json
import random
import logging
import datetime
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
HYPERPARAMS_PATH = Path(__file__).parent / "hyperparams.json"
DATA_PATH = ROOT / "training_data/preference_pairs.jsonl"
OUTPUT_DIR = Path(__file__).parent / "adapter"
LOG_DIR = Path(__file__).parent

SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def setup_logging(log_path: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(log_path)),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def detect_precision():
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            name = torch.cuda.get_device_name()
            if cap[0] >= 8:  # A100, A10, 4090 — bf16 capable
                logging.info(f"GPU {name} (compute {cap[0]}.{cap[1]}) supports bf16")
                return {"bf16": True, "fp16": False}
            else:  # T4, V100 — fp16 only
                logging.info(f"GPU {name} (compute {cap[0]}.{cap[1]}) using fp16")
                return {"bf16": False, "fp16": True}
    except Exception:
        pass
    return {"bf16": False, "fp16": False}


def load_dataset(data_path: Path, logger):
    from datasets import Dataset
    pairs = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    logger.info(f"Loaded {len(pairs)} preference pairs from {data_path}")
    for p in pairs:
        p.pop("task_id", None)
        p.pop("dimension", None)
    return Dataset.from_list(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--hub-token", type=str, default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    set_seed(SEED)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_path = LOG_DIR / f"training_run_seed{SEED}_{timestamp}.log"
    logger = setup_logging(log_path)

    with open(HYPERPARAMS_PATH) as f:
        hp = json.load(f)
    logger.info(f"Hyperparameters: {json.dumps(hp, indent=2)}")

    precision = detect_precision()
    logger.info(f"Precision: {precision}")

    # Load Unsloth model
    logger.info("Loading Unsloth Qwen2.5-1.5B-Instruct with 4-bit quantization...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hp["model_id"],
        max_seq_length=hp["orpo_trainer"]["max_length"],
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    # Apply LoRA
    logger.info(f"Applying LoRA: r={hp['lora']['r']}, alpha={hp['lora']['lora_alpha']}, "
                f"targets={hp['lora']['target_modules']}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=hp["lora"]["r"],
        target_modules=hp["lora"]["target_modules"],
        lora_alpha=hp["lora"]["lora_alpha"],
        lora_dropout=hp["lora"]["lora_dropout"],
        bias=hp["lora"]["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # Load dataset
    dataset = load_dataset(Path(args.data_path), logger)
    logger.info(f"Dataset size: {len(dataset)}")

    # Training arguments
    from trl import ORPOConfig, ORPOTrainer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = ORPOConfig(
        output_dir=str(output_dir),
        learning_rate=hp["orpo_trainer"]["learning_rate"],
        per_device_train_batch_size=hp["orpo_trainer"]["per_device_train_batch_size"],
        gradient_accumulation_steps=hp["orpo_trainer"]["gradient_accumulation_steps"],
        num_train_epochs=hp["orpo_trainer"]["num_train_epochs"],
        warmup_ratio=hp["orpo_trainer"]["warmup_ratio"],
        lr_scheduler_type=hp["orpo_trainer"]["lr_scheduler_type"],
        beta=hp["orpo_trainer"]["beta"],
        max_length=hp["orpo_trainer"]["max_length"],
        max_prompt_length=hp["orpo_trainer"]["max_prompt_length"],
        logging_steps=hp["orpo_trainer"]["logging_steps"],
        save_steps=hp["orpo_trainer"]["save_steps"],
        seed=SEED,
        bf16=precision["bf16"],
        fp16=precision["fp16"],
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting ORPO training...")
    train_result = trainer.train()
    logger.info(f"Training complete. Metrics: {train_result.metrics}")

    # Save adapter locally
    logger.info(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training run log (copy log file to standard name)
    standard_log = LOG_DIR / "training_run.log"
    import shutil
    shutil.copy(str(log_path), str(standard_log))
    logger.info(f"Training log copied to {standard_log}")

    # Push to HuggingFace
    hub_model_id = hp.get("hub_model_id", "rafiakedir/tenacious-bench-adapter")
    hub_token = args.hub_token or os.environ.get("HF_TOKEN", "")
    if hub_token:
        logger.info(f"Pushing adapter to HuggingFace: {hub_model_id}")
        model.push_to_hub(hub_model_id, token=hub_token)
        tokenizer.push_to_hub(hub_model_id, token=hub_token)
        logger.info(f"Adapter pushed to https://huggingface.co/{hub_model_id}")
    else:
        logger.warning("HF_TOKEN not set — skipping HuggingFace push")

    logger.info("=== TRAINING COMPLETE ===")
    logger.info(f"Adapter saved to: {output_dir}")
    logger.info(f"Log: {standard_log}")

    return train_result.metrics


if __name__ == "__main__":
    main()
