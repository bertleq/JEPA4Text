"""LLM-JEPA Training Loop.

Usage:
    python train.py                               # defaults
    python train.py --config configs/default.yaml  # from YAML
    python train.py --model_name gpt2 --max_steps 2 --batch_size 2  # CLI overrides
"""

import argparse
import json
import logging
import math
import os
import random
from functools import partial
from typing import List, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader

from config import JEPAConfig
from dataset import JEPADataset, jepa_collate_fn
from loss import compute_combined_loss
from model import LLMJepa

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: JEPAConfig) -> torch.device:
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg.device)


def load_paired_data(cfg: JEPAConfig) -> Tuple[List[str], List[str]]:
    """Load (text, code) pairs from a dataset.

    Supports:
      - A local JSONL / JSON file  (cfg.dataset_files set)
      - A HuggingFace dataset id   (cfg.dataset_path is an id on the Hub)
      - Auto-generated synthetic demo data (when nothing is provided)
    """
    # ── Synthetic demo data (for smoke testing) ────────────────────────
    if cfg.dataset_path == "__synthetic__" or (
        cfg.dataset_path == "json" and cfg.dataset_files is None
    ):
        logger.info("No dataset provided — generating synthetic demo data")
        return _synthetic_data()

    # ── Local file ─────────────────────────────────────────────────────
    if cfg.dataset_files is not None:
        texts, codes = [], []
        path = cfg.dataset_files
        if path.endswith(".jsonl") or path.endswith(".json"):
            with open(path) as f:
                objs = json.load(f)
                for obj in objs:
                    texts.append(str(obj[cfg.text_field]))
                    codes.append(str(obj[cfg.code_field]))
        else:
            raise ValueError(f"Unsupported file format: {path}")
        logger.info(f"Loaded {len(texts)} samples from {path}")
        return texts, codes

    # ── HuggingFace Hub ────────────────────────────────────────────────
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset_path, split="train")
    texts = [str(row[cfg.text_field]) for row in ds]
    codes = [str(row[cfg.code_field]) for row in ds]
    logger.info(f"Loaded {len(texts)} samples from HF dataset '{cfg.dataset_path}'")
    return texts, codes


def _synthetic_data(n: int = 200) -> Tuple[List[str], List[str]]:
    """Tiny synthetic NL-to-regex-ish pairs for smoke testing."""
    pairs = [
        ("lines containing a digit", r".*[0-9].*"),
        ("lines starting with a vowel", r"^[AEIOUaeiou].*"),
        ("lines ending with a period", r".*\.$"),
        ("lines with the word dog", r".*dog.*"),
        ("lines that are empty", r"^$"),
        ("lines with at least 3 characters", r".{3,}"),
        ("lines starting with a capital letter", r"^[A-Z].*"),
        ("lines containing only digits", r"^[0-9]+$"),
        ("lines with a comma", r".*,.*"),
        ("lines ending with an exclamation mark", r".*!$"),
    ]
    texts, codes = [], []
    for i in range(n):
        t, c = pairs[i % len(pairs)]
        texts.append(t)
        codes.append(c)
    return texts, codes


# ── Training ───────────────────────────────────────────────────────────


def train(cfg: JEPAConfig) -> None:
    set_seed(cfg.seed)
    device = resolve_device(cfg)
    logger.info(f"Device: {device}")

    # ── Model ──────────────────────────────────────────────────────────
    llm_jepa = LLMJepa(cfg)
    llm_jepa.model.to(device)
    llm_jepa.projection_head.to(device)
    tokenizer = llm_jepa.tokenizer

    # ── Target Model (EMA) ─────────────────────────────────────────────
    import copy
    target_llm_jepa = copy.deepcopy(llm_jepa)
    # Freeze target model
    for p in target_llm_jepa.parameters():
        p.requires_grad = False
    target_llm_jepa.model.to(device)
    target_llm_jepa.projection_head.to(device)

    # ── Dataset & DataLoader ───────────────────────────────────────────
    texts, codes = load_paired_data(cfg)
    dataset = JEPADataset(
        texts=texts,
        codes=codes,
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
        max_text_len=cfg.max_text_len,
        max_code_len=cfg.max_code_len,
        num_pred_tokens=cfg.num_pred_tokens,
        pred_token_id=llm_jepa.pred_token_id,
        mask_ratio=cfg.mask_ratio,
        mask_token_id=getattr(dataset_jepa, "mask_token_id", None) if "dataset_jepa" in locals() else None,
    )
    # Re-instantiate dataset to get the mask token id correct if needed, but the class handles it.
    
    collate = partial(jepa_collate_fn, pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    # ── Optimizer & LR scheduler ───────────────────────────────────────
    optimizer = torch.optim.AdamW(
        llm_jepa.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    total_steps = (
        cfg.max_steps
        if cfg.max_steps > 0
        else cfg.epochs * math.ceil(len(dataloader) / cfg.gradient_accumulation_steps)
    )
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_steps
    )

    # ── (Optional) wandb ───────────────────────────────────────────────
    if cfg.use_wandb:
        import wandb
        from dataclasses import asdict

        wandb.init(project=cfg.wandb_project, config=asdict(cfg))

    # ── NVIDIA Optimization ────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled TF32 for NVIDIA GPUs")

    # ── Training loop ──────────────────────────────────────────────────
    os.makedirs(cfg.output_dir, exist_ok=True)
    global_step = 0
    llm_jepa.model.train()

    logger.info(
        f"Starting training — {cfg.epochs} epochs, "
        f"{len(dataloader)} batches/epoch, "
        f"λ={cfg.lambda_jepa}, α={cfg.jepa_alpha}, β={cfg.jepa_beta}, τ={cfg.jepa_temperature}"
    )

    from loss import info_nce_loss, cosine_distance

    # Mixed Precision Setup
    use_amp = cfg.fp16 or cfg.bf16
    amp_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    # Use generic torch.amp.GradScaler for newer PyTorch versions
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16)

    for epoch in range(1, cfg.epochs + 1):
        # Accumulate as tensors to avoid CPU sync every step
        epoch_ntp_loss = torch.tensor(0.0, device=device)
        epoch_jepa_fwd = torch.tensor(0.0, device=device)
        epoch_jepa_bwd = torch.tensor(0.0, device=device)
        epoch_total_loss = torch.tensor(0.0, device=device)
        num_batches = 0

        for step, batch in enumerate(dataloader, 1):
            # Move to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # ── Pass 1: Autoregressive NTP ─────────────────────────────
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                ntp_loss, _ = llm_jepa.forward_autoregressive(
                    input_ids=batch["ar_input_ids"],
                    attention_mask=batch["ar_attention_mask"],
                    labels=batch["ar_labels"],
                )
                scaled_ntp_loss = ntp_loss / cfg.gradient_accumulation_steps
            
            # Backward immediately to free graph
            scaler.scale(scaled_ntp_loss).backward()

            # ── Pass 2: Forward JEPA (Text -> Code) ────────────────────
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                # Uses Main Model for Text(Pred) and Target Model for Code
                pred_emb_fwd, target_emb_fwd = llm_jepa.forward_jepa_step(
                    pred_input_ids=batch["text_pred_input_ids"],
                    pred_attention_mask=batch["text_pred_attention_mask"],
                    target_encoder=target_llm_jepa,
                    target_input_ids=batch["code_target_input_ids"],
                    target_attention_mask=batch["code_target_attention_mask"],
                )
                
                if cfg.jepa_loss_type == "infonce":
                    jepa_loss_fwd = info_nce_loss(pred_emb_fwd, target_emb_fwd, cfg.jepa_temperature)
                else:
                    jepa_loss_fwd = cosine_distance(pred_emb_fwd, target_emb_fwd)
                
                scaled_jepa_fwd = jepa_loss_fwd * cfg.jepa_alpha / cfg.gradient_accumulation_steps

            # Backward immediately
            scaler.scale(scaled_jepa_fwd).backward()

            # ── Pass 3: Backward JEPA (Code -> Text) ───────────────────
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                pred_emb_bwd, target_emb_bwd = llm_jepa.forward_jepa_step(
                    pred_input_ids=batch["code_pred_input_ids"],
                    pred_attention_mask=batch["code_pred_attention_mask"],
                    target_encoder=target_llm_jepa,
                    target_input_ids=batch["text_target_input_ids"],
                    target_attention_mask=batch["text_target_attention_mask"],
                )
                
                if cfg.jepa_loss_type == "infonce":
                    jepa_loss_bwd = info_nce_loss(pred_emb_bwd, target_emb_bwd, cfg.jepa_temperature)
                else:
                    jepa_loss_bwd = cosine_distance(pred_emb_bwd, target_emb_bwd)
                
                scaled_jepa_bwd = jepa_loss_bwd * cfg.jepa_beta / cfg.gradient_accumulation_steps

            # Backward immediately
            scaler.scale(scaled_jepa_bwd).backward()

            # Calculate total loss for logging (detached)
            loss = ntp_loss.detach() + \
                   cfg.jepa_alpha * jepa_loss_fwd.detach() + \
                   cfg.jepa_beta * jepa_loss_bwd.detach()

            if step % cfg.gradient_accumulation_steps == 0:
                # Unscale before clipping
                scaler.unscale_(optimizer)
                
                # Gradient Clipping
                if cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(llm_jepa.parameters(), cfg.max_grad_norm)
                
                # Scaler step
                scaler.step(optimizer)
                scaler.update()
                
                # Update EMA Target
                if cfg.use_ema_target:
                    with torch.no_grad():
                        m = cfg.ema_decay
                        for p_src, p_tgt in zip(llm_jepa.parameters(), target_llm_jepa.parameters()):
                            p_tgt.data.mul_(m).add_(p_src.data, alpha=1 - m)
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # ── Logging ────────────────────────────────────────────────
            # Detach to avoid graph retention, but keep on device to avoid sync
            epoch_ntp_loss += ntp_loss.detach()
            epoch_jepa_fwd += jepa_loss_fwd.detach()
            epoch_jepa_bwd += jepa_loss_bwd.detach()
            epoch_total_loss += loss.detach() * cfg.gradient_accumulation_steps
            num_batches += 1

            if step % cfg.log_every == 0 or step == 1:
                # Sync only when logging
                avg_ntp = epoch_ntp_loss.item() / num_batches
                avg_fwd = epoch_jepa_fwd.item() / num_batches
                avg_bwd = epoch_jepa_bwd.item() / num_batches
                avg_total = epoch_total_loss.item() / num_batches
                lr_now = scheduler.get_last_lr()[0]
                logger.info(
                    f"Ep {epoch}/{cfg.epochs} St {step} "
                    f"L={avg_total:.3f} NTP={avg_ntp:.3f} "
                    f"Fwd={avg_fwd:.3f} Bwd={avg_bwd:.3f} "
                    f"lr={lr_now:.2e}"
                )
                if cfg.use_wandb:
                    import wandb

                    wandb.log(
                        {
                            "loss/total": avg_total,
                            "loss/ntp": avg_ntp,
                            "loss/jepa_fwd": avg_fwd,
                            "loss/jepa_bwd": avg_bwd,
                            "lr": lr_now,
                            "epoch": epoch,
                            "global_step": global_step,
                        },
                        step=global_step,
                    )

            # Early exit for max_steps
            if 0 < cfg.max_steps <= global_step:
                break

        # ── End of epoch ───────────────────────────────────────────────
        avg_ntp = epoch_ntp_loss.item() / max(num_batches, 1)
        avg_fwd = epoch_jepa_fwd.item() / max(num_batches, 1)
        avg_bwd = epoch_jepa_bwd.item() / max(num_batches, 1)
        avg_total = epoch_total_loss.item() / max(num_batches, 1)
        logger.info(
            f"Epoch {epoch} done — "
            f"L={avg_total:.3f} NTP={avg_ntp:.3f} Fwd={avg_fwd:.3f} Bwd={avg_bwd:.3f}"
        )

        if cfg.save_every_epoch:
            ckpt_dir = os.path.join(cfg.output_dir, f"epoch_{epoch}")
            llm_jepa.model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"Checkpoint saved → {ckpt_dir}")

        if 0 < cfg.max_steps <= global_step:
            logger.info(f"Reached max_steps={cfg.max_steps}, stopping.")
            break

    logger.info("Training complete ✓")

    if cfg.use_wandb:
        import wandb

        wandb.finish()


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args() -> JEPAConfig:
    parser = argparse.ArgumentParser(description="LLM-JEPA Training")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")

    # Allow any config field to be overridden from CLI
    for field_name, field_obj in JEPAConfig.__dataclass_fields__.items():
        ftype = field_obj.type
        if ftype == "bool" or ftype is bool:
            parser.add_argument(f"--{field_name}", type=lambda x: x.lower() == "true", default=None)
        elif ftype == "list" or "list" in str(ftype).lower():
            parser.add_argument(f"--{field_name}", nargs="+", default=None)
        elif ftype == "float" or ftype is float:
            parser.add_argument(f"--{field_name}", type=float, default=None)
        elif ftype == "int" or ftype is int:
            parser.add_argument(f"--{field_name}", type=int, default=None)
        else:
            parser.add_argument(f"--{field_name}", type=str, default=None)

    args = parser.parse_args()

    # Start from YAML or defaults
    if args.config:
        cfg = JEPAConfig.from_yaml(args.config)
    else:
        cfg = JEPAConfig()

    # Override with any CLI args
    for field_name in JEPAConfig.__dataclass_fields__:
        val = getattr(args, field_name, None)
        if val is not None:
            setattr(cfg, field_name, val)

    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
