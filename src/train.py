"""
train.py — JEPA-LoRA Training Script (Qwen2.5 edition)
========================================================

Reads JSONL files where every line is {"text": "..."}.

Precision policy
────────────────
The backbone runs in bfloat16 by default (set via --torch_dtype).
ALL loss computations inside the model are upcast to float32 (see model.py).
AMP (torch.autocast) is only enabled for float16, never for bfloat16, because
bfloat16 already has the same exponent range as float32 and does not benefit
from the GradScaler.  For float32 training (--torch_dtype float32), everything
runs natively in float32 with no autocast at all.

Quick start (Qwen2.5-0.5B):
─────────────────────────────
python train.py \\
    --train_file ./data/train.jsonl \\
    --output_dir ./checkpoints/qwen-jepa

Full config (Qwen2.5-7B, 4× GPU):
────────────────────────────────────
torchrun --nproc_per_node=4 train.py \\
    --model_name_or_path Qwen/Qwen2.5-7B \\
    --torch_dtype bfloat16 \\
    --train_file ./data/train.jsonl \\
    --output_dir ./checkpoints/qwen25-7b-jepa \\
    --per_device_train_batch_size 2 \\
    --gradient_accumulation_steps 8
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from model import JEPALoRAModel, JEPALoRAConfig, QWEN25_LORA_TARGETS, SPAN_JEPA_LENGTHS
from collator import build_collator
from loss_schedule import LossSchedule

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

from typing import Optional

def generate_example(text, model):
    inputs = model.tokenizer(text, return_tensors="pt").to("cuda")

    # Merge LoRA weights into base for faster inference

    gen_kwargs = dict(
        max_new_tokens=512,
        temperature=0.7,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
        do_sample=True,
        repetition_penalty=1.15,  # penalise tokens already seen in context
        no_repeat_ngram_size=5,
    )

    with torch.no_grad():
        output_ids = model.backbone.generate(**inputs, **gen_kwargs)

    # Only print the newly generated tokens (after the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[0][prompt_len:]
    generated = model.tokenizer.decode(new_ids, skip_special_tokens=True)
    return generated

class JSONLDataset(Dataset):
    """
    JSONL dataset supporting two training modes:

    Pretrain mode (--completion_column not set)
        Each line: {"text": "..."}
        Tokens pre-computed at init. Collator produces labels = input_ids,
        LM loss trains on every token.

    SFT mode (--completion_column set)
        Each line: {"prompt": "...", "completion": "..."}
        Raw strings are returned; SFTCollator tokenises on the fly and sets
        labels=-100 for prompt tokens so LM loss only supervises the completion.
        JEPA auxiliary losses still run on the full sequence (prompt + completion).
    """

    def __init__(
        self,
        path: str,
        text_column: str,
        tokenizer,
        max_length: int,
        completion_column: Optional[str] = None,
    ):
        self.sft_mode          = (completion_column is not None)
        self.text_column       = text_column
        self.completion_column = completion_column
        self.examples: list    = []
        """
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning(f"Skipping malformed line {i}: {e}")
                    continue
        """
        with open(path) as f:
            objs = json.load(f)
            for obj in objs:
                if self.sft_mode:
                    prompt     = obj.get(text_column, "")
                    completion = obj.get(completion_column, "")
                    if prompt.strip() and completion.strip():
                        self.examples.append({
                            text_column:       prompt,
                            completion_column: completion,
                        })
                else:
                    text = obj.get(text_column, "")
                    if text.strip():
                        self.examples.append(text)

        if not self.examples:
            cols = text_column + (f" + {completion_column}" if completion_column else "")
            raise ValueError(f"No valid examples in {path}. Expected column(s): {cols}")

        mode_str = (f"SFT  prompt={text_column!r}  completion={completion_column!r}"
                    if self.sft_mode else f"pretrain  text={text_column!r}")
        log.info(f"Loaded {len(self.examples):,} examples  [{mode_str}]  from {path}")

        # Pre-tokenise only in pretrain mode (SFTCollator tokenises on the fly)
        if not self.sft_mode:
            log.info("Tokenising...")
            self._encodings = tokenizer(
                self.examples, truncation=True, max_length=max_length,
                padding=False, return_tensors=None,
            )
            log.info("Tokenisation complete.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if self.sft_mode:
            return self.examples[idx]          # raw dict — SFTCollator tokenises
        return {"input_ids": self._encodings["input_ids"][idx]}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info(f"Device: {device}  |  backbone dtype: {args.torch_dtype}")

    cfg = JEPALoRAConfig(
        model_name_or_path=args.model_name_or_path,
        torch_dtype=args.torch_dtype,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        # Loss weights
        lm_loss_weight=args.lm_loss_weight,
        jepa_pred_token_weight=args.jepa_pred_token_weight,
        span_jepa_weight=args.span_jepa_weight,
        jepa_offset_head_weight=args.jepa_offset_head_weight,
        jepa_pred_head_weight=args.jepa_pred_head_weight,
        layerwise_jepa_weight=args.layerwise_jepa_weight,
        self_consistency_weight=args.self_consistency_weight,
        contrastive_weight=args.contrastive_weight,
        # Offset head
        offset_head_mode=args.offset_head_mode,
        offset_head_hidden_dim=args.offset_head_hidden_dim,
        offset_head_num_layers=args.offset_head_num_layers,
        # Feature flags
        use_pred_token=args.use_pred_token,
        use_span_jepa=args.use_span_jepa,
        use_offset_head=args.use_offset_head,
        use_layerwise_jepa=args.use_layerwise_jepa,
        use_self_consistency=args.use_self_consistency,
        # Layer-wise JEPA
        layerwise_jepa_layers=args.layerwise_jepa_layers,
        layerwise_use_projector=args.layerwise_use_projector,
        # Span JEPA
        span_jepa_lengths=args.span_jepa_lengths,
        # Self-consistency
        self_consistency_prob=args.self_consistency_prob,
        self_consistency_overlap=args.self_consistency_overlap,
        # Contrastive
        contrastive_mode=args.contrastive_mode,
        infonce_temperature=args.infonce_temperature,
        infonce_max_pairs=args.infonce_max_pairs,
        vicreg_lambda=args.vicreg_lambda,
        vicreg_nu=args.vicreg_nu,
        max_seq_len=args.max_seq_len,
    )

    log.info(f"Loading model: {cfg.model_name_or_path}")
    model = JEPALoRAModel(cfg).to(device)
    model.print_trainable_parameters()

        # ── Loss schedule ────────────────────────────────────────────────────────
    schedule = LossSchedule.from_config(cfg, schedule_preset=args.schedule_preset)
    if args.schedule_scale != 1.0:
        schedule = schedule.scale_steps(args.schedule_scale)

    if args.schedule_preset == "lm_only":
        log.info("Loss schedule: LM only (no auxiliary losses)")
    elif args.schedule_preset == "all_on":
        log.info("Loss schedule: all losses at full weight from step 0")
    else:
        log.info(f"Loss schedule preset={args.schedule_preset!r}  scale={args.schedule_scale}")
        log.info(f"  Step 0:    {schedule.log_str(0)}")
        next_ev = schedule.next_event_step(0)
        if next_ev:
            log.info(f"  Next event at step {next_ev}")


    # Log active losses
    active = []
    if cfg.use_pred_token:         active.append("pred_token_embed")
    if cfg.use_span_jepa:          active.append(f"span_jepa{cfg.span_jepa_lengths}")
    if cfg.use_offset_head:        active.append(f"offset_head({cfg.offset_head_mode})")
    if cfg.use_layerwise_jepa:     active.append(f"layerwise_jepa{cfg.layerwise_jepa_layers}")
    if cfg.use_self_consistency:   active.append(f"self_consistency(p={cfg.self_consistency_prob})")
    if cfg.contrastive_mode != "none": active.append(f"contrastive({cfg.contrastive_mode})")
    log.info(f"Active auxiliary losses: {', '.join(active) or 'none'}")

    # ── Mode detection ───────────────────────────────────────────────────────
    sft_mode = bool(args.completion_column)
    if sft_mode:
        log.info(
            f"Mode: SFT  |  prompt={args.text_column!r}  "
            f"completion={args.completion_column!r}"
        )
    else:
        log.info("Mode: pretrain (full-sequence LM)")

    # Data
    dataset = JSONLDataset(
        path=args.train_file,
        text_column=args.text_column,
        tokenizer=model.tokenizer,
        max_length=cfg.max_seq_len,
        completion_column=args.completion_column or None,
    )

    collator = build_collator(
        tokenizer=model.tokenizer,
        use_pred_token=cfg.use_pred_token,
        pred_token_id=model.pred_token_id,
        span_pred_ids=model.span_pred_ids,
        max_length=cfg.max_seq_len,
        pred_interval=args.pred_interval,
        pad_to_multiple_of=8,
        # SFT params
        sft_mode=sft_mode,
        prompt_key=args.text_column       if sft_mode else None,
        completion_key=args.completion_column if sft_mode else None,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Optimizer — three param groups with separate LRs
    def _get_params(model, *name_fragments):
        return [
            p for n, p in model.named_parameters()
            if any(f in n for f in name_fragments) and p.requires_grad
        ]

    aux_head_names = ["mlp_offset_head", "pred_token_head", "layerwise_projectors"]
    backbone_params = [
        p for n, p in model.named_parameters()
        if not any(f in n for f in aux_head_names) and p.requires_grad
    ]
    head_params = _get_params(model, *aux_head_names)

    param_groups = [{"params": backbone_params, "lr": args.learning_rate}]
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": args.learning_rate * args.offset_head_lr_multiplier,
        })

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = (
        len(dataloader) * args.num_train_epochs
        // args.gradient_accumulation_steps
    )
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Precision / AMP setup ────────────────────────────────────────────────
    # bfloat16 backbone → no AMP scaler needed (bf16 has same exponent as fp32)
    # float16 backbone  → use AMP autocast + GradScaler
    # float32 backbone  → no AMP at all (everything already float32)
    # In all cases, losses are computed in float32 inside model.py.
    use_amp    = (args.torch_dtype == "float16") and (device.type == "cuda")
    amp_dtype  = torch.float16 if use_amp else None
    scaler     = torch.cuda.amp.GradScaler(enabled=use_amp)

    log.info(
        f"Training: {args.num_train_epochs} epochs | "
        f"{len(dataloader)} steps/epoch | "
        f"{total_steps} optimizer steps | "
        f"{warmup_steps} warmup | "
        f"AMP={'float16' if use_amp else 'off (losses always float32)'}"
    )

    global_step = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        running: dict = {}
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # autocast only for float16; bfloat16 and float32 skip it
            ctx = (
                torch.cuda.amp.autocast(dtype=amp_dtype)
                if use_amp
                else torch.cuda.amp.autocast(enabled=False)
            )
            with ctx:
                loss_weights = schedule.weights(global_step)
                loss_dict = model(**batch)

            loss = loss_dict["loss"] / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + v.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg = {k: v / args.logging_steps for k, v in running.items()}
                    loss_str = "  ".join(f"{k}={v:.4f}" for k, v in avg.items())
                    sched_str = schedule.log_str(global_step)
                    log.info(
                        f"Epoch {epoch+1}/{args.num_train_epochs} | "
                        f"Step {global_step}/{total_steps} | {loss_str}"
                    )
                    log.info(f"  schedule: {sched_str}")
                    running = {}

                if args.save_steps and global_step % args.save_steps == 0:
                    model.save(os.path.join(args.output_dir, f"checkpoint-{global_step}"))

        example = generate_example("How much is 1+3?", model)
        print(example)
        model.save(os.path.join(args.output_dir, f"epoch-{epoch+1}"))
        log.info(f"Epoch {epoch+1} complete.")

    model.save(args.output_dir)
    log.info(f"Training complete → {args.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="JEPA-LoRA Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Data")
    g.add_argument("--train_file",   type=str, default="dataset.json")
    g.add_argument("--text_column",  type=str, default="prompt",
                   help="JSONL key for the text (pretrain) or prompt (SFT)")
    g.add_argument("--completion_column", type=str, default="completion",
                   help=(
                       "JSONL key for the completion/response. "
                       "Set this to enable SFT mode. "
                       "Leave empty for pretrain mode. "
                       "Example: --completion_column completion"
                   ))

    g = p.add_argument_group("Model")
    g.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    g.add_argument("--torch_dtype",  type=str, default="float32",
                   choices=["float32", "bfloat16", "float16"],
                   help=(
                       "Backbone weight dtype. Loss computations are ALWAYS float32.\n"
                       "float32  → safest, most memory\n"
                       "bfloat16 → recommended for Qwen2.5 / LLaMA (same exponent range as fp32)\n"
                       "float16  → enables AMP GradScaler, more fragile"
                   ))
    g.add_argument("--max_seq_len",  type=int, default=1024)

    g = p.add_argument_group("LoRA")
    g.add_argument("--lora_r",     type=int, default=16)
    g.add_argument("--lora_alpha", type=int, default=32)
    g.add_argument("--lora_dropout", type=float, default=0.05)
    g.add_argument("--lora_target_modules", nargs="+",
                   default=list(QWEN25_LORA_TARGETS))

    g = p.add_argument_group("JEPA feature flags")
    g.add_argument("--use_pred_token",       action="store_true",  default=True)
    g.add_argument("--no_pred_token",        dest="use_pred_token", action="store_false")
    g.add_argument("--use_span_jepa",        action="store_true",  default=True)
    g.add_argument("--no_span_jepa",         dest="use_span_jepa",  action="store_false")
    g.add_argument("--use_offset_head",      action="store_true",  default=True)
    g.add_argument("--no_offset_head",       dest="use_offset_head", action="store_false")
    g.add_argument("--use_layerwise_jepa",   action="store_true",  default=True)
    g.add_argument("--no_layerwise_jepa",    dest="use_layerwise_jepa", action="store_false")
    g.add_argument("--use_self_consistency", action="store_true",  default=True)
    g.add_argument("--no_self_consistency",  dest="use_self_consistency", action="store_false")

    g = p.add_argument_group("Offset head")
    g.add_argument("--offset_head_mode", type=str, default="both",
                   choices=["pred_token", "mlp_head", "both"])
    g.add_argument("--offset_head_hidden_dim", type=int, default=512)
    g.add_argument("--offset_head_num_layers", type=int, default=2)
    g.add_argument("--pred_interval", type=int, default=4,
                   help="Insert prediction token block every N real tokens")

    g = p.add_argument_group("Layer-wise JEPA")
    g.add_argument("--layerwise_jepa_layers", nargs="+", type=int,
                   default=[4, 8, 16, 24],
                   help="Intermediate layer indices to supervise with [PRED] embed loss")
    g.add_argument("--layerwise_use_projector", action="store_true", default=True)
    g.add_argument("--no_layerwise_projector",  dest="layerwise_use_projector",
                   action="store_false")

    g = p.add_argument_group("Span JEPA")
    g.add_argument("--span_jepa_lengths", nargs="+", type=int,
                   default=list(SPAN_JEPA_LENGTHS),
                   help="Span lengths k for [PRED_k] tokens (each predicts mean embed of next k tokens)")

    g = p.add_argument_group("Self-consistency")
    g.add_argument("--self_consistency_prob",    type=float, default=0.5,
                   help="Fraction of batches on which to run the 2nd forward pass")
    g.add_argument("--self_consistency_overlap", type=float, default=0.5,
                   help="Overlap fraction between the two windows")

    g = p.add_argument_group("Contrastive")
    g.add_argument("--contrastive_mode", type=str, default="both",
                   choices=["infonce", "vicreg", "both", "none"])
    g.add_argument("--contrastive_weight",   type=float, default=0.3)
    g.add_argument("--infonce_temperature",  type=float, default=0.07)
    g.add_argument("--infonce_max_pairs",    type=int,   default=4096)
    g.add_argument("--vicreg_lambda",  type=float, default=25.0)
    g.add_argument("--vicreg_mu",      type=float, default=25.0)
    g.add_argument("--vicreg_nu",      type=float, default=1.0)
    g.add_argument("--vicreg_gamma",   type=float, default=1.0)

    g = p.add_argument_group("Loss weights")
    g.add_argument("--lm_loss_weight",           type=float, default=1.0)
    g.add_argument("--jepa_pred_token_weight",   type=float, default=0.5)
    g.add_argument("--span_jepa_weight",         type=float, default=0.4)
    g.add_argument("--jepa_offset_head_weight",  type=float, default=0.5)
    g.add_argument("--jepa_pred_head_weight",    type=float, default=0.5)
    g.add_argument("--layerwise_jepa_weight",    type=float, default=0.3)
    g.add_argument("--self_consistency_weight",  type=float, default=0.4)

    g = p.add_argument_group("Training")
    g.add_argument(
        "--schedule_preset", type=str, default="default",
        choices=["default", "fast", "all_on", "lm_only"],
        help=(
            "Loss curriculum preset:\n"
            "  default — 6-phase curriculum, all losses active by step ~2500\n"
            "  fast    — compressed, all losses active by step ~500\n"
            "  all_on  — no curriculum, all losses at full weight from step 0\n"
            "  lm_only — only LM loss (useful as baseline)"
        ),
    )
    g.add_argument(
        "--schedule_scale", type=float, default=.25,
        help=(
            "Multiply all schedule step thresholds by this factor. "
            "Use 0.5 for short runs, 2.0 for longer runs. "
            "Has no effect when --schedule_preset=all_on or lm_only."
        ),
    )
    g.add_argument("--output_dir",                  type=str,   default="checkpoints")
    g.add_argument("--num_train_epochs",            type=int,   default=3)
    g.add_argument("--per_device_train_batch_size", type=int,   default=4)
    g.add_argument("--gradient_accumulation_steps", type=int,   default=4)
    g.add_argument("--learning_rate",               type=float, default=5e-5)
    g.add_argument("--offset_head_lr_multiplier",   type=float, default=5.0,
                   help="Aux heads LR = learning_rate × this")
    g.add_argument("--weight_decay",                type=float, default=0.01)
    g.add_argument("--max_grad_norm",               type=float, default=1.0)
    g.add_argument("--warmup_ratio",                type=float, default=0.05)
    g.add_argument("--logging_steps",               type=int,   default=50)
    g.add_argument("--save_steps",                  type=int,   default=500)
    g.add_argument("--dataloader_num_workers",       type=int,   default=0)
    g.add_argument("--device", type=str, default=None)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
