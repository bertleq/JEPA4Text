# JEPA-LoRA (Qwen2.5 edition)

Fine-tune **Qwen2.5** (or any HuggingFace causal LM) with **LoRA** and up to four simultaneous training objectives: standard next-token CE loss plus three complementary JEPA-style latent-prediction losses.

---

## Architecture overview

```
Input tokens: t0  t1  t2  t3 [PRED] t4  t5  t6  t7 [PRED] ...
                                ↑ inserted by JEPAPredTokenCollator every N tokens

                     Qwen2.5 (frozen except LoRA adapters)
                              │
                    hidden states h0..hT
                              │
          ┌───────────────────┼───────────────────────┐
          │                   │                       │
    lm_loss              jepa losses              infonce_loss
   (CE on logits)    ┌────────┴────────┐        (cosine NCE on
                     │                 │         consecutive h_t)
              [PRED]-token         Offset heads
              raw cosine       ┌────────┴────────┐
              loss             │                 │
                          pred_token          mlp_head
                          head (MLP):         head (MLP):
                          h[PRED]→h[prev]     h_t → h_{t+1}
```

---

## The four losses

| Loss | Key | When active | Description |
|------|-----|-------------|-------------|
| Language modelling | `lm_loss` | always | Cross-entropy next-token prediction |
| [PRED]-token cosine | `jepa_pred_token_loss` | `use_pred_token=True` | Backbone's raw `h[PRED]` is pushed toward `h[preceding_token]` via cosine distance. No CE at [PRED] positions. |
| Offset MLP head | `jepa_offset_head_loss` | `offset_head_mode` ∈ {`mlp_head`, `both`} | A tiny MLP maps `h_t → h_{t+1}` for every consecutive pair. Smooth-L1 on normalised vectors. Targets detached. |
| Pred-token head | `jepa_pred_head_loss` | `offset_head_mode` ∈ {`pred_token`, `both`} | A different MLP maps `h[PRED] → h[preceding_token]`. More expressive than the raw cosine loss — adds a learned projection. |
| InfoNCE | `infonce_loss` | `use_infonce=True` | Contrastive objective over all `(h_t, h_{t+1})` pairs in the batch. Diagonal = positives, off-diagonal = negatives. |

Total loss:
```
L = w_lm  × lm_loss
  + w_pt   × jepa_pred_token_loss
  + w_oh   × jepa_offset_head_loss
  + w_ph   × jepa_pred_head_loss
  + w_nce  × infonce_loss
```

---

## Repo layout

```
jepa-lora/
├── src/
│   ├── model.py          # JEPALoRAModel, JEPALoRAConfig, all loss logic
│   └── collator.py       # StandardCollator, JEPAPredTokenCollator
├── configs/
│   ├── qwen25_0.5b.yaml  # Quick-start: Qwen2.5-0.5B
│   ├── qwen25_7b.yaml    # Production: Qwen2.5-7B
│   └── gpt2_debug.yaml   # Tiny GPT-2 for CPU testing
├── train.py              # Training script (JSONL → checkpoint)
├── generate.py           # Inference script (checkpoint → text)
└── requirements.txt
```

---

## Install

```bash
pip install -r requirements.txt
```

---

## Data format

Every line of your JSONL file must be a JSON object with a `text` key:

```jsonl
{"text": "The Eiffel Tower was completed in 1889 and stands 330 metres tall."}
{"text": "Quantum entanglement refers to correlations between particles..."}
```

Other keys are silently ignored. Empty lines and lines missing the key are skipped.

---

## Quick start

### Qwen2.5-0.5B (single GPU or CPU)

```bash
python train.py \
    --train_file ./data/train.jsonl \
    --output_dir ./checkpoints/qwen25-0.5b-jepa
```

All Qwen2.5 defaults are pre-configured: `bfloat16`, 2048-token context,
LoRA on all 7 projection types, `offset_head_mode=both`.

### Qwen2.5-7B (multi-GPU)

```bash
torchrun --nproc_per_node=4 train.py \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --train_file ./data/train.jsonl \
    --output_dir ./checkpoints/qwen25-7b-jepa \
    --lora_r 64 \
    --lora_alpha 128 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_seq_len 2048
```

### Any other HuggingFace causal LM

```bash
python train.py \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --train_file ./data/train.jsonl \
    --output_dir ./checkpoints/llama3-jepa
```

---

## offset_head_mode explained

| Mode | Heads instantiated | What they learn |
|------|--------------------|-----------------|
| `pred_token` | `PredTokenOffsetHead` only | MLP: `h[PRED] → h[preceding token]`. Requires `use_pred_token=True`. Most targeted. |
| `mlp_head` | `MLPOffsetHead` only | MLP: `h_t → h_{t+1}` for every token pair. Dense, works even without [PRED]. |
| `both` *(default)* | Both heads, independent weights | Combines both objectives. Best of both worlds. |

Switch at training time:

```bash
--offset_head_mode pred_token   # targeted only
--offset_head_mode mlp_head     # dense only
--offset_head_mode both         # default
```

---

## Disabling individual losses

```bash
# Pure LoRA (no JEPA at all)
python train.py --no_pred_token --no_offset_head --no_infonce ...

# LM + [PRED] cosine only (lightest JEPA)
python train.py --no_offset_head --no_infonce ...

# LM + InfoNCE only
python train.py --no_pred_token --no_offset_head ...

# Everything on (default)
python train.py ...
```

---

## Generate

```bash
python generate.py \
    --checkpoint ./checkpoints/qwen25-0.5b-jepa \
    --prompt "The key insight about language model training is" \
    --max_new_tokens 200 \
    --temperature 0.7
```

LoRA weights are merged into the base model before generation — no adapter overhead.

---

## Save / Load API

```python
from src.model import JEPALoRAModel, JEPALoRAConfig

# During training (called automatically each epoch)
model.save("./my_checkpoint")

# Reload for inference or continued training
model = JEPALoRAModel.load("./my_checkpoint", device="cuda")
```

Checkpoint contents:
- `adapter_config.json` + `adapter_model.safetensors` — LoRA weights (PEFT format, compatible with `merge_and_unload()`)
- `jepa_extras.pt` — `mlp_offset_head` and/or `pred_token_head` weights
- `jepa_config.json` — full `JEPALoRAConfig` (used to reconstruct model architecture on load)
- tokenizer files — includes the `[PRED]` and `<|pad|>` special tokens

---

## Recommended hyperparameters

| Param | 0.5B | 7B | Notes |
|-------|------|----|-------|
| `lora_r` | 16 | 64 | Higher r = more capacity but more VRAM |
| `lm_loss_weight` | 1.0 | 1.0 | Anchor — keep at 1.0 |
| `jepa_*_weight` | 0.5 | 0.4 | Reduce if JEPA losses destabilise early training |
| `infonce_weight` | 0.3 | 0.2 | Lower for larger models (huge N → already informative) |
| `infonce_temperature` | 0.07 | 0.07 | Increase to 0.1–0.2 for small batches (<8 seq) |
| `pred_interval` | 4 | 8 | Sparser [PRED] for longer sequences to limit overhead |
| `offset_head_lr_multiplier` | 5× | 5× | Heads are small and cold-started |
