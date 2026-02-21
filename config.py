"""Configuration dataclass for LLM-JEPA training."""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class JEPAConfig:
    """All hyperparameters for LLM-JEPA training."""

    # ── Model ──────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # ── Dataset ────────────────────────────────────────────────────────
    dataset_path: str = "json"  # HF dataset id or loader ("json", "csv")
    dataset_files: Optional[str] = "data/dataset.json"  # path to local file(s) if loader
    text_field: str = "prompt"  # column name for the "text" view
    code_field: str = "completion"  # column name for the "code" view

    # ── JEPA-specific ──────────────────────────────────────────────────
    lambda_jepa: float = 1.0  # (Legacy) or total weight
    num_pred_tokens: int = 0  # k  — number of [PRED] tokens appended
    pred_token_str: str = "[PRED]"  # surface form of the predictor token
    
    # New enhancements
    jepa_loss_type: str = "cosine"  # "cosine" or "infonce"
    jepa_temperature: float = 0.5   # τ for InfoNCE
    
    # Joint Optimization (Bidirectional)
    # L = L_LLM + alpha * L(Text->Code) + beta * L(Code->Text)
    jepa_alpha: float = 0.02
    jepa_beta: float = 0.02
    
    # Architecture
    projection_dim: int = 256  # Dimensionality of the projection head
    use_last_n_layers: int = 4 # Average hidden states of last N layers
    
    # EMA Target
    use_ema_target: bool = True
    ema_decay: float = 0.99
    proto_ema_tau: float = 0.99
    
    # Masking
    mask_ratio: float = 0.10   # Ratio of tokens to mask in the input view

    # ── Sequence lengths ───────────────────────────────────────────────
    max_seq_len: int = 512  # max tokens for the autoregressive pass
    max_text_len: int = 512  # max tokens for the text-only encoder pass
    max_code_len: int = 512  # max tokens for the code-only encoder pass

    # ── Training ───────────────────────────────────────────────────────
    lr: float = 1e-5#8e-6
    weight_decay: float = 0.01
    epochs: int = 7
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.05
    max_steps: int = -1  # -1 means train for full epochs
    max_grad_norm: float = 1.0  # Gradient clipping threshold
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = False

    # ── LoRA (optional) ────────────────────────────────────────────────
    use_lora: bool = False
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # ── Logging / checkpointing ────────────────────────────────────────
    output_dir: str = "./checkpoints_energy"
    log_every: int = 10
    save_every_epoch: bool = True
    use_wandb: bool = False
    wandb_project: str = "llm-jepa"

    # ── Device ─────────────────────────────────────────────────────────
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"

    @classmethod
    def from_yaml(cls, path: str) -> "JEPAConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str) -> None:
        from dataclasses import asdict

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
