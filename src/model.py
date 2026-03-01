"""
JEPALoRAModel
=============
Wraps any CausalLM (default: Qwen2.5) with a suite of JEPA-style auxiliary
losses on top of standard next-token CE:

  1.  LoRA adapters (PEFT)
  2.  [PRED]-token embed loss     — h[PRED] predicts embed(next_token) via cosine
  3.  Span-JEPA loss              — h[PRED_k] predicts mean(embed(t+1..t+k))
  4.  Causal-offset heads         — MLP: h_t → h_{t+1}  and/or  h[PRED] → embed
  5.  Layer-wise JEPA             — [PRED] embed loss at intermediate layers too
  6.  InfoNCE (false-neg masked)  — contrastive over consecutive (h_t, h_{t+1})
  7.  VICReg                      — variance + invariance + covariance regulariser
  8.  Self-consistency loss       — overlapping windows of same doc must produce
                                   consistent hidden states for the shared span

Precision policy
────────────────
The backbone runs in whatever dtype is configured (bfloat16 by default for
Qwen2.5).  ALL loss computations are upcast to float32 before any arithmetic,
so cosine similarities, InfoNCE softmax, VICReg covariance, etc. are always
numerically stable regardless of backbone dtype.  Gradients flow back through
float32 loss → autocast → backbone parameters.
"""

import os
import json
import dataclasses
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Literal, List

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QWEN25_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

OffsetHeadMode = Literal["pred_token", "mlp_head", "both"]

# Span lengths for Span-JEPA: [PRED_k] is a special token that predicts the
# mean embedding of the next k tokens.  We create one token per span length.
SPAN_JEPA_LENGTHS: List[int] = [2, 4, 8]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class JEPALoRAConfig:
    # ── Base model ──────────────────────────────────────────────────────────
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B"
    # Backbone dtype (memory / speed trade-off).
    # Loss computations are ALWAYS done in float32 regardless of this setting.
    torch_dtype: str = "bfloat16"

    # ── LoRA ────────────────────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: list(QWEN25_LORA_TARGETS)
    )

    # ── Loss weights ────────────────────────────────────────────────────────
    lm_loss_weight: float = 1.0
    jepa_pred_token_weight: float = 0.5     # [PRED]-token → embed(next) cosine
    span_jepa_weight: float = 0.4           # [PRED_k] → mean embed of span
    jepa_offset_head_weight: float = 0.5    # mlp_head offset head
    jepa_pred_head_weight: float = 0.5      # pred_token offset head
    layerwise_jepa_weight: float = 0.3      # per-layer [PRED] cosine losses
    self_consistency_weight: float = 0.4    # overlapping-window consistency
    contrastive_weight: float = 0.3         # InfoNCE / VICReg

    # ── Offset head ─────────────────────────────────────────────────────────
    offset_head_mode: OffsetHeadMode = "both"
    offset_head_hidden_dim: int = 512
    offset_head_num_layers: int = 2

    # ── Feature flags ────────────────────────────────────────────────────────
    use_pred_token: bool = True
    use_span_jepa: bool = True
    use_offset_head: bool = True
    use_layerwise_jepa: bool = True
    use_self_consistency: bool = True

    # ── Layer-wise JEPA ──────────────────────────────────────────────────────
    # Which intermediate layer indices to supervise with [PRED] embed loss.
    # Index 0 = embedding layer output, -1 = last layer (always used separately).
    # These are automatically clamped to the model's actual layer count at init.
    # Example for 28-layer Qwen2.5-0.5B: [4, 8, 16, 24]
    layerwise_jepa_layers: list = field(default_factory=lambda: [4, 8, 16, 24])
    # Each intermediate layer gets its own small linear projector (hidden→hidden).
    # If False, uses the raw hidden state directly (cheaper, slightly weaker).
    layerwise_use_projector: bool = True

    # ── Span JEPA ────────────────────────────────────────────────────────────
    # Span lengths to use.  One [PRED_k] special token is created per length.
    # During training, the collator randomly samples one span length per [PRED]
    # position (uniform over this list).
    span_jepa_lengths: list = field(default_factory=lambda: list(SPAN_JEPA_LENGTHS))

    # ── Self-consistency ─────────────────────────────────────────────────────
    # Fraction of batches on which to run the self-consistency pass.
    # 1.0 = every batch (expensive: 2× forward passes), 0.5 = every other batch.
    self_consistency_prob: float = 0.5
    # Overlap between the two windows as a fraction of max_seq_len.
    self_consistency_overlap: float = 0.5

    # ── Contrastive ──────────────────────────────────────────────────────────
    contrastive_mode: str = "both"       # "infonce" | "vicreg" | "both" | "none"
    infonce_temperature: float = 0.07
    infonce_max_pairs: int = 4096
    # Barlow-Twins-style loss (invariance + covariance on unit sphere).
    # The VICReg variance term is omitted — on a unit sphere the max per-dim
    # std is 1/√D (≈0.033 for D=896), so any γ > that fires permanently.
    # Collapse prevention is already handled by InfoNCE running alongside.
    vicreg_lambda: float = 1.0    # invariance  — cosine distance between pairs
    vicreg_nu: float = 0.04       # covariance  — off-diagonal decorrelation / D
    # vicreg_mu, vicreg_gamma: removed (variance term dropped, see above)

    # ── Training ────────────────────────────────────────────────────────────
    max_seq_len: int = 2048


# ---------------------------------------------------------------------------
# Utility: always-float32 cosine distance
# ---------------------------------------------------------------------------

def cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    1 − cos(a, b), computed in float32 for numerical stability.
    Inputs can be any dtype; output is float32 scalar.
    """
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    return (1.0 - (a * b).sum(dim=-1)).mean()


# ---------------------------------------------------------------------------
# Shared MLP
# ---------------------------------------------------------------------------

def _build_mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int) -> nn.Module:
    if num_layers <= 1 or hidden_dim == 0:
        return nn.Linear(in_dim, out_dim)
    layers: list = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    for _ in range(num_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Offset heads
# ---------------------------------------------------------------------------

class MLPOffsetHead(nn.Module):
    """h_t → h_{t+1} prediction for all consecutive pairs (dense mode)."""

    def __init__(self, hidden_size: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.net = _build_mlp(hidden_size, hidden_size, hidden_dim, num_layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)

    def loss(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h_src = hidden_states[:, :-1]
        h_tgt = hidden_states[:, 1:].detach()
        mask = (attention_mask[:, :-1] * attention_mask[:, 1:]).bool()
        if not mask.any():
            return hidden_states.new_zeros(1).squeeze()
        pred = self.forward(h_src)
        # upcast to float32 for loss computation
        return F.smooth_l1_loss(
            F.normalize(pred[mask].float(), dim=-1),
            F.normalize(h_tgt[mask].float(), dim=-1),
        )


class PredTokenOffsetHead(nn.Module):
    """
    MLP(h[PRED]) → embed(next_token).
    Level-2 supervision: adds learned projection on top of the raw cosine loss.
    """

    def __init__(self, hidden_size: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.net = _build_mlp(hidden_size, hidden_size, hidden_dim, num_layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)

    def loss(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        pred_token_id: int,
        embed_matrix: torch.Tensor,   # (V, D) float32, detached
        pad_token_id: int,
    ) -> torch.Tensor:
        pred_mask = (input_ids == pred_token_id)
        if not pred_mask.any():
            return hidden_states.new_zeros(1).squeeze()

        positions = pred_mask.nonzero(as_tuple=False)
        losses = []
        for b, p in positions:
            if p + 1 >= input_ids.size(1):
                continue
            next_tok = input_ids[b, p + 1].item()
            if next_tok == pad_token_id or next_tok == pred_token_id:
                continue
            proj     = self.forward(hidden_states[b, p])
            target_e = embed_matrix[next_tok]
            losses.append(cosine_loss(proj.unsqueeze(0), target_e.unsqueeze(0)))

        if not losses:
            return hidden_states.new_zeros(1).squeeze()
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Layer-wise projector bank
# ---------------------------------------------------------------------------

class LayerwiseProjectors(nn.Module):
    """
    One small linear projector per supervised intermediate layer.
    Projects h_layer → hidden_size so cosine loss is computed in a
    consistent space.  Can be disabled (use identity) via use_projector=False.
    """

    def __init__(self, hidden_size: int, layer_indices: List[int], use_projector: bool):
        super().__init__()
        self.layer_indices = layer_indices
        self.use_projector = use_projector
        if use_projector:
            # One linear per layer index, stored in a ModuleDict for proper
            # registration (important for optimizer / save/load)
            self.projectors = nn.ModuleDict({
                str(i): nn.Linear(hidden_size, hidden_size, bias=False)
                for i in layer_indices
            })

    def project(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor:
        if self.use_projector:
            return self.projectors[str(layer_idx)](h)
        return h


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class JEPALoRAModel(nn.Module):

    PRED_TOKEN = "[PRED]"

    def __init__(self, cfg: JEPALoRAConfig):
        super().__init__()
        self.cfg = cfg

        # ── Tokenizer ───────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=True,
        )
        # Qwen2.5 shares eos/pad — add a dedicated pad token
        if self.tokenizer.pad_token is None or (
            self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        ):
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        # Main [PRED] token
        new_special = []
        if cfg.use_pred_token or cfg.use_span_jepa:
            new_special.append(self.PRED_TOKEN)

        # [PRED_k] tokens for each span length
        self.span_pred_tokens: dict = {}   # span_len → token string
        self.span_pred_ids: dict = {}      # span_len → token id
        if cfg.use_span_jepa:
            for k in cfg.span_jepa_lengths:
                tok = f"[PRED_{k}]"
                self.span_pred_tokens[k] = tok
                new_special.append(tok)

        if new_special:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": new_special}
            )

        # ── Base model ──────────────────────────────────────────────────────
        dtype_map = {
            "float32":  torch.float32,
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
        }
        torch_dtype = dtype_map[cfg.torch_dtype]

        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        base.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

        # ── LoRA ────────────────────────────────────────────────────────────
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
        )
        self.backbone = get_peft_model(base, lora_cfg)
        self.hidden_size: int = self.backbone.config.hidden_size

        # ── [PRED] token IDs ────────────────────────────────────────────────
        self.pred_token_id: Optional[int] = (
            self.tokenizer.convert_tokens_to_ids(self.PRED_TOKEN)
            if (cfg.use_pred_token or cfg.use_span_jepa) else None
        )
        for k, tok in self.span_pred_tokens.items():
            self.span_pred_ids[k] = self.tokenizer.convert_tokens_to_ids(tok)

        # ── Offset heads ────────────────────────────────────────────────────
        self.mlp_offset_head: Optional[MLPOffsetHead] = None
        self.pred_token_head: Optional[PredTokenOffsetHead] = None
        if cfg.use_offset_head:
            h, d, n = self.hidden_size, cfg.offset_head_hidden_dim, cfg.offset_head_num_layers
            if cfg.offset_head_mode in ("mlp_head", "both"):
                self.mlp_offset_head = MLPOffsetHead(h, d, n)
            if cfg.offset_head_mode in ("pred_token", "both"):
                self.pred_token_head = PredTokenOffsetHead(h, d, n)

        # ── Layer-wise JEPA projectors ───────────────────────────────────────
        self.layerwise_projectors: Optional[LayerwiseProjectors] = None
        if cfg.use_layerwise_jepa:
            num_layers = self.backbone.config.num_hidden_layers
            valid_layers = [
                l for l in cfg.layerwise_jepa_layers if 0 <= l < num_layers
            ]
            if valid_layers:
                self.layerwise_projectors = LayerwiseProjectors(
                    self.hidden_size, valid_layers, cfg.layerwise_use_projector
                )

    # ──────────────────────────────────────────────────────────────────────
    # Embedding matrix helper
    # ──────────────────────────────────────────────────────────────────────

    def _get_embedding_matrix(self) -> torch.Tensor:
        """
        Returns W_e (V, D) in float32, detached.
        Upcast here so all embed-space targets are always float32.
        """
        base = self.backbone.base_model if hasattr(self.backbone, "base_model") else self.backbone
        return base.get_input_embeddings().weight.detach().float()

    # ──────────────────────────────────────────────────────────────────────
    # Loss 1: Standard LM (CE)
    # ──────────────────────────────────────────────────────────────────────

    def _lm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        # CE internally upcast on modern PyTorch, but be explicit
        return F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Loss 2: [PRED]-token embed loss
    # ──────────────────────────────────────────────────────────────────────

    def _jepa_pred_token_embed_loss(
        self,
        hidden_states: torch.Tensor,  # (B, T, D)
        input_ids: torch.Tensor,      # (B, T)
        W_e: torch.Tensor,            # (V, D) float32 detached
    ) -> torch.Tensor:
        """
        For every [PRED] at position p, push h[PRED] toward embed(token_{p+1}).
        Computed entirely in float32.
        """
        pred_mask = (input_ids == self.pred_token_id)
        if not pred_mask.any():
            return hidden_states.new_zeros(1).float().squeeze()

        positions = pred_mask.nonzero(as_tuple=False)
        losses = []
        for b, p in positions:
            if p + 1 >= input_ids.size(1):
                continue
            next_tok = input_ids[b, p + 1].item()
            if next_tok == self.tokenizer.pad_token_id or next_tok == self.pred_token_id:
                continue
            losses.append(cosine_loss(
                hidden_states[b, p].unsqueeze(0),
                W_e[next_tok].unsqueeze(0),
            ))

        if not losses:
            return hidden_states.new_zeros(1).float().squeeze()
        return torch.stack(losses).mean()

    # ──────────────────────────────────────────────────────────────────────
    # Loss 3: Span-JEPA
    # ──────────────────────────────────────────────────────────────────────

    def _span_jepa_loss(
        self,
        hidden_states: torch.Tensor,  # (B, T, D)
        input_ids: torch.Tensor,      # (B, T)
        W_e: torch.Tensor,            # (V, D) float32 detached
    ) -> torch.Tensor:
        """
        For every [PRED_k] token at position p, h[PRED_k] should predict:
            mean(W_e[token_{p+1}], ..., W_e[token_{p+k}])

        This forces the model to maintain a compressed representation of the
        next k tokens *before* seeing them — a multi-step look-ahead in
        embedding space.

        If fewer than k real tokens follow (sequence end or padding), we use
        however many are available (minimum 1).

        Target = mean of available embed vectors, normalised.
        Loss   = cosine distance between h[PRED_k] and that mean.
        """
        losses = []
        pad_id = self.tokenizer.pad_token_id

        for k, span_id in self.span_pred_ids.items():
            span_mask = (input_ids == span_id)
            if not span_mask.any():
                continue

            positions = span_mask.nonzero(as_tuple=False)
            for b, p in positions:
                # Gather up to k real tokens after position p
                embeds = []
                for offset in range(1, k + 1):
                    pos = p + offset
                    if pos >= input_ids.size(1):
                        break
                    tok = input_ids[b, pos].item()
                    # Stop at padding or another special token
                    if tok == pad_id or tok in self.span_pred_ids.values() or tok == self.pred_token_id:
                        break
                    embeds.append(W_e[tok])

                if not embeds:
                    continue

                # Mean of available embeddings as target (float32)
                target = torch.stack(embeds).mean(dim=0)
                losses.append(cosine_loss(
                    hidden_states[b, p].unsqueeze(0),
                    target.unsqueeze(0),
                ))

        if not losses:
            return hidden_states.new_zeros(1).float().squeeze()
        return torch.stack(losses).mean()

    # ──────────────────────────────────────────────────────────────────────
    # Loss 4: Layer-wise JEPA
    # ──────────────────────────────────────────────────────────────────────

    def _layerwise_jepa_loss(
        self,
        all_hidden_states: tuple,     # tuple of (B, T, D), one per layer
        input_ids: torch.Tensor,      # (B, T)
        W_e: torch.Tensor,            # (V, D) float32 detached
    ) -> torch.Tensor:
        """
        Apply the [PRED] embed loss at each configured intermediate layer,
        optionally through a small linear projector.

        Rationale: early layers encode syntax, middle layers semantics, late
        layers task-specific features.  Supervising all of them forces the
        model to develop predictive representations at every depth, not just
        at the output.

        The loss from each layer is averaged (not summed) so the total weight
        stays stable regardless of how many layers are configured.
        """
        if self.layerwise_projectors is None:
            return torch.tensor(0.0)

        pred_mask = (input_ids == self.pred_token_id)
        if not pred_mask.any():
            return torch.tensor(0.0)

        positions = pred_mask.nonzero(as_tuple=False)
        pad_id = self.tokenizer.pad_token_id
        layer_losses = []

        for layer_idx in self.layerwise_projectors.layer_indices:
            # all_hidden_states[0] = embedding output, [1] = layer 1, etc.
            h_layer = all_hidden_states[layer_idx]   # (B, T, D)
            pos_losses = []

            for b, p in positions:
                if p + 1 >= input_ids.size(1):
                    continue
                next_tok = input_ids[b, p + 1].item()
                if next_tok == pad_id or next_tok == self.pred_token_id:
                    continue

                h_proj = self.layerwise_projectors.project(layer_idx, h_layer[b, p])
                target = W_e[next_tok]
                pos_losses.append(cosine_loss(h_proj.unsqueeze(0), target.unsqueeze(0)))

            if pos_losses:
                layer_losses.append(torch.stack(pos_losses).mean())

        if not layer_losses:
            return torch.tensor(0.0)
        return torch.stack(layer_losses).mean()

    # ──────────────────────────────────────────────────────────────────────
    # Loss 5: Self-consistency
    # ──────────────────────────────────────────────────────────────────────

    def _self_consistency_loss(
        self,
        input_ids: torch.Tensor,      # (B, T)
        attention_mask: torch.Tensor, # (B, T)
    ) -> torch.Tensor:
        """
        Self-consistency loss: run the model on two overlapping windows of
        the same document and penalise divergence in the shared region.

        Window layout (overlap = 50% of T):
            Window A: tokens [0  .. T-1]        ← original batch (already computed)
            Window B: tokens [T//2 .. T//2 + T] ← shifted by T//2

        For the shared span [T//2 .. T-1] we compare hidden states from both
        windows using VICReg's variance + invariance + covariance loss.

        This forces the model to produce *stable* representations for a token
        span regardless of what comes before it — a strong global coherence
        signal that pure NTP never provides.

        Because this requires a second forward pass it is gated by
        cfg.self_consistency_prob (default 0.5).  The batch must have at
        least 2 × overlap tokens to be meaningful; short sequences are skipped.

        Note: Window B is constructed by rolling input_ids along dim=1.
        This keeps everything in-batch (no extra data needed) and naturally
        creates the shifted context.
        """
        T = input_ids.size(1)
        overlap = int(T * self.cfg.self_consistency_overlap)
        shift   = T - overlap   # how far we shift: T // 2 for 50% overlap

        if overlap < 4:   # too short to be meaningful
            return torch.tensor(0.0)

        # Build Window B by shifting tokens along the sequence dimension.
        # We roll input_ids by `shift` positions: the last `overlap` tokens
        # of the original become the first `overlap` of the new window.
        # Padding is handled by using attention_mask on both.
        input_ids_b   = torch.roll(input_ids,   shifts=-shift, dims=1)
        attn_mask_b   = torch.roll(attention_mask, shifts=-shift, dims=1)

        # The rolled region [T-shift:] wraps around and is invalid — zero it out
        attn_mask_b[:, -shift:] = 0

        with torch.no_grad():
            out_b = self.backbone(
                input_ids=input_ids_b,
                attention_mask=attn_mask_b,
                output_hidden_states=False,
            )

        # Hidden states of Window A for the overlap region: positions [0 : overlap]
        # (these correspond to tokens [shift : T] of the original document)
        # Hidden states of Window B for the overlap region: positions [0 : overlap]
        # (Window B starts at token [shift], so first `overlap` positions match)
        # We fetch Window A's hidden states from the stored forward pass result.
        # IMPORTANT: We cannot use the already-computed hidden_states here because
        # self-consistency is called from forward() which passes h_A. We receive
        # it as an argument.
        # → This method returns a sentinel; actual call in forward() passes h_A.
        raise NotImplementedError("Call _self_consistency_loss_with_ha instead")

    def _self_consistency_loss_with_ha(
        self,
        h_a: torch.Tensor,            # (B, T, D) — Window A hidden states (last layer)
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """The actual implementation, called from forward() with h_A in hand."""
        T = input_ids.size(1)
        overlap = int(T * self.cfg.self_consistency_overlap)
        shift   = T - overlap

        if overlap < 4:
            return torch.tensor(0.0, device=h_a.device)

        input_ids_b = torch.roll(input_ids,     shifts=-shift, dims=1)
        attn_mask_b = torch.roll(attention_mask, shifts=-shift, dims=1)
        attn_mask_b[:, -shift:] = 0

        # Second forward pass — no grad for backbone; loss flows through h_A
        with torch.no_grad():
            out_b = self.backbone(
                input_ids=input_ids_b,
                attention_mask=attn_mask_b,
                output_hidden_states=True,
            )
        h_b = out_b.hidden_states[-1]   # (B, T, D)

        # Shared region in Window A: positions [0 : overlap] correspond to
        # the original tokens [0 : overlap].
        # Shared region in Window B: positions [0 : overlap] correspond to
        # original tokens [shift : shift + overlap] = [shift : T].
        # These are NOT the same tokens — the shared region is by construction
        # the LAST `overlap` tokens of A and the FIRST `overlap` tokens of B
        # (after the roll), which are both original tokens [shift : T].
        h_a_shared = h_a[:, shift:, :]           # (B, overlap, D) — last overlap of A
        h_b_shared = h_b[:, :overlap, :].detach()# (B, overlap, D) — first overlap of B

        # Flatten to (B*overlap, D) for VICReg
        valid_a = attention_mask[:, shift:].bool()   # (B, overlap)
        valid_b = attn_mask_b[:,  :overlap].bool()   # (B, overlap)
        valid   = valid_a & valid_b                   # (B, overlap)

        if not valid.any():
            return torch.tensor(0.0, device=h_a.device)

        z_a = h_a_shared[valid].float()   # (N, D)
        z_b = h_b_shared[valid].float()   # (N, D)

        if z_a.size(0) < 2:
            return torch.tensor(0.0, device=h_a.device)

        return self._vicreg_tensors(z_a, z_b)

    # ──────────────────────────────────────────────────────────────────────
    # Loss 6/7: InfoNCE + VICReg (contrastive)
    # ──────────────────────────────────────────────────────────────────────

    def _gather_consecutive_pairs(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_pairs: int = 4096,
    ) -> tuple:
        h_src_list, h_tgt_list, tok_list = [], [], []
        for b in range(hidden_states.size(0)):
            valid = attention_mask[b].bool()
            valid_pairs = valid[:-1] & valid[1:]
            idx = valid_pairs.nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue
            h_src_list.append(hidden_states[b, idx])
            h_tgt_list.append(hidden_states[b, idx + 1])
            tok_list.append(input_ids[b, idx + 1])

        if not h_src_list:
            return None, None, None

        h_src       = torch.cat(h_src_list).float()
        h_tgt       = torch.cat(h_tgt_list).float()
        next_tokens = torch.cat(tok_list)

        N = h_src.size(0)
        if N > max_pairs:
            idx = torch.randperm(N, device=h_src.device)[:max_pairs]
            h_src, h_tgt, next_tokens = h_src[idx], h_tgt[idx], next_tokens[idx]

        return h_src, h_tgt, next_tokens

    def _infonce_loss(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        tau = self.cfg.infonce_temperature
        h_src, h_tgt, next_tokens = self._gather_consecutive_pairs(
            hidden_states, input_ids, attention_mask, self.cfg.infonce_max_pairs
        )
        if h_src is None:
            return torch.tensor(0.0)

        N = h_src.size(0)
        h_src = F.normalize(h_src, dim=-1)
        h_tgt = F.normalize(h_tgt, dim=-1)

        sim = torch.matmul(h_src, h_tgt.T) / tau   # (N, N) float32
        # False-negative masking: same next-token → not a valid negative
        same_tok = next_tokens.unsqueeze(0) == next_tokens.unsqueeze(1)
        same_tok.fill_diagonal_(False)
        sim = sim.masked_fill(same_tok, float("-inf"))

        labels = torch.arange(N, device=sim.device)
        return F.cross_entropy(sim, labels)

    def _vicreg_tensors(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Barlow-Twins-style loss on two float32 tensors (N, D).

        Why we dropped the VICReg variance term
        ─────────────────────────────────────────
        VICReg's variance hinge  relu(γ − std)  was designed for unnormalised
        projector outputs stabilised by BatchNorm, where std per dimension is
        O(1).  After L2-normalisation to the unit sphere, the maximum possible
        std per dimension is 1/√D (≈ 0.033 for D=896).  Any γ > 1/√D means
        the hinge fires at its maximum value on EVERY dimension FOREVER — it
        becomes a large constant that never decreases, which is where the ~250
        loss came from.

        Collapse prevention on the unit sphere is already handled by InfoNCE
        (which runs alongside this loss): the InfoNCE denominator forces
        representations apart, making the variance hinge redundant.

        What remains is invariance + covariance — equivalent to Barlow Twins:
          • Invariance  — cosine distance between paired vectors, ∈ [0, 2]
          • Covariance  — off-diagonal cross-correlation penalty, decorrelates
                          feature dimensions so each encodes something distinct

        Expected loss range at training start: 0.05 – 0.5 (well-behaved).
        """
        lam = self.cfg.vicreg_lambda   # invariance weight
        nu  = self.cfg.vicreg_nu       # covariance weight
        # vicreg_mu and vicreg_gamma are intentionally unused after this fix

        # L2-normalise to unit sphere
        z_a = F.normalize(z_a, dim=-1)   # (N, D)
        z_b = F.normalize(z_b, dim=-1)   # (N, D)

        # Invariance: mean cosine distance between paired representations
        # Perfectly aligned pairs → 0.  Orthogonal → 1.  Anti-aligned → 2.
        inv = (1.0 - (z_a * z_b).sum(dim=-1)).mean()

        def _cov(z):
            N, D = z.shape
            z = z - z.mean(dim=0)             # centre each dimension
            c = (z.T @ z) / (N - 1)           # (D, D) sample covariance
            # Penalise squared off-diagonal entries only.
            # Diagonal = per-dimension variance (not penalised here).
            # Divide by D so the scale doesn't grow with model size.
            off_diag = c.pow(2).sum() - c.diagonal().pow(2).sum()
            return off_diag / D

        cov = _cov(z_a) + _cov(z_b)
        return lam * inv + nu * cov

    def _vicreg_loss(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        h_src, h_tgt, _ = self._gather_consecutive_pairs(
            hidden_states, input_ids, attention_mask, self.cfg.infonce_max_pairs
        )
        if h_src is None:
            return torch.tensor(0.0)
        return self._vicreg_tensors(h_src, h_tgt)

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        run_self_consistency: Optional[bool] = None,
        loss_weights: Optional[dict] = None,
    ) -> dict:
        """
        loss_weights: optional dict mapping loss name → effective weight.
            Overrides the static weights in cfg for any key present.
            Keys not present fall back to cfg values.
            Produced by LossSchedule.weights(global_step) in the training loop.
            None → use cfg weights as-is (old behaviour / all_on mode).

            Example:
                {"jepa_pred_token_loss": 0.25,   # 50% ramped in
                 "span_jepa_loss":       0.0,    # not yet started
                 "infonce_loss":         0.0}

        run_self_consistency: override the stochastic gate.
          None  → use cfg.self_consistency_prob to decide randomly
          True  → always run it
          False → never run it
        """
        need_all_hidden = self.cfg.use_layerwise_jepa and (self.layerwise_projectors is not None)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            output_hidden_states=True,   # always needed (layer-wise JEPA or fallback)
        )

        logits            = outputs.logits             # (B, T, V)
        all_hidden_states = outputs.hidden_states      # tuple of (B, T, D), len = n_layers+1
        hidden_states     = all_hidden_states[-1]      # last layer (B, T, D)

        loss_dict: dict = {}
        total_loss = torch.tensor(0.0, device=input_ids.device)

        # Effective weight lookup — schedule overrides cfg if provided
        def _w(loss_name: str, cfg_weight: float) -> float:
            if loss_weights is not None and loss_name in loss_weights:
                return loss_weights[loss_name]
            return cfg_weight

        # ── 1. LM loss — always at full cfg weight, never scheduled ─────────
        if labels is not None:
            lm = self._lm_loss(logits, labels)
            loss_dict["lm_loss"] = lm
            total_loss = total_loss + self.cfg.lm_loss_weight * lm

        # ── Fetch embedding matrix once (shared by losses 2, 3, 4) ─────────
        W_e = self._get_embedding_matrix() if (
            self.cfg.use_pred_token or self.cfg.use_span_jepa or self.cfg.use_layerwise_jepa
        ) else None

        # ── 2. [PRED]-token embed loss ───────────────────────────────────────
        w = _w("jepa_pred_token_loss", self.cfg.jepa_pred_token_weight)
        if self.cfg.use_pred_token and w > 0 and labels is not None and self.pred_token_id is not None:
            pt = self._jepa_pred_token_embed_loss(hidden_states, input_ids, W_e)
            loss_dict["jepa_pred_token_loss"] = pt
            total_loss = total_loss + w * pt

        # ── 3. Span-JEPA ─────────────────────────────────────────────────────
        w = _w("span_jepa_loss", self.cfg.span_jepa_weight)
        if self.cfg.use_span_jepa and w > 0 and self.span_pred_ids:
            sj = self._span_jepa_loss(hidden_states, input_ids, W_e)
            loss_dict["span_jepa_loss"] = sj
            total_loss = total_loss + w * sj

        # ── 4. Causal-offset heads ───────────────────────────────────────────
        if self.cfg.use_offset_head:
            w_oh = _w("jepa_offset_head_loss", self.cfg.jepa_offset_head_weight)
            if self.mlp_offset_head is not None and w_oh > 0:
                oh = self.mlp_offset_head.loss(hidden_states, attention_mask)
                loss_dict["jepa_offset_head_loss"] = oh
                total_loss = total_loss + w_oh * oh

            w_ph = _w("jepa_pred_head_loss", self.cfg.jepa_pred_head_weight)
            if self.pred_token_head is not None and w_ph > 0 and self.pred_token_id is not None:
                ph = self.pred_token_head.loss(
                    hidden_states, input_ids,
                    self.pred_token_id, W_e, self.tokenizer.pad_token_id,
                )
                loss_dict["jepa_pred_head_loss"] = ph
                total_loss = total_loss + w_ph * ph

        # ── 5. Layer-wise JEPA ───────────────────────────────────────────────
        w = _w("layerwise_jepa_loss", self.cfg.layerwise_jepa_weight)
        if self.cfg.use_layerwise_jepa and w > 0 and self.layerwise_projectors is not None:
            lw = self._layerwise_jepa_loss(all_hidden_states, input_ids, W_e)
            loss_dict["layerwise_jepa_loss"] = lw
            total_loss = total_loss + w * lw

        # ── 6/7. Contrastive losses ──────────────────────────────────────────
        cmode = self.cfg.contrastive_mode
        w_nce = _w("infonce_loss", self.cfg.contrastive_weight)
        if cmode in ("infonce", "both") and w_nce > 0:
            nce = self._infonce_loss(hidden_states, input_ids, attention_mask)
            loss_dict["infonce_loss"] = nce
            total_loss = total_loss + w_nce * nce

        w_vic = _w("vicreg_loss", self.cfg.contrastive_weight)
        if cmode in ("vicreg", "both") and w_vic > 0:
            vic = self._vicreg_loss(hidden_states, input_ids, attention_mask)
            loss_dict["vicreg_loss"] = vic
            total_loss = total_loss + self.cfg.contrastive_weight * vic

        # ── 8. Self-consistency ──────────────────────────────────────────────
        w_sc = _w("self_consistency_loss", self.cfg.self_consistency_weight)
        if self.cfg.use_self_consistency and w_sc > 0:
            do_sc = (
                run_self_consistency
                if run_self_consistency is not None
                else (random.random() < self.cfg.self_consistency_prob)
            )
            if do_sc:
                sc = self._self_consistency_loss_with_ha(
                    hidden_states, input_ids, attention_mask
                )
                loss_dict["self_consistency_loss"] = sc
                total_loss = total_loss + w_sc * sc

        loss_dict["loss"] = total_loss
        return loss_dict

    # ──────────────────────────────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────────────────────────────

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.backbone.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        extras: dict = {}
        if self.mlp_offset_head is not None:
            extras["mlp_offset_head"] = self.mlp_offset_head.state_dict()
        if self.pred_token_head is not None:
            extras["pred_token_head"] = self.pred_token_head.state_dict()
        if self.layerwise_projectors is not None:
            extras["layerwise_projectors"] = self.layerwise_projectors.state_dict()

        torch.save(extras, os.path.join(output_dir, "jepa_extras.pt"))
        with open(os.path.join(output_dir, "jepa_config.json"), "w") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=2)
        print(f"[JEPALoRAModel] Saved → {output_dir}")

    @classmethod
    def load(cls, output_dir: str, device: str = "cpu") -> "JEPALoRAModel":
        with open(os.path.join(output_dir, "jepa_config.json")) as f:
            cfg_dict = json.load(f)
        cfg = JEPALoRAConfig(**cfg_dict)

        tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
        dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=dtype_map[cfg.torch_dtype],
            trust_remote_code=True,
        )
        base.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        peft_backbone = PeftModel.from_pretrained(base, output_dir)

        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.cfg = cfg
        model.tokenizer = tokenizer
        model.backbone = peft_backbone
        model.hidden_size = peft_backbone.config.hidden_size

        model.pred_token_id = (
            tokenizer.convert_tokens_to_ids(cls.PRED_TOKEN)
            if (cfg.use_pred_token or cfg.use_span_jepa) else None
        )
        model.span_pred_tokens = {}
        model.span_pred_ids = {}
        if cfg.use_span_jepa:
            for k in cfg.span_jepa_lengths:
                tok = f"[PRED_{k}]"
                model.span_pred_tokens[k] = tok
                model.span_pred_ids[k] = tokenizer.convert_tokens_to_ids(tok)

        model.mlp_offset_head = None
        model.pred_token_head = None
        model.layerwise_projectors = None

        h, d, n = model.hidden_size, cfg.offset_head_hidden_dim, cfg.offset_head_num_layers
        if cfg.use_offset_head:
            if cfg.offset_head_mode in ("mlp_head", "both"):
                model.mlp_offset_head = MLPOffsetHead(h, d, n)
            if cfg.offset_head_mode in ("pred_token", "both"):
                model.pred_token_head = PredTokenOffsetHead(h, d, n)

        if cfg.use_layerwise_jepa:
            num_layers = peft_backbone.config.num_hidden_layers
            valid_layers = [l for l in cfg.layerwise_jepa_layers if 0 <= l < num_layers]
            if valid_layers:
                model.layerwise_projectors = LayerwiseProjectors(
                    model.hidden_size, valid_layers, cfg.layerwise_use_projector
                )

        extras_path = os.path.join(output_dir, "jepa_extras.pt")
        if os.path.exists(extras_path):
            extras = torch.load(extras_path, map_location=device, weights_only=True)
            for attr, key in [
                ("mlp_offset_head", "mlp_offset_head"),
                ("pred_token_head", "pred_token_head"),
                ("layerwise_projectors", "layerwise_projectors"),
            ]:
                obj = getattr(model, attr)
                if obj is not None and key in extras:
                    obj.load_state_dict(extras[key])

        model = model.to(device)
        model.eval()
        print(f"[JEPALoRAModel] Loaded ← {output_dir}")
        return model

    def print_trainable_parameters(self) -> None:
        self.backbone.print_trainable_parameters()
        heads = [
            ("mlp_offset_head",     self.mlp_offset_head),
            ("pred_token_head",     self.pred_token_head),
            ("layerwise_projectors",self.layerwise_projectors),
        ]
        for name, mod in heads:
            if mod is not None:
                n = sum(p.numel() for p in mod.parameters())
                print(f"  + {name}: {n:,} trainable params")
