"""LLM-JEPA model wrapper.

Wraps any HuggingFace causal LM and provides:
  - Autoregressive NTP forward pass
  - Encoder: last-token hidden-state extractor
  - JEPA forward: computes Pred(Enc(Text)) vs Enc(Code)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from config import JEPAConfig
from loss import cosine_distance


class LLMJepa(nn.Module):
    """Thin wrapper around a causal LM that adds the JEPA prediction head."""

    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.config = config

        # ── Load base model & tokenizer ────────────────────────────────
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if config.bf16 else (torch.float16 if config.fp16 else torch.float32),
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # ── Register [PRED] token(s) ───────────────────────────────────
        self.pred_token_id: Optional[int] = None
        if config.num_pred_tokens > 0:
            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [config.pred_token_str]}
            )
            if num_added > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
            self.pred_token_id = self.tokenizer.convert_tokens_to_ids(
                config.pred_token_str
            )

        # ── Optional LoRA ──────────────────────────────────────────────
        if config.use_lora:
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            self.model.print_trainable_parameters()

        # ── Projection Head ────────────────────────────────────────────
        self.projection_head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, config.projection_dim),
            nn.GELU(),
            nn.Linear(config.projection_dim, config.projection_dim),
            nn.LayerNorm(config.projection_dim),
        )

    # ── Autoregressive NTP pass ────────────────────────────────────────

    def forward_autoregressive(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard causal LM forward.

        Returns:
            (loss, logits)
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out.loss, out.logits

    # ── Encoder ────────────────────────────────────────────────────────

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract last-token hidden state.
        
        If use_last_n_layers > 1, averages the hidden states of those layers.
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Determine which layers to use
        # hidden_states tuple includes (embeddings, layer_1, ..., layer_N)
        # So we want [-N:]
        n = self.config.use_last_n_layers
        relevant_hidden_states = out.hidden_states[-n:]
        
        # Average across layers: (B, L, D)
        avg_hidden = torch.stack(relevant_hidden_states).mean(dim=0)

        # Index of the last *non-pad* token for each sample
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        batch_idx = torch.arange(avg_hidden.size(0), device=avg_hidden.device)
        last_token_hidden = avg_hidden[batch_idx, seq_lengths]  # (B, D)
        
        return last_token_hidden

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply projection head and normalize."""
        proj = self.projection_head(hidden_states)
        return F.normalize(proj, p=2, dim=-1)

    # ── JEPA forward ───────────────────────────────────────────────────
    
    def forward_jepa_step(
        self,
        pred_input_ids: torch.Tensor,
        pred_attention_mask: torch.Tensor,
        target_encoder: "LLMJepa",
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One direction of JEPA: Pred(Enc(Source)) vs Enc_Target(Target)."""
        
        # Predictor/Source Representation
        pred_raw = self.encode(pred_input_ids, pred_attention_mask)
        pred_emb = self.project(pred_raw)
        
        # Target Representation (using target_encoder, likely EMA)
        with torch.no_grad():
            target_raw = target_encoder.encode(target_input_ids, target_attention_mask)
            target_emb = target_encoder.project(target_raw)
            
        return pred_emb, target_emb
