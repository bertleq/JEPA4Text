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

def safe_normalize(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)
    
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
            # For BFloat16, weights can be BF16.
            # For FP16 (AMP), weights must be FP32 to support GradScaler/Optimizer stability,
            # unless using specialized optimizers. Autocast handles the op precision.
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        )

        # Enable Gradient Checkpointing for memory savings
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

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
        # Match dtype of the base model
        self.projection_head = self.projection_head.float()

        self.register_buffer(
            "proto_ema",
            safe_normalize(self.model.lm_head.weight.detach().clone())
        )
    def get_prototypes(self):
        with torch.no_grad():
            proto = self.model.lm_head.weight
            return safe_normalize(proto)

    @torch.no_grad()
    def update_proto_ema(self, tau=0.999):
        tau = self.config.proto_ema_tau
        w = self.model.lm_head.weight.detach()
        # move lm_head weights onto proto device BEFORE normalize
        w = w.to(self.proto_ema.device)
        w = F.normalize(w, dim=-1)
        self.proto_ema.mul_(tau).add_(w, alpha=1 - tau)

    @torch.no_grad()
    def predict_next_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float = 1.0,
    ):
        """
        Softmax-free energy decoding
        Returns:
            next_token_ids: (B,)
        """
    
        hidden = self.encode(input_ids, attention_mask)     # (B, H)
        pred   = safe_normalize(hidden)                      # predictor output
    
        proto  = self.get_prototypes()                       # (V, H)
    
        energy = pred @ proto.T                             # cosine sim
    
        if temperature != 1.0:
            energy = energy / temperature
    
        next_token = energy.argmax(dim=-1)
        return next_token
        
    @torch.no_grad()
    def generate_energy(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 50,
    ):
        for _ in range(max_new_tokens):
    
            next_token = self.predict_next_token(
                input_ids,
                attention_mask
            )
    
            input_ids = torch.cat(
                [input_ids, next_token.unsqueeze(-1)],
                dim=1
            )
    
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones_like(next_token).unsqueeze(-1)
                ],
                dim=1
            )
    
        return input_ids
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
        # Run forward pass without labels first to get logits
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = out.logits
        
        # Calculate loss manually in float32 for stability in FP16/AMP
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss, logits

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
        return safe_normalize(proj)

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

    def fully_forward_jepa_step(
        self,
        pred_input_ids: torch.Tensor,
        pred_attention_mask: torch.Tensor,
        target_encoder: "LLMJepa",
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ):
        """
        One direction of JEPA:
        Pred(Enc(Source)) vs Enc_Target(Target)
        
        Now also returns the future token id t_{k+Δ}
        for prototype alignment (energy decoding).
        """
    
        # ── Predictor Branch ─────────────────────────────────────────────
        pred_raw = self.encode(pred_input_ids, pred_attention_mask)
        pred_proto_hidden_state = safe_normalize(pred_raw)
        pred_emb = self.project(pred_raw)
    
        # ── Target Branch (EMA encoder) ───────────────────────────────────
        with torch.no_grad():
            target_raw = target_encoder.encode(
                target_input_ids,
                target_attention_mask
            )
            target_emb = target_encoder.project(target_raw)
    
        # ── Extract Future Token t_{k+Δ} ──────────────────────────────────
        # last non-pad position in target_input_ids
        seq_lengths = target_attention_mask.sum(dim=1) - 1   # (B,)
        batch_idx   = torch.arange(
            target_input_ids.size(0),
            device=target_input_ids.device
        )
    
        future_token_ids = target_input_ids[batch_idx, seq_lengths]  # (B,)
    
        return pred_emb, target_emb, pred_proto_hidden_state, future_token_ids

    def save_jepa(self, path: str):
        """
        Saves:
          - base LM (+LoRA if used)
          - projection head
          - proto EMA
          - tokenizer
          - config
        """
    
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
        torch.save(
            {
                "projection_head": self.projection_head.state_dict(),
                "proto_ema": self.proto_ema,
                "config": self.config.__dict__,
            },
            f"{path}/jepa_state.pt"
        )

    @classmethod
    def load_jepa(cls, path: str, device="cuda"):
        """
        Loads full JEPA inference model:
          - LM
          - projection head
          - proto EMA
        """
    
        from config import JEPAConfig
    
        jepa_ckpt = torch.load(
            f"{path}/jepa_state.pt",
            map_location=device
        )
    
        config = JEPAConfig(**jepa_ckpt["config"])
    
        model = cls(config)
    
        # load LM
        model.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        )
    
        # load tokenizer
        model.tokenizer = AutoTokenizer.from_pretrained(path)
    
        # load projection head
        model.projection_head.load_state_dict(
            jepa_ckpt["projection_head"]
        )
    
        # restore proto EMA (🚨 critical for energy decoding)
        model.register_buffer(
            "proto_ema",
            jepa_ckpt["proto_ema"].to(device)
        )
    
        model.to(device)
        model.eval()
    
        return model
