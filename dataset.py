"""Paired-view dataset for LLM-JEPA training.

Each sample yields three tokenized views:
  1. autoregressive — [Text SEP Code]         for L_LLM (NTP on Code tokens)
  2. text_pred     — [Text + k × [PRED]]      for Pred(Enc(Text))
  3. code_only     — [Code]                   for Enc(Code)
"""

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


SEPARATOR = "\n\n### Code:\n"


class JEPADataset(Dataset):
    """Wraps paired (text, code) samples into the three views needed by LLM-JEPA."""

    def __init__(
        self,
        texts: List[str],
        codes: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 512,
        max_text_len: int = 256,
        max_code_len: int = 256,
        num_pred_tokens: int = 1,
        pred_token_id: Optional[int] = None,
        mask_ratio: float = 0.0,
        mask_token_id: Optional[int] = None,
    ):
        assert len(texts) == len(codes)
        self.texts = texts
        self.codes = codes
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.max_code_len = max_code_len
        self.num_pred_tokens = num_pred_tokens
        self.mask_ratio = mask_ratio
        
        # If a dedicated [PRED] token was registered, use it; otherwise fall back
        self.pred_token_id = pred_token_id or tokenizer.pad_token_id
        self.mask_token_id = mask_token_id or tokenizer.mask_token_id or tokenizer.unk_token_id or tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.texts)

    def _apply_masking(self, input_ids: List[int]) -> List[int]:
        """Apply random token masking."""
        if self.mask_ratio <= 0.0:
            return input_ids

        num_to_mask = int(len(input_ids) * self.mask_ratio)
        if num_to_mask == 0:
            return input_ids

        # Select random indices to mask
        import random
        indices = random.sample(range(len(input_ids)), num_to_mask)
        masked_ids = list(input_ids)
        for idx in indices:
            masked_ids[idx] = self.mask_token_id
        return masked_ids

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        code = self.codes[idx]

        # ── View 1: autoregressive [Text SEP Code] ────────────────────
        combined = text + SEPARATOR + code
        ar = self.tokenizer(
            combined,
            truncation=True,
            max_length=self.max_seq_len,
            padding=False,
            return_tensors=None,
        )
        ar_input_ids = ar["input_ids"]
        ar_attention_mask = ar["attention_mask"]

        # Labels: mask the Text portion so NTP loss is only on Code tokens
        text_prefix = text + SEPARATOR
        text_prefix_ids = self.tokenizer(
            text_prefix, truncation=True, max_length=self.max_seq_len, padding=False
        )["input_ids"]
        num_text_tokens = len(text_prefix_ids)

        ar_labels = [-100] * num_text_tokens + ar_input_ids[num_text_tokens:]
        # Align lengths (tokenizer might merge boundary tokens)
        ar_labels = ar_labels[: len(ar_input_ids)]
        if len(ar_labels) < len(ar_input_ids):
            ar_labels += [-100] * (len(ar_input_ids) - len(ar_labels))

        # ── View 2: text + k [PRED] tokens (Masked input) ──────────────
        text_tok = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len - self.num_pred_tokens,
            padding=False,
            return_tensors=None,
        )
        # Apply masking to the text input ids
        masked_text_ids = self._apply_masking(text_tok["input_ids"])
        
        text_pred_ids = masked_text_ids + [self.pred_token_id] * self.num_pred_tokens
        text_pred_mask = text_tok["attention_mask"] + [1] * self.num_pred_tokens

        # ── View 3: code only (Target) ────────────────────────────────
        code_tok = self.tokenizer(
            code,
            truncation=True,
            max_length=self.max_code_len,
            padding=False,
            return_tensors=None,
        )
        
        # ── View 4: code + k [PRED] tokens (Masked input for reverse direction) ──
        # Optional: Apply masking to code input for reverse direction
        masked_code_ids = self._apply_masking(code_tok["input_ids"])
        
        code_pred_ids = masked_code_ids + [self.pred_token_id] * self.num_pred_tokens
        code_pred_mask = code_tok["attention_mask"] + [1] * self.num_pred_tokens

        # ── View 5: text only (Target for reverse direction) ──────────
        # (Already computed text_tok above, just need untokenized for target)
        
        return {
            # autoregressive
            "ar_input_ids": ar_input_ids,
            "ar_attention_mask": ar_attention_mask,
            "ar_labels": ar_labels,
            
            # Text -> Code (Forward JEPA)
            "text_pred_input_ids": text_pred_ids,
            "text_pred_attention_mask": text_pred_mask,
            "code_target_input_ids": code_tok["input_ids"],
            "code_target_attention_mask": code_tok["attention_mask"],
            
            # Code -> Text (Backward JEPA)
            "code_pred_input_ids": code_pred_ids,
            "code_pred_attention_mask": code_pred_mask,
            "text_target_input_ids": text_tok["input_ids"],
            "text_target_attention_mask": text_tok["attention_mask"],
        }


# ── Collator ──────────────────────────────────────────────────────────


def jepa_collate_fn(
    batch: List[Dict[str, Any]], pad_token_id: int
) -> Dict[str, torch.Tensor]:
    """Pads each of the views independently and stacks into tensors."""

    def _pad(sequences: List[List[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(s) for s in sequences)
        padded = [s + [pad_value] * (max_len - len(s)) for s in sequences]
        return torch.tensor(padded, dtype=torch.long)

    return {
        "ar_input_ids": _pad([b["ar_input_ids"] for b in batch], pad_token_id),
        "ar_attention_mask": _pad([b["ar_attention_mask"] for b in batch], 0),
        "ar_labels": _pad([b["ar_labels"] for b in batch], -100),
        
        # Forward: Text(masked) -> Code
        "text_pred_input_ids": _pad(
            [b["text_pred_input_ids"] for b in batch], pad_token_id
        ),
        "text_pred_attention_mask": _pad(
            [b["text_pred_attention_mask"] for b in batch], 0
        ),
        "code_target_input_ids": _pad([b["code_target_input_ids"] for b in batch], pad_token_id),
        "code_target_attention_mask": _pad([b["code_target_attention_mask"] for b in batch], 0),
        
        # Backward: Code(masked) -> Text
        "code_pred_input_ids": _pad(
            [b["code_pred_input_ids"] for b in batch], pad_token_id
        ),
        "code_pred_attention_mask": _pad(
            [b["code_pred_attention_mask"] for b in batch], 0
        ),
        "text_target_input_ids": _pad([b["text_target_input_ids"] for b in batch], pad_token_id),
        "text_target_attention_mask": _pad([b["text_target_attention_mask"] for b in batch], 0),
    }
