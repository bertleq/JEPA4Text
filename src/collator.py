"""
Collators for JEPA-LoRA training.

StandardCollator
    Plain causal LM collation — padding, labels = input_ids with -100 on pad.

JEPAPredTokenCollator
    Inserts [PRED] and [PRED_k] tokens into sequences.

    Layout (pred_interval=4, span_lengths=[2,4]):
        t0 t1 t2 t3 [PRED] [PRED_2] [PRED_4] t4 t5 t6 t7 [PRED] [PRED_2] [PRED_4] ...
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    inserted block every pred_interval real tokens

    All prediction-token positions get label = -100 (excluded from CE loss).
    The model learns to predict at these positions via the JEPA auxiliary losses
    in model.py, not via the LM head.

    When span_pred_ids is empty (span JEPA disabled), only [PRED] is inserted.

build_collator
    Factory that returns the right collator based on config.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class StandardCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], str):
            encoded = self.tokenizer(
                examples, truncation=True, max_length=self.max_length,
                padding=False, return_tensors=None,
            )
            input_ids_list = encoded["input_ids"]
        else:
            input_ids_list = [ex["input_ids"][:self.max_length] for ex in examples]
        return self._pad_and_label(input_ids_list)

    def _pad_and_label(self, input_ids_list):
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(ids) for ids in input_ids_list)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1)
                       // self.pad_to_multiple_of * self.pad_to_multiple_of)
        input_ids, attention_mask, labels = [], [], []
        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(ids + [-100] * pad_len)
        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }


@dataclass
class JEPAPredTokenCollator:
    """
    Inserts prediction tokens every pred_interval real tokens.

    Inserted block at each interval:
        [PRED]  — always (if pred_token_id is set)
        [PRED_2], [PRED_4], ... — one per configured span length

    The block is inserted AFTER every pred_interval-th real token.
    Labels at all special-token positions are set to -100.

    pred_interval=0  →  insert block only at end of sequence (summary mode)
    pred_interval=1  →  after every real token (dense, expensive)
    pred_interval=4  →  after every 4 tokens (recommended)
    """
    tokenizer: PreTrainedTokenizerBase
    pred_token_id: Optional[int]               # [PRED] token id, or None
    span_pred_ids: Dict[int, int]              # {span_len: token_id}
    max_length: int = 512
    pred_interval: int = 4
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], str):
            encoded = self.tokenizer(
                examples, truncation=True, max_length=self.max_length,
                padding=False, return_tensors=None,
            )
            raw_ids_list = encoded["input_ids"]
        else:
            raw_ids_list = [ex["input_ids"][:self.max_length] for ex in examples]

        input_ids_list, labels_list = [], []
        for raw_ids in raw_ids_list:
            inp, lbl = self._insert_pred_tokens(raw_ids)
            # Hard truncate to max_length after insertion
            input_ids_list.append(inp[:self.max_length])
            labels_list.append(lbl[:self.max_length])

        return self._pad_and_label(input_ids_list, labels_list)

    def _build_pred_block(self) -> List[int]:
        """Returns the list of special token ids to insert at each interval."""
        block = []
        if self.pred_token_id is not None:
            block.append(self.pred_token_id)
        # Insert span tokens in ascending order of span length
        for k in sorted(self.span_pred_ids.keys()):
            block.append(self.span_pred_ids[k])
        return block

    def _insert_pred_tokens(self, token_ids: List[int]):
        pred_block = self._build_pred_block()
        if not pred_block:
            # Nothing to insert — passthrough
            return list(token_ids), list(token_ids)

        n_special = len(pred_block)
        inp, lbl = [], []

        if self.pred_interval <= 0:
            # End-only mode
            inp = list(token_ids) + pred_block
            lbl = list(token_ids) + [-100] * n_special
            return inp, lbl

        for i, tok in enumerate(token_ids):
            inp.append(tok)
            lbl.append(tok)
            # After every pred_interval real tokens, insert the prediction block
            if (i + 1) % self.pred_interval == 0 and i < len(token_ids) - 1:
                inp.extend(pred_block)
                lbl.extend([-100] * n_special)

        return inp, lbl

    def _pad_and_label(self, input_ids_list, labels_list):
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(ids) for ids in input_ids_list)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1)
                       // self.pad_to_multiple_of * self.pad_to_multiple_of)
        input_ids, attention_mask, labels = [], [], []
        for ids, lbls in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(lbls + [-100] * pad_len)
        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }


def build_collator(
    tokenizer: PreTrainedTokenizerBase,
    use_pred_token: bool,
    pred_token_id: Optional[int],
    span_pred_ids: Dict[int, int],
    max_length: int,
    pred_interval: int = 4,
    pad_to_multiple_of: Optional[int] = 8,
    # SFT params — set both to enable SFT mode
    sft_mode: bool = False,
    prompt_key: Optional[str] = None,    # key for the prompt field in each example
    completion_key: Optional[str] = None,
) -> Union["StandardCollator", "JEPAPredTokenCollator", "SFTCollator", "SFTJEPACollator"]:
    """
    Factory.  Returns the right collator based on mode and JEPA flags.

    Pretrain mode (sft_mode=False):
        JEPAPredTokenCollator  if any [PRED] tokens are active
        StandardCollator       otherwise

    SFT mode (sft_mode=True):
        SFTJEPACollator  if any [PRED] tokens are active
        SFTCollator      otherwise

    In SFT mode the prompt tokens receive label=-100 so the LM loss only
    supervises the completion.  JEPA auxiliary losses still run on the full
    sequence (prompt + completion) — the richer context in the prompt makes
    them more informative, not less.
    """
    has_pred_tokens = (use_pred_token and pred_token_id is not None) or bool(span_pred_ids)

    if not sft_mode:
        if has_pred_tokens:
            return JEPAPredTokenCollator(
                tokenizer=tokenizer,
                pred_token_id=pred_token_id if use_pred_token else None,
                span_pred_ids=span_pred_ids,
                max_length=max_length,
                pred_interval=pred_interval,
                pad_to_multiple_of=pad_to_multiple_of,
            )
        return StandardCollator(
            tokenizer=tokenizer,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    # SFT mode
    assert prompt_key and completion_key, \
        "sft_mode=True requires both prompt_key and completion_key"
    if has_pred_tokens:
        return SFTJEPACollator(
            tokenizer=tokenizer,
            prompt_key=prompt_key,
            completion_key=completion_key,
            pred_token_id=pred_token_id if use_pred_token else None,
            span_pred_ids=span_pred_ids,
            max_length=max_length,
            pred_interval=pred_interval,
            pad_to_multiple_of=pad_to_multiple_of,
        )
    return SFTCollator(
        tokenizer=tokenizer,
        prompt_key=prompt_key,
        completion_key=completion_key,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
    )


# ---------------------------------------------------------------------------
# SFT collators
# ---------------------------------------------------------------------------

@dataclass
class SFTCollator:
    """
    Supervised fine-tuning collator (no JEPA tokens).

    Each example dict must contain two keys (configurable):
        prompt_key     — the instruction / context  (labels masked to -100)
        completion_key — the target response         (labels = token ids)

    The two parts are concatenated as:
        [prompt tokens] [completion tokens] [EOS]

    Truncation strategy: if prompt + completion exceeds max_length, the
    PROMPT is truncated from the right first (keeping the full completion).
    If the completion alone exceeds max_length it is truncated too, but a
    warning is logged.  This mirrors the common practice of preserving as
    much of the target as possible.
    """
    tokenizer: PreTrainedTokenizerBase
    prompt_key: str
    completion_key: str
    max_length: int = 2048
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list, labels_list = [], []

        for ex in examples:
            prompt_ids, completion_ids = self._encode_pair(ex)
            inp, lbl = self._build_labels(prompt_ids, completion_ids)
            input_ids_list.append(inp)
            labels_list.append(lbl)

        return self._pad_and_label(input_ids_list, labels_list)

    def _encode_pair(self, ex: Dict):
        prompt_text     = ex[self.prompt_key]
        completion_text = ex[self.completion_key]

        prompt_ids     = self.tokenizer.encode(prompt_text,     add_special_tokens=True)
        completion_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)

        eos_id = self.tokenizer.eos_token_id

        # Separate EOS from the completion body so truncation never removes it.
        # We re-attach it after truncation, guaranteeing it is always present.
        has_eos = eos_id is not None
        body    = completion_ids  # no EOS yet

        # Truncate: shorten prompt first to preserve as much completion as possible,
        # then truncate completion body if still too long.
        # Reserve 1 slot for EOS at the end.
        eos_reserve = 1 if has_eos else 0
        budget      = self.max_length - eos_reserve

        total = len(prompt_ids) + len(body)
        if total > budget:
            excess        = total - budget
            prompt_budget = max(0, len(prompt_ids) - excess)
            prompt_ids    = prompt_ids[:prompt_budget]
            remaining     = budget - len(prompt_ids)
            body          = body[:remaining]

        if has_eos:
            completion_ids = body + [eos_id]
        else:
            completion_ids = body

        return prompt_ids, completion_ids

    def _build_labels(self, prompt_ids: List[int], completion_ids: List[int]):
        inp = prompt_ids + completion_ids
        # Mask prompt — only supervise on completion
        lbl = [-100] * len(prompt_ids) + list(completion_ids)
        return inp, lbl

    def _pad_and_label(self, input_ids_list, labels_list):
        pad_id  = self.tokenizer.pad_token_id
        max_len = max(len(ids) for ids in input_ids_list)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1)
                       // self.pad_to_multiple_of * self.pad_to_multiple_of)

        input_ids, attention_mask, labels = [], [], []
        for ids, lbls in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(lbls + [-100] * pad_len)

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }


@dataclass
class SFTJEPACollator(SFTCollator):
    """
    SFT collator that also inserts [PRED] / [PRED_k] tokens.

    JEPA tokens are inserted only into the **completion** portion — inserting
    them into the prompt would corrupt the instruction context and introduce
    noise into the prompt's causal attention.  The prediction block sits
    between completion tokens, so [PRED] still has real next tokens to predict.

    Layout (pred_interval=4, completion has tokens c0..c11, then EOS):
        [prompt tokens] c0 c1 c2 c3 [PRED][PRED_2][PRED_4] c4 ... c11 EOS

    Labels:
        prompt           → all -100
        completion real  → token ids  (LM loss)
        [PRED] / [PRED_k]→ -100       (JEPA losses only)
        EOS              → eos_token_id (LM loss — critical for stop-token learning)

    EOS protection
    ──────────────
    EOS must always be the last token of every sequence and must always have
    a real label (not -100).  Two bugs that would hide it:

    1. SFTJEPACollator inflates the sequence with JEPA blocks AFTER
       _encode_pair budgets for max_length, so a naive [:max_length] truncation
       chops EOS off first.  Fix: strip EOS before insertion, reattach after,
       truncate the middle, then append EOS unconditionally.

    2. _encode_pair must budget for JEPA expansion when computing how much
       completion to keep, otherwise EOS gets pushed past max_length.
       Fix: compute the JEPA overhead and subtract it from the budget first.
    """
    pred_token_id: Optional[int] = None
    span_pred_ids: Dict[int, int] = field(default_factory=dict)
    pred_interval: int = 4

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list, labels_list = [], []
        n_jepa = len(self._build_pred_block())

        for ex in examples:
            prompt_ids, completion_ids = self._encode_pair_jepa(ex, n_jepa)

            # Separate EOS from the rest of the completion — protect it from
            # being truncated or displaced by JEPA token insertion
            eos_id = self.tokenizer.eos_token_id
            if (eos_id is not None
                    and completion_ids
                    and completion_ids[-1] == eos_id):
                completion_body = completion_ids[:-1]
                has_eos = True
            else:
                completion_body = completion_ids
                has_eos = False

            # Insert JEPA tokens into the completion body (not EOS)
            comp_with_pred, comp_labels = self._insert_pred_tokens(completion_body)

            # Reattach EOS after JEPA expansion — always at the very end
            if has_eos:
                comp_with_pred = comp_with_pred + [eos_id]
                comp_labels    = comp_labels    + [eos_id]

            full_inp = prompt_ids + comp_with_pred
            full_lbl = [-100] * len(prompt_ids) + comp_labels

            # Hard safety truncation (should rarely fire given _encode_pair_jepa)
            # but if it does, protect EOS by truncating from before it
            if len(full_inp) > self.max_length:
                full_inp = full_inp[:self.max_length - 1] + ([eos_id] if has_eos else [full_inp[-1]])
                full_lbl = full_lbl[:self.max_length - 1] + ([eos_id] if has_eos else [full_lbl[-1]])

            input_ids_list.append(full_inp)
            labels_list.append(full_lbl)

        return self._pad_and_label(input_ids_list, labels_list)

    def _encode_pair_jepa(self, ex: Dict, n_jepa_per_block: int):
        """
        Like SFTCollator._encode_pair but accounts for JEPA token expansion
        when computing the budget, so EOS is never pushed past max_length.

        JEPA expansion overhead: every pred_interval completion tokens add
        n_jepa_per_block extra tokens.  We compute the worst-case overhead
        and subtract it from the completion budget before encoding.
        """
        prompt_text     = ex[self.prompt_key]
        completion_text = ex[self.completion_key]

        prompt_ids     = self.tokenizer.encode(prompt_text,     add_special_tokens=True)
        completion_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            completion_ids = completion_ids + [eos_id]

        if self.pred_interval > 0 and n_jepa_per_block > 0:
            # Worst-case: one JEPA block per pred_interval completion tokens
            # +1 because EOS is reattached separately and doesn't count here
            max_comp_tokens = self.max_length - len(prompt_ids) - 1  # -1 for EOS
            if max_comp_tokens > 0:
                # How many real tokens fit given JEPA expansion?
                # real_tokens + floor(real_tokens / pred_interval) * n_jepa <= budget
                # real_tokens <= budget / (1 + n_jepa / pred_interval)
                expansion_ratio = 1.0 + n_jepa_per_block / self.pred_interval
                max_real = int(max_comp_tokens / expansion_ratio)
                # Keep EOS: truncate body, reattach EOS
                body = completion_ids[:-1] if (eos_id and completion_ids and completion_ids[-1] == eos_id) else completion_ids
                if len(body) > max_real:
                    body = body[:max_real]
                completion_ids = body + ([eos_id] if eos_id else [])
        else:
            # No JEPA expansion — use standard prompt-first truncation
            total = len(prompt_ids) + len(completion_ids)
            if total > self.max_length:
                excess        = total - self.max_length
                prompt_budget = max(0, len(prompt_ids) - excess)
                prompt_ids    = prompt_ids[:prompt_budget]
                if len(prompt_ids) + len(completion_ids) > self.max_length:
                    body = completion_ids[:-1] if (eos_id and completion_ids and completion_ids[-1] == eos_id) else completion_ids
                    body = body[:self.max_length - len(prompt_ids) - 1]
                    completion_ids = body + ([eos_id] if eos_id else [])

        # Truncate prompt if needed
        avail = self.max_length - len(completion_ids)
        prompt_ids = prompt_ids[:max(0, avail)]

        return prompt_ids, completion_ids

    def _build_pred_block(self) -> List[int]:
        block = []
        if self.pred_token_id is not None:
            block.append(self.pred_token_id)
        for k in sorted(self.span_pred_ids.keys()):
            block.append(self.span_pred_ids[k])
        return block

    def _insert_pred_tokens(self, token_ids: List[int]):
        """Insert JEPA blocks into token_ids (EOS already removed by caller)."""
        pred_block = self._build_pred_block()
        if not pred_block or self.pred_interval <= 0:
            return list(token_ids), list(token_ids)

        n_special = len(pred_block)
        inp, lbl = [], []
        for i, tok in enumerate(token_ids):
            inp.append(tok)
            lbl.append(tok)
            # Insert block after every pred_interval tokens, but NOT after the
            # last token (EOS is reattached separately by the caller)
            if (i + 1) % self.pred_interval == 0 and i < len(token_ids) - 1:
                inp.extend(pred_block)
                lbl.extend([-100] * n_special)
        return inp, lbl
