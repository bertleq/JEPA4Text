"""
loss_schedule.py — Curriculum loss scheduling for JEPA-LoRA
=============================================================

Motivation
──────────
Starting all 7+ auxiliary losses at full weight from step 0 causes:
  1. Gradient conflict during early training — losses that haven't converged
     yet push representations in competing directions.
  2. Obscured diagnostics — when everything fires at once it's hard to tell
     which losses are helping or hurting.
  3. Sub-optimal local minima — the backbone carves out a representation
     geometry shaped by random LoRA weights trying to satisfy too many
     objectives simultaneously.

The natural curriculum:
  Phase 1 — Learn to predict completions (LM loss only).
  Phase 2 — Align last-layer representations to embed space ([PRED] token).
  Phase 3 — Learn causal structure (offset heads).
  Phase 4 — Multi-scale look-ahead (span JEPA + layer-wise JEPA).
  Phase 5 — Representation geometry (InfoNCE / VICReg).
  Phase 6 — Global coherence (self-consistency — most expensive, last).

Each auxiliary loss has a (start_step, ramp_steps) pair.  Its effective
weight linearly ramps from 0 to its configured maximum over [start_step,
start_step + ramp_steps], then stays at the maximum.

LM loss is ALWAYS at full weight — it never ramps.

Usage
──────
    schedule = LossSchedule.from_config(cfg)   # build from JEPALoRAConfig
    # ... or build manually:
    schedule = LossSchedule(entries=[
        LossEntry("jepa_pred_token_loss", max_weight=0.5, start_step=200,  ramp_steps=400),
        LossEntry("span_jepa_loss",       max_weight=0.4, start_step=600,  ramp_steps=400),
        ...
    ])

    # In the training loop:
    weights = schedule.weights(global_step)   # dict[loss_name → float]
    loss_dict = model(input_ids, attention_mask, labels=labels,
                      loss_weights=weights)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


# ---------------------------------------------------------------------------
# Single loss entry
# ---------------------------------------------------------------------------

@dataclass
class LossEntry:
    """
    Describes the schedule for one auxiliary loss.

    name        — must match the key used in model.forward()'s loss_dict
    max_weight  — the fully-ramped weight (= cfg.*_weight)
    start_step  — global step at which this loss starts ramping in
    ramp_steps  — number of steps to linearly ramp from 0 → max_weight
                  (0 means: jump to max_weight immediately at start_step)
    """
    name: str
    max_weight: float
    start_step: int
    ramp_steps: int = 400

    def weight_at(self, step: int) -> float:
        if step < self.start_step:
            return 0.0
        if self.ramp_steps <= 0:
            return self.max_weight
        progress = min(1.0, (step - self.start_step) / self.ramp_steps)
        # Linear ramp — simple, predictable, easy to reason about
        return self.max_weight * progress


# ---------------------------------------------------------------------------
# Schedule container
# ---------------------------------------------------------------------------

@dataclass
class LossSchedule:
    """
    Collection of LossEntry objects.  Call .weights(step) to get the
    current effective weight dict for all registered losses.

    LM loss is not registered here — it's always 1.0 and is handled
    separately in the model's forward().
    """
    entries: List[LossEntry] = field(default_factory=list)

    def weights(self, step: int) -> Dict[str, float]:
        """
        Returns {loss_name: effective_weight} for the given global step.
        Losses that haven't started yet have weight 0.0.
        """
        return {e.name: e.weight_at(step) for e in self.entries}

    def log_str(self, step: int) -> str:
        """Human-readable summary of active weights at this step."""
        parts = []
        for e in self.entries:
            w = e.weight_at(step)
            if w > 0:
                frac = min(1.0, (step - e.start_step) / max(1, e.ramp_steps))
                ramp_str = f"{frac*100:.0f}%" if frac < 1.0 else "full"
                parts.append(f"{e.name}={w:.3f}({ramp_str})")
            else:
                parts.append(f"{e.name}=0(pending@{e.start_step})")
        return "  ".join(parts) if parts else "(no auxiliary losses)"

    def next_event_step(self, step: int) -> Optional[int]:
        """Returns the next step at which a loss starts or finishes ramping."""
        candidates = []
        for e in self.entries:
            if e.start_step > step:
                candidates.append(e.start_step)
            end = e.start_step + e.ramp_steps
            if end > step:
                candidates.append(end)
        return min(candidates) if candidates else None

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg, schedule_preset: str = "default") -> "LossSchedule":
        """
        Build a LossSchedule from a JEPALoRAConfig.

        schedule_preset options
        ───────────────────────
        "default"   — the 6-phase curriculum described in the module docstring.
                      Good for most training runs of 2000+ steps.

        "fast"      — compressed schedule for short runs or debugging.
                      All losses active by step 500.

        "all_on"    — no curriculum; all losses at full weight from step 0.
                      Equivalent to the old behaviour. Useful for ablations.

        "lm_only"   — only LM loss (no auxiliary losses).
                      Useful as a baseline or when diagnosing issues.
        """
        if schedule_preset == "lm_only":
            return cls(entries=[])

        if schedule_preset == "all_on":
            return cls._all_on(cfg)

        if schedule_preset == "fast":
            return cls._fast(cfg)

        return cls._default(cfg)

    @classmethod
    def _all_on(cls, cfg) -> "LossSchedule":
        """All auxiliary losses at full weight from step 0."""
        entries = []
        _add = entries.append
        if cfg.use_pred_token:
            _add(LossEntry("jepa_pred_token_loss", cfg.jepa_pred_token_weight, 0, 0))
        if cfg.use_span_jepa:
            _add(LossEntry("span_jepa_loss",       cfg.span_jepa_weight,       0, 0))
        if cfg.use_offset_head:
            _add(LossEntry("jepa_offset_head_loss",cfg.jepa_offset_head_weight,0, 0))
            _add(LossEntry("jepa_pred_head_loss",  cfg.jepa_pred_head_weight,  0, 0))
        if cfg.use_layerwise_jepa:
            _add(LossEntry("layerwise_jepa_loss",  cfg.layerwise_jepa_weight,  0, 0))
        if cfg.contrastive_mode in ("infonce", "both"):
            _add(LossEntry("infonce_loss",         cfg.contrastive_weight,     0, 0))
        if cfg.contrastive_mode in ("vicreg", "both"):
            _add(LossEntry("vicreg_loss",          cfg.contrastive_weight,     0, 0))
        if cfg.use_self_consistency:
            _add(LossEntry("self_consistency_loss",cfg.self_consistency_weight,0, 0))
        return cls(entries=entries)

    @classmethod
    def _default(cls, cfg) -> "LossSchedule":
        """
        6-phase curriculum.  Step numbers are sensible defaults for a
        training run of ~3000 steps; scale them via --schedule_scale if
        your run is shorter or longer.

        Phase 1 (0–200):      LM only
        Phase 2 (200–600):    + [PRED] embed loss
        Phase 3 (600–1000):   + offset heads
        Phase 4 (1000–1500):  + span JEPA + layer-wise JEPA
        Phase 5 (1500–2000):  + InfoNCE / VICReg
        Phase 6 (2000–2500):  + self-consistency
        """
        entries = []
        _add = entries.append
        ramp = 400   # ramp duration for all losses

        if cfg.use_pred_token:
            _add(LossEntry("jepa_pred_token_loss", cfg.jepa_pred_token_weight,
                           start_step=200,  ramp_steps=ramp))
        if cfg.use_offset_head:
            _add(LossEntry("jepa_offset_head_loss",cfg.jepa_offset_head_weight,
                           start_step=600,  ramp_steps=ramp))
            _add(LossEntry("jepa_pred_head_loss",  cfg.jepa_pred_head_weight,
                           start_step=600,  ramp_steps=ramp))
        if cfg.use_span_jepa:
            _add(LossEntry("span_jepa_loss",       cfg.span_jepa_weight,
                           start_step=1000, ramp_steps=ramp))
        if cfg.use_layerwise_jepa:
            _add(LossEntry("layerwise_jepa_loss",  cfg.layerwise_jepa_weight,
                           start_step=1000, ramp_steps=ramp))
        if cfg.contrastive_mode in ("infonce", "both"):
            _add(LossEntry("infonce_loss",         cfg.contrastive_weight,
                           start_step=1500, ramp_steps=ramp))
        if cfg.contrastive_mode in ("vicreg", "both"):
            _add(LossEntry("vicreg_loss",          cfg.contrastive_weight,
                           start_step=1500, ramp_steps=ramp))
        if cfg.use_self_consistency:
            _add(LossEntry("self_consistency_loss",cfg.self_consistency_weight,
                           start_step=2000, ramp_steps=ramp))
        return cls(entries=entries)

    @classmethod
    def _fast(cls, cfg) -> "LossSchedule":
        """Compressed schedule — all losses active by step 500."""
        entries = []
        _add = entries.append
        ramp = 100

        if cfg.use_pred_token:
            _add(LossEntry("jepa_pred_token_loss", cfg.jepa_pred_token_weight,
                           start_step=50,  ramp_steps=ramp))
        if cfg.use_offset_head:
            _add(LossEntry("jepa_offset_head_loss",cfg.jepa_offset_head_weight,
                           start_step=100, ramp_steps=ramp))
            _add(LossEntry("jepa_pred_head_loss",  cfg.jepa_pred_head_weight,
                           start_step=100, ramp_steps=ramp))
        if cfg.use_span_jepa:
            _add(LossEntry("span_jepa_loss",       cfg.span_jepa_weight,
                           start_step=200, ramp_steps=ramp))
        if cfg.use_layerwise_jepa:
            _add(LossEntry("layerwise_jepa_loss",  cfg.layerwise_jepa_weight,
                           start_step=200, ramp_steps=ramp))
        if cfg.contrastive_mode in ("infonce", "both"):
            _add(LossEntry("infonce_loss",         cfg.contrastive_weight,
                           start_step=300, ramp_steps=ramp))
        if cfg.contrastive_mode in ("vicreg", "both"):
            _add(LossEntry("vicreg_loss",          cfg.contrastive_weight,
                           start_step=300, ramp_steps=ramp))
        if cfg.use_self_consistency:
            _add(LossEntry("self_consistency_loss",cfg.self_consistency_weight,
                           start_step=400, ramp_steps=ramp))
        return cls(entries=entries)

    def scale_steps(self, factor: float) -> "LossSchedule":
        """
        Return a new schedule with all start_step and ramp_steps multiplied
        by factor.  Useful for adapting the default schedule to longer/shorter
        runs without respecifying everything.

        Example: schedule.scale_steps(2.0) doubles all step thresholds.
        """
        return LossSchedule(entries=[
            LossEntry(
                name=e.name,
                max_weight=e.max_weight,
                start_step=int(e.start_step * factor),
                ramp_steps=int(e.ramp_steps * factor),
            )
            for e in self.entries
        ])
