"""Loss functions for LLM-JEPA training.

Combined loss:  L = L_LLM  +  λ × d(Pred(Enc(Text)), Enc(Code))
where d is the cosine distance  1 - cos_sim.
"""

import torch
import torch.nn.functional as F


def cosine_distance(pred_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
    """Cosine distance averaged over the batch.

    Args:
        pred_emb:   (B, D) — Pred(Enc(Text))  predicted embedding
        target_emb: (B, D) — Enc(Code)         target embedding

    Returns:
        Scalar tensor — mean(1 − cos_sim) over the batch.
    """
    cos_sim = F.cosine_similarity(pred_emb, target_emb, dim=-1)  # (B,)
    return (1.0 - cos_sim).mean()


def info_nce_loss(
    query_emb: torch.Tensor,
    key_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE contrastive loss.

    Args:
        query_emb: (B, D) Normalized query embeddings (Predictor output)
        key_emb:   (B, D) Normalized key embeddings   (Target encoder output)
        temperature: Scalar temperature parameter

    Returns:
        Scalar InfoNCE loss.
    """
    # Cosine similarity matrix: (B, B)
    # logits[i, j] = sim(query[i], key[j]) / temp
    logits = torch.matmul(query_emb, key_emb.t()) / temperature
    
    # Labels are [0, 1, ..., B-1] (diagonal elements are positives)
    labels = torch.arange(logits.size(0), device=logits.device)
    
    return F.cross_entropy(logits, labels)


def compute_combined_loss(
    ntp_loss: torch.Tensor,
    jepa_loss_fwd: torch.Tensor,
    jepa_loss_bwd: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """L_total = L_LLM + α * L(Text->Code) + β * L(Code->Text)."""
    return ntp_loss + alpha * jepa_loss_fwd + beta * jepa_loss_bwd
