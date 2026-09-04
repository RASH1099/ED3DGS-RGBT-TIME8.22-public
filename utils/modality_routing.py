import torch
import torch.nn.functional as F


def compute_modality_routing(logits, tau, hard=True, independent_samples=False,
                             stochastic=True):
    """
    Args:
        logits: [N, 3] modality identity logits (shared, rgb-only, thermal-only).
        tau: Gumbel-Softmax temperature.
        hard: Whether to return a straight-through hard one-hot for forward.
    Returns:
        s_hard: [N, 3] one-hot (straight-through if hard=True).
        s_soft: [N, 3] soft probabilities (for loss/stat).
    """
    if tau <= 0:
        raise ValueError("tau must be positive")

    if not stochastic:
        s_soft = F.softmax(logits / tau, dim=-1)
        if hard:
            index = s_soft.max(dim=-1, keepdim=True).indices
            one_hot = torch.zeros_like(s_soft).scatter_(-1, index, 1.0)
            s_hard = one_hot - s_soft.detach() + s_soft
        else:
            s_hard = s_soft
        return s_hard, s_soft

    if independent_samples:
        s_soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        if hard:
            s_hard = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        else:
            s_hard = s_soft
        return s_hard, s_soft

    # Draw one Gumbel sample so the hard route used by rendering and the
    # soft route used by regularizers describe the same stochastic choice.
    gumbels = -torch.empty_like(logits).exponential_().log()
    s_soft = F.softmax((logits + gumbels) / tau, dim=-1)
    if hard:
        index = s_soft.max(dim=-1, keepdim=True).indices
        one_hot = torch.zeros_like(s_soft).scatter_(-1, index, 1.0)
        s_hard = one_hot - s_soft.detach() + s_soft
    else:
        s_hard = s_soft
    return s_hard, s_soft
