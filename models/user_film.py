"""
MMVP2 User Personalization
==========================

Two pieces:

1. UserEmbedding
       Plain `nn.Embedding(num_users, d_user)`; the index 0 is reserved as
       the "unknown user" entry in case a held-out user ID is queried.

2. UserFiLM
       Generates per-user (gamma, beta) feature-wise modulation parameters
       from the user embedding and applies them to a feature tensor:
           x' = gamma * x + beta
       This is FiLM (Perez et al., 2018), a standard, low-cost way of
       conditioning shared features on a user embedding without adding
       per-user duplicated parameters.

Both modules are tiny:
    UserEmbedding (25 users x d=24)            -> ~600 params
    UserFiLM      (24 -> 2 * d_target=256)     -> ~12K params
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  User embedding lookup
# --------------------------------------------------------------------------- #

class UserEmbedding(nn.Module):
    """
    Embedding(num_users, d_user) with mean-zero initialization (so the
    untrained or empty case gracefully degrades to "no personalization").

    Index 0 is reserved as the unknown-user / fallback entry.
    """
    def __init__(self, num_users: int, d_user: int = 24, init_std: float = 0.02):
        super().__init__()
        # +1 slot for the "unknown user" pad entry at index 0.
        self.embed = nn.Embedding(num_users + 1, d_user, padding_idx=0)
        nn.init.normal_(self.embed.weight, mean=0.0, std=init_std)
        with torch.no_grad():
            self.embed.weight[0].zero_()              # zero pad row

        self.num_users = num_users
        self.d_user    = d_user

    def forward(self, user_id: torch.Tensor) -> torch.Tensor:
        """user_id [B] (long) -> [B, d_user]"""
        return self.embed(user_id)


# --------------------------------------------------------------------------- #
#  Feature-wise Linear Modulation (FiLM) from user embedding
# --------------------------------------------------------------------------- #

class UserFiLM(nn.Module):#Feature-wise Linear Modulation.
    """
    Modulate a [B, *, d_target] feature tensor with per-user gamma, beta:

        x' = (1 + gamma) * x + beta

    The "1 +" form means the default (untrained) mapping is the identity,
    which keeps training stable when the embedding is small.

    d_user: size of the user embedding. In MMVP2, usually 24.
d_target: size of the feature vector to modify. Often 256.
init_scale: how small the initial weights are.
    """
    def __init__(self, d_user: int, d_target: int, init_scale: float = 0.02):
        super().__init__()
        self.fc = nn.Linear(d_user, 2 * d_target)
        nn.init.normal_(self.fc.weight, mean=0.0, std=init_scale)
        nn.init.zeros_(self.fc.bias)
        self.d_user   = d_user
        self.d_target = d_target

    def forward(self, x: torch.Tensor, user_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : [B, ..., d_target]
            user_emb : [B, d_user]
        Returns:
            x'       : same shape as x, FiLM-modulated.
        """
        gb = self.fc(user_emb)                              # [B, 2*d_target]
        gamma, beta = gb.chunk(2, dim=-1)                   # each [B, d_target]
        # broadcast across the middle dimensions (e.g. time, tokens, ...)
        # x has shape [B, ..., d_target]; we add singleton dims to gamma/beta
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(1)
            beta  = beta.unsqueeze(1)
        return (1.0 + gamma) * x + beta

    def extra_repr(self) -> str:
        return f"d_user={self.d_user}, d_target={self.d_target}"
