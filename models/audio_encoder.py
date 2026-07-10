"""
MMVP2 Audio Encoder
===================

Lightweight 1D temporal encoder that turns the video's log-mel spectrogram
features into a per-timestep "audio context" vector:

    [B, T, n_mels=64]   --conv1d-->  [B, T, d_audio]

The encoder is purposely tiny (~5K params) because:
  * audio is an auxiliary modality - the trajectory + saliency stack already
    contributes most of the signal,
  * we use it only to **gate** trajectory-vs-content predictions, not to
    predict positions directly,
  * a tiny encoder reduces over-fitting on a 14-video dataset.

Design choices follow the SalViT360-AV paper's audio branch (downmixed
mono log-mel spectrogram -> small temporal CNN), but at much lower
dimensionality because we only need a per-frame context vector for gating.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioEncoder(nn.Module):
    """
    Per-timestep audio context encoder.

    Forward:
        log_mel : [B, T, n_mels]   per-frame log-mel features (5 fps)
    Returns:
        ctx     : [B, T, d_audio]  per-frame audio context
    """

    def __init__(
        self,
        n_mels: int = 64,
        d_audio: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd for symmetric padding")
        pad = kernel_size // 2

        self.proj_in = nn.Linear(n_mels, d_audio)
        self.conv1   = nn.Conv1d(d_audio, d_audio, kernel_size=kernel_size, padding=pad)
        self.conv2   = nn.Conv1d(d_audio, d_audio, kernel_size=kernel_size, padding=pad)
        self.norm1   = nn.LayerNorm(d_audio)
        self.norm2   = nn.LayerNorm(d_audio)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.n_mels  = n_mels
        self.d_audio = d_audio

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_mel: [B, T, n_mels]
        Returns:
            ctx:     [B, T, d_audio]
        """
        x = self.proj_in(log_mel)                  # [B, T, D]
        x = self.norm1(x)

        # Conv1d expects [B, D, T]
        h = x.transpose(1, 2)                      # [B, D, T]
        h = F.gelu(self.conv1(h))
        h = F.gelu(self.conv2(h))
        h = h.transpose(1, 2)                      # [B, T, D]
        h = self.norm2(h + x)                      # residual + LN
        h = self.dropout(h)
        return h

    def extra_repr(self) -> str:
        return f"n_mels={self.n_mels}, d_audio={self.d_audio}"
