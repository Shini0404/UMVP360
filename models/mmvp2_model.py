"""
=============================================================================
MMVP2 — Personalized Multi-Modal Viewport Prediction (built on STAR-VP)
=============================================================================

MMVP2 is a strict extension of STAR-VP.  We do NOT change the backbone
architecture (LSTM + spatial attention + temporal attention + gating).  We
only:
  1. Replace top-K saliency (D_P=128) with a DENSE 16x32 token grid (N=512)
     that directly carries SalViT360 saliency.  Optionally, we add a second
     channel per token = behavioural prior (cross-user view-frequency).
  2. Enrich the LSTM input from 3D head trajectory to 11D
     [head(3) + eye(3) + offset(3) + is_fix(1) + dwell(1)].
  3. Initialise the LSTM hidden state with a per-user embedding.
  4. Feed an audio context vector (log-mel CNN) and a face engagement scalar
     into the gating module so it can shift the trajectory-vs-content blend
     based on what is happening in the video / how engaged the user is.
  5. Apply FiLM modulation (gamma, beta) on the spatial-attention viewport
     output `p_s_out` using the user embedding (per-user re-scaling).

Every new component is OFF by default (dim=0 / disabled) so the same class
can also reproduce the STAR-VP baseline -- this is what we use for the
ablation rows.

Inputs (forward kwargs):
    past_positions     [B, T_M, head_dim+aux_dim]   trajectory features
    sal_xyz            [B, T_M+T_H, D_P, d_in_s]    dense saliency tokens
    user_id            [B] (long)                    per-sample user index (1..num_users)
    audio              [B, T_audio, n_mels]          log-mel sequence (over past+future)
    face               [B, T, d_face_in]             per-frame face features
Returns:
    p_hat              [B, T_H, head_dim]            head positions (xyz, unit sphere)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lstm_module import LSTMModule
from .spatial_attention import SpatialAttentionModule
from .temporal_attention import TemporalAttentionModule
from .gating_fusion import GatingFusionModule
from .audio_encoder import AudioEncoder
from .user_film import UserEmbedding, UserFiLM


# ========================================================================= #
#  Output container
# ========================================================================= #

@dataclass
class MMVP2Output:
    p_hat: torch.Tensor
    p_prime: torch.Tensor | None = None
    p_double_prime: torch.Tensor | None = None
    p_s_out: torch.Tensor | None = None
    s_s_out: torch.Tensor | None = None
    w_prime: torch.Tensor | None = None
    w_double_prime: torch.Tensor | None = None


# ========================================================================= #
#  MMVP2 model
# ========================================================================= #

class MMVP2(nn.Module):
    """
    Personalized Multi-Modal Viewport Prediction model.

    All STAR-VP hyperparameters are preserved as defaults; the new components
    (audio, face, user, behavioural prior) can each be toggled individually.
    """

    def __init__(
        self,
        # ---- Trajectory inputs ----
        head_dim: int = 3,
        aux_dim:  int = 0,                # eye(3) + offset(3) + is_fix(1) + dwell(1) = 8
        # ---- Saliency tokens ----
        d_in_s: int = 4,                  # 4 = STAR-VP top-K   |  5 = dense + behav
        d_in_p: int = 4,                  # viewport (xyz + te)
        # ---- LSTM (Table 1) ----
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        # ---- Spatial Attention (Table 1) ----
        d_c: int = 256,
        n_heads: int = 8,
        spatial_enc_layers: int = 2,
        spatial_dec_layers: int = 2,
        # ---- Temporal Attention (Table 1) ----
        d_pe: int = 129,
        d_te: int = 127,
        temporal_enc_layers: int = 2,
        temporal_dec_layers: int = 2,
        # ---- Gating Fusion (Table 1) ----
        d_g: int = 128,
        # ---- MMVP2 personalization / multi-modal ----
        num_users: int = 0,               # 0 = disable user embedding
        d_user:    int = 0,
        use_user_film: bool = False,
        n_mels:    int = 64,
        d_audio:   int = 0,               # 0 = disable audio gate
        audio_kernel_size: int = 5,
        d_face:    int = 0,               # 0 = disable face gate
        face_input_dim: int = 5,           # 5 = MUSE-VP's face features
        # ---- Global ----
        t_m: int = 25,
        t_h: int = 25,
        dropout: float = 0.0,
        normalize_output: bool = True,
    ):
        super().__init__()

        self.t_m = t_m
        self.t_h = t_h
        self.head_dim = head_dim
        self.aux_dim  = aux_dim
        self.d_c = d_c
        self.normalize_output = normalize_output

        self.num_users = num_users
        self.d_user    = d_user
        self.use_user_film = use_user_film and (num_users > 0 and d_user > 0)
        self.d_audio   = d_audio
        self.d_face    = d_face

        lstm_input_dim = head_dim + aux_dim

        # ---- Stage 1: LSTM (richer input + optional user-init) ----
        self.lstm = LSTMModule(
            input_dim=lstm_input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            t_m=t_m,
            t_h=t_h,
            dropout=dropout,
            head_dim=head_dim,
            d_user=d_user if num_users > 0 else 0,
        )

        # ---- Stage 2: Spatial Attention (dense or top-K saliency) ----
        self.spatial_attn = SpatialAttentionModule(
            d_c=d_c,
            n_heads=n_heads,
            n_enc_layers=spatial_enc_layers,
            n_dec_layers=spatial_dec_layers,
            dropout=dropout,
            d_in_s=d_in_s,
            d_in_p=d_in_p,
        )

        # ---- Stage 3: Temporal Attention ----
        self.temporal_attn = TemporalAttentionModule(
            d_c=d_c,
            d_pe=d_pe,
            d_te=d_te,
            n_heads=n_heads,
            n_enc_layers=temporal_enc_layers,
            n_dec_layers=temporal_dec_layers,
            t_m=t_m,
            t_h=t_h,
            output_dim=head_dim,
            dropout=dropout,
        )

        # ---- Stage 4: Gating Fusion (audio + face context optional) ----
        self.gating = GatingFusionModule(
            t_h=t_h,
            d_g=d_g,
            input_dim=head_dim,
            d_audio=d_audio,
            d_face=d_face,
        )

        # ---- Personalization ----
        if num_users > 0 and d_user > 0:
            self.user_emb = UserEmbedding(num_users=num_users, d_user=d_user)
        else:
            self.user_emb = None
        if self.use_user_film:
            self.user_film = UserFiLM(d_user=d_user, d_target=d_c)
        else:
            self.user_film = None

        # ---- Audio encoder ----
        if d_audio > 0:
            self.audio_enc = AudioEncoder(
                n_mels=n_mels,
                d_audio=d_audio,
                kernel_size=audio_kernel_size,
                dropout=dropout,
            )
        else:
            self.audio_enc = None

        # ---- Face engagement scalar(s) ----
        # Mean-pool face features over past timesteps then compress with
        # a small linear -> d_face.  When d_face == 0 the gating module
        # ignores it.
        if d_face > 0:
            self.face_proj = nn.Linear(face_input_dim, d_face)
        else:
            self.face_proj = None

    # --------------------------------------------------------------------- #
    #  Forward
    # --------------------------------------------------------------------- #

    def forward(
        self,
        past_positions: torch.Tensor,                # [B, T_M, head_dim + aux_dim]
        sal_xyz:        torch.Tensor,                # [B, T, D_P, d_in_s]
        user_id:        torch.Tensor | None = None,  # [B]
        audio:          torch.Tensor | None = None,  # [B, T, n_mels]
        face:           torch.Tensor | None = None,  # [B, T, face_in]
        return_intermediates: bool = False,
    ) -> torch.Tensor | MMVP2Output:
        B = past_positions.shape[0]

        # Past head-only view for spatial attention's viewport stream
        past_head = past_positions[:, :, : self.head_dim]                 # [B, T_M, head_dim]

        # ---- User embedding ----
        if self.user_emb is not None and user_id is not None:
            u = self.user_emb(user_id)                                    # [B, d_user]
        else:
            u = None

        # ---- Stage 1: LSTM ----
        p_prime = self.lstm(past_positions, user_emb=u)                   # [B, T_H, head_dim]

        # ---- Stage 2: Spatial Attention ----
        p_s_out, s_s_out = self.spatial_attn(
            sal_xyz, past_head, p_prime,
        )                                                                  # both [B, T, D_C]

        # FiLM personalization on the viewport stream
        if self.user_film is not None and u is not None:
            p_s_out = self.user_film(p_s_out, u)

        # ---- Stage 3: Temporal Attention ----
        p_double_prime = self.temporal_attn(p_s_out, s_s_out)              # [B, T_H, head_dim]

        # ---- Optional audio / face context for the gate ----
        audio_ctx = None
        face_ctx  = None
        if self.audio_enc is not None:
            if audio is None:
                raise ValueError("audio is required (d_audio > 0)")
            audio_seq = self.audio_enc(audio)                              # [B, T, d_audio]
            # Pool over the *past* part only (no future leakage):
            T_total = audio_seq.shape[1]
            T_past = max(1, T_total - self.t_h)
            audio_ctx = audio_seq[:, :T_past].mean(dim=1)                  # [B, d_audio]
        if self.face_proj is not None:
            if face is None:
                raise ValueError("face is required (d_face > 0)")
            T_total = face.shape[1]
            T_past = max(1, T_total - self.t_h)
            face_pool = face[:, :T_past].mean(dim=1)                       # [B, face_in]
            face_ctx = self.face_proj(face_pool)                           # [B, d_face]

        # ---- Stage 4: Gating Fusion ----
        p_hat, w_prime, w_double_prime = self.gating(
            p_prime, p_double_prime,
            audio_ctx=audio_ctx, face_ctx=face_ctx,
            return_weights=True,
        )                                                                  # [B, T_H, head_dim]

        if self.normalize_output:
            p_hat = F.normalize(p_hat, p=2, dim=-1)

        if return_intermediates:
            return MMVP2Output(
                p_hat=p_hat,
                p_prime=p_prime,
                p_double_prime=p_double_prime,
                p_s_out=p_s_out,
                s_s_out=s_s_out,
                w_prime=w_prime,
                w_double_prime=w_double_prime,
            )
        return p_hat

    # --------------------------------------------------------------------- #
    #  Summaries
    # --------------------------------------------------------------------- #

    def count_parameters(self) -> dict[str, int]:
        out = {
            "lstm":          sum(p.numel() for p in self.lstm.parameters()),
            "spatial_attn":  sum(p.numel() for p in self.spatial_attn.parameters()),
            "temporal_attn": sum(p.numel() for p in self.temporal_attn.parameters()),
            "gating":        sum(p.numel() for p in self.gating.parameters()),
        }
        if self.user_emb is not None:
            out["user_emb"]  = sum(p.numel() for p in self.user_emb.parameters())
        if self.user_film is not None:
            out["user_film"] = sum(p.numel() for p in self.user_film.parameters())
        if self.audio_enc is not None:
            out["audio_enc"] = sum(p.numel() for p in self.audio_enc.parameters())
        if self.face_proj is not None:
            out["face_proj"] = sum(p.numel() for p in self.face_proj.parameters())
        out["total"] = sum(p.numel() for p in self.parameters())
        return out

    def extra_repr(self) -> str:
        return (
            f"t_m={self.t_m}, t_h={self.t_h}, head_dim={self.head_dim}, "
            f"aux_dim={self.aux_dim}, d_c={self.d_c}, "
            f"users={self.num_users}, d_user={self.d_user}, "
            f"d_audio={self.d_audio}, d_face={self.d_face}, "
            f"film={self.user_film is not None}"
        )
