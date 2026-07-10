"""
=============================================================================
STAR-VP Full Model — Complete Pipeline
=============================================================================

Paper:  "STAR-VP: Improving Long-term Viewport Prediction in 360° Videos
         via Space-aligned and Time-varying Fusion"
        MM '24 (ACM Multimedia 2024)

This module wires together all five sub-modules into the complete STAR-VP
architecture.  The data flow follows Sections 3.2–3.6 of the paper exactly:

    ┌──────────────────────────────────────────────────────────────────┐
    │                        STAR-VP Pipeline                         │
    │                                                                  │
    │   past_positions ──┬──→ [LSTM Module] ──→ P̂'  ──────────────┐  │
    │   [B, 15, 3]       │    (Section 3.3)    [B,25,3]            │  │
    │                    │                                          │  │
    │                    └──→ [Spatial Attention] ──→ P_{s-out}    │  │
    │   sal_xyz ─────────────→ (Section 3.4)        [B,40,256]    │  │
    │   [B, 40, 128, 4]      Uses past_pos + P̂'     S_{s-out}    │  │
    │                                                [B,40,256]    │  │
    │                                                   │          │  │
    │                         [Temporal Attention] ←────┘          │  │
    │                          (Section 3.5)                       │  │
    │                          P̂'' [B,25,3] ─────────────────┐    │  │
    │                                                         │    │  │
    │                         [Gating Fusion]  ←──── P̂' ─────┘    │  │
    │                          (Section 3.6)                       │  │
    │                          P̂ [B,25,3] ← output                │  │
    └──────────────────────────────────────────────────────────────────┘

Inputs (expected to be pre-processed):
    past_positions:  [B, T_M, 3]         Head positions on unit sphere
    sal_xyz:         [B, T_M+T_H, D_P, 4]  Pre-processed saliency from
                                          SalMapProcessor (offline step)

Output:
    p_hat:  [B, T_H, 3]   Final predicted future head positions

All hyperparameters default to the paper's Table 1 values.
=============================================================================
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


# ========================================================================= #
#  Output container
# ========================================================================= #

@dataclass
class STARVPOutput:
    """
    Structured output from the STAR-VP model.

    Always populated:
        p_hat            [B, T_H, 3]  Final blended prediction.

    Populated when return_intermediates=True:
        p_prime          [B, T_H, 3]  LSTM short-term prediction.
        p_double_prime   [B, T_H, 3]  Temporal Attention long-term prediction.
        p_s_out          [B, T, D_C]  Spatially-aligned viewport features.
        s_s_out          [B, T, D_C]  Aggregated saliency features.
        w_prime          [B, T_H]     Gating weight for LSTM.
        w_double_prime   [B, T_H]     Gating weight for Temporal.
    """
    p_hat: torch.Tensor
    p_prime: torch.Tensor | None = None
    p_double_prime: torch.Tensor | None = None
    p_s_out: torch.Tensor | None = None
    s_s_out: torch.Tensor | None = None
    w_prime: torch.Tensor | None = None
    w_double_prime: torch.Tensor | None = None


# ========================================================================= #
#  STAR-VP Model  (Sections 3.2–3.6)
# ========================================================================= #

class STARVP(nn.Module):
    """
    Complete STAR-VP model for 360° video viewport prediction.

    Sub-modules (each faithfully reproducing the paper):
        1. LSTMModule           — autoregressive short-term prediction (Sec 3.3)
        2. SpatialAttentionModule  — per-timestep HT-VC spatial alignment (Sec 3.4)
        3. TemporalAttentionModule — time-varying fusion over 80 tokens (Sec 3.5)
        4. GatingFusionModule      — convex blend of short/long-term (Sec 3.6)

    SalMapProcessor (Sec 3.2) is NOT included here because:
        - It has zero learnable parameters.
        - Saliency processing is a slow per-frame operation best done offline.
        - The model expects pre-processed sal_xyz as input.
    """

    def __init__(
        self,
        # --- LSTM (Table 1) ---
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        # --- Spatial Attention (Table 1) ---
        d_c: int = 256,
        n_heads: int = 8,
        spatial_enc_layers: int = 2,
        spatial_dec_layers: int = 2,
        # --- Temporal Attention (Table 1) ---
        d_pe: int = 129,
        d_te: int = 127,
        temporal_enc_layers: int = 2,
        temporal_dec_layers: int = 2,
        # --- Gating Fusion (Table 1) ---
        d_g: int = 128,
        # --- Global ---
        t_m: int = 15,
        t_h: int = 25,
        input_dim: int = 3,
        dropout: float = 0.0,
        normalize_output: bool = True,
    ):
        """
        All defaults are the exact values from the paper's Table 1.

        Args:
            lstm_hidden_dim:     D_cL = 256.
            lstm_num_layers:     N_layers_L = 2.
            d_c:                 D_cT = 256 (channel dim in attention modules).
            n_heads:             N_heads = 8.
            spatial_enc_layers:  Spatial encoder PerceiverBlocks.
            spatial_dec_layers:  Spatial decoder PerceiverBlocks.
            d_pe:                D_PE = 129 (Fourier PE dim).
            d_te:                D_TE = 127 (modality embedding dim).
            temporal_enc_layers: Temporal encoder PerceiverBlocks.
            temporal_dec_layers: Temporal decoder PerceiverBlocks.
            d_g:                 D_G = 128 (gating MLP hidden).
            t_m:                 T_M = 15 (past timesteps).
            t_h:                 T_H = 25 (future timesteps).
            input_dim:           3 for unit-sphere xyz.
            dropout:             Dropout rate (paper doesn't specify, default 0).
            normalize_output:    If True, L2-normalize the final output to the
                                 unit sphere.  Required for Orthodromic Distance.
        """
        super().__init__()
        self.t_m = t_m
        self.t_h = t_h
        self.d_c = d_c
        self.normalize_output = normalize_output

        # ---- Stage 1: LSTM — short-term trajectory prediction (Sec 3.3) ----
        self.lstm = LSTMModule(
            input_dim=input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            t_m=t_m,
            t_h=t_h,
            dropout=dropout,
        )

        # ---- Stage 2: Spatial Attention — per-timestep alignment (Sec 3.4) ----
        self.spatial_attn = SpatialAttentionModule(
            d_in=4,          # (x,y,z,s) or (x,y,z,te)
            d_c=d_c,
            n_heads=n_heads,
            n_enc_layers=spatial_enc_layers,
            n_dec_layers=spatial_dec_layers,
            dropout=dropout,
        )

        # ---- Stage 3: Temporal Attention — time-varying fusion (Sec 3.5) ----
        self.temporal_attn = TemporalAttentionModule(
            d_c=d_c,
            d_pe=d_pe,
            d_te=d_te,
            n_heads=n_heads,
            n_enc_layers=temporal_enc_layers,
            n_dec_layers=temporal_dec_layers,
            t_m=t_m,
            t_h=t_h,
            output_dim=input_dim,
            dropout=dropout,
        )

        # ---- Stage 4: Gating Fusion — short/long-term blend (Sec 3.6) ----
        self.gating = GatingFusionModule(
            t_h=t_h,
            d_g=d_g,
            input_dim=input_dim,
        )

    # --------------------------------------------------------------------- #
    #  Forward
    # --------------------------------------------------------------------- #

    def forward(
        self,
        past_positions: torch.Tensor,
        sal_xyz: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | STARVPOutput:
        """
        Full STAR-VP forward pass.

        Args:
            past_positions:       [B, T_M, 3]          Past head positions.
            sal_xyz:              [B, T_M+T_H, D_P, 4] Pre-processed saliency.
            return_intermediates: If True, return STARVPOutput with all
                                  intermediate tensors.  Otherwise return
                                  only p_hat [B, T_H, 3].

        Returns:
            p_hat [B, T_H, 3] or STARVPOutput.
        """
        # ============================================================= #
        #  Stage 1: LSTM — autoregressive short-term prediction
        # ============================================================= #
        p_prime = self.lstm(past_positions)                    # [B, T_H, 3]

        # ============================================================= #
        #  Stage 2: Spatial Attention — per-timestep HT-VC alignment
        # ============================================================= #
        p_s_out, s_s_out = self.spatial_attn(
            sal_xyz, past_positions, p_prime,
        )                                                      # both [B, T, D_C]

        # ============================================================= #
        #  Stage 3: Temporal Attention — time-varying fusion
        # ============================================================= #
        p_double_prime = self.temporal_attn(p_s_out, s_s_out)  # [B, T_H, 3]

        # ============================================================= #
        #  Stage 4: Gating Fusion — blend short & long-term
        # ============================================================= #
        p_hat, w_prime, w_double_prime = self.gating(
            p_prime, p_double_prime, return_weights=True,
        )                                                      # [B, T_H, 3]

        # ============================================================= #
        #  Optional: normalize to unit sphere
        # ============================================================= #
        if self.normalize_output:
            p_hat = F.normalize(p_hat, p=2, dim=-1)

        # ============================================================= #
        #  Return
        # ============================================================= #
        if return_intermediates:
            return STARVPOutput(
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
    #  Convenience
    # --------------------------------------------------------------------- #

    def count_parameters(self) -> dict[str, int]:
        """Count trainable parameters per sub-module and total."""
        counts = {}
        for name, module in [
            ("lstm", self.lstm),
            ("spatial_attn", self.spatial_attn),
            ("temporal_attn", self.temporal_attn),
            ("gating", self.gating),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    def extra_repr(self) -> str:
        return (
            f"t_m={self.t_m}, t_h={self.t_h}, d_c={self.d_c}, "
            f"normalize_output={self.normalize_output}"
        )


# ========================================================================= #
#  Self-Test — exhaustive verification
# ========================================================================= #

def _self_test() -> bool:
    """
    Exhaustive verification of the complete STAR-VP model.

    Tests:
      A — Sub-module instantiation
        1.  LSTM module exists and has correct type
        2.  Spatial Attention module exists and has correct type
        3.  Temporal Attention module exists and has correct type
        4.  Gating Fusion module exists and has correct type

      B — Default hyperparameters match Table 1
        5.  T_M = 15, T_H = 25
        6.  LSTM: hidden=256, layers=2, input=3
        7.  Spatial: d_c=256, n_heads=8
        8.  Temporal: d_c=256, d_pe=129, d_te=127
        9.  Gating: d_g=128, t_h=25

      C — End-to-end forward pass shapes
       10.  Output shape [B, T_H, 3] = [4, 25, 3]
       11.  Output is on unit sphere (normalize_output=True)
       12.  Output NOT on unit sphere when normalize_output=False

      D — Intermediate outputs (return_intermediates=True)
       13.  Returns STARVPOutput dataclass
       14.  p_hat shape [B, 25, 3]
       15.  p_prime shape [B, 25, 3]
       16.  p_double_prime shape [B, 25, 3]
       17.  p_s_out shape [B, 40, 256]
       18.  s_s_out shape [B, 40, 256]
       19.  w_prime shape [B, 25]
       20.  w_double_prime shape [B, 25]
       21.  w_prime + w_double_prime = 1

      E — Gradient flow (end-to-end)
       22.  All model parameters receive gradients
       23.  Gradient flows to past_positions input
       24.  Gradient flows to sal_xyz input
       25.  LSTM parameters receive gradients
       26.  Spatial Attention parameters receive gradients
       27.  Temporal Attention parameters receive gradients
       28.  Gating parameters receive gradients

      F — Parameter counts
       29.  Per-module counts are positive
       30.  Total = sum of sub-modules
       31.  Total parameter count verification

      G — Robustness
       32.  No NaN/Inf in output
       33.  Deterministic in eval mode
       34.  Batch size 1 works
       35.  Batch size 8 works

      H — Integration
       36.  Output usable for OD loss (unit vectors, dot product in [-1,1])
       37.  Intermediates usable for auxiliary losses
    """
    torch.manual_seed(42)

    T_M = 15
    T_H = 25
    T = T_M + T_H
    D_C = 256
    D_P = 128
    B = 4

    all_passed = True
    test_num = 0

    def check(condition: bool, name: str) -> None:
        nonlocal all_passed, test_num
        test_num += 1
        status = "OK" if condition else "FAIL"
        print(f"  [{test_num:>2}] {name}: {status}")
        if not condition:
            all_passed = False

    print("=" * 72)
    print("STAR-VP Full Model Self-Test")
    print("=" * 72)

    model = STARVP(normalize_output=True)
    model.eval()

    past_pos = torch.randn(B, T_M, 3)
    sal_xyz = torch.randn(B, T, D_P, 4)

    # =================================================================== #
    #  Test Group A: Sub-module instantiation
    # =================================================================== #
    print("\n--- A. Sub-module instantiation ---")

    check(isinstance(model.lstm, LSTMModule), "LSTM module type")
    check(isinstance(model.spatial_attn, SpatialAttentionModule), "Spatial Attn type")
    check(isinstance(model.temporal_attn, TemporalAttentionModule), "Temporal Attn type")
    check(isinstance(model.gating, GatingFusionModule), "Gating Fusion type")

    # =================================================================== #
    #  Test Group B: Default hyperparameters
    # =================================================================== #
    print("\n--- B. Default hyperparameters (Table 1) ---")

    check(model.t_m == 15 and model.t_h == 25, f"T_M={model.t_m}, T_H={model.t_h}")
    check(
        model.lstm.hidden_dim == 256
        and model.lstm.num_layers == 2
        and model.lstm.input_dim == 3,
        f"LSTM: hidden={model.lstm.hidden_dim}, layers={model.lstm.num_layers}",
    )
    check(
        model.spatial_attn.d_c == 256 and model.spatial_attn.n_heads == 8,
        f"Spatial: d_c={model.spatial_attn.d_c}, heads={model.spatial_attn.n_heads}",
    )
    check(
        model.temporal_attn.d_c == 256
        and model.temporal_attn.d_pe == 129
        and model.temporal_attn.d_te == 127,
        f"Temporal: d_c={model.temporal_attn.d_c}, d_pe={model.temporal_attn.d_pe}, d_te={model.temporal_attn.d_te}",
    )
    check(
        model.gating.d_g == 128 and model.gating.t_h == 25,
        f"Gating: d_g={model.gating.d_g}, t_h={model.gating.t_h}",
    )

    # =================================================================== #
    #  Test Group C: End-to-end forward pass shapes
    # =================================================================== #
    print("\n--- C. End-to-end forward pass ---")

    with torch.no_grad():
        out = model(past_pos, sal_xyz)

    check(
        tuple(out.shape) == (B, T_H, 3),
        f"Output shape: {tuple(out.shape)} == ({B}, {T_H}, 3)",
    )

    # Check unit sphere (normalize_output=True)
    norms = out.norm(dim=-1)
    check(
        torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
        f"Unit sphere: norms in [{norms.min():.6f}, {norms.max():.6f}]",
    )

    # Check NOT unit sphere when normalize_output=False
    model_no_norm = STARVP(normalize_output=False)
    model_no_norm.eval()
    with torch.no_grad():
        out_nn = model_no_norm(past_pos, sal_xyz)
    norms_nn = out_nn.norm(dim=-1)
    not_all_unit = not torch.allclose(norms_nn, torch.ones_like(norms_nn), atol=1e-3)
    check(not_all_unit, "normalize_output=False → NOT all unit vectors")

    # =================================================================== #
    #  Test Group D: Intermediate outputs
    # =================================================================== #
    print("\n--- D. Intermediate outputs ---")

    with torch.no_grad():
        result = model(past_pos, sal_xyz, return_intermediates=True)

    check(isinstance(result, STARVPOutput), "Returns STARVPOutput")
    check(tuple(result.p_hat.shape) == (B, T_H, 3), f"p_hat: {tuple(result.p_hat.shape)}")
    check(tuple(result.p_prime.shape) == (B, T_H, 3), f"p_prime: {tuple(result.p_prime.shape)}")
    check(
        tuple(result.p_double_prime.shape) == (B, T_H, 3),
        f"p_double_prime: {tuple(result.p_double_prime.shape)}",
    )
    check(tuple(result.p_s_out.shape) == (B, T, D_C), f"p_s_out: {tuple(result.p_s_out.shape)}")
    check(tuple(result.s_s_out.shape) == (B, T, D_C), f"s_s_out: {tuple(result.s_s_out.shape)}")
    check(tuple(result.w_prime.shape) == (B, T_H), f"w_prime: {tuple(result.w_prime.shape)}")
    check(
        tuple(result.w_double_prime.shape) == (B, T_H),
        f"w_double_prime: {tuple(result.w_double_prime.shape)}",
    )
    check(
        torch.allclose(
            result.w_prime + result.w_double_prime,
            torch.ones_like(result.w_prime),
            atol=1e-6,
        ),
        "w_prime + w_double_prime = 1",
    )

    # =================================================================== #
    #  Test Group E: Gradient flow
    # =================================================================== #
    print("\n--- E. Gradient flow (end-to-end) ---")

    model.train()
    model.zero_grad()

    past_g = torch.randn(B, T_M, 3, requires_grad=True)
    sal_g = torch.randn(B, T, D_P, 4, requires_grad=True)
    out_g = model(past_g, sal_g)
    out_g.sum().backward()

    all_grads = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    check(all_grads, "All model parameters receive gradients")

    check(
        past_g.grad is not None and past_g.grad.abs().sum() > 0,
        "Gradient flows to past_positions",
    )
    check(
        sal_g.grad is not None and sal_g.grad.abs().sum() > 0,
        "Gradient flows to sal_xyz",
    )

    # Per-module gradient check
    for name, module in [
        ("LSTM", model.lstm),
        ("Spatial Attn", model.spatial_attn),
        ("Temporal Attn", model.temporal_attn),
        ("Gating", model.gating),
    ]:
        has_grads = all(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters()
        )
        check(has_grads, f"{name} parameters receive gradients")

    # =================================================================== #
    #  Test Group F: Parameter counts
    # =================================================================== #
    print("\n--- F. Parameter counts ---")

    counts = model.count_parameters()
    check(
        all(v > 0 for v in counts.values()),
        "All sub-module counts > 0",
    )
    sub_sum = counts["lstm"] + counts["spatial_attn"] + counts["temporal_attn"] + counts["gating"]
    check(sub_sum == counts["total"], f"Sum of sub-modules = total: {sub_sum:,} == {counts['total']:,}")

    # Verify against direct count
    direct_total = sum(p.numel() for p in model.parameters())
    check(counts["total"] == direct_total, f"Total: {counts['total']:,} == {direct_total:,}")

    print(f"       LSTM:          {counts['lstm']:>10,}")
    print(f"       Spatial Attn:  {counts['spatial_attn']:>10,}")
    print(f"       Temporal Attn: {counts['temporal_attn']:>10,}")
    print(f"       Gating:        {counts['gating']:>10,}")
    print(f"       ─────────────────────────")
    print(f"       Total:         {counts['total']:>10,}")

    # =================================================================== #
    #  Test Group G: Robustness
    # =================================================================== #
    print("\n--- G. Robustness ---")

    model.eval()

    check(
        not (torch.isnan(out).any() or torch.isinf(out).any()),
        "No NaN/Inf in output",
    )

    with torch.no_grad():
        det_a = model(past_pos, sal_xyz)
        det_b = model(past_pos, sal_xyz)
    check(torch.equal(det_a, det_b), "Deterministic in eval mode")

    for bs in [1, 8]:
        with torch.no_grad():
            out_bs = model(
                torch.randn(bs, T_M, 3),
                torch.randn(bs, T, D_P, 4),
            )
        check(
            tuple(out_bs.shape) == (bs, T_H, 3),
            f"Batch={bs} → shape {tuple(out_bs.shape)}",
        )

    # =================================================================== #
    #  Test Group H: Integration
    # =================================================================== #
    print("\n--- H. Integration ---")

    model.eval()
    target = F.normalize(torch.randn(B, T_H, 3), dim=-1)
    with torch.no_grad():
        pred = model(past_pos, sal_xyz)

    # OD loss: arccos(dot product)
    cos_sim = (pred * target).sum(dim=-1)
    check(
        (cos_sim >= -1.0 - 1e-5).all() and (cos_sim <= 1.0 + 1e-5).all(),
        f"Dot products in [-1,1]: [{cos_sim.min():.4f}, {cos_sim.max():.4f}]",
    )
    cos_sim_clamped = cos_sim.clamp(-1 + 1e-7, 1 - 1e-7)
    od = torch.acos(cos_sim_clamped).mean()
    check(
        not (torch.isnan(od) or torch.isinf(od)) and od >= 0,
        f"OD loss computable: {od.item():.4f} rad",
    )

    # =================================================================== #
    #  I — T_M = 25 (fair comparison with MFTR/MUSE window length)
    # =================================================================== #
    print("\n--- I. T_M=25 forward (fair alignment) ---")

    T_M25, T_H25 = 25, 25
    T25 = T_M25 + T_H25
    model_25 = STARVP(
        normalize_output=True,
        t_m=T_M25,
        t_h=T_H25,
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        d_c=D_C,
        n_heads=8,
        spatial_enc_layers=2,
        spatial_dec_layers=2,
        temporal_enc_layers=2,
        temporal_dec_layers=2,
        d_pe=129,
        d_te=127,
        d_g=128,
    )
    model_25.eval()
    pp25 = torch.randn(2, T_M25, 3)
    ss25 = torch.randn(2, T25, D_P, 4)
    with torch.no_grad():
        o25 = model_25(pp25, ss25)
    check(
        tuple(o25.shape) == (2, T_H25, 3),
        f"T_M=25 forward output {tuple(o25.shape)} == (2, {T_H25}, 3)",
    )

    # =================================================================== #
    #  Summary
    # =================================================================== #
    print(f"\n{'=' * 72}")
    print(f"{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"Total: {test_num} tests")
    print(f"{'=' * 72}")
    return all_passed


if __name__ == "__main__":
    _self_test()
