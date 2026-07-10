"""
=============================================================================
STAR-VP Temporal Attention Module — Time-varying HT-VC Fusion (1st Stage)
=============================================================================

Paper:  Section 3.5, Equations 4-5
        Table 1 — D_cT=256, D_PE=129, D_TE=127, N_heads=8, N_layers_T=2

The Temporal Attention Module captures how the relative importance of
viewpoint (HT) and saliency (VC) features changes over time.  It is
the "first-stage fusion" that produces long-term-optimized predictions.

Architecture
------------

    STEP A — Positional & Modality Embeddings:

        PE    ∈ R^{T×D_PE}   = [40, 129]   fixed 1D Fourier (sinusoidal)
        TE_P  ∈ R^{T×D_TE}   = [40, 127]   learnable viewport-modality emb.
        TE_S  ∈ R^{T×D_TE}   = [40, 127]   learnable saliency-modality emb.

    STEP B — Construct 80-token encoder input PS (Eq. 4):

        P_tokens = C_f(P_{s-out}, PE, TE_P)     [B, 40, D_PS=512]
        S_tokens = C_f(S_{s-out}, PE, TE_S)     [B, 40, D_PS=512]
        PS       = C_t(P_tokens, S_tokens)       [B, 80, 512]

        D_PS = D_C + D_PE + D_TE = 256 + 129 + 127 = 512

    STEP C — Encoder (self-attention over all 80 tokens):

        PS' = Encoder(Q=K=V=PS)                  [B, 80, 512]

        Each token can attend to every other token across both time and
        modality, discovering which timesteps and modalities are related.

    STEP D — Decoder query embeddings E (Eq. 5):

        E = C_f(PE_{future}, TE_P_{future})      [B, T_H=25, D_E=256]

        Uses only the FUTURE timesteps' positional + viewport-modality
        embeddings.  D_E = D_PE + D_TE = 129 + 127 = 256.

    STEP E — Decoder (cross-attention: queries attend to encoder memory):

        P̂''_raw = Decoder(Q=E, K=V=PS')          [B, 25, 256]

        Note: d_q=256 ≠ d_kv=512, so this is the MIXED-DIMENSION case
        handled by our PerceiverBlock.

    STEP F — Output projection:

        P̂'' = Linear(P̂''_raw)                    [B, 25, 3]

Downstream:
    P̂'' feeds into the Gating Fusion Module (Section 3.6) as the
    long-term prediction to be blended with the LSTM short-term P̂'.

Hyperparameters from Table 1:
    D_C   = 256   Channel dimension (Spatial Attention output)
    D_PE  = 129   Fourier positional embedding dimension
    D_TE  = 127   Trainable modality-specific embedding dimension
    D_PS  = 512   Encoder token dimension (D_C + D_PE + D_TE)
    D_E   = 256   Decoder query dimension (D_PE + D_TE)
    N_heads = 8   Attention heads
=============================================================================
"""

import math
import torch
import torch.nn as nn

from .encoder_decoder import PerceiverBlock


# ========================================================================= #
#  Temporal Attention Module  (Paper Section 3.5, Eqs. 4-5)
# ========================================================================= #

class TemporalAttentionModule(nn.Module):
    """
    Time-varying fusion of spatially-aligned viewport and saliency features.

    Constructs an 80-token sequence (40 viewport + 40 saliency), enriches
    each token with Fourier positional and learnable modality embeddings,
    then uses Transformer-style encoder (self-attention) and decoder
    (cross-attention) to produce 25 future viewport predictions.
    """

    def __init__(
        self,
        d_c: int = 256,
        d_pe: int = 129,
        d_te: int = 127,
        n_heads: int = 8,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        t_m: int = 15,
        t_h: int = 25,
        output_dim: int = 3,
        dropout: float = 0.0,
    ):
        """
        Args:
            d_c:           Spatial Attention output dim.  Table 1: D_cT = 256.
            d_pe:          Fourier PE dimension.           Table 1: D_PE = 129.
            d_te:          Modality embedding dimension.   Table 1: D_TE = 127.
            n_heads:       Attention heads.                Table 1: 8.
            n_enc_layers:  PerceiverBlocks in encoder.     Table 1: N_layers_T = 2.
            n_dec_layers:  PerceiverBlocks in decoder.     Table 1: N_layers_T = 2.
            t_m:           Past timesteps.                 15 (3s at 5fps).
            t_h:           Future timesteps.               25 (5s at 5fps).
            output_dim:    Position dimension (3 for xyz). Default 3.
            dropout:       Dropout in attention/MLP.       Default 0.
        """
        super().__init__()
        self.d_c = d_c
        self.d_pe = d_pe
        self.d_te = d_te
        self.t_m = t_m
        self.t_h = t_h

        t_total = t_m + t_h              # 40
        d_ps = d_c + d_pe + d_te         # 512  (encoder token dim)
        d_e = d_pe + d_te                # 256  (decoder query dim)

        self.d_ps = d_ps
        self.d_e = d_e
        self.t_total = t_total

        # ---- Fixed Fourier positional embeddings (Eq. 4) ----
        pe = self._make_fourier_pe(t_total, d_pe)      # [T, D_PE]
        self.register_buffer("pe", pe)

        # ---- Learnable modality-specific embeddings (Eq. 4) ----
        self.te_p = nn.Parameter(torch.empty(t_total, d_te))    # viewport
        self.te_s = nn.Parameter(torch.empty(t_total, d_te))    # saliency
        nn.init.normal_(self.te_p, std=0.02)
        nn.init.normal_(self.te_s, std=0.02)

        # ---- Encoder: self-attention over 80 tokens at D_PS=512 ----
        # d_attn = D_C = 256 (internal attention dim from Table 1)
        self.encoder_layers = nn.ModuleList([
            PerceiverBlock(
                d_q=d_ps, d_kv=d_ps, d_attn=d_c,
                n_heads=n_heads, dropout=dropout,
            )
            for _ in range(n_enc_layers)
        ])

        # ---- Decoder: cross-attention, Q=D_E=256, K/V=D_PS=512 ----
        self.decoder_layers = nn.ModuleList([
            PerceiverBlock(
                d_q=d_e, d_kv=d_ps, d_attn=d_c,
                n_heads=n_heads, dropout=dropout,
            )
            for _ in range(n_dec_layers)
        ])

        # ---- Output projection: D_E=256 → 3 ----
        self.output_proj = nn.Linear(d_e, output_dim)

    # --------------------------------------------------------------------- #
    #  Fourier PE
    # --------------------------------------------------------------------- #

    @staticmethod
    def _make_fourier_pe(max_len: int, d_pe: int) -> torch.Tensor:
        """
        Standard sinusoidal positional encoding (Vaswani et al., 2017).

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_pe))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_pe))

        For odd d_pe the last column is sin-only (no paired cos).
        """
        pe = torch.zeros(max_len, d_pe)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

        # Frequencies: 1 / 10000^(2i / d_pe)  for i = 0, 1, …, ceil(d_pe/2)-1
        half = (d_pe + 1) // 2      # number of sin terms
        div_term = torch.exp(
            torch.arange(half, dtype=torch.float32) * -(2.0 * math.log(10000.0) / d_pe)
        )

        angles = position * div_term                 # [max_len, half]
        pe[:, 0::2] = torch.sin(angles)              # 65 sin columns for d_pe=129
        pe[:, 1::2] = torch.cos(angles[:, :d_pe // 2])   # 64 cos columns
        return pe

    # --------------------------------------------------------------------- #
    #  Forward
    # --------------------------------------------------------------------- #

    def forward(
        self,
        p_s_out: torch.Tensor,
        s_s_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            p_s_out: [B, T, D_C]   Viewport features from Spatial Attention.
                     T must equal t_m + t_h.
            s_s_out: [B, T, D_C]   Saliency features from Spatial Attention.

        Returns:
            p_pred: [B, T_H, 3]    Long-term viewport predictions.
        """
        B, T, _ = p_s_out.shape
        assert T == self.t_total, (
            f"Expected T={self.t_total} (T_M+T_H), got {T}"
        )

        # ============================================================= #
        #  STEP A: Expand fixed PE and learnable TE to batch  (Eq. 4)
        # ============================================================= #
        pe  = self.pe.unsqueeze(0).expand(B, -1, -1)        # [B, T, D_PE]
        te_p = self.te_p.unsqueeze(0).expand(B, -1, -1)     # [B, T, D_TE]
        te_s = self.te_s.unsqueeze(0).expand(B, -1, -1)     # [B, T, D_TE]

        # ============================================================= #
        #  STEP B: Construct 80-token encoder input PS  (Eq. 4)
        # ============================================================= #
        # Viewport tokens: C_f(P_{s-out}, PE, TE_P) → [B, T, D_PS]
        p_tokens = torch.cat([p_s_out, pe, te_p], dim=-1)

        # Saliency tokens: C_f(S_{s-out}, PE, TE_S) → [B, T, D_PS]
        s_tokens = torch.cat([s_s_out, pe, te_s], dim=-1)

        # C_t — interleave along time → [B, 2T, D_PS] = [B, 80, 512]
        PS = torch.cat([p_tokens, s_tokens], dim=1)

        # ============================================================= #
        #  STEP C: Encoder — self-attention over all 80 tokens
        # ============================================================= #
        for enc in self.encoder_layers:
            PS = enc(PS, PS, PS)                             # [B, 80, D_PS]

        # ============================================================= #
        #  STEP D: Decoder query embeddings E  (Eq. 5)
        # ============================================================= #
        # Future timesteps: indices T_M … T_M+T_H-1
        pe_future  = pe[:, self.t_m:, :]                     # [B, T_H, D_PE]  #keeps all batches, time from 15 onward, all D_PE dims
        te_p_future = te_p[:, self.t_m:, :]                  # [B, T_H, D_TE]  #keeps all batches, time from 15 onward, all D_TE dims

        E = torch.cat([pe_future, te_p_future], dim=-1)      # [B, T_H, D_E]

        # ============================================================= #
        #  STEP E: Decoder — cross-attention (Q=E, K=V=PS')
        # ============================================================= #
        for dec in self.decoder_layers:
            E = dec(E, PS, PS)                               # [B, T_H, D_E]

        # ============================================================= #
        #  STEP F: Output projection → 3D coordinates
        # ============================================================= #
        p_pred = self.output_proj(E)                         # [B, T_H, 3]

        return p_pred

    # --------------------------------------------------------------------- #
    #  Repr
    # --------------------------------------------------------------------- #

    def extra_repr(self) -> str:
        return (
            f"d_c={self.d_c}, d_pe={self.d_pe}, d_te={self.d_te}, "
            f"d_ps={self.d_ps}, d_e={self.d_e}, "
            f"t_m={self.t_m}, t_h={self.t_h}, "
            f"n_enc={len(self.encoder_layers)}, n_dec={len(self.decoder_layers)}"
        )


# ========================================================================= #
#  Self-Test — exhaustive verification
# ========================================================================= #

def _self_test() -> bool:
    """
    Exhaustive verification of TemporalAttentionModule for STAR-VP.

    Tests:
      A — Basic shapes and derived dims
        1.  Output shape [B, T_H, 3]
        2.  D_PS = D_C + D_PE + D_TE = 512
        3.  D_E  = D_PE + D_TE = 256
        4.  Default T_M=15, T_H=25

      B — Fourier Positional Embeddings (PE)
        5.  PE shape [T, D_PE] = [40, 129]
        6.  PE is not trainable (buffer, not parameter)
        7.  PE[0, 0] = sin(0) = 0
        8.  PE[0, 1] = cos(0) = 1
        9.  PE values bounded in [-1, 1]
       10.  PE different across timesteps (not constant)

      C — Modality Embeddings (TE_P, TE_S)
       11.  TE_P shape [T, D_TE] = [40, 127]
       12.  TE_S shape [T, D_TE] = [40, 127]
       13.  TE_P is trainable parameter
       14.  TE_S is trainable parameter
       15.  TE_P ≠ TE_S (different modalities have different embeddings)

      D — Encoder architecture
       16.  Encoder has n_enc_layers PerceiverBlocks
       17.  Encoder d_q = d_kv = D_PS = 512
       18.  Encoder d_attn = D_C = 256

      E — Decoder architecture
       19.  Decoder has n_dec_layers PerceiverBlocks
       20.  Decoder d_q = D_E = 256
       21.  Decoder d_kv = D_PS = 512
       22.  Decoder d_attn = D_C = 256

      F — Output projection
       23.  output_proj: D_E=256 → 3

      G — Gradient flow
       24.  All model parameters receive gradients
       25.  Gradient flows to p_s_out input
       26.  Gradient flows to s_s_out input
       27.  TE_P receives gradient
       28.  TE_S receives gradient

      H — Parameter counts
       29.  TE_P params = 40 * 127 = 5,080
       30.  TE_S params = 40 * 127 = 5,080
       31.  output_proj params = 256*3 + 3 = 771
       32.  Total parameter count verification

      I — Robustness
       33.  No NaN/Inf in output
       34.  Zero inputs → valid output
       35.  Determinism in eval mode
       36.  Different batch sizes (1, 4, 8)

      J — Integration shapes
       37.  Output [B, 25, 3] for Gating Fusion
       38.  Accepts Spatial Attention outputs [B, 40, 256]
    """
    torch.manual_seed(42)

    T_M = 15
    T_H = 25
    T = T_M + T_H
    D_C = 256
    D_PE = 129
    D_TE = 127
    D_PS = D_C + D_PE + D_TE   # 512
    D_E = D_PE + D_TE           # 256
    N_HEADS = 8
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
    print("STAR-VP Temporal Attention Module Self-Test")
    print("=" * 72)

    model = TemporalAttentionModule(
        d_c=D_C, d_pe=D_PE, d_te=D_TE, n_heads=N_HEADS,
        n_enc_layers=2, n_dec_layers=2, t_m=T_M, t_h=T_H,
    )
    model.eval()

    p_in = torch.randn(B, T, D_C)
    s_in = torch.randn(B, T, D_C)

    with torch.no_grad():
        out = model(p_in, s_in)

    # =================================================================== #
    #  Test Group A: Basic shapes and derived dims
    # =================================================================== #
    print("\n--- A. Basic shapes ---")

    check(
        tuple(out.shape) == (B, T_H, 3),
        f"Output shape {tuple(out.shape)} == ({B}, {T_H}, 3)",
    )
    check(model.d_ps == D_PS, f"D_PS = {model.d_ps} == {D_PS}")
    check(model.d_e == D_E, f"D_E = {model.d_e} == {D_E}")
    check(model.t_m == T_M and model.t_h == T_H, f"T_M={model.t_m}, T_H={model.t_h}")

    # =================================================================== #
    #  Test Group B: Fourier Positional Embeddings
    # =================================================================== #
    print("\n--- B. Fourier PE ---")

    pe = model.pe
    check(tuple(pe.shape) == (T, D_PE), f"PE shape {tuple(pe.shape)} == ({T}, {D_PE})")

    # PE should not be a trainable parameter
    pe_is_buffer = not any(p is pe for p in model.parameters())
    check(pe_is_buffer, "PE is buffer (not trainable)")

    # PE[0, 0] = sin(0) = 0
    check(abs(pe[0, 0].item()) < 1e-6, f"PE[0,0] = sin(0) = {pe[0, 0].item():.6f}")

    # PE[0, 1] = cos(0) = 1
    check(abs(pe[0, 1].item() - 1.0) < 1e-6, f"PE[0,1] = cos(0) = {pe[0, 1].item():.6f}")

    # PE bounded in [-1, 1]
    check(
        pe.min() >= -1.0 - 1e-6 and pe.max() <= 1.0 + 1e-6,
        f"PE in [-1,1]: [{pe.min():.4f}, {pe.max():.4f}]",
    )

    # PE varies across timesteps (not constant)
    check(
        not torch.allclose(pe[0], pe[1], atol=1e-6),
        "PE differs across timesteps",
    )

    # =================================================================== #
    #  Test Group C: Modality Embeddings
    # =================================================================== #
    print("\n--- C. Modality embeddings ---")

    check(
        tuple(model.te_p.shape) == (T, D_TE),
        f"TE_P shape {tuple(model.te_p.shape)} == ({T}, {D_TE})",
    )
    check(
        tuple(model.te_s.shape) == (T, D_TE),
        f"TE_S shape {tuple(model.te_s.shape)} == ({T}, {D_TE})",
    )
    check(model.te_p.requires_grad, "TE_P is trainable")
    check(model.te_s.requires_grad, "TE_S is trainable")
    check(
        not torch.allclose(model.te_p, model.te_s, atol=1e-6),
        "TE_P ≠ TE_S",
    )

    # =================================================================== #
    #  Test Group D: Encoder architecture
    # =================================================================== #
    print("\n--- D. Encoder architecture ---")

    check(
        len(model.encoder_layers) == 2,
        f"Encoder layers: {len(model.encoder_layers)} (expect 2)",
    )
    enc0 = model.encoder_layers[0]
    check(
        enc0.d_q == D_PS and enc0.d_kv == D_PS,
        f"Encoder: d_q={enc0.d_q}, d_kv={enc0.d_kv} (both {D_PS})",
    )
    check(enc0.d_attn == D_C, f"Encoder: d_attn={enc0.d_attn} == D_C={D_C}")

    # =================================================================== #
    #  Test Group E: Decoder architecture
    # =================================================================== #
    print("\n--- E. Decoder architecture ---")

    check(
        len(model.decoder_layers) == 2,
        f"Decoder layers: {len(model.decoder_layers)} (expect 2)",
    )
    dec0 = model.decoder_layers[0]
    check(dec0.d_q == D_E, f"Decoder: d_q={dec0.d_q} == D_E={D_E}")
    check(dec0.d_kv == D_PS, f"Decoder: d_kv={dec0.d_kv} == D_PS={D_PS}")
    check(dec0.d_attn == D_C, f"Decoder: d_attn={dec0.d_attn} == D_C={D_C}")

    # =================================================================== #
    #  Test Group F: Output projection
    # =================================================================== #
    print("\n--- F. Output projection ---")

    check(
        model.output_proj.in_features == D_E and model.output_proj.out_features == 3,
        f"output_proj: {model.output_proj.in_features}→{model.output_proj.out_features}",
    )

    # =================================================================== #
    #  Test Group G: Gradient flow
    # =================================================================== #
    print("\n--- G. Gradient flow ---")

    model.train()
    model.zero_grad()

    p_g = torch.randn(B, T, D_C, requires_grad=True)
    s_g = torch.randn(B, T, D_C, requires_grad=True)
    out_g = model(p_g, s_g)
    out_g.sum().backward()

    all_grads = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    check(all_grads, "All model parameters receive gradients")

    check(
        p_g.grad is not None and p_g.grad.abs().sum() > 0,
        "Gradient flows to p_s_out input",
    )
    check(
        s_g.grad is not None and s_g.grad.abs().sum() > 0,
        "Gradient flows to s_s_out input",
    )
    check(
        model.te_p.grad is not None and model.te_p.grad.abs().sum() > 0,
        "TE_P receives gradient",
    )
    check(
        model.te_s.grad is not None and model.te_s.grad.abs().sum() > 0,
        "TE_S receives gradient",
    )

    # =================================================================== #
    #  Test Group H: Parameter counts
    # =================================================================== #
    print("\n--- H. Parameter counts ---")

    n_te_p = model.te_p.numel()
    n_te_s = model.te_s.numel()
    expected_te = T * D_TE  # 40 * 127 = 5080
    check(n_te_p == expected_te, f"TE_P params: {n_te_p:,} == {expected_te:,}")
    check(n_te_s == expected_te, f"TE_S params: {n_te_s:,} == {expected_te:,}")

    n_proj = sum(p.numel() for p in model.output_proj.parameters())
    expected_proj = D_E * 3 + 3  # 256*3 + 3 = 771
    check(n_proj == expected_proj, f"output_proj params: {n_proj:,} == {expected_proj:,}")

    # Encoder PerceiverBlock (d_q=d_kv=512, d_attn=256):
    # norm_q: 512*2 = 1024
    # norm_kv: 512*2 = 1024
    # MHA: q_proj(512→256)+k_proj(512→256)+v_proj(512→256)+o_proj(256→512)
    #   = (512*256+256) + (512*256+256) + (512*256+256) + (256*512+512)
    #   = 131328 + 131328 + 131328 + 131584 = 525568
    # norm_ff: 512*2 = 1024
    # mlp.0: 512*1024+1024 = 525312   (d_ff = 4*d_attn = 1024)
    # mlp.3: 1024*512+512 = 524800
    # Total = 1024+1024+525568+1024+525312+524800 = 1578752
    n_enc_block = sum(p.numel() for p in model.encoder_layers[0].parameters())
    expected_enc_block = (
        D_PS * 2                          # norm_q
        + D_PS * 2                        # norm_kv
        + (D_PS * D_C + D_C)             # q_proj
        + (D_PS * D_C + D_C)             # k_proj
        + (D_PS * D_C + D_C)             # v_proj
        + (D_C * D_PS + D_PS)            # o_proj
        + D_PS * 2                        # norm_ff
        + (D_PS * 4 * D_C + 4 * D_C)    # mlp linear1 (d_ff=4*d_attn=1024)
        + (4 * D_C * D_PS + D_PS)        # mlp linear2
    )
    check(
        n_enc_block == expected_enc_block,
        f"Encoder block params: {n_enc_block:,} == {expected_enc_block:,}",
    )

    # Decoder PerceiverBlock (d_q=256, d_kv=512, d_attn=256):
    # norm_q: 256*2 = 512
    # norm_kv: 512*2 = 1024
    # MHA: q_proj(256→256)+k_proj(512→256)+v_proj(512→256)+o_proj(256→256)
    #   = (256*256+256) + (512*256+256) + (512*256+256) + (256*256+256)
    #   = 65792 + 131328 + 131328 + 65792 = 394240
    # norm_ff: 256*2 = 512
    # mlp.0: 256*1024+1024 = 263168
    # mlp.3: 1024*256+256 = 262400
    # Total = 512+1024+394240+512+263168+262400 = 921856
    n_dec_block = sum(p.numel() for p in model.decoder_layers[0].parameters())
    expected_dec_block = (
        D_E * 2                           # norm_q
        + D_PS * 2                        # norm_kv
        + (D_E * D_C + D_C)              # q_proj
        + (D_PS * D_C + D_C)             # k_proj
        + (D_PS * D_C + D_C)             # v_proj
        + (D_C * D_E + D_E)              # o_proj
        + D_E * 2                         # norm_ff
        + (D_E * 4 * D_C + 4 * D_C)     # mlp linear1
        + (4 * D_C * D_E + D_E)          # mlp linear2
    )
    check(
        n_dec_block == expected_dec_block,
        f"Decoder block params: {n_dec_block:,} == {expected_dec_block:,}",
    )

    n_total = sum(p.numel() for p in model.parameters())
    expected_total = (
        n_te_p + n_te_s
        + 2 * expected_enc_block
        + 2 * expected_dec_block
        + expected_proj
    )
    check(n_total == expected_total, f"Total params: {n_total:,} == {expected_total:,}")
    print(f"       Encoder block: {n_enc_block:,}")
    print(f"       Decoder block: {n_dec_block:,}")
    print(f"       Total model:   {n_total:,}")

    # =================================================================== #
    #  Test Group I: Robustness
    # =================================================================== #
    print("\n--- I. Robustness ---")

    model.eval()

    check(
        not (torch.isnan(out).any() or torch.isinf(out).any()),
        "No NaN/Inf in output",
    )

    with torch.no_grad():
        out_z = model(torch.zeros(2, T, D_C), torch.zeros(2, T, D_C))
    check(not torch.isnan(out_z).any(), "Zero inputs → no NaN")

    with torch.no_grad():
        det_a = model(p_in, s_in)
        det_b = model(p_in, s_in)
    check(torch.equal(det_a, det_b), "Deterministic in eval mode")

    for bs in [1, 4, 8]:
        with torch.no_grad():
            out_bs = model(torch.randn(bs, T, D_C), torch.randn(bs, T, D_C))
        check(
            tuple(out_bs.shape) == (bs, T_H, 3),
            f"Batch={bs} → shape {tuple(out_bs.shape)}",
        )

    # =================================================================== #
    #  Test Group J: Integration shapes
    # =================================================================== #
    print("\n--- J. Integration shapes ---")

    model.eval()
    with torch.no_grad():
        out_j = model(p_in, s_in)

    check(
        tuple(out_j.shape) == (B, T_H, 3),
        f"Gating Fusion input: {tuple(out_j.shape)} == ({B}, {T_H}, 3)",
    )
    check(
        p_in.shape == (B, T, D_C) and s_in.shape == (B, T, D_C),
        f"Accepts Spatial Attn outputs: p={tuple(p_in.shape)}, s={tuple(s_in.shape)}",
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
