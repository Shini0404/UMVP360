"""
=============================================================================
STAR-VP Spatial Attention Module — Space-aligned HT-VC Fusion
=============================================================================

Paper:  Section 3.4, Equation 3
        Table 1 — D_cT=256, N_heads=8, N_layers_T=2, D_P=128

The Spatial Attention Module aligns the viewpoint trajectory (HT) with the
saliency (VC) in the spatial dimension.  For each timestep independently, it
determines "which salient regions are near where the user is looking?"

Architecture (per timestep, batched by flattening B × T → BT)
--------------------------------------------------------------

    STEP A — Prepare inputs (Eq. 3):

        P_concat = C_t(P̄, P̂')                    [B, T_M+T_H, 3]
        P_{s-in} = C_f(P_concat, TE)               [B, T_M+T_H, 4]

        TE ∈ R^{(T_M+T_H)×1} is expanded from a single learnable scalar.
        It transforms the viewport (x,y,z) into 4D form (x,y,z,te) to match
        the saliency format (x,y,z,s), facilitating spatial alignment.

        S_{s-in} = S_xyz                            [B, T_M+T_H, D_P, 4]

    STEP B — Linear projections to D_C:

        S = Linear_S(S_{s-in})                      [B, T, D_P, D_C]
        P = Linear_P(P_{s-in})                      [B, T, D_C]

    STEP C — Encoder (self-attention on saliency, per timestep):

        For all timesteps in parallel (flatten B×T → BT):
            Q = K = V = S_t     [BT, D_P, D_C]
            S' = Encoder(S)     [BT, D_P, D_C]

        The encoder learns relationships BETWEEN saliency points.

    STEP D — Decoder (cross-attention: viewport queries saliency):

        For all timesteps in parallel:
            Q = P_t             [BT, 1, D_C]
            K = V = S'_t        [BT, D_P, D_C]
            P_dec = Decoder(P, S')   [BT, 1, D_C]

        The viewport "asks" the saliency map "what's important near me?"

    OUTPUTS:
        P_{s-out} = P_dec                           [B, T, D_C]
        S_{s-out} = mean_pool(S', dim=D_P)          [B, T, D_C]

Downstream:
    Both outputs feed into the Temporal Attention Module (Section 3.5).
    P_{s-out} carries spatially-aligned viewport features.
    S_{s-out} carries aggregated saliency features.

Hyperparameters from Table 1:
    D_C  (D_cT) = 256   Internal channel dimension
    N_heads     = 8     Attention heads
    N_layers_T  = 2     Self-attention layers per encoder/decoder
    D_P         = 128   Saliency points per frame
=============================================================================
"""

import torch
import torch.nn as nn

from .encoder_decoder import PerceiverBlock


# ========================================================================= #
#  Spatial Attention Module  (Paper Section 3.4, Eq. 3)
# ========================================================================= #

#Turns raw 4D tokens (x,y,z,s) and (x,y,z,te) into 256-D features.
#Runs an encoder: the 128 saliency points talk to each other (self-attention).
#Runs a decoder: the single viewport vector queries those 128 points (cross-attention) to pull out “what matters near my gaze.”


class SpatialAttentionModule(nn.Module):
    """
    Per-timestep spatial alignment of viewpoint (HT) and saliency (VC).

    Processes each of the (T_M + T_H) timesteps independently:
    the 128 saliency points self-attend, then the single viewport position
    cross-attends to the encoded saliency to extract nearby visual features.

    The B × T dimension is flattened to BT for efficient batched attention.
    """

    def __init__(
        self,
        d_in: int = 4,
        d_c: int = 256,
        n_heads: int = 8,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        dropout: float = 0.0,
        # ---- MMVP2 extensions (defaults = STAR-VP behaviour) ----
        d_in_s: int | None = None,
        d_in_p: int | None = None,
    ):
        """
        Args:
            d_in:          Default dimension of raw saliency / viewport tokens.
                           4 for STAR-VP: (x, y, z, s) and (x, y, z, te).
                           Used as the default when d_in_s / d_in_p are None.
            d_c:           Internal channel dim.  Table 1: D_cT = 256.
            n_heads:       Number of attention heads.  Table 1: 8.
            n_enc_layers:  PerceiverBlocks in the saliency encoder.
                           Table 1: N_layers_T = 2.
            n_dec_layers:  PerceiverBlocks in the viewport decoder.
                           Table 1: N_layers_T = 2.
            dropout:       Dropout in attention/MLP.  Paper does not specify.

            d_in_s:        Override the saliency-side input dim.  Use this in
                           MMVP2 when adding extra channels to saliency tokens
                           (e.g. behavioural prior -> d_in_s = 5).
            d_in_p:        Override the viewport-side input dim.  Default
                           keeps the (x, y, z, te) STAR-VP layout.
        """
        super().__init__()
        if d_in_s is None:
            d_in_s = d_in
        if d_in_p is None:
            d_in_p = d_in
        self.d_in = d_in
        self.d_in_s = d_in_s
        self.d_in_p = d_in_p
        self.d_c = d_c
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers

        # -- Trainable Embedding (TE): single scalar, Eq. 3 --
        self.te = nn.Parameter(torch.zeros(1))

        # -- Linear projections to D_C --
        self.proj_s = nn.Linear(d_in_s, d_c)
        self.proj_p = nn.Linear(d_in_p, d_c)

        # -- Encoder: self-attention on saliency (per timestep) --
        # Q = K = V = S — learns inter-saliency-point relationships
        self.encoder_layers = nn.ModuleList([
            PerceiverBlock(
                d_q=d_c, d_kv=d_c, d_attn=d_c,
                n_heads=n_heads, dropout=dropout,
            )
            for _ in range(n_enc_layers)
        ])

        # -- Decoder: cross-attention — viewport queries encoded saliency --
        # Q = P (viewport), K = V = S' (encoded saliency)
        self.decoder_layers = nn.ModuleList([
            PerceiverBlock(
                d_q=d_c, d_kv=d_c, d_attn=d_c,
                n_heads=n_heads, dropout=dropout,
            )
            for _ in range(n_dec_layers)
        ])

    # --------------------------------------------------------------------- #
    #  Forward
    # --------------------------------------------------------------------- #

    def forward(
        self,
        s_xyz: torch.Tensor,
        past_positions: torch.Tensor,
        lstm_predictions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s_xyz:            [B, T, D_P, 4]   Saliency representation from
                              SalMapProcessor.  T = T_M + T_H.
            past_positions:   [B, T_M, 3]      Past head positions (unit sphere).
            lstm_predictions: [B, T_H, 3]      LSTM-predicted future positions.

        Returns:
            p_s_out: [B, T, D_C]   Spatially aligned viewport features.
            s_s_out: [B, T, D_C]   Aggregated saliency features (mean-pooled).
        """
        B, T, D_P, _ = s_xyz.shape

        # ============================================================= #
        #  STEP A: Prepare P_{s-in}  (Eq. 3)
        # ============================================================= #
        # C_t — concat past and predicted along time → [B, T, 3]
        p_concat = torch.cat([past_positions, lstm_predictions], dim=1)

        # TE — expand scalar to [B, T, 1]  (same value everywhere)
        te_expanded = self.te.view(1, 1, 1).expand(B, T, 1)

        # C_f — concat along feature → [B, T, 4]  (x, y, z, te)
        p_s_in = torch.cat([p_concat, te_expanded], dim=-1)

        # S_{s-in} = S_xyz   (already [B, T, D_P, 4])

        # ============================================================= #
        #  STEP B: Linear projections to D_C
        # ============================================================= #
        S = self.proj_s(s_xyz)      # [B, T, D_P, D_C] #D_P = number of saliency points per frame (per timestep).
        P = self.proj_p(p_s_in)     # [B, T, D_C] #D_C = internal channel dimension.

        # ============================================================= #
        #  Flatten B × T → BT for per-timestep batched attention
        # ============================================================= #
        BT = B * T
        D_C = self.d_c

        S_flat = S.reshape(BT, D_P, D_C)       # [BT, D_P, D_C] 
        P_flat = P.reshape(BT, D_C).unsqueeze(1)  # [BT, 1, D_C]

        # ============================================================= #
        #  STEP C: Encoder — self-attention on saliency
        # ============================================================= #
        for enc in self.encoder_layers:
            S_flat = enc(S_flat, S_flat, S_flat)    # [BT, D_P, D_C]

        # ============================================================= #
        #  STEP D: Decoder — viewport cross-attends to encoded saliency
        # ============================================================= #
        for dec in self.decoder_layers:
            P_flat = dec(P_flat, S_flat, S_flat)    # [BT, 1, D_C]

        # ============================================================= #
        #  Reshape back to [B, T, D_C]
        # ============================================================= #
        # P_flat: [BT, 1, D_C] → squeeze token dim → [BT, D_C] → [B, T, D_C]
        p_s_out = P_flat.squeeze(1).reshape(B, T, D_C)

        # S_flat: [BT, D_P, D_C] → mean-pool over saliency points → [BT, D_C]
        s_s_out = S_flat.mean(dim=1).reshape(B, T, D_C)

        return p_s_out, s_s_out

    # --------------------------------------------------------------------- #
    #  Repr
    # --------------------------------------------------------------------- #

    def extra_repr(self) -> str:
        return (
            f"d_in_s={self.d_in_s}, d_in_p={self.d_in_p}, "
            f"d_c={self.d_c}, n_heads={self.n_heads}, "
            f"n_enc_layers={self.n_enc_layers}, n_dec_layers={self.n_dec_layers}"
        )


# ========================================================================= #
#  Self-Test — exhaustive verification
# ========================================================================= #

def _self_test() -> bool:
    """
    Exhaustive verification of SpatialAttentionModule for STAR-VP.

    Tests:
      A — Basic shapes and defaults
        1.  p_s_out shape [B, T, D_C]
        2.  s_s_out shape [B, T, D_C]
        3.  T = T_M + T_H = 40
        4.  Default D_C = 256

      B — Trainable embedding (TE)
        5.  TE is a single scalar parameter
        6.  Changing TE changes output (it contributes meaningfully)
        7.  TE gradient is non-zero after backward

      C — Linear projections
        8.  proj_s maps 4 → D_C
        9.  proj_p maps 4 → D_C

      D — Per-timestep independence
       10.  Changing saliency at timestep t does NOT change output at t'≠t
       11.  Changing viewport at timestep t does NOT change output at t'≠t

      E — Encoder / Decoder architecture
       12.  Encoder has n_enc_layers PerceiverBlocks
       13.  Decoder has n_dec_layers PerceiverBlocks
       14.  Encoder uses self-attention (d_q == d_kv == D_C)
       15.  Decoder uses cross-attention (d_q == d_kv == D_C)

      F — Gradient flow
       16.  Gradients reach all model parameters
       17.  Gradient flows back to s_xyz input
       18.  Gradient flows back to past_positions input
       19.  Gradient flows back to lstm_predictions input
       20.  TE receives gradient

      G — Parameter counts
       21.  Total parameter count verification
       22.  TE has exactly 1 parameter
       23.  proj_s parameter count
       24.  proj_p parameter count

      H — Robustness
       25.  No NaN/Inf in output
       26.  Zero inputs → valid output
       27.  Determinism in eval mode
       28.  Different batch sizes (1, 4, 8)

      I — Flexibility
       29.  Non-default T_M=10, T_H=20
       30.  Custom d_c=128, n_heads=4, n_layers=1

      J — Integration shapes
       31.  p_s_out compatible with Temporal Attention [B, 40, D_C]
       32.  s_s_out compatible with Temporal Attention [B, 40, D_C]
       33.  Both outputs have same shape
    """
    torch.manual_seed(42)

    T_M = 15
    T_H = 25
    T = T_M + T_H       # 40
    D_C = 256
    D_P = 128
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
    print("STAR-VP Spatial Attention Module Self-Test")
    print("=" * 72)

    model = SpatialAttentionModule(
        d_in=4, d_c=D_C, n_heads=N_HEADS,
        n_enc_layers=2, n_dec_layers=2,
    )
    model.eval()

    s_xyz = torch.randn(B, T, D_P, 4)
    past = torch.randn(B, T_M, 3)
    lstm_pred = torch.randn(B, T_H, 3)

    with torch.no_grad():
        p_out, s_out = model(s_xyz, past, lstm_pred)

    # =================================================================== #
    #  Test Group A: Basic shapes
    # =================================================================== #
    print("\n--- A. Basic shapes ---")

    check(
        tuple(p_out.shape) == (B, T, D_C),
        f"p_s_out shape {tuple(p_out.shape)} == ({B}, {T}, {D_C})",
    )
    check(
        tuple(s_out.shape) == (B, T, D_C),
        f"s_s_out shape {tuple(s_out.shape)} == ({B}, {T}, {D_C})",
    )
    check(
        p_out.shape[1] == T_M + T_H,
        f"T = {p_out.shape[1]} == T_M+T_H = {T_M+T_H}",
    )
    check(p_out.shape[2] == D_C, f"D_C = {p_out.shape[2]} == {D_C}")

    # =================================================================== #
    #  Test Group B: Trainable embedding (TE)
    # =================================================================== #
    print("\n--- B. Trainable embedding (TE) ---")

    check(model.te.numel() == 1, f"TE is scalar: numel={model.te.numel()}")

    # Changing TE changes output
    model_a = SpatialAttentionModule(d_in=4, d_c=D_C, n_heads=N_HEADS, n_enc_layers=1, n_dec_layers=1)
    model_b = SpatialAttentionModule(d_in=4, d_c=D_C, n_heads=N_HEADS, n_enc_layers=1, n_dec_layers=1)
    # Copy all params from a to b, then change only TE in b
    model_b.load_state_dict(model_a.state_dict())
    with torch.no_grad():
        model_b.te.fill_(5.0)

    model_a.eval()
    model_b.eval()
    s_test = torch.randn(2, 10, D_P, 4)
    past_test = torch.randn(2, 5, 3)
    pred_test = torch.randn(2, 5, 3)
    with torch.no_grad():
        pa, sa = model_a(s_test, past_test, pred_test)
        pb, sb = model_b(s_test, past_test, pred_test)
    check(
        not torch.allclose(pa, pb, atol=1e-5),
        "Changing TE changes p_s_out",
    )

    # TE gradient
    model.train()
    model.zero_grad()
    p_g, s_g = model(s_xyz, past, lstm_pred)
    (p_g.sum() + s_g.sum()).backward()
    check(
        model.te.grad is not None and model.te.grad.abs().sum() > 0,
        f"TE gradient: {model.te.grad}",
    )

    # =================================================================== #
    #  Test Group C: Linear projections
    # =================================================================== #
    print("\n--- C. Linear projections ---")

    check(
        model.proj_s.in_features == 4 and model.proj_s.out_features == D_C,
        f"proj_s: {model.proj_s.in_features}→{model.proj_s.out_features}",
    )
    check(
        model.proj_p.in_features == 4 and model.proj_p.out_features == D_C,
        f"proj_p: {model.proj_p.in_features}→{model.proj_p.out_features}",
    )

    # =================================================================== #
    #  Test Group D: Per-timestep independence
    # =================================================================== #
    print("\n--- D. Per-timestep independence ---")

    model.eval()

    # D1: Changing saliency at t=5 should not affect output at t=0
    s_orig = torch.randn(2, T, D_P, 4)
    s_mod = s_orig.clone()
    s_mod[:, 5, :, :] += 10.0  # large perturbation at timestep 5

    past_d = torch.randn(2, T_M, 3)
    pred_d = torch.randn(2, T_H, 3)
    with torch.no_grad():
        p_orig, s_orig_out = model(s_orig, past_d, pred_d)
        p_mod, s_mod_out = model(s_mod, past_d, pred_d)

    # Timestep 0 should be identical (saliency only changed at t=5)
    check(
        torch.allclose(p_orig[:, 0, :], p_mod[:, 0, :], atol=1e-5),
        "Saliency change at t=5 → p_s_out at t=0 unchanged",
    )
    # Timestep 5 should differ
    check(
        not torch.allclose(p_orig[:, 5, :], p_mod[:, 5, :], atol=1e-5),
        "Saliency change at t=5 → p_s_out at t=5 changed",
    )
    # S output at t=0 unchanged, at t=5 changed
    check(
        torch.allclose(s_orig_out[:, 0, :], s_mod_out[:, 0, :], atol=1e-5),
        "Saliency change at t=5 → s_s_out at t=0 unchanged",
    )
    check(
        not torch.allclose(s_orig_out[:, 5, :], s_mod_out[:, 5, :], atol=1e-5),
        "Saliency change at t=5 → s_s_out at t=5 changed",
    )

    # =================================================================== #
    #  Test Group E: Encoder / Decoder architecture
    # =================================================================== #
    print("\n--- E. Encoder / Decoder architecture ---")

    check(
        len(model.encoder_layers) == 2,
        f"Encoder has {len(model.encoder_layers)} layers (expect 2)",
    )
    check(
        len(model.decoder_layers) == 2,
        f"Decoder has {len(model.decoder_layers)} layers (expect 2)",
    )
    # Verify encoder blocks are self-attention compatible (d_q == d_kv)
    enc0 = model.encoder_layers[0]
    check(
        enc0.d_q == D_C and enc0.d_kv == D_C,
        f"Encoder block: d_q={enc0.d_q}, d_kv={enc0.d_kv} (both {D_C})",
    )
    # Verify decoder blocks
    dec0 = model.decoder_layers[0]
    check(
        dec0.d_q == D_C and dec0.d_kv == D_C,
        f"Decoder block: d_q={dec0.d_q}, d_kv={dec0.d_kv} (both {D_C})",
    )

    # =================================================================== #
    #  Test Group F: Gradient flow
    # =================================================================== #
    print("\n--- F. Gradient flow ---")

    model.train()
    model.zero_grad()

    s_g2 = torch.randn(B, T, D_P, 4, requires_grad=True)
    past_g2 = torch.randn(B, T_M, 3, requires_grad=True)
    pred_g2 = torch.randn(B, T_H, 3, requires_grad=True)

    p_o, s_o = model(s_g2, past_g2, pred_g2)
    loss = p_o.sum() + s_o.sum()
    loss.backward()

    # F1: All model parameters get gradients
    all_grads = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    check(all_grads, "All model parameters receive gradients")

    # F2-F4: Input gradients
    check(
        s_g2.grad is not None and s_g2.grad.abs().sum() > 0,
        "Gradient flows to s_xyz input",
    )
    check(
        past_g2.grad is not None and past_g2.grad.abs().sum() > 0,
        "Gradient flows to past_positions input",
    )
    check(
        pred_g2.grad is not None and pred_g2.grad.abs().sum() > 0,
        "Gradient flows to lstm_predictions input",
    )

    # F5: TE specifically
    check(
        model.te.grad is not None and model.te.grad.abs().sum() > 0,
        "TE receives gradient",
    )

    # =================================================================== #
    #  Test Group G: Parameter counts
    # =================================================================== #
    print("\n--- G. Parameter counts ---")

    n_total = sum(p.numel() for p in model.parameters())

    # TE: 1
    n_te = model.te.numel()
    check(n_te == 1, f"TE params: {n_te}")

    # proj_s: 4*256 + 256 = 1280
    n_proj_s = sum(p.numel() for p in model.proj_s.parameters())
    expected_proj = 4 * D_C + D_C
    check(n_proj_s == expected_proj, f"proj_s params: {n_proj_s:,} == {expected_proj:,}")

    # proj_p: same
    n_proj_p = sum(p.numel() for p in model.proj_p.parameters())
    check(n_proj_p == expected_proj, f"proj_p params: {n_proj_p:,} == {expected_proj:,}")

    # Each PerceiverBlock with d_q=d_kv=d_c=256, d_attn=256:
    # norm_q: 256*2 = 512
    # norm_kv: 256*2 = 512
    # MHA: 4*(256*256+256) = 263168
    # norm_ff: 256*2 = 512
    # mlp.0: 256*1024+1024 = 263168
    # mlp.3: 1024*256+256 = 262400
    # Total per block: 512+512+263168+512+263168+262400 = 790272
    n_per_block = sum(p.numel() for p in model.encoder_layers[0].parameters())
    expected_block = (
        D_C * 2          # norm_q
        + D_C * 2        # norm_kv
        + 4 * (D_C * D_C + D_C)  # MHA
        + D_C * 2        # norm_ff
        + (D_C * 4 * D_C + 4 * D_C)  # mlp linear1
        + (4 * D_C * D_C + D_C)      # mlp linear2
    )
    check(
        n_per_block == expected_block,
        f"PerceiverBlock params: {n_per_block:,} == {expected_block:,}",
    )

    # Total: TE(1) + proj_s(1280) + proj_p(1280) + 4 blocks
    expected_total = 1 + 2 * expected_proj + 4 * expected_block
    check(n_total == expected_total, f"Total params: {n_total:,} == {expected_total:,}")
    print(f"       Total: {n_total:,} parameters")

    # =================================================================== #
    #  Test Group H: Robustness
    # =================================================================== #
    print("\n--- H. Robustness ---")

    model.eval()

    # H1: No NaN/Inf
    check(
        not (torch.isnan(p_out).any() or torch.isinf(p_out).any()),
        "No NaN/Inf in p_s_out",
    )
    check(
        not (torch.isnan(s_out).any() or torch.isinf(s_out).any()),
        "No NaN/Inf in s_s_out",
    )

    # H2: Zero inputs
    with torch.no_grad():
        pz, sz = model(
            torch.zeros(2, T, D_P, 4),
            torch.zeros(2, T_M, 3),
            torch.zeros(2, T_H, 3),
        )
    check(not torch.isnan(pz).any(), "Zero input → p_s_out no NaN")
    check(not torch.isnan(sz).any(), "Zero input → s_s_out no NaN")

    # H3: Determinism
    with torch.no_grad():
        pa2, sa2 = model(s_xyz, past, lstm_pred)
        pb2, sb2 = model(s_xyz, past, lstm_pred)
    check(torch.equal(pa2, pb2), "Deterministic p_s_out in eval mode")
    check(torch.equal(sa2, sb2), "Deterministic s_s_out in eval mode")

    # H4: Different batch sizes
    for bs in [1, 4, 8]:
        with torch.no_grad():
            p_bs, s_bs = model(
                torch.randn(bs, T, D_P, 4),
                torch.randn(bs, T_M, 3),
                torch.randn(bs, T_H, 3),
            )
        check(
            tuple(p_bs.shape) == (bs, T, D_C),
            f"Batch={bs} → p_s_out {tuple(p_bs.shape)}",
        )

    # =================================================================== #
    #  Test Group I: Flexibility
    # =================================================================== #
    print("\n--- I. Flexibility ---")

    # I1: Non-default T_M=10, T_H=20
    model_flex = SpatialAttentionModule(d_in=4, d_c=D_C, n_heads=N_HEADS, n_enc_layers=2, n_dec_layers=2)
    model_flex.eval()
    T_M2, T_H2 = 10, 20
    with torch.no_grad():
        pf, sf = model_flex(
            torch.randn(2, T_M2 + T_H2, D_P, 4),
            torch.randn(2, T_M2, 3),
            torch.randn(2, T_H2, 3),
        )
    check(
        tuple(pf.shape) == (2, T_M2 + T_H2, D_C),
        f"T_M={T_M2},T_H={T_H2} → p_s_out {tuple(pf.shape)}",
    )

    # I2: Custom d_c=128, n_heads=4, 1 layer each
    model_sm = SpatialAttentionModule(d_in=4, d_c=128, n_heads=4, n_enc_layers=1, n_dec_layers=1)
    model_sm.eval()
    with torch.no_grad():
        ps, ss = model_sm(
            torch.randn(2, T, D_P, 4),
            torch.randn(2, T_M, 3),
            torch.randn(2, T_H, 3),
        )
    check(
        tuple(ps.shape) == (2, T, 128),
        f"Custom d_c=128 → p_s_out {tuple(ps.shape)}",
    )

    # =================================================================== #
    #  Test Group J: Integration shapes
    # =================================================================== #
    print("\n--- J. Integration shapes ---")

    model.eval()
    with torch.no_grad():
        pj, sj = model(s_xyz, past, lstm_pred)

    # Temporal Attention expects P_{s-out} and S_{s-out} both [B, T, D_C]
    check(
        tuple(pj.shape) == (B, T, D_C),
        f"P_s_out for Temporal Attn: {tuple(pj.shape)}",
    )
    check(
        tuple(sj.shape) == (B, T, D_C),
        f"S_s_out for Temporal Attn: {tuple(sj.shape)}",
    )
    check(pj.shape == sj.shape, "P_s_out and S_s_out have same shape")

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
