"""
=============================================================================
STAR-VP Gating Fusion Module — Second-Stage Fusion
=============================================================================

Paper:  Section 3.6, Equation 6
        Table 1 — D_G = 128

The Gating Fusion Module blends the LSTM short-term predictions P̂' with
the Temporal Attention long-term predictions P̂''.  It learns a SCALAR
gate weight per timestep so the model can automatically shift from
trusting trajectory inertia (early steps) to trusting content-based
saliency attraction (later steps).

Architecture (Eq. 6)
--------------------

    STEP A — Flatten and concatenate:

        x = Flatten(C_f(P̂', P̂''))                  [B, 6·T_H] = [B, 150]

    STEP B — Two-layer MLP → per-timestep scalar gate:

        hidden = σ_r(W_r · x + b_r)                 [B, D_G=128]
        W'     = σ_s(W_s · hidden + b_s)             [B, T_H=25]

        σ_r = ReLU,  σ_s = Sigmoid
        W_r ∈ R^{D_G × 6T_H} = [128, 150]
        W_s ∈ R^{T_H × D_G}  = [25, 128]

    STEP C — Complementary weight:

        W'' = 1 − W'                                 [B, T_H]

        W' + W'' = 1 at every timestep (convex combination guaranteed).

    STEP D — Weighted combination:

        P̂ = W' ⊗ P̂' ⊕ W'' ⊗ P̂''                   [B, T_H, 3]

        where ⊗ is element-wise multiplication (broadcast over 3 coords)
        and   ⊕ is element-wise addition.

    NOTE — The paper does NOT normalize the final output to the unit sphere.
    The output is a convex combination of two 3D vectors.  Downstream code
    (training/evaluation) may normalize to unit sphere for the Orthodromic
    Distance metric if desired.

Expected behavior (paper Figure 6):
    Step 0-1s  →  W' ≈ 1.0  (trust LSTM)
    Step 3-5s  →  W' ≈ 0.0  (trust Temporal Attention)

Hyperparameters from Table 1:
    T_H = 25    Future prediction horizon
    D_G = 128   Gating MLP hidden size
=============================================================================
"""

import torch
import torch.nn as nn


# ========================================================================= #
#  Gating Fusion Module  (Paper Section 3.6, Eq. 6)
# ========================================================================= #

class GatingFusionModule(nn.Module):
    """
    Blends LSTM short-term predictions with Temporal Attention long-term
    predictions using a learned per-timestep scalar gate.

    The gate weight W' ∈ [0, 1]^{T_H} determines the blend ratio at each
    prediction step.  W'' = 1 − W' is the complementary weight.
    """

    def __init__(
        self,
        t_h: int = 25,
        d_g: int = 128,
        input_dim: int = 3,
        # ---- MMVP2 extensions (defaults preserve original STAR-VP behaviour) ----
        d_audio: int = 0,
        d_face:  int = 0,
    ):
        """
        Args:
            t_h:        Future prediction horizon.    Table 1: T_H = 25.
            d_g:        Gating MLP hidden dimension.  Table 1: D_G = 128.
            input_dim:  Coordinate dimension (3 for xyz).

            d_audio:    Optional dim of an audio context vector that is
                        passed to the gate (mean-pooled across past timesteps).
                        If 0 (default) no audio is used and behaviour matches
                        the original STAR-VP gate.
            d_face:     Optional dim of a face context scalar/vector
                        (mean-pooled engagement features).  If 0 no face is
                        used.  Common choice: d_face=1 (single engagement
                        scalar), `d_face=5` for the raw blendshape vector.
        """
        super().__init__()
        self.t_h = t_h
        self.d_g = d_g
        self.input_dim = input_dim
        self.d_audio = d_audio
        self.d_face  = d_face

        flat_dim = t_h * input_dim * 2 + d_audio + d_face   # default 150

        self.gate = nn.Sequential(
            nn.Linear(flat_dim, d_g),
            nn.ReLU(),
            nn.Linear(d_g, t_h),
            nn.Sigmoid(),
        )

    def forward(
        self,
        p_lstm: torch.Tensor,
        p_temporal: torch.Tensor,
        audio_ctx: torch.Tensor | None = None,
        face_ctx:  torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            p_lstm:       [B, T_H, 3]  Short-term predictions from LSTM.
            p_temporal:   [B, T_H, 3]  Long-term predictions from Temporal Attention.
            audio_ctx:    [B, d_audio] or None.  Pooled audio context;
                          required when self.d_audio > 0.
            face_ctx:     [B, d_face] or None.  Pooled face engagement;
                          required when self.d_face > 0.
            return_weights:  If True, also return (w_prime, w_double_prime).

        Returns:
            p_hat:           [B, T_H, 3]  Final blended predictions.
            w_prime:         [B, T_H]     LSTM gate weight  (only if return_weights).
            w_double_prime:  [B, T_H]     Temporal gate weight (only if return_weights).
        """
        B = p_lstm.shape[0]

        # ============================================================= #
        #  STEP A: Flatten and concatenate  (Eq. 6, extended for MMVP2)
        # ============================================================= #
        x = torch.cat([p_lstm, p_temporal], dim=-1)   # [B, T_H, 6]
        x = x.reshape(B, -1)                          # [B, 6·T_H = 150]
        if self.d_audio > 0:
            if audio_ctx is None:
                raise ValueError("audio_ctx is required (d_audio > 0)")
            x = torch.cat([x, audio_ctx], dim=-1)     # [B, 150 + d_audio]
        if self.d_face > 0:
            if face_ctx is None:
                raise ValueError("face_ctx is required (d_face > 0)")
            x = torch.cat([x, face_ctx], dim=-1)      # [B, ... + d_face]

        # ============================================================= #
        #  STEP B: Two-layer MLP → per-timestep scalar gate  (Eq. 6)
        # ============================================================= #
        w_prime = self.gate(x)                         # [B, T_H]

        # ============================================================= #
        #  STEP C: Complementary weight  (Eq. 6)
        # ============================================================= #
        w_double_prime = 1.0 - w_prime                 # [B, T_H]

        # ============================================================= #
        #  STEP D: Weighted combination  (Eq. 6)
        # ============================================================= #
        # Unsqueeze to [B, T_H, 1] for broadcast over the 3 coordinates
        w_p = w_prime.unsqueeze(-1)                    # [B, T_H, 1]
        w_dp = w_double_prime.unsqueeze(-1)            # [B, T_H, 1]

        p_hat = w_p * p_lstm + w_dp * p_temporal       # [B, T_H, 3]

        if return_weights:
            return p_hat, w_prime, w_double_prime
        return p_hat

    def extra_repr(self) -> str:
        return (
            f"t_h={self.t_h}, d_g={self.d_g}, input_dim={self.input_dim}, "
            f"d_audio={self.d_audio}, d_face={self.d_face}"
        )


# ========================================================================= #
#  Self-Test — exhaustive verification
# ========================================================================= #

def _self_test() -> bool:
    """
    Exhaustive verification of GatingFusionModule for STAR-VP.

    Tests:
      A — Basic shapes and hyperparameters
        1.  Output shape [B, T_H, 3]
        2.  Default T_H=25, D_G=128
        3.  Flat input dim = 150

      B — Gate MLP architecture
        4.  First linear: 150 → 128
        5.  Activation: ReLU
        6.  Second linear: 128 → 25
        7.  Activation: Sigmoid

      C — Gate weight properties
        8.  W' ∈ [0, 1] (sigmoid output)
        9.  W'' ∈ [0, 1]
       10.  W' + W'' = 1 at every timestep and batch element
       11.  W' shape = [B, T_H]
       12.  W'' shape = [B, T_H]

      D — Convex combination properties
       13.  Output is between p_lstm and p_temporal (element-wise)
       14.  When W'→1: output ≈ p_lstm
       15.  When W'→0: output ≈ p_temporal
       16.  Identical inputs → output = input (regardless of weights)

      E — Gradient flow
       17.  All model parameters receive gradients
       18.  Gradient flows to p_lstm input
       19.  Gradient flows to p_temporal input

      F — Parameter counts
       20.  Linear1: 150*128 + 128 = 19,328
       21.  Linear2: 128*25 + 25 = 3,225
       22.  Total: 22,553

      G — Robustness
       23.  No NaN/Inf in output
       24.  Zero inputs → valid output
       25.  Large inputs → valid output (no NaN from sigmoid)
       26.  Deterministic in eval mode
       27.  Different batch sizes (1, 4, 8)

      H — Integration shapes
       28.  Accepts [B, 25, 3] from LSTM module
       29.  Accepts [B, 25, 3] from Temporal Attention module
       30.  Output [B, 25, 3] is the final STAR-VP prediction

      I — return_weights flag
       31.  return_weights=False → single tensor
       32.  return_weights=True → tuple of 3 tensors
       33.  Returned weights shapes are correct
    """
    torch.manual_seed(42)

    T_H = 25
    D_G = 128
    INPUT_DIM = 3
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
    print("STAR-VP Gating Fusion Module Self-Test")
    print("=" * 72)

    model = GatingFusionModule(t_h=T_H, d_g=D_G, input_dim=INPUT_DIM)
    model.eval()

    p_lstm = torch.randn(B, T_H, INPUT_DIM)
    p_temp = torch.randn(B, T_H, INPUT_DIM)

    with torch.no_grad():
        out = model(p_lstm, p_temp)

    # =================================================================== #
    #  Test Group A: Basic shapes and hyperparameters
    # =================================================================== #
    print("\n--- A. Basic shapes ---")

    check(
        tuple(out.shape) == (B, T_H, INPUT_DIM),
        f"Output shape {tuple(out.shape)} == ({B}, {T_H}, {INPUT_DIM})",
    )
    check(model.t_h == T_H and model.d_g == D_G, f"T_H={model.t_h}, D_G={model.d_g}")

    flat_dim = T_H * INPUT_DIM * 2   # 150
    check(
        model.gate[0].in_features == flat_dim,
        f"Flat dim: {model.gate[0].in_features} == {flat_dim}",
    )

    # =================================================================== #
    #  Test Group B: Gate MLP architecture
    # =================================================================== #
    print("\n--- B. Gate MLP architecture ---")

    check(
        model.gate[0].in_features == flat_dim and model.gate[0].out_features == D_G,
        f"Linear1: {model.gate[0].in_features}→{model.gate[0].out_features}",
    )
    check(isinstance(model.gate[1], nn.ReLU), f"Activation1: {type(model.gate[1]).__name__}")
    check(
        model.gate[2].in_features == D_G and model.gate[2].out_features == T_H,
        f"Linear2: {model.gate[2].in_features}→{model.gate[2].out_features}",
    )
    check(isinstance(model.gate[3], nn.Sigmoid), f"Activation2: {type(model.gate[3]).__name__}")

    # =================================================================== #
    #  Test Group C: Gate weight properties
    # =================================================================== #
    print("\n--- C. Gate weight properties ---")

    with torch.no_grad():
        _, w_p, w_dp = model(p_lstm, p_temp, return_weights=True)

    check(
        (w_p >= 0.0).all() and (w_p <= 1.0).all(),
        f"W' ∈ [0,1]: min={w_p.min():.6f}, max={w_p.max():.6f}",
    )
    check(
        (w_dp >= 0.0).all() and (w_dp <= 1.0).all(),
        f"W'' ∈ [0,1]: min={w_dp.min():.6f}, max={w_dp.max():.6f}",
    )
    check(
        torch.allclose(w_p + w_dp, torch.ones_like(w_p), atol=1e-6),
        "W' + W'' = 1 everywhere",
    )
    check(tuple(w_p.shape) == (B, T_H), f"W' shape: {tuple(w_p.shape)}")
    check(tuple(w_dp.shape) == (B, T_H), f"W'' shape: {tuple(w_dp.shape)}")

    # =================================================================== #
    #  Test Group D: Convex combination properties
    # =================================================================== #
    print("\n--- D. Convex combination properties ---")

    with torch.no_grad():
        out_d, w_p_d, _ = model(p_lstm, p_temp, return_weights=True)

    # For each element: out should lie between p_lstm and p_temp in the
    # sense that it's a weighted average.  Verify the algebraic identity:
    #   out = W' * p_lstm + (1-W') * p_temporal
    w_expanded = w_p_d.unsqueeze(-1)
    expected = w_expanded * p_lstm + (1.0 - w_expanded) * p_temp
    check(
        torch.allclose(out_d, expected, atol=1e-5),
        "Output matches W'·P̂' + W''·P̂'' algebraically",
    )

    # When W' → 1: output ≈ p_lstm.  Force gate bias so W' → 1.
    model_hi = GatingFusionModule(t_h=T_H, d_g=D_G)
    with torch.no_grad():
        model_hi.gate[2].bias.fill_(10.0)   # sigmoid(10) ≈ 1
        model_hi.gate[0].weight.zero_()
        model_hi.gate[0].bias.zero_()
        model_hi.gate[2].weight.zero_()
        out_hi = model_hi(p_lstm, p_temp)
    # W' ≈ sigmoid(10) ≈ 0.99995 → output ≈ p_lstm
    check(
        torch.allclose(out_hi, p_lstm, atol=1e-3),
        "W'→1: output ≈ p_lstm",
    )

    # When W' → 0: output ≈ p_temporal
    model_lo = GatingFusionModule(t_h=T_H, d_g=D_G)
    with torch.no_grad():
        model_lo.gate[2].bias.fill_(-10.0)  # sigmoid(-10) ≈ 0
        model_lo.gate[0].weight.zero_()
        model_lo.gate[0].bias.zero_()
        model_lo.gate[2].weight.zero_()
        out_lo = model_lo(p_lstm, p_temp)
    check(
        torch.allclose(out_lo, p_temp, atol=1e-3),
        "W'→0: output ≈ p_temporal",
    )

    # Identical inputs → output = input (regardless of weights)
    p_same = torch.randn(B, T_H, INPUT_DIM)
    with torch.no_grad():
        out_same = model(p_same, p_same)
    check(
        torch.allclose(out_same, p_same, atol=1e-5),
        "Identical inputs → output = input",
    )

    # =================================================================== #
    #  Test Group E: Gradient flow
    # =================================================================== #
    print("\n--- E. Gradient flow ---")

    model.train()
    model.zero_grad()

    p_lg = torch.randn(B, T_H, INPUT_DIM, requires_grad=True)
    p_tg = torch.randn(B, T_H, INPUT_DIM, requires_grad=True)
    out_g = model(p_lg, p_tg)
    out_g.sum().backward()

    all_grads = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    check(all_grads, "All model parameters receive gradients")

    check(
        p_lg.grad is not None and p_lg.grad.abs().sum() > 0,
        "Gradient flows to p_lstm input",
    )
    check(
        p_tg.grad is not None and p_tg.grad.abs().sum() > 0,
        "Gradient flows to p_temporal input",
    )

    # =================================================================== #
    #  Test Group F: Parameter counts
    # =================================================================== #
    print("\n--- F. Parameter counts ---")

    n_linear1 = flat_dim * D_G + D_G     # 150*128 + 128 = 19328
    n_linear2 = D_G * T_H + T_H          # 128*25 + 25 = 3225
    n_total = n_linear1 + n_linear2       # 22553

    actual_l1 = sum(p.numel() for p in model.gate[0].parameters())
    actual_l2 = sum(p.numel() for p in model.gate[2].parameters())
    actual_total = sum(p.numel() for p in model.parameters())

    check(actual_l1 == n_linear1, f"Linear1 params: {actual_l1:,} == {n_linear1:,}")
    check(actual_l2 == n_linear2, f"Linear2 params: {actual_l2:,} == {n_linear2:,}")
    check(actual_total == n_total, f"Total params: {actual_total:,} == {n_total:,}")
    print(f"       Linear1:  {actual_l1:,}")
    print(f"       Linear2:  {actual_l2:,}")
    print(f"       Total:    {actual_total:,}")

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
        out_z = model(torch.zeros(2, T_H, 3), torch.zeros(2, T_H, 3))
    check(not torch.isnan(out_z).any(), "Zero inputs → no NaN")

    with torch.no_grad():
        out_big = model(
            torch.full((2, T_H, 3), 1e4),
            torch.full((2, T_H, 3), -1e4),
        )
    check(
        not (torch.isnan(out_big).any() or torch.isinf(out_big).any()),
        "Large inputs → no NaN/Inf (sigmoid saturation safe)",
    )

    with torch.no_grad():
        det_a = model(p_lstm, p_temp)
        det_b = model(p_lstm, p_temp)
    check(torch.equal(det_a, det_b), "Deterministic in eval mode")

    for bs in [1, 4, 8]:
        with torch.no_grad():
            out_bs = model(torch.randn(bs, T_H, 3), torch.randn(bs, T_H, 3))
        check(
            tuple(out_bs.shape) == (bs, T_H, 3),
            f"Batch={bs} → shape {tuple(out_bs.shape)}",
        )

    # =================================================================== #
    #  Test Group H: Integration shapes
    # =================================================================== #
    print("\n--- H. Integration shapes ---")

    lstm_out = torch.randn(B, T_H, 3)
    temp_out = torch.randn(B, T_H, 3)
    with torch.no_grad():
        final = model(lstm_out, temp_out)

    check(
        tuple(lstm_out.shape) == (B, T_H, 3),
        f"Accepts LSTM output: {tuple(lstm_out.shape)}",
    )
    check(
        tuple(temp_out.shape) == (B, T_H, 3),
        f"Accepts Temporal Attn output: {tuple(temp_out.shape)}",
    )
    check(
        tuple(final.shape) == (B, T_H, 3),
        f"Final prediction: {tuple(final.shape)} (STAR-VP output)",
    )

    # =================================================================== #
    #  Test Group I: return_weights flag
    # =================================================================== #
    print("\n--- I. return_weights flag ---")

    with torch.no_grad():
        result_no = model(p_lstm, p_temp, return_weights=False)
        result_yes = model(p_lstm, p_temp, return_weights=True)

    check(isinstance(result_no, torch.Tensor), "return_weights=False → tensor")
    check(isinstance(result_yes, tuple) and len(result_yes) == 3, "return_weights=True → 3-tuple")

    p_hat_y, wp_y, wdp_y = result_yes
    check(tuple(wp_y.shape) == (B, T_H), f"W' shape: {tuple(wp_y.shape)}")
    check(tuple(wdp_y.shape) == (B, T_H), f"W'' shape: {tuple(wdp_y.shape)}")

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
