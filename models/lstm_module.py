"""
=============================================================================
STAR-VP LSTM Module — Autoregressive Encoder-Decoder for Short-term Prediction
=============================================================================

Paper:  Section 3.3, Equation 2
        Table 1 — D_cL=256, N_layers_L=2
        Section 4.1.2 — deep-pos-only baseline description

The LSTM module predicts the future T_H viewpoint positions given only the
past T_M viewpoint positions.  It uses an encoder-decoder architecture with
autoregressive decoding (the prediction at each step is fed back as input
for the next step).

Architecture
------------
    Input:   P̄  ∈ R^{B × T_M × 3}    past head positions (unit sphere xyz)
    Output:  P̂' ∈ R^{B × T_H × 3}    predicted future positions

    Encoding phase:
        Feed all T_M past positions through the LSTM at once.
        This encodes the trajectory history into the LSTM hidden state (h, c).

    Decoding phase (autoregressive):
        Starting from the last observed position P̄[:, -1, :], repeatedly:
            1. Feed the current input through the LSTM (single step)
            2. Project the LSTM output (256-d) to 3D via a linear layer
            3. Use that 3D prediction as input for the next step
        This produces T_H predictions, each depending on all previous ones.

Paper description (Section 3.3):
    "we use LSTM, a lightweight and efficient temporal prediction model,
     to take only the viewpoint positions from the past T_M time steps as
     input and output predictions for the future T_H time steps in an
     autoregressive manner"

Hyperparameters from Table 1:
    D_cL        = 256   Channel dimension of the hidden state
    N_layers_L  = 2     Number of LSTM layers

Downstream usage of P̂':
    1. Spatial Attention Module (Section 3.4):
       P̂' is concatenated with P̄ along time → [B, T_M+T_H, 3]
       Used as the viewport position input for spatial alignment.

    2. Gating Fusion Module (Section 3.6):
       P̂' is blended with P̂'' (temporal attention output) using
       learned scalar weights:  P̂ = W'⊗P̂' + W''⊗P̂''
=============================================================================
"""

import torch
import torch.nn as nn


# ========================================================================= #
#  LSTM Module  (Paper Section 3.3, Eq. 2)
# ========================================================================= #

class LSTMModule(nn.Module):
    """
    Autoregressive LSTM encoder-decoder for viewpoint trajectory prediction.

    Encoding:  Processes the full past trajectory (T_M steps of 3D positions)
               in a single forward pass, capturing the hidden state.
    Decoding:  Generates T_H future predictions step-by-step, feeding each
               prediction back as input for the next (autoregressive).

    The LSTM input_size is 3 (raw xyz on the unit sphere) and hidden_size is
    D_cL (256 by default).  A single linear layer projects the 256-d LSTM
    output back to 3D coordinates at each decoding step.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 2,
        t_m: int = 15,
        t_h: int = 25,
        dropout: float = 0.0,
        # ---- MMVP2 extensions (defaults preserve original STAR-VP behaviour) ----
        head_dim: int = 3,
        d_user: int = 0,
    ):
        """
        Args:
            input_dim:   Total per-step input dim (head + eye + offset + fix...).
                         Defaults to 3, matching the original STAR-VP behaviour.
            hidden_dim:  LSTM hidden state size.  Paper Table 1: D_cL = 256.
            num_layers:  Number of stacked LSTM layers.  Paper: N_layers_L = 2.
            t_m:         Number of past (memory) timesteps.  Paper: T_M = 15.
            t_h:         Number of future (horizon) timesteps.  Paper: T_H = 25.
            dropout:     Dropout between LSTM layers (only active when
                         num_layers > 1 and model.training is True).
                         Paper does not specify; defaults to 0.

            head_dim:    The number of leading channels of `input_dim` that
                         correspond to the predicted head position (3 = xyz).
                         Used during autoregressive decoding: only those
                         channels are replaced by the model's prediction; the
                         remaining `input_dim - head_dim` channels are filled
                         from the **last observed** value (no future leakage).
                         When input_dim == 3 this is just the original
                         STAR-VP loop.
            d_user:      If > 0, the LSTM hidden / cell states are
                         initialised from a user embedding rather than zeros.
                         Use 0 to keep the STAR-VP behaviour unchanged.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.t_m = t_m
        self.t_h = t_h
        self.head_dim = head_dim
        self.d_user = d_user

        if head_dim > input_dim:
            raise ValueError(f"head_dim={head_dim} cannot exceed input_dim={input_dim}")

        # Core LSTM — shared between encoding and decoding phases
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection: hidden_dim → head_dim  (256 → 3)
        self.output_proj = nn.Linear(hidden_dim, head_dim)

        # Optional user-conditioned initial hidden / cell state.
        if d_user > 0:
            self.user_to_hc = nn.Linear(d_user, 2 * num_layers * hidden_dim)
            nn.init.normal_(self.user_to_hc.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.user_to_hc.bias)
        else:
            self.user_to_hc = None

    # --------------------------------------------------------------------- #
    #  Forward
    # --------------------------------------------------------------------- #

    def forward(
        self,
        past_positions: torch.Tensor,
        user_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode past trajectory, then autoregressively decode future positions.

        Args:
            past_positions: [B, T_M, input_dim]
                When input_dim == 3 (default) this is just the head trajectory,
                identical to the STAR-VP baseline.
                When input_dim > 3 (MMVP2), the leading `head_dim` channels
                are the head xyz and the trailing channels carry auxiliary
                signals (eye, offset, fixation flag, dwell duration, ...).
                T_M can differ from self.t_m at runtime (the module adapts).

            user_emb: [B, d_user] or None
                Optional user embedding used to initialise (h_0, c_0).
                Only used when self.user_to_hc is present (i.e. constructed
                with d_user > 0).  If d_user > 0 but user_emb is None, h_0/c_0
                fall back to zeros (the STAR-VP behaviour).

        Returns:
            predictions: [B, T_H, head_dim]
                Predicted future head positions (xyz).
        """
        B = past_positions.shape[0]

        # ============================================================= #
        #  Initial hidden / cell state
        # ============================================================= #
        if self.user_to_hc is not None and user_emb is not None:
            hc = self.user_to_hc(user_emb)                              # [B, 2*L*H]
            hc = hc.view(B, 2, self.num_layers, self.hidden_dim)
            h0 = hc[:, 0].permute(1, 0, 2).contiguous()                 # [L, B, H]
            c0 = hc[:, 1].permute(1, 0, 2).contiguous()
        else:
            h0 = past_positions.new_zeros(self.num_layers, B, self.hidden_dim)
            c0 = past_positions.new_zeros(self.num_layers, B, self.hidden_dim)

        # ============================================================= #
        #  Encoding Phase
        # ============================================================= #
        _, (h, c) = self.lstm(past_positions, (h0, c0))

        # ============================================================= #
        #  Decoding Phase — autoregressive, T_H steps
        # ============================================================= #
        # First decoder input = last observed step (full input_dim).
        last_step = past_positions[:, -1:, :]                           # [B, 1, input_dim]
        # Auxiliary tail (eye, offset, fixation, ...) is held constant during
        # decoding -- this avoids using future values we don't have at inference.
        aux_tail = (
            last_step[:, :, self.head_dim :]
            if self.input_dim > self.head_dim
            else None
        )                                                                # [B, 1, input_dim - head_dim] or None
        decoder_input = last_step

        predictions: list[torch.Tensor] = []
        for _ in range(self.t_h):
            lstm_out, (h, c) = self.lstm(decoder_input, (h, c))         # [B, 1, hidden_dim]
            pred_head = self.output_proj(lstm_out.squeeze(1))            # [B, head_dim]
            predictions.append(pred_head)

            if aux_tail is None:
                decoder_input = pred_head.unsqueeze(1)                   # [B, 1, head_dim]
            else:
                decoder_input = torch.cat(
                    [pred_head.unsqueeze(1), aux_tail], dim=-1
                )                                                        # [B, 1, input_dim]

        return torch.stack(predictions, dim=1)                           # [B, T_H, head_dim]

    # --------------------------------------------------------------------- #
    #  Repr
    # --------------------------------------------------------------------- #

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, t_m={self.t_m}, t_h={self.t_h}"
        )


# ========================================================================= #
#  Self-Test — exhaustive verification
# ========================================================================= #

def _self_test() -> bool:
    """
    Exhaustive verification of LSTMModule for STAR-VP.

    Tests:
      A — Basic shapes and defaults
        1.  Output shape [B, T_H, 3] with default params
        2.  Default hyperparameters match paper Table 1
        3.  Different batch sizes (1, 4, 16)
        4.  Output dtype matches input dtype

      B — Autoregressive behaviour
        5.  Changing last past position changes first prediction
        6.  Each prediction depends on the previous (gradient chain)
        7.  Encoding uses full history (changing early position changes output)
        8.  Decoder predictions form a causal chain

      C — Gradient flow
        9.  Gradients reach all LSTM parameters
       10.  Gradients reach output_proj parameters
       11.  Gradients flow back to the input tensor

      D — Parameter counts
       12.  Total parameter count verification
       13.  LSTM parameter breakdown
       14.  Output projection parameter count

      E — Robustness
       15.  No NaN/Inf in output
       16.  Zero input → valid output (no NaN)
       17.  Large-magnitude input → valid output (no NaN)
       18.  Determinism: same input → identical output (eval mode)
       19.  Training vs eval mode differ when dropout > 0

      F — Flexibility
       20.  Custom T_M at runtime (not just default 15)
       21.  Custom hidden_dim and num_layers
       22.  Single-step horizon (T_H=1)
       23.  Large horizon (T_H=50)

      G — Integration shapes
       24.  Output concat with past → [B, T_M+T_H, 3] for Spatial Attention
       25.  Output usable by Gating Fusion (just shape check)
    """
    torch.manual_seed(42)

    T_M = 15
    T_H = 25
    D_cL = 256
    N_LAYERS = 2

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
    print("STAR-VP LSTM Module Self-Test")
    print("=" * 72)

    model = LSTMModule(
        input_dim=3, hidden_dim=D_cL, num_layers=N_LAYERS,
        t_m=T_M, t_h=T_H,
    )
    model.eval()

    # =================================================================== #
    #  Test Group A: Basic shapes and defaults
    # =================================================================== #
    print("\n--- A. Basic shapes and defaults ---")

    B = 4
    past = torch.randn(B, T_M, 3)
    with torch.no_grad():
        pred = model(past)

    # A1: Output shape
    check(
        tuple(pred.shape) == (B, T_H, 3),
        f"Output shape {tuple(pred.shape)} == ({B}, {T_H}, 3)",
    )

    # A2: Default hyperparameters
    check(model.hidden_dim == 256, f"hidden_dim={model.hidden_dim} == 256 (D_cL)")
    check(model.num_layers == 2, f"num_layers={model.num_layers} == 2 (N_layers_L)")
    check(model.input_dim == 3, f"input_dim={model.input_dim} == 3")
    check(model.t_h == 25, f"t_h={model.t_h} == 25 (T_H)")
    check(model.t_m == 15, f"t_m={model.t_m} == 15 (T_M)")

    # A3: Different batch sizes
    for bs in [1, 4, 16]:
        with torch.no_grad():
            out = model(torch.randn(bs, T_M, 3))
        check(
            tuple(out.shape) == (bs, T_H, 3),
            f"Batch={bs} → shape {tuple(out.shape)}",
        )

    # A4: Dtype preservation
    check(pred.dtype == past.dtype, f"Output dtype {pred.dtype} == input dtype {past.dtype}")

    # =================================================================== #
    #  Test Group B: Autoregressive behaviour
    # =================================================================== #
    print("\n--- B. Autoregressive behaviour ---")

    model.train()

    # B1: Changing last past position changes first prediction
    torch.manual_seed(10)
    past_a = torch.randn(2, T_M, 3)
    past_b = past_a.clone()
    past_b[:, -1, :] += 0.5  # perturb last position

    model.eval()
    with torch.no_grad():
        pred_a = model(past_a)
        pred_b = model(past_b)
    check(
        not torch.allclose(pred_a[:, 0, :], pred_b[:, 0, :], atol=1e-6),
        "Perturbing last past position changes 1st prediction",
    )

    # B2: Autoregressive gradient chain — later predictions depend on earlier
    model.train()
    past_grad = torch.randn(2, T_M, 3, requires_grad=True)
    preds = model(past_grad)
    # The last prediction should depend on the first prediction's path
    # through the autoregressive chain.  We check by taking gradient of
    # the last prediction w.r.t. past (which flows through all intermediate
    # predictions during decoding).
    loss_last = preds[:, -1, :].sum()
    loss_last.backward()
    check(
        past_grad.grad is not None and past_grad.grad.abs().sum() > 0,
        "Last prediction gradient reaches input (autoregressive chain)",
    )

    # B3: Encoding uses full history — changing first position changes output
    model.eval()
    past_c = past_a.clone()
    past_c[:, 0, :] += 0.5  # perturb FIRST position
    with torch.no_grad():
        pred_c = model(past_c)
    check(
        not torch.allclose(pred_a, pred_c, atol=1e-6),
        "Perturbing first past position changes predictions (full encoding)",
    )

    # B4: Causal chain — prediction t+1 differs from prediction t
    # (if decoder were non-autoregressive, all steps could be identical)
    model.eval()
    with torch.no_grad():
        pred_d = model(torch.randn(2, T_M, 3))
    # Check that not all timesteps are identical
    timesteps_differ = not torch.allclose(pred_d[:, 0, :], pred_d[:, -1, :], atol=1e-6)
    check(timesteps_differ, "Predictions at different timesteps differ (causal chain)")

    # =================================================================== #
    #  Test Group C: Gradient flow
    # =================================================================== #
    print("\n--- C. Gradient flow ---")

    model.train()
    model.zero_grad()

    past_g = torch.randn(2, T_M, 3, requires_grad=True)
    out_g = model(past_g)
    loss_g = out_g.sum()
    loss_g.backward()

    # C1: LSTM parameters have gradients
    lstm_grad_ok = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.lstm.parameters()
    )
    check(lstm_grad_ok, "Gradients reach all LSTM parameters")

    # C2: output_proj has gradients
    proj_grad_ok = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.output_proj.parameters()
    )
    check(proj_grad_ok, "Gradients reach output_proj parameters")

    # C3: Gradient flows back to input
    check(
        past_g.grad is not None and past_g.grad.abs().sum() > 0,
        "Gradient flows back to input tensor",
    )

    # =================================================================== #
    #  Test Group D: Parameter counts
    # =================================================================== #
    print("\n--- D. Parameter counts ---")

    n_total = sum(p.numel() for p in model.parameters())

    # LSTM parameter count for multi-layer LSTM:
    # Layer 0: W_ih [4*H, I] + b_ih [4*H] + W_hh [4*H, H] + b_hh [4*H]
    #        = 4*H*I + 4*H + 4*H*H + 4*H
    # Layer k (k>0): W_ih [4*H, H] + b_ih [4*H] + W_hh [4*H, H] + b_hh [4*H]
    #        = 4*H*H + 4*H + 4*H*H + 4*H
    I, H_dim = 3, 256
    layer0 = 4 * H_dim * I + 4 * H_dim + 4 * H_dim * H_dim + 4 * H_dim
    layer1 = 4 * H_dim * H_dim + 4 * H_dim + 4 * H_dim * H_dim + 4 * H_dim
    expected_lstm = layer0 + layer1
    # output_proj: H_dim * I + I = 256*3 + 3 = 771
    expected_proj = H_dim * I + I
    expected_total = expected_lstm + expected_proj

    n_lstm = sum(p.numel() for p in model.lstm.parameters())
    n_proj = sum(p.numel() for p in model.output_proj.parameters())

    # D1
    check(n_total == expected_total, f"Total params: {n_total:,} == {expected_total:,}")
    # D2
    check(n_lstm == expected_lstm, f"LSTM params: {n_lstm:,} == {expected_lstm:,}")
    print(f"       Layer 0: {layer0:,}  (input_size=3 → hidden=256)")
    print(f"       Layer 1: {layer1:,}  (hidden=256 → hidden=256)")
    # D3
    check(n_proj == expected_proj, f"output_proj params: {n_proj:,} == {expected_proj:,}")
    print(f"       Total model: {n_total:,} parameters")

    # =================================================================== #
    #  Test Group E: Robustness
    # =================================================================== #
    print("\n--- E. Robustness ---")

    model.eval()

    # E1: No NaN/Inf
    with torch.no_grad():
        out_e = model(torch.randn(4, T_M, 3))
    check(
        not (torch.isnan(out_e).any() or torch.isinf(out_e).any()),
        "No NaN/Inf in output",
    )

    # E2: Zero input
    with torch.no_grad():
        out_zero = model(torch.zeros(2, T_M, 3))
    check(
        not torch.isnan(out_zero).any(),
        f"Zero input → no NaN (output range [{out_zero.min():.4f}, {out_zero.max():.4f}])",
    )

    # E3: Large-magnitude input
    with torch.no_grad():
        out_large = model(torch.randn(2, T_M, 3) * 100)
    check(
        not torch.isnan(out_large).any(),
        f"Large input → no NaN (output range [{out_large.min():.4f}, {out_large.max():.4f}])",
    )

    # E4: Determinism (eval mode, same input)
    torch.manual_seed(0)
    test_input = torch.randn(2, T_M, 3)
    with torch.no_grad():
        det_a = model(test_input)
        det_b = model(test_input)
    check(
        torch.equal(det_a, det_b),
        "Deterministic in eval mode",
    )

    # E5: Training vs eval differ with dropout
    model_drop = LSTMModule(
        input_dim=3, hidden_dim=D_cL, num_layers=N_LAYERS,
        t_m=T_M, t_h=T_H, dropout=0.5,
    )
    test_in = torch.randn(4, T_M, 3)

    model_drop.eval()
    with torch.no_grad():
        eval_out1 = model_drop(test_in)
        eval_out2 = model_drop(test_in)
    eval_match = torch.equal(eval_out1, eval_out2)

    model_drop.train()
    with torch.no_grad():
        train_out1 = model_drop(test_in)
        train_out2 = model_drop(test_in)
    train_match = torch.equal(train_out1, train_out2)

    check(eval_match, "Eval mode with dropout: deterministic")
    check(not train_match, "Train mode with dropout: stochastic")

    # =================================================================== #
    #  Test Group F: Flexibility
    # =================================================================== #
    print("\n--- F. Flexibility ---")

    model.eval()

    # F1: Runtime T_M differs from constructor default
    with torch.no_grad():
        out_tm10 = model(torch.randn(2, 10, 3))  # T_M=10 instead of 15
    check(
        tuple(out_tm10.shape) == (2, T_H, 3),
        f"Runtime T_M=10 → shape {tuple(out_tm10.shape)}",
    )

    with torch.no_grad():
        out_tm30 = model(torch.randn(2, 30, 3))  # T_M=30
    check(
        tuple(out_tm30.shape) == (2, T_H, 3),
        f"Runtime T_M=30 → shape {tuple(out_tm30.shape)}",
    )

    # F1b: Constructor t_m=25 (fair MFTR/MUSE alignment)
    model_tm25 = LSTMModule(
        input_dim=3, hidden_dim=D_cL, num_layers=N_LAYERS,
        t_m=25, t_h=25,
    )
    model_tm25.eval()
    with torch.no_grad():
        out_25 = model_tm25(torch.randn(2, 25, 3))
    check(
        tuple(out_25.shape) == (2, 25, 3),
        f"Constructor t_m=t_h=25 → shape {tuple(out_25.shape)}",
    )

    # F2: Custom hidden_dim and num_layers
    model_custom = LSTMModule(input_dim=3, hidden_dim=128, num_layers=1, t_m=10, t_h=5)
    model_custom.eval()
    with torch.no_grad():
        out_custom = model_custom(torch.randn(2, 10, 3))
    check(
        tuple(out_custom.shape) == (2, 5, 3),
        f"Custom (H=128, L=1, T_H=5) → shape {tuple(out_custom.shape)}",
    )

    # F3: Single-step horizon
    model_1step = LSTMModule(input_dim=3, hidden_dim=64, num_layers=1, t_m=5, t_h=1)
    model_1step.eval()
    with torch.no_grad():
        out_1 = model_1step(torch.randn(2, 5, 3))
    check(
        tuple(out_1.shape) == (2, 1, 3),
        f"T_H=1 → shape {tuple(out_1.shape)}",
    )

    # F4: Large horizon
    model_large = LSTMModule(input_dim=3, hidden_dim=64, num_layers=1, t_m=5, t_h=50)
    model_large.eval()
    with torch.no_grad():
        out_50 = model_large(torch.randn(2, 5, 3))
    check(
        tuple(out_50.shape) == (2, 50, 3),
        f"T_H=50 → shape {tuple(out_50.shape)}",
    )
    check(not torch.isnan(out_50).any(), "T_H=50: no NaN (long autoregressive chain)")

    # =================================================================== #
    #  Test Group G: Integration shapes (downstream compatibility)
    # =================================================================== #
    print("\n--- G. Integration shapes ---")

    model.eval()
    past_int = torch.randn(4, T_M, 3)
    with torch.no_grad():
        pred_int = model(past_int)

    # G1: Concat for Spatial Attention — [B, T_M+T_H, 3]
    p_concat = torch.cat([past_int, pred_int], dim=1)
    check(
        tuple(p_concat.shape) == (4, T_M + T_H, 3),
        f"Past+Pred concat → {tuple(p_concat.shape)} == (4, {T_M+T_H}, 3)",
    )

    # G2: Gating Fusion compatibility — P̂' has [B, T_H, 3]
    check(
        pred_int.shape[1] == T_H and pred_int.shape[2] == 3,
        f"Gating Fusion compatible: [{pred_int.shape[1]}, {pred_int.shape[2]}] == [{T_H}, 3]",
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
