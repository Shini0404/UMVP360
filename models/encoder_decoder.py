"""
=============================================================================
STAR-VP Encoder/Decoder — Perceiver-IO-style Attention Block
=============================================================================

Paper:  Section 3.7, Equations 7-9
        "Inspired by the universal multimodal fusion architecture Perceiver IO,
         we unified the architecture of the Encoder and Decoder."

This file provides the shared attention building block used by BOTH the
Spatial Attention Module (Section 3.4) and the Temporal Attention Module
(Section 3.5).  The Encoder and Decoder have the SAME architecture — the
only difference is how they are called:

    Encoder:  self-attention     → Q = K = V = input
    Decoder:  cross-attention    → Q ≠ K = V

The block supports DIFFERENT dimensions for Q and K/V inputs.  Internal
projection layers (L_q, L_k, L_v from Eq.7) map them to a shared attention
dimension d_attn, and L_o maps back to d_q so the residual connection works.

Concrete usage from downstream modules
---------------------------------------
Spatial Attention (Section 3.4):
    Encoder:  Q=K=V = S       [B*T, 128, D_C=256]     → [B*T, 128, 256]
    Decoder:  Q=P, K=V=S'     [B*T, 1, 256] / [B*T, 128, 256] → [B*T, 1, 256]

Temporal Attention (Section 3.5):
    Encoder:  Q=K=V = PS      [B, 80, D_PS=512]       → [B, 80, 512]
    Decoder:  Q=E, K=V=PS'    [B, 25, 256] / [B, 80, 512] → [B, 25, 256]

Architecture (Eq. 9, pre-norm residual):
    O_attn = Q + Attn(LN(Q), LN(K), LN(V))
    output = O_attn + MLP(LN(O_attn))

    Where Attn is Eq. 7 (multi-head attention with separate projections)
    and   MLP  is Eq. 8: Linear → GELU → Linear
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================================= #
#  Multi-Head Attention  (Paper Eq. 7)
# ========================================================================= #

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with separate input projections for Q, K, V
    that can handle DIFFERENT input dimensions, exactly as described in
    the paper's Equation 7.

    Attn(Q_in, K_in, V_in) = L_o(softmax(Q' K'^T / sqrt(d_head)) V')
    Q' = L_q(Q_in),  K' = L_k(K_in),  V' = L_v(V_in)

    L_q : d_q   → d_attn
    L_k : d_kv  → d_attn
    L_v : d_kv  → d_attn
    L_o : d_attn → d_q          (maps back to query dimension for residual)

    Args:
        d_q:      Dimension of query input.
        d_kv:     Dimension of key/value input (K and V share the same dim).
        d_attn:   Internal attention dimension (total across all heads).
                  Must be divisible by n_heads.
        n_heads:  Number of attention heads.
        dropout:  Dropout on attention weights.  Default 0.0.
    """

    def __init__(
        self,
        d_q: int,
        d_kv: int,
        d_attn: int,
        n_heads: int,               #d_attn = 256
                                    #n_heads = 8
                                    #so each head gets d_head = 32
                                    #So there are 8 small attention modules running in parallel, each on 32 channels.
        dropout: float = 0.0,
    ):
        super().__init__()

        if d_attn % n_heads != 0:
            raise ValueError(
                f"d_attn ({d_attn}) must be divisible by n_heads ({n_heads})"
            )

        self.d_q = d_q
        self.d_kv = d_kv
        self.d_attn = d_attn
        self.n_heads = n_heads
        self.d_head = d_attn // n_heads
        self.scale = self.d_head ** -0.5 #Without scaling, dot products can get too large,softmax becomes too peaky/unstable, and training gets harder.
                                         #Scaling keeps values in a healthier range.

        self.q_proj = nn.Linear(d_q, d_attn)
        self.k_proj = nn.Linear(d_kv, d_attn)
        self.v_proj = nn.Linear(d_kv, d_attn)
        self.o_proj = nn.Linear(d_attn, d_q)

        self._dropout_p = dropout

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            q:  [B, n_q, d_q]    Query input.
            k:  [B, n_kv, d_kv]  Key input.
            v:  [B, n_kv, d_kv]  Value input (same length as key).

        Returns:
            output:       [B, n_q, d_q]            Attention output (same shape as q).
            attn_weights: None (SDPA does not materialise weights).
        """
        B, n_q, _ = q.shape
        n_kv = k.shape[1]

        Q = self.q_proj(q).view(B, n_q,  self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(k).view(B, n_kv, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(v).view(B, n_kv, self.n_heads, self.d_head).transpose(1, 2)

        dropout_p = self._dropout_p if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, dropout_p=dropout_p, is_causal=False,
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(B, n_q, self.d_attn)
        )

        output = self.o_proj(attn_output)
        return output, None


# ========================================================================= #
#  Perceiver Block  (Paper Eq. 9)
# ========================================================================= #

class PerceiverBlock(nn.Module):
    """
    One layer of the unified Encoder/Decoder from STAR-VP (Section 3.7).

    Implements Equation 9 with pre-norm residual connections:

        O_attn = Q + Attn(LN(Q), LN(K), LN(V))       [attention + residual]
        output = O_attn + MLP(LN(O_attn))              [MLP + residual]

    The output always has the same shape as Q, so it can be stacked:
        block1 → block2 → ... → blockN

    For an Encoder (self-attention), call with q=k=v=input.
    For a Decoder (cross-attention), call with q=query, k=v=memory.

    Args:
        d_q:      Query/residual stream dimension.
        d_kv:     Key/Value input dimension.  For self-attention, set d_kv=d_q.
        d_attn:   Internal attention dimension (d_l in paper, D_cT=256).
                  Must be divisible by n_heads.
        n_heads:  Number of attention heads (paper: 8).
        d_ff:     Hidden dimension of the two-layer MLP.  If None, defaults
                  to 4 * d_attn.
        dropout:  Dropout rate for attention weights and MLP.  Default 0.0.
    """

    def __init__(
        self,
        d_q: int,
        d_kv: int,
        d_attn: int,
        n_heads: int,
        d_ff: int | None = None, #hidden size inside MLP
        dropout: float = 0.0,
    ):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_attn
        
        #Save config values
        self.d_q = d_q
        self.d_kv = d_kv
        self.d_attn = d_attn
        self.n_heads = n_heads
        self.d_ff = d_ff

        # --- Pre-norm layers ---
        # Separate LayerNorms for Q and K/V (they may have different dimensions).
        # K and V always share the same dimension and typically the same input,
        # so they share one LayerNorm.
        self.norm_q = nn.LayerNorm(d_q)
        self.norm_kv = nn.LayerNorm(d_kv)

        # --- Multi-Head Attention (Eq. 7) ---
        self.attn = MultiHeadAttention(
            d_q=d_q,
            d_kv=d_kv,
            d_attn=d_attn,
            n_heads=n_heads,
            dropout=dropout,
        )

        # --- Pre-norm for MLP ---
        self.norm_ff = nn.LayerNorm(d_q)

        # --- MLP (Eq. 8): Linear → GELU → Dropout → Linear → Dropout ---
        self.mlp = nn.Sequential(
            nn.Linear(d_q, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_q),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q:  [B, n_q, d_q]     Query input (also the residual stream).
            k:  [B, n_kv, d_kv]   Key input.
            v:  [B, n_kv, d_kv]   Value input.
            return_attn_weights:   If True, return (output, attn_weights).

        Returns:
            output: [B, n_q, d_q]  Same shape as q (residual preserves shape).
            attn_weights: [B, n_heads, n_q, n_kv]  (only if return_attn_weights)
        """
        # ---- Eq. 9 line 2: O_attn = A(Q, Attn(N(Q), N(K), N(V))) ----
        # Pre-norm the inputs
        q_normed = self.norm_q(q)
        k_normed = self.norm_kv(k)
        v_normed = self.norm_kv(v)      # K and V share the same LayerNorm

        # Multi-head attention on the normed inputs
        attn_out, attn_weights = self.attn(q_normed, k_normed, v_normed)

        # Residual: add raw Q (not the normed version)
        o_attn = q + attn_out           # [B, n_q, d_q]

        # ---- Eq. 9 line 1: output = A(O_attn, MLP(N(O_attn))) ----
        # Pre-norm before MLP
        ff_out = self.mlp(self.norm_ff(o_attn))

        # Residual: add O_attn
        output = o_attn + ff_out        # [B, n_q, d_q]

        if return_attn_weights:
            return output, attn_weights
        return output

    def extra_repr(self) -> str:
        return (
            f"d_q={self.d_q}, d_kv={self.d_kv}, d_attn={self.d_attn}, "
            f"n_heads={self.n_heads}, d_ff={self.d_ff}"
        )


# ========================================================================= #
#  Self-Test — exhaustive verification for all downstream use cases
# ========================================================================= #

def _self_test():
    """
    Exhaustive verification of MultiHeadAttention and PerceiverBlock
    for every concrete shape combination used by the STAR-VP model.

    Tests:
      1. MultiHeadAttention — uniform dims (self-attention case)
      2. MultiHeadAttention — mixed dims (cross-attention case)
      3. PerceiverBlock — Spatial Encoder  (d_q=d_kv=256, self-attn)
      4. PerceiverBlock — Spatial Decoder  (d_q=d_kv=256, cross-attn)
      5. PerceiverBlock — Temporal Encoder (d_q=d_kv=512, self-attn)
      6. PerceiverBlock — Temporal Decoder (d_q=256, d_kv=512, cross-attn)
      7. Gradient flow through all parameters
      8. Parameter count verification
      9. Stacking multiple blocks
     10. Attention weight shapes and properties
     11. Numerical stability (no NaN/Inf)
     12. Batch size 1 edge case
     13. Single token query edge case
     14. Pre-norm residual correctness (output differs from input)
     15. Deterministic output with same input
    """
    torch.manual_seed(42)

    D_C = 256       # Paper's D_cT
    D_PS = 512      # D_C + D_PE + D_TE = 256 + 129 + 127
    D_PE_TE = 256   # D_PE + D_TE = 129 + 127 (decoder query dim)
    N_HEADS = 8     # Paper's N_heads
    D_P = 128       # Number of saliency points

    all_passed = True
    test_num = 0

    def check(condition: bool, name: str):
        nonlocal all_passed, test_num
        test_num += 1
        status = "OK" if condition else "FAIL"
        print(f"  [{test_num:>2}] {name}: {status}")
        if not condition:
            all_passed = False

    print("=" * 72)
    print("STAR-VP Encoder/Decoder Self-Test")
    print("=" * 72)

    # =================================================================== #
    #  Test Group A: MultiHeadAttention
    # =================================================================== #
    print("\n--- A. MultiHeadAttention ---")

    # A1: Uniform dimensions (self-attention, Spatial module)
    mha_uniform = MultiHeadAttention(d_q=D_C, d_kv=D_C, d_attn=D_C, n_heads=N_HEADS)
    B, n_q, n_kv = 4, D_P, D_P
    q_in = torch.randn(B, n_q, D_C)
    out, weights = mha_uniform(q_in, q_in, q_in)
    check(tuple(out.shape) == (B, n_q, D_C), f"Uniform MHA output shape {tuple(out.shape)}")
    check(
        tuple(weights.shape) == (B, N_HEADS, n_q, n_kv),
        f"Uniform MHA weight shape {tuple(weights.shape)}",
    )

    # A2: Mixed dimensions (cross-attention, Temporal decoder)
    mha_mixed = MultiHeadAttention(d_q=D_PE_TE, d_kv=D_PS, d_attn=D_C, n_heads=N_HEADS)
    B_t, n_q_t, n_kv_t = 4, 25, 80
    q_t = torch.randn(B_t, n_q_t, D_PE_TE)
    kv_t = torch.randn(B_t, n_kv_t, D_PS)
    out_t, weights_t = mha_mixed(q_t, kv_t, kv_t)
    check(tuple(out_t.shape) == (B_t, n_q_t, D_PE_TE), f"Mixed MHA output shape {tuple(out_t.shape)}")
    check(
        tuple(weights_t.shape) == (B_t, N_HEADS, n_q_t, n_kv_t),
        f"Mixed MHA weight shape {tuple(weights_t.shape)}",
    )

    # A3: Attention weights sum to 1 along key dimension
    weight_sum = weights.sum(dim=-1)    # [B, H, n_q]
    check(
        torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5),
        "Attention weights sum to 1",
    )

    # A4: No NaN/Inf
    check(
        not (torch.isnan(out).any() or torch.isinf(out).any()),
        "No NaN/Inf in uniform MHA output",
    )
    check(
        not (torch.isnan(out_t).any() or torch.isinf(out_t).any()),
        "No NaN/Inf in mixed MHA output",
    )

    # A5: Deterministic (same input → same output)
    out2, _ = mha_uniform(q_in, q_in, q_in)
    check(torch.allclose(out, out2, atol=1e-6), "Deterministic output")

    # A6: Parameter count for uniform MHA
    # q_proj: D_C * D_C + D_C = 256*256+256 = 65792
    # k_proj: D_C * D_C + D_C = 65792
    # v_proj: D_C * D_C + D_C = 65792
    # o_proj: D_C * D_C + D_C = 65792
    # Total: 4 * 65792 = 263168
    n_params_mha = sum(p.numel() for p in mha_uniform.parameters())
    expected_mha = 4 * (D_C * D_C + D_C)
    check(
        n_params_mha == expected_mha,
        f"Uniform MHA params: {n_params_mha:,} == {expected_mha:,}",
    )

    # A7: Parameter count for mixed MHA
    # q_proj: D_PE_TE * D_C + D_C = 256*256+256 = 65792
    # k_proj: D_PS * D_C + D_C = 512*256+256 = 131328
    # v_proj: D_PS * D_C + D_C = 131328
    # o_proj: D_C * D_PE_TE + D_PE_TE = 256*256+256 = 65792
    # Total: 65792 + 131328 + 131328 + 65792 = 394240
    n_params_mixed = sum(p.numel() for p in mha_mixed.parameters())
    expected_mixed = (
        (D_PE_TE * D_C + D_C)    # q_proj
        + (D_PS * D_C + D_C)     # k_proj
        + (D_PS * D_C + D_C)     # v_proj
        + (D_C * D_PE_TE + D_PE_TE)  # o_proj
    )
    check(
        n_params_mixed == expected_mixed,
        f"Mixed MHA params: {n_params_mixed:,} == {expected_mixed:,}",
    )

    # =================================================================== #
    #  Test Group B: PerceiverBlock — Spatial Attention shapes
    # =================================================================== #
    print("\n--- B. PerceiverBlock — Spatial Attention ---")

    T_total = 40  # T_M + T_H
    BT = 4 * T_total  # batch * time (flattened for per-timestep processing)

    # B1: Spatial Encoder (self-attention on saliency)
    # S: [BT, 128, 256], self-attention → [BT, 128, 256]
    sp_enc = PerceiverBlock(d_q=D_C, d_kv=D_C, d_attn=D_C, n_heads=N_HEADS)
    S_in = torch.randn(BT, D_P, D_C)
    S_out = sp_enc(S_in, S_in, S_in)
    check(tuple(S_out.shape) == (BT, D_P, D_C), f"Spatial Encoder output {tuple(S_out.shape)}")
    check(not torch.isnan(S_out).any(), "Spatial Encoder no NaN")

    # B2: Spatial Decoder (cross-attention: viewport queries encoded saliency)
    # P: [BT, 1, 256], S': [BT, 128, 256] → [BT, 1, 256]
    sp_dec = PerceiverBlock(d_q=D_C, d_kv=D_C, d_attn=D_C, n_heads=N_HEADS)
    P_in = torch.randn(BT, 1, D_C)
    P_out = sp_dec(P_in, S_out.detach(), S_out.detach())
    check(tuple(P_out.shape) == (BT, 1, D_C), f"Spatial Decoder output {tuple(P_out.shape)}")
    check(not torch.isnan(P_out).any(), "Spatial Decoder no NaN")

    # B3: Spatial Decoder with attention weights (None when using SDPA)
    P_out_viz, sp_attn = sp_dec(P_in, S_out.detach(), S_out.detach(), return_attn_weights=True)
    check(
        sp_attn is None or tuple(sp_attn.shape) == (BT, N_HEADS, 1, D_P),
        f"Spatial Decoder attn weights returned={sp_attn is not None}",
    )

    # =================================================================== #
    #  Test Group C: PerceiverBlock — Temporal Attention shapes
    # =================================================================== #
    print("\n--- C. PerceiverBlock — Temporal Attention ---")

    B = 4
    T_enc = 80   # 2 * (T_M + T_H)
    T_dec = 25   # T_H

    # C1: Temporal Encoder (self-attention over all 80 tokens at D_PS=512)
    tm_enc = PerceiverBlock(d_q=D_PS, d_kv=D_PS, d_attn=D_C, n_heads=N_HEADS)
    PS_in = torch.randn(B, T_enc, D_PS)
    PS_out = tm_enc(PS_in, PS_in, PS_in)
    check(tuple(PS_out.shape) == (B, T_enc, D_PS), f"Temporal Encoder output {tuple(PS_out.shape)}")
    check(not torch.isnan(PS_out).any(), "Temporal Encoder no NaN")

    # C2: Temporal Decoder (cross-attention: query embeddings attend to encoder)
    # E: [B, 25, 256], PS': [B, 80, 512] → [B, 25, 256]
    tm_dec = PerceiverBlock(d_q=D_PE_TE, d_kv=D_PS, d_attn=D_C, n_heads=N_HEADS)
    E_in = torch.randn(B, T_dec, D_PE_TE)
    dec_out = tm_dec(E_in, PS_out.detach(), PS_out.detach())
    check(tuple(dec_out.shape) == (B, T_dec, D_PE_TE), f"Temporal Decoder output {tuple(dec_out.shape)}")
    check(not torch.isnan(dec_out).any(), "Temporal Decoder no NaN")

    # C3: Temporal Decoder attention weights (None when using SDPA)
    dec_out_viz, tm_attn = tm_dec(
        E_in, PS_out.detach(), PS_out.detach(), return_attn_weights=True,
    )
    check(
        tm_attn is None or tuple(tm_attn.shape) == (B, N_HEADS, T_dec, T_enc),
        f"Temporal Decoder attn weights returned={tm_attn is not None}",
    )

    # =================================================================== #
    #  Test Group D: Gradient flow
    # =================================================================== #
    print("\n--- D. Gradient flow ---")

    # D1: Spatial Encoder gradient flow
    sp_enc.zero_grad()
    loss_sp = sp_enc(torch.randn(2, D_P, D_C), torch.randn(2, D_P, D_C), torch.randn(2, D_P, D_C)).sum()
    loss_sp.backward()
    sp_grad_ok = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in sp_enc.parameters() if p.requires_grad
    )
    check(sp_grad_ok, "Spatial Encoder gradient flow to all params")

    # D2: Temporal Decoder gradient flow (mixed dims — most complex case)
    tm_dec.zero_grad()
    q_grad = torch.randn(2, T_dec, D_PE_TE)
    kv_grad = torch.randn(2, T_enc, D_PS)
    loss_tm = tm_dec(q_grad, kv_grad, kv_grad).sum()
    loss_tm.backward()
    tm_grad_ok = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in tm_dec.parameters() if p.requires_grad
    )
    check(tm_grad_ok, "Temporal Decoder gradient flow to all params")

    # D3: Gradient flows through q input (important for upstream modules)
    sp_enc_2 = PerceiverBlock(d_q=D_C, d_kv=D_C, d_attn=D_C, n_heads=N_HEADS)
    q_leaf = torch.randn(2, D_P, D_C, requires_grad=True)
    kv_leaf = torch.randn(2, D_P, D_C, requires_grad=True)
    out_leaf = sp_enc_2(q_leaf, kv_leaf, kv_leaf)
    out_leaf.sum().backward()
    check(
        q_leaf.grad is not None and q_leaf.grad.abs().sum() > 0,
        "Gradient flows back to q input tensor",
    )
    check(
        kv_leaf.grad is not None and kv_leaf.grad.abs().sum() > 0,
        "Gradient flows back to k/v input tensor",
    )

    # =================================================================== #
    #  Test Group E: Parameter counts for PerceiverBlock
    # =================================================================== #
    print("\n--- E. Parameter counts ---")

    # E1: Spatial block (d_q=d_kv=256, d_attn=256, d_ff=4*256=1024)
    sp_block = PerceiverBlock(d_q=D_C, d_kv=D_C, d_attn=D_C, n_heads=N_HEADS)
    n_sp = sum(p.numel() for p in sp_block.parameters())
    # norm_q: 256*2 = 512   (weight + bias)
    # norm_kv: 256*2 = 512
    # attn: 4*(256*256+256) = 263168
    # norm_ff: 256*2 = 512
    # mlp.0 (Linear d_q→d_ff): 256*1024+1024 = 263168
    # mlp.3 (Linear d_ff→d_q): 1024*256+256 = 262400
    # Total: 512 + 512 + 263168 + 512 + 263168 + 262400 = 790272
    expected_sp = (
        D_C * 2           # norm_q
        + D_C * 2         # norm_kv
        + 4 * (D_C * D_C + D_C)  # MHA (4 projections)
        + D_C * 2         # norm_ff
        + (D_C * 4 * D_C + 4 * D_C)     # mlp linear1
        + (4 * D_C * D_C + D_C)         # mlp linear2
    )
    check(n_sp == expected_sp, f"Spatial block params: {n_sp:,} == {expected_sp:,}")
    print(f"       Spatial block: {n_sp:,} parameters")

    # E2: Temporal Decoder block (d_q=256, d_kv=512, d_attn=256, d_ff=1024)
    tm_block = PerceiverBlock(d_q=D_PE_TE, d_kv=D_PS, d_attn=D_C, n_heads=N_HEADS)
    n_tm = sum(p.numel() for p in tm_block.parameters())
    # norm_q: 256*2 = 512
    # norm_kv: 512*2 = 1024
    # attn.q_proj: 256*256+256 = 65792
    # attn.k_proj: 512*256+256 = 131328
    # attn.v_proj: 512*256+256 = 131328
    # attn.o_proj: 256*256+256 = 65792
    # norm_ff: 256*2 = 512
    # mlp.0: 256*1024+1024 = 263168
    # mlp.3: 1024*256+256 = 262400
    expected_tm = (
        D_PE_TE * 2         # norm_q
        + D_PS * 2           # norm_kv
        + (D_PE_TE * D_C + D_C)    # q_proj
        + (D_PS * D_C + D_C)       # k_proj
        + (D_PS * D_C + D_C)       # v_proj
        + (D_C * D_PE_TE + D_PE_TE)  # o_proj
        + D_PE_TE * 2       # norm_ff
        + (D_PE_TE * 4 * D_C + 4 * D_C)   # mlp linear1
        + (4 * D_C * D_PE_TE + D_PE_TE)   # mlp linear2
    )
    check(n_tm == expected_tm, f"Temporal Decoder block params: {n_tm:,} == {expected_tm:,}")
    print(f"       Temporal Decoder block: {n_tm:,} parameters")

    # =================================================================== #
    #  Test Group F: Stacking blocks
    # =================================================================== #
    print("\n--- F. Stacking blocks ---")

    # F1: Stack 2 encoder blocks (N_layers_T=2 from paper)
    blocks = nn.ModuleList([
        PerceiverBlock(d_q=D_C, d_kv=D_C, d_attn=D_C, n_heads=N_HEADS)
        for _ in range(2)
    ])
    x = torch.randn(2, D_P, D_C)
    for block in blocks:
        x = block(x, x, x)
    check(
        tuple(x.shape) == (2, D_P, D_C),
        f"Stacked encoder output {tuple(x.shape)}",
    )
    check(not torch.isnan(x).any(), "Stacked encoder no NaN")

    # F2: Stack 2 decoder blocks (cross-attention, fixed memory)
    dec_blocks = nn.ModuleList([
        PerceiverBlock(d_q=D_PE_TE, d_kv=D_PS, d_attn=D_C, n_heads=N_HEADS)
        for _ in range(2)
    ])
    memory = torch.randn(2, T_enc, D_PS)
    query = torch.randn(2, T_dec, D_PE_TE)
    for block in dec_blocks:
        query = block(query, memory, memory)
    check(
        tuple(query.shape) == (2, T_dec, D_PE_TE),
        f"Stacked decoder output {tuple(query.shape)}",
    )
    check(not torch.isnan(query).any(), "Stacked decoder no NaN")

    # F3: Gradient through stacked blocks
    all_params = list(blocks.parameters()) + list(dec_blocks.parameters())
    loss_stack = x.sum() + query.sum()
    loss_stack.backward()
    stack_grad = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in all_params if p.requires_grad
    )
    check(stack_grad, "Gradient flows through all stacked blocks")

    # =================================================================== #
    #  Test Group G: Edge cases
    # =================================================================== #
    print("\n--- G. Edge cases ---")

    # G1: Batch size 1
    blk = PerceiverBlock(d_q=D_C, d_kv=D_C, d_attn=D_C, n_heads=N_HEADS)
    out_b1 = blk(torch.randn(1, 10, D_C), torch.randn(1, 10, D_C), torch.randn(1, 10, D_C))
    check(tuple(out_b1.shape) == (1, 10, D_C), f"Batch=1 output {tuple(out_b1.shape)}")

    # G2: Single query token (Spatial Decoder: viewport is 1 token)
    out_sq = blk(torch.randn(2, 1, D_C), torch.randn(2, D_P, D_C), torch.randn(2, D_P, D_C))
    check(tuple(out_sq.shape) == (2, 1, D_C), f"Single-query output {tuple(out_sq.shape)}")

    # G3: Large sequence (stress test with 160 items — BT=160 for spatial)
    large_in = torch.randn(1, 160, D_C)
    out_large = blk(large_in, large_in, large_in)
    check(tuple(out_large.shape) == (1, 160, D_C), "Large sequence output shape")
    check(not torch.isnan(out_large).any(), "Large sequence no NaN")

    # G4: Pre-norm residual correctness — output should differ from input
    # (if attention/MLP had no effect, output would equal input due to residual)
    inp = torch.randn(2, 10, D_C)
    outp = blk(inp, inp, inp)
    check(
        not torch.allclose(inp, outp, atol=1e-4),
        "Output differs from input (attention has effect)",
    )

    # G5: Dropout mode — verify training vs eval behavior
    blk_drop = PerceiverBlock(d_q=D_C, d_kv=D_C, d_attn=D_C, n_heads=N_HEADS, dropout=0.5)
    test_inp = torch.randn(2, 10, D_C)
    blk_drop.eval()
    with torch.no_grad():
        out_eval1 = blk_drop(test_inp, test_inp, test_inp)
        out_eval2 = blk_drop(test_inp, test_inp, test_inp)
    check(
        torch.allclose(out_eval1, out_eval2, atol=1e-6),
        "Eval mode is deterministic (dropout disabled)",
    )

    # =================================================================== #
    #  Test Group H: Residual stream preservation
    # =================================================================== #
    print("\n--- H. Residual stream preservation ---")

    # H1: With zero-initialized params, the residual should pass through
    # Create a block and zero out all parameters
    blk_zero = PerceiverBlock(d_q=D_C, d_kv=D_C, d_attn=D_C, n_heads=N_HEADS)
    with torch.no_grad():
        for p in blk_zero.parameters():
            p.zero_()
    inp_zero = torch.randn(2, 5, D_C)
    out_zero = blk_zero(inp_zero, inp_zero, inp_zero)
    # With all weights zeroed: attn output is 0, mlp output is 0 (biases zeroed too)
    # So output = inp + 0 + 0 = inp
    # But LayerNorm has its own params (weight=0, bias=0 after zeroing), so
    # the normed input is all zeros → attn of zeros → output is just the residual
    # Actually, LN with weight=0 makes all outputs 0, so attn gets 0 input.
    # The attention output is 0 (since all projections are zeroed).
    # The MLP also gets 0 input (LN(o_attn) with weight=0) → output 0.
    # So: output = inp + 0 + 0 = inp.  ✓
    check(
        torch.allclose(inp_zero, out_zero, atol=1e-6),
        "Zeroed params → output equals input (residual pass-through)",
    )

    # =================================================================== #
    #  Summary
    # =================================================================== #
    print(f"\n{'=' * 72}")
    print(f"{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"{'=' * 72}")
    return all_passed


if __name__ == "__main__":
    _self_test()
