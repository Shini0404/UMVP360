"""
Forward pass test with dummy data for the full MUSE-VP pipeline.

Simulates realistic input data matching the preprocessing output format,
runs a full forward + backward pass for every ablation row, and performs
a single training step to confirm the entire pipeline is end-to-end
differentiable and numerically stable.

Run:
    python tests/test_forward_pass.py
    python tests/test_forward_pass.py --device cuda   # GPU test
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from muse_vp.models import (
    MUSEVP,
    geodesic_loss,
    create_lstm_only,
    create_star_vp,
    create_plus_eye_lstm,
    create_plus_offset,
    create_plus_eye_spatial,
    create_plus_face,
    create_plus_user,
    create_full_muse_vp,
)

# ========================================================================= #
#  Dummy data that mimics real preprocessing output
# ========================================================================= #

def make_dummy_batch(
    B: int = 8,
    T_M: int = 25,
    T_H: int = 25,
    K: int = 32,
    num_users: int = 24,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """
    Generate a single batch of dummy data with realistic value ranges.

    Returns a dict whose keys match the MUSEVP.forward() argument names.
    """

    # --- PAST tracking features (T_M frames of observation) ---

    # Head direction: 3-D unit vectors on the sphere, smooth trajectory
    # (simulate slow panning by adding small perturbations)
    head_base = F.normalize(torch.randn(B, 1, 3, device=device), dim=-1)
    head_noise = torch.randn(B, T_M, 3, device=device) * 0.05
    head_3d = F.normalize(head_base + head_noise, dim=-1)

    # Eye gaze: 3-D unit vectors, close to head but with small offsets
    eye_offset = torch.randn(B, T_M, 3, device=device) * 0.15
    eye_3d = F.normalize(head_3d + eye_offset, dim=-1)

    # Normalized eye-head offset: (relH/30°, relV/30°, magnitude/30°)
    # Typical range ~ [-1, 1]
    offset = torch.randn(B, T_M, 3, device=device) * 0.3

    # Fixation features: (IsFixating ∈ {0,1}, duration ∈ [0,1])
    is_fixating = (torch.rand(B, T_M, 1, device=device) > 0.4).float()
    fix_duration = torch.rand(B, T_M, 1, device=device) * is_fixating
    fixation = torch.cat([is_fixating, fix_duration], dim=-1)

    # --- FUTURE / SALIENCY features (T_H prediction frames) ---

    # Saliency 3-D coordinates: K points on the unit sphere per frame
    s_xyz = F.normalize(torch.randn(B, T_H, K, 3, device=device), dim=-1)

    # Saliency weights: sum to 1 per frame (importance distribution)
    s_weight = torch.softmax(torch.randn(B, T_H, K, device=device), dim=-1)

    # Future eye gaze: unit vectors (ground truth for training)
    e_gaze_3d = F.normalize(torch.randn(B, T_H, 3, device=device), dim=-1)

    # Face features: 5 blendshape values in [0, 1]
    # (FaceActivity, BrowLower, JawDrop, EyesClosed, BrowRaise)
    face_raw = torch.rand(B, 5, device=device)

    # User IDs
    user_id = torch.randint(0, num_users, (B,), device=device)

    # --- Ground-truth future head positions ---
    target_base = F.normalize(torch.randn(B, 1, 3, device=device), dim=-1)
    target_noise = torch.randn(B, T_H, 3, device=device) * 0.05
    target = F.normalize(target_base + target_noise, dim=-1)

    return {
        "head_3d": head_3d,
        "eye_3d": eye_3d,
        "offset": offset,
        "fixation": fixation,
        "s_xyz": s_xyz,
        "s_weight": s_weight,
        "e_gaze_3d": e_gaze_3d,
        "face_raw": face_raw,
        "user_id": user_id,
        "target": target,
    }


# ========================================================================= #
#  Tests
# ========================================================================= #

def test_all_ablation_rows(device: torch.device):
    """Forward + backward pass for every ablation configuration."""

    B, T_M, T_H, K, NUM_USERS = 8, 25, 25, 32, 24
    common = dict(T_M=T_M, T_H=T_H, K=K, num_users=NUM_USERS)
    data = make_dummy_batch(B, T_M, T_H, K, NUM_USERS, device)

    rows = [
        ("Row 1  LSTM-only",       create_lstm_only),
        ("Row 2  STAR-VP",         create_star_vp),
        ("Row 3  +Eye LSTM",       create_plus_eye_lstm),
        ("Row 4  +Offset",         create_plus_offset),
        ("Row 5  +Eye Spatial",    create_plus_eye_spatial),
        ("Row 6  +Face",           create_plus_face),
        ("Row 7  +User",           create_plus_user),
        ("Row 8  Full MUSE-VP",    create_full_muse_vp),
    ]

    print(f"\n{'='*72}")
    print(f"  Forward + backward pass — all ablation rows")
    print(f"  device={device}, B={B}, T_M={T_M}, T_H={T_H}, K={K}")
    print(f"{'='*72}")

    all_ok = True
    for name, factory in rows:
        model = factory(**common).to(device)
        model.train()

        n_params = sum(p.numel() for p in model.parameters())
        t0 = time.perf_counter()

        p_hat = model(
            data["head_3d"], data["eye_3d"], data["offset"], data["fixation"],
            data["s_xyz"], data["s_weight"], data["e_gaze_3d"],
            data["face_raw"], data["user_id"],
        )

        fwd_ms = (time.perf_counter() - t0) * 1000

        loss = geodesic_loss(p_hat, data["target"])

        t0 = time.perf_counter()
        loss.backward()
        bwd_ms = (time.perf_counter() - t0) * 1000

        # Checks
        shape_ok = tuple(p_hat.shape) == (B, T_H, 3)
        unit_ok = torch.allclose(
            p_hat.norm(dim=-1), torch.ones(B, T_H, device=device), atol=1e-5,
        )
        nan_ok = not (torch.isnan(p_hat).any() or torch.isinf(p_hat).any())
        loss_ok = loss.isfinite().item()
        grad_ok = all(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters() if p.requires_grad
        )
        passed = shape_ok and unit_ok and nan_ok and loss_ok and grad_ok
        status = "PASS" if passed else "FAIL"

        print(
            f"\n  {name:30s}  {status}"
            f"\n    params={n_params:>10,}  "
            f"fwd={fwd_ms:6.1f}ms  bwd={bwd_ms:6.1f}ms  "
            f"loss={loss.item():.4f} rad ({loss.item()*57.2958:.1f}°)"
        )
        if not passed:
            all_ok = False
            if not shape_ok:  print(f"    FAIL: shape {tuple(p_hat.shape)}")
            if not unit_ok:   print(f"    FAIL: not unit vectors")
            if not nan_ok:    print(f"    FAIL: NaN/Inf in output")
            if not loss_ok:   print(f"    FAIL: loss not finite")
            if not grad_ok:   print(f"    FAIL: gradients missing or NaN")

        model.zero_grad()

    return all_ok


def test_training_step(device: torch.device):
    """Simulate a full training step: forward → loss → backward → optimizer."""

    B, T_M, T_H, K, NUM_USERS = 8, 25, 25, 32, 24
    data = make_dummy_batch(B, T_M, T_H, K, NUM_USERS, device)

    model = create_full_muse_vp(
        T_M=T_M, T_H=T_H, K=K, num_users=NUM_USERS,
    ).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    print(f"\n{'='*72}")
    print(f"  Training step simulation (3 steps, full MUSE-VP)")
    print(f"  device={device}, B={B}")
    print(f"{'='*72}")

    all_ok = True
    losses = []

    for step in range(3):
        optimizer.zero_grad()

        p_hat = model(
            data["head_3d"], data["eye_3d"], data["offset"], data["fixation"],
            data["s_xyz"], data["s_weight"], data["e_gaze_3d"],
            data["face_raw"], data["user_id"],
        )

        loss = geodesic_loss(p_hat, data["target"])
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        print(f"    step {step}: loss={loss.item():.6f} rad  "
              f"grad_norm={grad_norm:.4f}")

        if not loss.isfinite().item():
            print(f"    FAIL: loss diverged at step {step}")
            all_ok = False
            break

    # After 3 steps on the same data, loss should decrease (overfitting)
    if len(losses) == 3 and losses[2] < losses[0]:
        print(f"  Loss decreased: {losses[0]:.6f} → {losses[2]:.6f}  OK")
    elif len(losses) == 3:
        print(f"  Loss did not decrease: {losses[0]:.6f} → {losses[2]:.6f}  "
              f"(acceptable — only 3 steps)")

    return all_ok


def test_intermediates(device: torch.device):
    """Verify forward_with_intermediates returns correct shapes."""

    B, T_M, T_H, K, NUM_USERS = 4, 25, 25, 32, 24
    d_model = 64
    data = make_dummy_batch(B, T_M, T_H, K, NUM_USERS, device)

    model = create_full_muse_vp(
        T_M=T_M, T_H=T_H, K=K, d_model=d_model, num_users=NUM_USERS,
    ).to(device)
    model.eval()

    print(f"\n{'='*72}")
    print(f"  Intermediate output verification")
    print(f"{'='*72}")

    with torch.no_grad():
        p_hat, inter = model.forward_with_intermediates(
            data["head_3d"], data["eye_3d"], data["offset"], data["fixation"],
            data["s_xyz"], data["s_weight"], data["e_gaze_3d"],
            data["face_raw"], data["user_id"],
        )

    expected = {
        "p_prime":         (B, T_H, 3),
        "p_double_prime":  (B, T_H, 3),
        "s_s_out":         (B, T_H, K, d_model),
        "p_s_out":         (B, T_H, d_model),
    }

    all_ok = True
    for key, exp_shape in expected.items():
        actual = tuple(inter[key].shape)
        ok = actual == exp_shape
        print(f"  {key:20s}: {str(actual):30s}  {'OK' if ok else 'FAIL'}")
        if not ok:
            all_ok = False

    # p_hat should match forward() output
    p_hat_fwd = model(
        data["head_3d"], data["eye_3d"], data["offset"], data["fixation"],
        data["s_xyz"], data["s_weight"], data["e_gaze_3d"],
        data["face_raw"], data["user_id"],
    )
    match_ok = torch.allclose(p_hat, p_hat_fwd, atol=1e-6)
    print(f"  forward() == forward_with_intermediates(): "
          f"{'OK' if match_ok else 'FAIL'}")
    if not match_ok:
        all_ok = False

    return all_ok


def test_eval_mode(device: torch.device):
    """Verify model works in eval mode (dropout disabled, deterministic)."""

    B, T_M, T_H, K, NUM_USERS = 4, 25, 25, 32, 24
    data = make_dummy_batch(B, T_M, T_H, K, NUM_USERS, device)

    model = create_full_muse_vp(
        T_M=T_M, T_H=T_H, K=K, num_users=NUM_USERS,
    ).to(device)
    model.eval()

    print(f"\n{'='*72}")
    print(f"  Eval mode determinism")
    print(f"{'='*72}")

    with torch.no_grad():
        out1 = model(
            data["head_3d"], data["eye_3d"], data["offset"], data["fixation"],
            data["s_xyz"], data["s_weight"], data["e_gaze_3d"],
            data["face_raw"], data["user_id"],
        )
        out2 = model(
            data["head_3d"], data["eye_3d"], data["offset"], data["fixation"],
            data["s_xyz"], data["s_weight"], data["e_gaze_3d"],
            data["face_raw"], data["user_id"],
        )

    det_ok = torch.allclose(out1, out2, atol=1e-6)
    print(f"  Two identical forward passes match: {'OK' if det_ok else 'FAIL'}")

    unit_ok = torch.allclose(
        out1.norm(dim=-1), torch.ones(B, T_H, device=device), atol=1e-5,
    )
    print(f"  Output unit vectors: {'OK' if unit_ok else 'FAIL'}")

    return det_ok and unit_ok


def test_batch_size_one(device: torch.device):
    """Verify model handles B=1 (single sample inference)."""

    T_M, T_H, K, NUM_USERS = 25, 25, 32, 24
    data = make_dummy_batch(1, T_M, T_H, K, NUM_USERS, device)

    model = create_full_muse_vp(
        T_M=T_M, T_H=T_H, K=K, num_users=NUM_USERS,
    ).to(device)
    model.eval()

    print(f"\n{'='*72}")
    print(f"  Single-sample inference (B=1)")
    print(f"{'='*72}")

    with torch.no_grad():
        p_hat = model(
            data["head_3d"], data["eye_3d"], data["offset"], data["fixation"],
            data["s_xyz"], data["s_weight"], data["e_gaze_3d"],
            data["face_raw"], data["user_id"],
        )

    shape_ok = tuple(p_hat.shape) == (1, T_H, 3)
    unit_ok = torch.allclose(
        p_hat.norm(dim=-1), torch.ones(1, T_H, device=device), atol=1e-5,
    )
    nan_ok = not torch.isnan(p_hat).any()

    print(f"  Shape: {tuple(p_hat.shape)}  {'OK' if shape_ok else 'FAIL'}")
    print(f"  Unit vectors: {'OK' if unit_ok else 'FAIL'}")
    print(f"  No NaN: {'OK' if nan_ok else 'FAIL'}")

    return shape_ok and unit_ok and nan_ok


# ========================================================================= #
#  Main
# ========================================================================= #

def main():
    parser = argparse.ArgumentParser(description="MUSE-VP forward pass test")
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run on: cpu or cuda (default: cpu)",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    torch.manual_seed(42)

    print(f"\n  MUSE-VP Forward Pass Test Suite")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    results = {}
    t_start = time.perf_counter()

    results["ablation_rows"] = test_all_ablation_rows(device)
    results["training_step"] = test_training_step(device)
    results["intermediates"] = test_intermediates(device)
    results["eval_mode"] = test_eval_mode(device)
    results["batch_one"] = test_batch_size_one(device)

    elapsed = time.perf_counter() - t_start

    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s}  {status}")
        if not passed:
            all_passed = False

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"{'='*72}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
