"""
=============================================================================
MMVP2 Evaluation Script (extends STAR-VP)
=============================================================================

Computes the same paper metrics as STAR-VP / MUSE-VP:

    Orthodromic Distance (OD)        — radians (lower is better)
    Intersection over Union  (IoU)    — 9x16 tile grid, FOV 100° (higher better)

Reported per prediction step (0.2s … 5.0s), short-term (<1s), long-term
(2-5s) and overall.

Usage
-----
    python -m MMVP2.evaluate \\
        --checkpoint runs/mmvp2_rowFull_*/checkpoints/best.pt \\
        --output runs/mmvp2_rowFull_*/eval.json

The script auto-instantiates MMVP2 with the same toggles as the training run
by reading the `config` blob saved in the checkpoint, so a single CLI works
for every ablation row.
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .compat import torch_load
from .dataset import (
    MUSE_VP_PROCESSED, DENSE_SAL_DIR, BEHAV_PRIOR_DIR, AUDIO_DIR,
    create_dataloaders,
)
from .models import MMVP2

logger = logging.getLogger("mmvp2.evaluate")
FPS = 5

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):  # type: ignore[misc]
        return iterable


# ========================================================================= #
#  Tile centres on the unit sphere (matches STAR-VP convention)
# ========================================================================= #

def compute_tile_centers(n_rows: int = 9, n_cols: int = 16) -> torch.Tensor:
    """Returns [n_rows*n_cols, 3] unit vectors at the centre of each ERP tile.

    Convention matches paper Eq. 1 (physics spherical):
        phi   = pi  * (i + 0.5) / n_rows
        theta = 2pi * (j + 0.5) / n_cols
        x = cos(theta) * sin(phi)
        y = sin(theta) * sin(phi)
        z = cos(phi)
    """
    row_idx = torch.arange(n_rows, dtype=torch.float32)
    col_idx = torch.arange(n_cols, dtype=torch.float32)
    phi = math.pi * (row_idx + 0.5) / n_rows
    theta = 2.0 * math.pi * (col_idx + 0.5) / n_cols
    phi_g = phi.unsqueeze(1).expand(n_rows, n_cols)
    th_g  = theta.unsqueeze(0).expand(n_rows, n_cols)
    x = torch.cos(th_g) * torch.sin(phi_g)
    y = torch.sin(th_g) * torch.sin(phi_g)
    z = torch.cos(phi_g)
    return torch.stack([x, y, z], dim=-1).reshape(-1, 3)

    #9 * 16 = 144 tiles, [144, 3]
_TILE_CENTERS_9x16: torch.Tensor | None = None


def _get_tile_centers(device: torch.device) -> torch.Tensor:
    global _TILE_CENTERS_9x16
    if _TILE_CENTERS_9x16 is None:
        _TILE_CENTERS_9x16 = compute_tile_centers(9, 16)
    return _TILE_CENTERS_9x16.to(device)


# ========================================================================= #
#  Metrics
# ========================================================================= #

@torch.no_grad()
def compute_od_per_step(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cos_sim = (pred * target).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_sim)              # [B, T_H]


@torch.no_grad()
def compute_iou_per_step(
    pred: torch.Tensor,
    target: torch.Tensor,
    fov_half_deg: float = 50.0,
    n_rows: int = 9,
    n_cols: int = 16,
) -> torch.Tensor:
    tc = _get_tile_centers(pred.device).unsqueeze(0).unsqueeze(0)   # [1, 1, N, 3]
    fov = fov_half_deg * (math.pi / 180.0)
    cos_p = (tc * pred.unsqueeze(2)).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    cos_t = (tc * target.unsqueeze(2)).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    pred_in = torch.acos(cos_p) < fov
    tgt_in  = torch.acos(cos_t) < fov
    inter = (pred_in & tgt_in).float().sum(dim=-1)
    union = (pred_in | tgt_in).float().sum(dim=-1)
    return inter / union.clamp(min=1.0)


def aggregate_results(
    od_per_step: torch.Tensor,
    iou_per_step: torch.Tensor,
    fps: int = FPS,
) -> dict:
    T_H = od_per_step.shape[0]
    time_labels = [(t + 1) / fps for t in range(T_H)]
    short_end  = min(fps - 2, T_H - 1)              # 3 at 5fps
    long_start = 2 * fps - 1                         # 9 at 5fps
    short_idx = list(range(0, short_end + 1))
    long_idx  = list(range(long_start, T_H))
    return {
        "fps": fps,
        "T_H": T_H,
        "time_labels": time_labels,
        "od_per_step":  od_per_step.tolist(),
        "iou_per_step": iou_per_step.tolist(),
        "od_short_term":  od_per_step[short_idx].mean().item() if short_idx else float("nan"),
        "od_long_term":   od_per_step[long_idx].mean().item()  if long_idx  else float("nan"),
        "od_overall":     od_per_step.mean().item(),
        "iou_short_term": iou_per_step[short_idx].mean().item() if short_idx else float("nan"),
        "iou_long_term":  iou_per_step[long_idx].mean().item()  if long_idx  else float("nan"),
        "iou_overall":    iou_per_step.mean().item(),
    }


def print_results(results: dict, title: str = "MMVP2 Evaluation") -> None:
    T_H = results["T_H"]
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"\n{'Step':>6s} {'Time(s)':>8s} {'OD (rad)':>10s} {'OD (deg)':>10s} {'IoU (%)':>10s}")
    print("-" * 48)
    for t in range(T_H):
        od_deg = results["od_per_step"][t] * (180.0 / math.pi)
        print(
            f"{t + 1:>6d} {results['time_labels'][t]:>8.1f} "
            f"{results['od_per_step'][t]:>10.4f} {od_deg:>10.2f} "
            f"{results['iou_per_step'][t] * 100:>10.2f}"
        )
    print("\n" + "-" * 48)
    print(
        f"{'Short-term (<1s)':>24s}: OD={results['od_short_term']:.4f} rad "
        f"({results['od_short_term'] * 180 / math.pi:.2f}\u00b0), "
        f"IoU={results['iou_short_term'] * 100:.2f}%"
    )
    print(
        f"{'Long-term (2-5s)':>24s}: OD={results['od_long_term']:.4f} rad "
        f"({results['od_long_term'] * 180 / math.pi:.2f}\u00b0), "
        f"IoU={results['iou_long_term'] * 100:.2f}%"
    )
    print(
        f"{'Overall':>24s}: OD={results['od_overall']:.4f} rad "
        f"({results['od_overall'] * 180 / math.pi:.2f}\u00b0), "
        f"IoU={results['iou_overall'] * 100:.2f}%"
    )
    print("=" * 80)


# ========================================================================= #
#  Eval loop
# ========================================================================= #

@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_aux: bool,
    use_audio: bool,
    use_face: bool,
    use_user: bool,
    use_amp: bool = False,
) -> dict:
    model.eval()
    all_od:  list[torch.Tensor] = []
    all_iou: list[torch.Tensor] = []
    use_amp = bool(use_amp and device.type == "cuda")

    n_batches = len(loader)
    logger.info(
        "Running test set: %d batches (no per-batch log until done — watch the progress bar).",
        n_batches,
    )

    for batch in tqdm(loader, desc="Test", unit="batch", ncols=100, mininterval=0.5):
        past = (batch["past_traj"] if use_aux else batch["past_head"]).to(device, non_blocking=True)
        sal  = batch["sal_xyz"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        kwargs = {}
        if use_audio: kwargs["audio"]   = batch["audio"].to(device, non_blocking=True)
        if use_face:  kwargs["face"]    = batch["face"].to(device, non_blocking=True)
        if use_user:  kwargs["user_id"] = batch["user_id"].to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                p_hat = model(past, sal, **kwargs)
        else:
            p_hat = model(past, sal, **kwargs)
        all_od.append(compute_od_per_step(p_hat, target))
        all_iou.append(compute_iou_per_step(p_hat, target))

    od = torch.cat(all_od,  dim=0).mean(dim=0)
    iou = torch.cat(all_iou, dim=0).mean(dim=0)
    return aggregate_results(od, iou)


# ========================================================================= #
#  Main
# ========================================================================= #

def _strip_compile_prefix(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        out[k.removeprefix("_orig_mod.") if k.startswith("_orig_mod.") else k] = v
    return out


def run_evaluation(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # ---- Checkpoint ----
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for evaluation")
    ckpt_path = Path(args.checkpoint).expanduser()
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch_load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config", {}) or {}

    # Use config from checkpoint to recreate the same MMVP2 config
    def _cfg(k, default):
        v = cfg.get(k, default)
        return v if v is not None else default

    use_aux       = bool(_cfg("use_aux", False))
    use_behav     = bool(_cfg("use_behav", False))
    use_audio     = bool(_cfg("use_audio", False))
    use_face      = bool(_cfg("use_face", False))
    use_user      = bool(_cfg("use_user", False))
    use_user_film = bool(_cfg("use_user_film", False))
    t_m   = int(_cfg("t_m", 25))
    t_h   = int(_cfg("t_h", 25))
    d_audio = int(_cfg("d_audio", 64)) if use_audio else 0
    d_face  = int(_cfg("d_face",   4)) if use_face  else 0
    d_user  = int(_cfg("d_user",  24)) if use_user  else 0

    sal_pool_factor = int(_cfg("sal_pool_factor", 1))

    # ---- Data ----
    behav_dir = BEHAV_PRIOR_DIR if use_behav else None
    audio_dir = AUDIO_DIR       if use_audio else None
    loaders = create_dataloaders(
        muse_vp_dir=args.muse_vp_dir,
        dense_sal_dir=args.dense_sal_dir,
        behav_prior_dir=behav_dir,
        audio_dir=audio_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        T_M=t_m, T_H=t_h,
        seed=int(_cfg("split_seed", 42)),
        return_test=True,
        sal_pool_factor=sal_pool_factor,
    )
    test_loader = loaders["test"]
    full_ds = loaders["full_dataset"]
    n_users = (max(r.user_id for r in full_ds.recordings) + 1) if use_user else 0
    logger.info(f"Test batches: {len(test_loader)}  num_users (eval): {n_users}")

    # ---- Model ----
    model = MMVP2(
        head_dim=3,
        aux_dim=8 if use_aux else 0,
        d_in_s=5 if use_behav else 4,
        d_in_p=4,
        lstm_hidden_dim=int(_cfg("lstm_hidden", 256)),
        lstm_num_layers=int(_cfg("lstm_layers", 2)),
        d_c=int(_cfg("d_c", 256)),
        n_heads=int(_cfg("n_heads", 8)),
        spatial_enc_layers=int(_cfg("spatial_enc_layers", _cfg("spatial_layers", 2))),
        spatial_dec_layers=int(_cfg("spatial_dec_layers", _cfg("spatial_layers", 2))),
        d_pe=int(_cfg("d_pe", 129)),
        d_te=int(_cfg("d_te", 127)),
        temporal_enc_layers=int(_cfg("temporal_layers", 2)),
        temporal_dec_layers=int(_cfg("temporal_layers", 2)),
        d_g=int(_cfg("d_g", 128)),
        num_users=n_users,
        d_user=d_user,
        use_user_film=use_user_film,
        n_mels=64,
        d_audio=d_audio,
        d_face=d_face,
        face_input_dim=5,
        t_m=t_m, t_h=t_h,
        dropout=0.0,
        normalize_output=True,
    ).to(device)

    state = _strip_compile_prefix(ckpt["model_state_dict"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing[:5]}...({len(missing)} total)")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}...({len(unexpected)} total)")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"Toggles: aux={use_aux} behav={use_behav} audio={use_audio} "
                f"face={use_face} user={use_user} film={use_user_film}")

    results = evaluate(
        model, test_loader, device, use_aux, use_audio, use_face, use_user,
        use_amp=getattr(args, "amp", False),
    )
    print_results(results, title=f"MMVP2 Evaluation: {ckpt_path.name}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"checkpoint": str(ckpt_path), "config": cfg, "results": results}, f, indent=2)
        logger.info(f"Results saved to {out_path}")

    return results


# ========================================================================= #
#  CLI
# ========================================================================= #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MMVP2 Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--muse_vp_dir",   type=str, default=str(MUSE_VP_PROCESSED))
    p.add_argument("--dense_sal_dir", type=str, default=str(DENSE_SAL_DIR))
    p.add_argument("--output",        type=str, default=None)
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--device",        type=str, default="auto")
    p.add_argument(
        "--amp", action="store_true",
        help="Use mixed precision on CUDA (faster eval, matches typical training).",
    )
    args = p.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


if __name__ == "__main__":
    run_evaluation(parse_args())
