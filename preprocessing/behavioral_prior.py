"""
Cross-user behavioural saliency prior for MMVP2 (Sparkle-inspired).

For every (video, timestep) we estimate the empirical viewing-frequency map
on the *same* 16x32 spherical grid used for SalViT360 saliency, but built
from the **head trajectories of TRAINING participants only**.

This is "what other users actually looked at on this video at this moment"
- a behavioural-saliency prior that is orthogonal to visual saliency.

CRITICAL: only training-set participants contribute to the prior, and the
same train/val/test split must be used everywhere downstream.  This script
therefore takes a `--train-participants` list (defaults to the first 20 of
P001..P025, matching the FAIR_PARTICIPANTS 8:1:1 sample-level split used
by STAR-VP / MUSE-VP / MMVP - we keep all 25 in the dataset and let the
loader split per-user, but for the prior we use a participant-level
training subset because we need per-video per-time aggregates).

Concretely, for each (content, t) we do:
    1. Collect head_3d[t] from every training participant who has that
       video, where head_3d is the 5fps unit-vector trajectory used by
       MUSE-VP / MMVP (under muse_vp/processed_data/<P>/<video_folder>.pt).
    2. Drop each (x, y, z) into the corresponding ERP cell on the same
       16x32 grid as dense_saliency.py.
    3. Apply a small angular Gaussian smoothing on the sphere
       (sigma = ~10 degrees ~ FoV-sized blob) so neighbouring cells share
       a vote.
    4. Normalize the resulting [16, 32] map to zero-mean unit-std (matches
       the normalization used in dense_saliency.py).

Output per content: [N_5fps, 512, 4] using the same (x, y, z) cell coords
as dense saliency, with `s` being the cross-user view-frequency value.

Run
---
    cd /media/user/HDD3/Shini/STAR_VP
    conda activate starvp
    python -m MMVP2.preprocessing.behavioral_prior
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch

from .dense_saliency import (
    GRID_H, GRID_W, ERP_H, ERP_W,
    compute_cell_coords,
)


logger = logging.getLogger("mmvp2.behavioral_prior")

REPO_ROOT      = Path("/media/user/HDD3/Shini/STAR_VP")
MUSE_VP_DIR    = REPO_ROOT / "muse_vp" / "processed_data"
DEFAULT_OUTPUT = REPO_ROOT / "MMVP2" / "processed_data" / "behavioral_prior"

# train participants for the prior (must match what fair_sample uses).
DEFAULT_TRAIN_PARTICIPANTS = [f"P{i:03d}" for i in range(1, 21)]
ALL_PARTICIPANTS = [f"P{i:03d}" for i in range(1, 26)]

# Angular smoothing kernel sigma (degrees of arc on the unit sphere).
DEFAULT_SIGMA_DEG = 10.0


# ----------------------------------------------------------------------------- #
#  Helpers
# ----------------------------------------------------------------------------- #

def _xyz_to_cell(xyz: torch.Tensor, grid_h: int, grid_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Map [N, 3] unit vectors to (row, col) integer cell indices on the same ERP grid
    as compute_cell_coords (= STAR-VP SalMapProcessor convention):

        phi_polar = arccos(z)          0 .. pi  (0 = north pole)
        theta     = atan2(y, x)        -pi .. +pi  -> wrap to [0, 2*pi)

    Cell indices:
        r = floor( phi_polar / pi * grid_h )
        c = floor( theta / (2*pi) * grid_w )
    """
    z = xyz[:, 2].clamp(-1.0, 1.0)
    phi_polar = torch.arccos(z)                          # 0 (north) .. pi (south)
    theta     = torch.atan2(xyz[:, 1], xyz[:, 0])         # [-pi, pi]
    theta_pos = (theta + 2.0 * torch.pi) % (2.0 * torch.pi)  # [0, 2*pi)

    r = torch.floor(phi_polar / torch.pi * grid_h).long().clamp(0, grid_h - 1)
    c = torch.floor(theta_pos / (2.0 * torch.pi) * grid_w).long().clamp(0, grid_w - 1)
    return r, c


def _angular_gaussian_kernel(coords: torch.Tensor, sigma_deg: float) -> torch.Tensor:
    """
    Build a [N, N] kernel where K[i, j] = exp(-arccos(c_i . c_j)^2 / 2*sigma^2)
    for cell centres `coords` [N, 3].  Used to smooth the empirical viewing
    histogram by a small angular Gaussian (FoV-sized blob).
    """
    cos_dist = (coords @ coords.T).clamp(-1.0, 1.0)
    arc = torch.arccos(cos_dist)                          # in radians
    sigma = sigma_deg * torch.pi / 180.0
    K = torch.exp(-0.5 * (arc / sigma) ** 2)
    K = K / K.sum(dim=1, keepdim=True)                    # row-normalize
    return K


# ----------------------------------------------------------------------------- #
#  Main aggregation per video
# ----------------------------------------------------------------------------- #

def _gather_train_trajectories(
    muse_vp_dir: Path,
    train_participants: list[str],
    content_name: str,
) -> list[torch.Tensor]:
    """
    Return list of [N_5fps, 3] head trajectories from training participants
    that watched this content.  We rely on muse_vp's per-participant per-video
    .pt files (same 5fps as everything else).
    """
    trajs = []
    for p in train_participants:
        p_dir = muse_vp_dir / p
        if not p_dir.exists():
            continue
        # video file name in muse_vp is the *folder* name, e.g. video_12_LOSC_Football
        # Multiple folders can share a content_name — e.g. exp1 vs exp2 — but
        # muse_vp/processed_data already canonicalizes to a single folder per
        # (participant, content). Find it by content_name field inside the .pt.
        for pt in sorted(p_dir.glob("video_*.pt")):
            d = torch.load(pt, weights_only=False, map_location="cpu")
            if d.get("content_name") != content_name:
                continue
            trajs.append(d["head_3d"].to(torch.float32))
            break
    return trajs


def build_prior_for_content(
    content_name: str,
    train_participants: list[str],
    muse_vp_dir: Path = MUSE_VP_DIR,
    grid_h: int = GRID_H,
    grid_w: int = GRID_W,
    sigma_deg: float = DEFAULT_SIGMA_DEG,
) -> torch.Tensor | None:
    """Return [N_5fps, 512, 4] behavioural prior tokens, or None if no train data."""
    trajs = _gather_train_trajectories(muse_vp_dir, train_participants, content_name)
    if not trajs:
        logger.warning("[%s] no training trajectories found", content_name)
        return None

    n_5 = min(t.shape[0] for t in trajs)
    if n_5 == 0:
        logger.warning("[%s] empty trajectory", content_name)
        return None

    # Build per-frame histogram on the grid using all train users' heads
    counts = torch.zeros(n_5, grid_h * grid_w, dtype=torch.float32)
    for t in trajs:
        t = t[:n_5]                                      # [n_5, 3]
        r, c = _xyz_to_cell(t, grid_h, grid_w)           # [n_5], [n_5]
        flat_idx = r * grid_w + c                        # [n_5]
        # accumulate per-frame
        counts[torch.arange(n_5), flat_idx] += 1.0

    counts = counts / max(1, len(trajs))                  # frequency in [0, 1]

    # Angular Gaussian smoothing across cells (FoV-sized blob)
    coords = compute_cell_coords(grid_h, grid_w)          # [N=512, 3]
    K = _angular_gaussian_kernel(coords, sigma_deg)       # [N, N]
    smoothed = counts @ K.T                               # [n_5, N]

    # Per-frame zero-mean unit-std (same as visual saliency)
    mu = smoothed.mean(dim=1, keepdim=True)
    sd = smoothed.std(dim=1, keepdim=True).clamp(min=1e-6)
    s_norm = (smoothed - mu) / sd                         # [n_5, N]

    s_flat = s_norm.unsqueeze(-1)                         # [n_5, N, 1]
    coords_exp = coords.unsqueeze(0).expand(n_5, -1, -1)  # [n_5, N, 3]
    out = torch.cat([coords_exp, s_flat], dim=-1).contiguous()
    return out


# ----------------------------------------------------------------------------- #
#  CLI
# ----------------------------------------------------------------------------- #

def _list_contents(muse_vp_dir: Path, train_participants: list[str]) -> list[str]:
    """Discover the set of distinct content names by scanning a couple of train Ps."""
    seen = set()
    for p in train_participants[:5]:
        p_dir = muse_vp_dir / p
        if not p_dir.exists():
            continue
        for pt in p_dir.glob("video_*.pt"):
            d = torch.load(pt, weights_only=False, map_location="cpu")
            cn = d.get("content_name")
            if cn:
                seen.add(cn)
    return sorted(seen)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--muse-vp-dir",  type=Path, default=MUSE_VP_DIR)
    parser.add_argument("--output-dir",   type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-h",       type=int,  default=GRID_H)
    parser.add_argument("--grid-w",       type=int,  default=GRID_W)
    parser.add_argument("--sigma-deg",    type=float, default=DEFAULT_SIGMA_DEG)
    parser.add_argument("--train-participants", type=str, nargs="+",
                        default=DEFAULT_TRAIN_PARTICIPANTS,
                        help="Participants used to build the prior (must match training split).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    contents = _list_contents(args.muse_vp_dir, args.train_participants)
    logger.info("contents: %s", contents)
    logger.info("train_participants (n=%d): %s", len(args.train_participants), args.train_participants)

    for cn in contents:
        out = args.output_dir / f"{cn}.pt"
        if out.exists() and not args.overwrite:
            logger.info("[skip] %s exists", out.name)
            continue
        logger.info("[run]  %s ...", cn)
        prior = build_prior_for_content(
            cn,
            train_participants=args.train_participants,
            muse_vp_dir=args.muse_vp_dir,
            grid_h=args.grid_h, grid_w=args.grid_w,
            sigma_deg=args.sigma_deg,
        )
        if prior is None:
            continue
        payload = {
            "behav_xyzs":    prior.to(torch.float32),     # [N_5fps, N=512, 4]
            "coords":        compute_cell_coords(args.grid_h, args.grid_w).to(torch.float32),
            "grid_h":        args.grid_h,
            "grid_w":        args.grid_w,
            "sigma_deg":     args.sigma_deg,
            "n_train":       len(args.train_participants),
            "train_participants": list(args.train_participants),
            "content_name":  cn,
            "num_frames_5fps": int(prior.shape[0]),
        }
        torch.save(payload, out)
        logger.info("[done] %s : N_5fps=%d  size=%.1f KB",
                    cn, prior.shape[0],
                    os.path.getsize(out) / 1024)

    logger.info("all behavioural priors written to %s", args.output_dir)


if __name__ == "__main__":
    main()
