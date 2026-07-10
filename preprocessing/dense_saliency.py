"""
Dense saliency pooling for MMVP2.

Replaces STAR-VP's `sal_xyz` top-K representation (128 points/frame) with a
fixed 16x32 spherical token grid (= 512 tokens/frame) that preserves the
spatial structure of SalViT360's predictions.

Each token carries:
    [x, y, z, s]
where (x, y, z) is the unit-sphere centre of the ERP cell (computed once and
shared across videos / time) and s is the mean saliency inside the cell at
the current timestep.

Inputs
------
SalViT360 saliency maps under
    /media/user/HDD3/Shini/STAR_VP/SalViT360/saliency_salvit360av/
    salvit360av_<video_name>_saliency.pt
Each file is a torch.FloatTensor [N_30fps, 480, 960].

Outputs
-------
MMVP2/processed_data/dense_sal/
    <content_name>.pt  -> dict with
        sal_xyzs    [N_5fps, 512, 4]   (x, y, z, s)
        coords      [512, 3]           cell centres (shared)
        grid_h      16
        grid_w      32
        content_name, num_frames_5fps, target_fps, source_video

Run
---
    cd /media/user/HDD3/Shini/STAR_VP
    conda activate starvp
    python -m MMVP2.preprocessing.dense_saliency
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- #
#  Config
# ----------------------------------------------------------------------------- #

VIDEO_FPS = 30
TARGET_FPS = 5
DOWNSAMPLE_STEP = VIDEO_FPS // TARGET_FPS    # 6

GRID_H = 16
GRID_W = 32
N_TOKENS = GRID_H * GRID_W                   # 512

ERP_H = 480
ERP_W = 960

REPO_ROOT = Path("/media/user/HDD3/Shini/STAR_VP")
DEFAULT_INPUT_DIR = REPO_ROOT / "SalViT360" / "saliency_salvit360av"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "MMVP2" / "processed_data" / "dense_sal"


# Map raw SalViT filename -> short content name we use throughout MMVP2.
# Filenames look like: salvit360av_video_12_LOSC_Football_saliency.pt
# Content name is everything between "video_NN_" and "_saliency".
def _content_name_from_filename(fname: str) -> str | None:
    m = re.match(r"salvit360av_video_\d{2}_(.*?)_saliency\.pt$", fname)
    return m.group(1) if m else None


logger = logging.getLogger("mmvp2.dense_saliency")


# ----------------------------------------------------------------------------- #
#  Cell-centre coordinate table (computed once, shared by all videos/frames)
# ----------------------------------------------------------------------------- #

def compute_cell_coords(grid_h: int = GRID_H, grid_w: int = GRID_W) -> torch.Tensor:
    """
    Compute 3D unit-sphere centres of an `grid_h x grid_w` ERP cell grid.

    Convention MUST match STAR-VP's SalMapProcessor (star_vp/models/salmap_processor.py)
    which uses the physics spherical convention from the paper:

        theta_polar = (2*pi / W) * (j + 0.5)          azimuth   in [0, 2*pi)
        phi_polar   = ( pi  / H) * (i + 0.5)          polar from +z in (0, pi)

        x = cos(theta_polar) * sin(phi_polar)
        y = sin(theta_polar) * sin(phi_polar)
        z = cos(phi_polar)

    With this convention:
        i=0    (top row)     -> phi  ~ 0     -> z = +1   (north pole)
        i=H-1  (bottom row)  -> phi  ~ pi    -> z = -1   (south pole)
        i=H/2, j=0           -> phi=pi/2, theta=0 -> x = +1  (equator front,
                                                              matches HeadYaw=0
                                                              -> head forward)

    This matters because STAR-VP's `head_3d` is computed with
    euler_to_unit_vector(yaw=0, pitch=0) -> (1, 0, 0).  We need our saliency
    coordinates to use the SAME forward axis so that head and saliency are
    spatially aligned at HeadYaw=0.

    Returns
    -------
    coords : Tensor [grid_h * grid_w, 3]   (unit vectors)
    """
    rows = torch.arange(grid_h, dtype=torch.float64)
    cols = torch.arange(grid_w, dtype=torch.float64)
    theta = (2.0 * torch.pi / grid_w) * (cols + 0.5)                    # [grid_w]
    phi   = (torch.pi       / grid_h) * (rows + 0.5)                    # [grid_h]

    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing="ij")    # both [gH, gW]
    sin_phi = torch.sin(phi_grid)
    x = torch.cos(theta_grid) * sin_phi
    y = torch.sin(theta_grid) * sin_phi
    z = torch.cos(phi_grid)

    coords = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(torch.float32)
    coords = coords / coords.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return coords


# ----------------------------------------------------------------------------- #
#  Pool a single video's saliency map
# ----------------------------------------------------------------------------- #

def _pool_saliency(
    sal_30fps: torch.Tensor,
    grid_h: int = GRID_H,
    grid_w: int = GRID_W,
    target_fps: int = TARGET_FPS,
    chunk: int = 256,
) -> torch.Tensor:
    """
    Pool a [N_30fps, 480, 960] saliency tensor to
    [N_5fps, grid_h, grid_w] via average pooling, in chunks to avoid RAM blow-up.
    """
    if sal_30fps.ndim != 3:
        raise ValueError(f"expected [N, H, W], got {tuple(sal_30fps.shape)}")
    n_30 = sal_30fps.shape[0]
    step = VIDEO_FPS // target_fps
    idxs = torch.arange(0, n_30, step)      # 5fps indices into raw 30fps stream
    n_5 = idxs.numel()

    out = torch.empty((n_5, grid_h, grid_w), dtype=torch.float32)

    for c0 in range(0, n_5, chunk):
        c1 = min(c0 + chunk, n_5)
        frames = sal_30fps[idxs[c0:c1]]           # [C, ERP_H, ERP_W]
        frames = frames.unsqueeze(1).float()      # [C, 1, ERP_H, ERP_W]
        pooled = F.adaptive_avg_pool2d(frames, (grid_h, grid_w))     # [C, 1, gH, gW]
        out[c0:c1] = pooled.squeeze(1)
        del frames, pooled

    return out                                    # [N_5fps, grid_h, grid_w]


def _normalize_per_frame(sal_5fps_grid: torch.Tensor) -> torch.Tensor:
    """
    Per-frame standardization: subtract mean, divide by std (eps clamped).
    Keeps the relative spatial structure of saliency while removing the
    SalViT360 per-frame intensity drift (range was [0, 5.2], not [0, 1]).
    The PAVER-based STAR-VP path used a similar normalization implicitly via
    SalMapProcessor's weight column.
    """
    flat = sal_5fps_grid.flatten(start_dim=1)              # [N, 512]
    mu = flat.mean(dim=1, keepdim=True)
    sd = flat.std(dim=1, keepdim=True).clamp(min=1e-6)
    flat = (flat - mu) / sd
    return flat.reshape_as(sal_5fps_grid)


def process_one_video(
    src_path: Path,
    dst_path: Path,
    grid_h: int = GRID_H,
    grid_w: int = GRID_W,
    target_fps: int = TARGET_FPS,
    coords: torch.Tensor | None = None,
    chunk: int = 256,
) -> dict:
    """Pool one SalViT saliency .pt file into the [N_5fps, 512, 4] format."""
    if coords is None:
        coords = compute_cell_coords(grid_h, grid_w)

    sal_30fps = torch.load(src_path, weights_only=False, map_location="cpu")
    if not isinstance(sal_30fps, torch.Tensor):
        raise TypeError(f"{src_path}: not a tensor (got {type(sal_30fps)})")
    n_30 = sal_30fps.shape[0]

    pooled = _pool_saliency(
        sal_30fps, grid_h=grid_h, grid_w=grid_w,
        target_fps=target_fps, chunk=chunk,
    )                                                # [N_5fps, gH, gW]
    norm = _normalize_per_frame(pooled)              # [N_5fps, gH, gW]
    s_flat = norm.flatten(start_dim=1).unsqueeze(-1)   # [N_5fps, 512, 1]

    coords_exp = coords.unsqueeze(0).expand(s_flat.shape[0], -1, -1)   # [N_5fps, 512, 3]
    sal_xyzs = torch.cat([coords_exp, s_flat], dim=-1).contiguous()    # [N_5fps, 512, 4]

    payload = {
        "sal_xyzs": sal_xyzs.to(torch.float32),
        "coords": coords.to(torch.float32),
        "grid_h": grid_h,
        "grid_w": grid_w,
        "num_frames_30fps": int(n_30),
        "num_frames_5fps": int(sal_xyzs.shape[0]),
        "target_fps": target_fps,
        "source_video": str(src_path.name),
        "normalization": "per-frame zero-mean unit-std",
    }
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst_path)

    del sal_30fps, pooled, norm, s_flat, coords_exp, sal_xyzs
    gc.collect()
    return {
        "src": str(src_path.name),
        "dst": str(dst_path.name),
        "n_30": n_30,
        "n_5":  payload["num_frames_5fps"],
    }


# ----------------------------------------------------------------------------- #
#  CLI
# ----------------------------------------------------------------------------- #

def _iter_input_files(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.glob("salvit360av_*_saliency.pt")):
        yield p


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-h",     type=int,  default=GRID_H)
    parser.add_argument("--grid-w",     type=int,  default=GRID_W)
    parser.add_argument("--target-fps", type=int,  default=TARGET_FPS)
    parser.add_argument("--chunk",      type=int,  default=256,
                        help="frames per pooling chunk (controls RAM peak)")
    parser.add_argument("--overwrite",  action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )

    if not args.input_dir.exists():
        raise SystemExit(f"input dir not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coords = compute_cell_coords(args.grid_h, args.grid_w)
    logger.info("cell coords: %s, |coord|=%.6f..%.6f",
                tuple(coords.shape),
                float(coords.norm(dim=-1).min()),
                float(coords.norm(dim=-1).max()))

    files = list(_iter_input_files(args.input_dir))
    logger.info("found %d source files in %s", len(files), args.input_dir)
    if not files:
        raise SystemExit("no input files matched")

    for src in files:
        content = _content_name_from_filename(src.name)
        if content is None:
            logger.warning("skip (cannot parse name): %s", src.name)
            continue
        dst = args.output_dir / f"{content}.pt"
        if dst.exists() and not args.overwrite:
            logger.info("[skip] %s already exists", dst.name)
            continue

        logger.info("processing %s -> %s ...", src.name, dst.name)
        info = process_one_video(
            src, dst,
            grid_h=args.grid_h, grid_w=args.grid_w,
            target_fps=args.target_fps,
            coords=coords, chunk=args.chunk,
        )
        logger.info(
            "[done]  %s : 30fps=%d -> 5fps=%d   (file: %.1f MB)",
            content, info["n_30"], info["n_5"],
            os.path.getsize(dst) / 1024 / 1024,
        )

    logger.info("all videos processed; output dir: %s", args.output_dir)


if __name__ == "__main__":
    main()
