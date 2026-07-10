"""
=============================================================================
MMVP2 Training Script  (extends STAR-VP)
=============================================================================

Builds on the STAR-VP training procedure but adds all MMVP2 levers as
command-line switches so a single script + a single model class can be used
for the full ablation study (run_ablation_mmvp2.sh).

Key additions over STAR-VP train.py:
  - MMVP2Dataset (rich multi-modal batches: past_traj, sal_xyz, audio,
    face, user_id, target).
  - MMVP2 model wired to optional components (aux trajectory inputs,
    user embedding, FiLM, audio gate, face gate, dense saliency +
    behavioural prior).
  - Horizon-weighted Orthodromic Distance loss (w_t = 1 + t/T_H), which
    emphasises long-horizon predictions where every prior model degrades.
  - Per-horizon error tracking, CSV/text logs, TensorBoard, AMP, cosine
    LR schedule, gradient clipping, early stopping (all carried over).

Typical training:
    python -m MMVP2.train \\
        --use_aux --use_audio --use_face --use_user --use_user_film \\
        --use_behav --horizon_weighted \\
        --tag rowFull

Each ablation row toggles exactly one component.  See
run_ablation_mmvp2.sh for the full table.
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .compat import torch_load
from .dataset import (
    MMVP2Dataset, MUSE_VP_PROCESSED, DENSE_SAL_DIR, BEHAV_PRIOR_DIR, AUDIO_DIR,
    create_dataloaders,
)
from .models import MMVP2

logger = logging.getLogger("mmvp2.train")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):  # type: ignore[misc]
        return iterable


# ========================================================================= #
#  Loss
# ========================================================================= #

def _angular_step_error(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """[B, T_H] geodesic error in radians."""
    cos_sim = (pred * target).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_sim)


def orthodromic_distance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    horizon_weighted: bool = False,
) -> torch.Tensor:
    """
    Mean Orthodromic (geodesic) distance between predicted and target unit
    vectors on the sphere.

    If horizon_weighted is True, weights the per-step error by
        w_t = 1 + t / T_H,        t = 0, 1, ..., T_H-1
    which emphasises later predictions where head motion is hardest to
    predict (Section: horizon-weighted loss).
    """
    err = _angular_step_error(pred, target)                  # [B, T_H]
    if horizon_weighted:
        T_H = err.shape[-1]
        t = torch.arange(T_H, device=err.device, dtype=err.dtype)
        w = 1.0 + t / max(T_H, 1)                             # [T_H]
        err = err * w
        return err.mean()
    return err.mean()


@torch.no_grad()
def per_horizon_error_deg(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """[T_H] mean angular error per horizon step, in degrees."""
    err = _angular_step_error(pred, target)
    return err.mean(dim=0) * (180.0 / math.pi)


# ========================================================================= #
#  Train / Validate
# ========================================================================= #

def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    use_aux: bool,
    use_audio: bool,
    use_face: bool,
    use_user: bool,
    horizon_weighted: bool,
    progress_desc: str | None = None,
) -> float:
    model.train()
    total = 0.0
    n_batches = 0
    batches = tqdm(loader, desc=progress_desc, dynamic_ncols=True) if progress_desc else loader

    for batch in batches:
        batch = _move_batch(batch, device)
        past = batch["past_traj"] if use_aux else batch["past_head"]
        target = batch["target"]
        sal_xyz = batch["sal_xyz"]
        kwargs = {}
        if use_audio:
            kwargs["audio"] = batch["audio"]
        if use_face:
            kwargs["face"] = batch["face"]
        if use_user:
            kwargs["user_id"] = batch["user_id"]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            p_hat = model(past, sal_xyz, **kwargs)
            loss = orthodromic_distance_loss(
                p_hat, target, horizon_weighted=horizon_weighted,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total += loss.item()
        n_batches += 1

    return total / max(n_batches, 1)


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    use_aux: bool,
    use_audio: bool,
    use_face: bool,
    use_user: bool,
) -> dict:
    model.eval()
    total = 0.0
    n_batches = 0
    horizons: list[torch.Tensor] = []

    for batch in loader:
        batch = _move_batch(batch, device)
        past = batch["past_traj"] if use_aux else batch["past_head"]
        target = batch["target"]
        sal_xyz = batch["sal_xyz"]
        kwargs = {}
        if use_audio:
            kwargs["audio"] = batch["audio"]
        if use_face:
            kwargs["face"] = batch["face"]
        if use_user:
            kwargs["user_id"] = batch["user_id"]

        with torch.amp.autocast("cuda", enabled=use_amp):
            p_hat = model(past, sal_xyz, **kwargs)
            loss = orthodromic_distance_loss(p_hat, target, horizon_weighted=False)

        total += loss.item()
        horizons.append(per_horizon_error_deg(p_hat, target))
        n_batches += 1

    mean = total / max(n_batches, 1)
    per_h = (
        torch.stack(horizons).mean(dim=0) if horizons else torch.zeros(1)
    )
    return {
        "loss_rad": mean,
        "loss_deg": mean * (180.0 / math.pi),
        "per_horizon_deg": per_h,
    }


# ========================================================================= #
#  Checkpoint helpers
# ========================================================================= #

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    best_val_loss: float,
    config: dict,
) -> None:
    state = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": (
            model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict:
    ckpt = torch_load(path, map_location="cpu", weights_only=False)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


class EarlyStopping:
    def __init__(self, patience: int = 15):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, v: float) -> bool:
        if v < self.best - 1e-8:
            self.best = v
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ========================================================================= #
#  Main
# ========================================================================= #

def train(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"mmvp2_{args.tag}_{timestamp}" if args.tag else f"mmvp2_{timestamp}"
    run_dir = Path(args.run_dir) / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "train.log"),
        ],
    )

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    except ImportError:
        logger.warning("tensorboard not installed; skipping TB logging.")

    config = vars(args)
    logger.info("=" * 72)
    logger.info(f"MMVP2 training run: {run_name}")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Device: {device}  AMP: {use_amp}")
    logger.info("=" * 72)
    logger.info("Toggles: " + json.dumps({
        "use_aux": args.use_aux,
        "use_audio": args.use_audio,
        "use_face": args.use_face,
        "use_user": args.use_user,
        "use_user_film": args.use_user_film,
        "use_behav": args.use_behav,
        "horizon_weighted": args.horizon_weighted,
    }))
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---- Per-epoch metrics files (training_log + CSV) ----
    training_log_path = run_dir / "training_log.txt"
    metrics_csv_path = run_dir / "metrics.csv"
    log_f = open(training_log_path, "w", encoding="utf-8")
    csv_f = open(metrics_csv_path, "w", newline="", encoding="utf-8")
    csv_writer: csv.DictWriter | None = None

    def _write_log(epoch, train_loss, val_loss, val_deg, lr, elapsed, per_h, is_best):
        nonlocal csv_writer
        T_H = len(per_h)
        fps = 5
        h1 = per_h[fps - 1].item()  if T_H >= fps      else float("nan")
        h3 = per_h[3 * fps - 1].item() if T_H >= 3 * fps else float("nan")
        h5 = per_h[-1].item()       if T_H > 0         else float("nan")

        block = (
            f"{'=' * 80}\n"
            f"Epoch {epoch + 1}/{args.epochs} | time={elapsed:.1f}s | lr={lr:.6e}\n"
            f"  train_loss_rad={train_loss:.8f}\n"
            f"  val_loss_rad={val_loss:.8f}  val_loss_deg={val_deg:.6f}\n"
            f"  per_horizon_deg @1s={h1:.6f} @3s={h3:.6f} @5s={h5:.6f}\n"
            f"  new_best={'yes' if is_best else 'no'}\n"
        )
        log_f.write(block); log_f.flush()

        row = {
            "epoch": epoch + 1,
            "train_loss_rad": f"{train_loss:.8f}",
            "val_loss_rad": f"{val_loss:.8f}",
            "val_loss_deg": f"{val_deg:.8f}",
            "lr": f"{lr:.8e}",
            "elapsed_s": f"{elapsed:.4f}",
            "new_best": int(is_best),
        }
        for t_idx in range(T_H):
            sec = (t_idx + 1) / fps
            row[f"err_{sec:.1f}s_deg"] = f"{per_h[t_idx].item():.8f}"

        if csv_writer is None:
            csv_writer = csv.DictWriter(csv_f, fieldnames=list(row.keys()))
            csv_writer.writeheader()
        csv_writer.writerow(row); csv_f.flush()

    log_f.write(
        "MMVP2 Training Log\n"
        f"Run: {run_name}\n"
        f"data_dir={args.muse_vp_dir} t_m={args.t_m} t_h={args.t_h} "
        f"batch_size={args.batch_size} lr={args.lr} epochs={args.epochs}\n"
        f"toggles: use_aux={args.use_aux} use_audio={args.use_audio} "
        f"use_face={args.use_face} use_user={args.use_user} "
        f"use_user_film={args.use_user_film} use_behav={args.use_behav} "
        f"horizon_weighted={args.horizon_weighted}\n"
        f"{'=' * 80}\n\n"
    )
    log_f.flush()

    # Resolve spatial layer counts (allow separate enc/dec or fall back to shared)
    spatial_enc_layers = args.spatial_enc_layers if args.spatial_enc_layers is not None else args.spatial_layers
    spatial_dec_layers = args.spatial_dec_layers if args.spatial_dec_layers is not None else args.spatial_layers
    args.spatial_enc_layers = spatial_enc_layers
    args.spatial_dec_layers = spatial_dec_layers

    # ---- Data ----
    behav_dir = BEHAV_PRIOR_DIR if args.use_behav else None
    audio_dir = AUDIO_DIR if args.use_audio else None
    loaders = create_dataloaders(
        muse_vp_dir=args.muse_vp_dir,
        dense_sal_dir=args.dense_sal_dir,
        behav_prior_dir=behav_dir,
        audio_dir=audio_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        T_M=args.t_m, T_H=args.t_h,
        seed=args.split_seed,
        return_test=False,
        sal_pool_factor=args.sal_pool_factor,
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    full_ds: MMVP2Dataset = loaders["full_dataset"]
    n_users = max(r.user_id for r in full_ds.recordings) + 1
    logger.info(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    logger.info(f"Detected num_users (max user_id + 1): {n_users}")

    # ---- Model ----
    head_dim = 3
    aux_dim  = 8 if args.use_aux else 0       # eye3 + offset3 + fix2
    d_in_s   = 5 if args.use_behav else 4     # behavioural prior adds a channel
    d_audio  = args.d_audio if args.use_audio else 0
    d_face   = args.d_face  if args.use_face  else 0
    n_users_used = n_users if args.use_user else 0
    d_user_used  = args.d_user if args.use_user else 0

    model = MMVP2(
        head_dim=head_dim,
        aux_dim=aux_dim,
        d_in_s=d_in_s,
        d_in_p=4,
        lstm_hidden_dim=args.lstm_hidden,
        lstm_num_layers=args.lstm_layers,
        d_c=args.d_c,
        n_heads=args.n_heads,
        spatial_enc_layers=args.spatial_enc_layers,
        spatial_dec_layers=args.spatial_dec_layers,
        d_pe=args.d_pe,
        d_te=args.d_te,
        temporal_enc_layers=args.temporal_layers,
        temporal_dec_layers=args.temporal_layers,
        d_g=args.d_g,
        num_users=n_users_used,
        d_user=d_user_used,
        use_user_film=args.use_user_film,
        n_mels=64,
        d_audio=d_audio,
        d_face=d_face,
        face_input_dim=5,
        t_m=args.t_m,
        t_h=args.t_h,
        dropout=args.dropout,
        normalize_output=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"Component params: {model.count_parameters()}")

    # ---- Optimizer / Scheduler / Scaler ----
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min,
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    early_stop = EarlyStopping(patience=args.patience)

    best_val = float("inf")
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = load_checkpoint(Path(args.resume), model, optimizer, scheduler, scaler)
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val_loss"]
        early_stop.best = best_val
        logger.info(f"Resumed at epoch {start_epoch}, best_val={best_val:.6f}")

    if not args.no_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            logger.info("torch.compile enabled")
        except Exception as e:
            logger.warning(f"torch.compile failed ({e})")

    # ---- Loop ----
    logger.info(f"Training for {args.epochs} epochs (from epoch {start_epoch})")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model, train_loader, optimizer, args.grad_clip, device, scaler, use_amp,
            args.use_aux, args.use_audio, args.use_face, args.use_user,
            args.horizon_weighted,
            progress_desc=f"Train {epoch + 1}/{args.epochs}",
        )

        val_metrics = validate(
            model, val_loader, device, use_amp,
            args.use_aux, args.use_audio, args.use_face, args.use_user,
        )
        val_loss = val_metrics["loss_rad"]
        val_deg  = val_metrics["loss_deg"]
        per_h    = val_metrics["per_horizon_deg"]

        scheduler.step()
        elapsed = time.perf_counter() - t0

        logger.info(
            f"Epoch {epoch:>3d}/{args.epochs} | "
            f"train={train_loss:.6f} rad | "
            f"val={val_loss:.6f} rad ({val_deg:.2f}\u00b0) | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )
        if writer is not None:
            writer.add_scalar("Loss/train_rad", train_loss, epoch)
            writer.add_scalar("Loss/val_rad", val_loss, epoch)
            writer.add_scalar("Loss/val_deg", val_deg, epoch)
            writer.add_scalar("LR", lr, epoch)
            for t_idx in range(len(per_h)):
                sec = (t_idx + 1) / 5.0
                writer.add_scalar(f"PerHorizon/{sec:.1f}s", per_h[t_idx].item(), epoch)
        if len(per_h) >= 5:
            logger.info(
                f"         per-horizon: 1s={per_h[4].item():.2f}\u00b0 | "
                f"3s={per_h[14].item() if len(per_h) >= 15 else float('nan'):.2f}\u00b0 | "
                f"5s={per_h[-1].item():.2f}\u00b0"
            )

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt",
                model, optimizer, scheduler, scaler,
                epoch, best_val, config,
            )
            logger.info(f"         \u2605 New best (val={val_deg:.2f}\u00b0)")
        save_checkpoint(
            ckpt_dir / "latest.pt",
            model, optimizer, scheduler, scaler,
            epoch, best_val, config,
        )

        _write_log(epoch, train_loss, val_loss, val_deg, lr, elapsed, per_h, is_best)

        if early_stop.step(val_loss):
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {args.patience} epochs)"
            )
            log_f.write(
                f"\nEarly stopping at epoch {epoch + 1} "
                f"(no improvement for {args.patience} epochs).\n"
            )
            log_f.flush()
            break

    if writer is not None:
        writer.close()
    log_f.write(
        f"\nTraining complete.\n"
        f"Best validation loss: {best_val:.8f} rad "
        f"({best_val * 180 / math.pi:.6f} deg)\n"
        f"Best checkpoint: {ckpt_dir / 'best.pt'}\n"
    )
    log_f.close()
    csv_f.close()
    logger.info(f"Per-epoch metrics: {training_log_path} | {metrics_csv_path}")
    logger.info("=" * 72)
    logger.info(
        f"Training complete. Best val: {best_val:.6f} rad "
        f"({best_val * 180 / math.pi:.2f}\u00b0)"
    )
    logger.info(f"Best checkpoint: {ckpt_dir / 'best.pt'}")
    logger.info(f"Run directory:   {run_dir}")
    logger.info("=" * 72)
    return model, best_val


# ========================================================================= #
#  CLI
# ========================================================================= #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MMVP2 training (extends STAR-VP)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Data ---
    p.add_argument("--muse_vp_dir",   type=str, default=str(MUSE_VP_PROCESSED))
    p.add_argument("--dense_sal_dir", type=str, default=str(DENSE_SAL_DIR))
    p.add_argument("--split_seed",    type=int, default=42)

    # --- Architecture (Table 1) ---
    p.add_argument("--t_m", type=int, default=25)
    p.add_argument("--t_h", type=int, default=25)
    p.add_argument("--lstm_hidden", type=int, default=256)
    p.add_argument("--lstm_layers", type=int, default=2)
    p.add_argument("--d_c", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--spatial_layers",     type=int, default=2,
                    help="Spatial enc AND dec layers (legacy; overridden by --spatial_enc_layers / --spatial_dec_layers)")
    p.add_argument("--spatial_enc_layers", type=int, default=None,
                    help="Spatial encoder self-attention layers. "
                         "Set to 0 to skip D_P² self-attn → ~16× faster with 512 tokens.")
    p.add_argument("--spatial_dec_layers", type=int, default=None,
                    help="Spatial decoder cross-attention layers (default = --spatial_layers).")
    p.add_argument("--sal_pool_factor",    type=int, default=1,
                    help="Pool saliency tokens at load time: 1=keep 512, 2→128 (8×16), 4→32. "
                         "Reduces attention cost by factor².")
    p.add_argument("--temporal_layers", type=int, default=2)
    p.add_argument("--d_pe", type=int, default=129)
    p.add_argument("--d_te", type=int, default=127)
    p.add_argument("--d_g", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)

    # --- MMVP2 ablation toggles ---
    p.add_argument("--use_aux",       action="store_true",
                    help="LSTM input = head(3)+eye(3)+offset(3)+fix(2) (11D)")
    p.add_argument("--use_behav",     action="store_true",
                    help="Add behavioural prior as 5th saliency channel")
    p.add_argument("--use_audio",     action="store_true",
                    help="Add log-mel audio context to gating module")
    p.add_argument("--d_audio",       type=int, default=64)
    p.add_argument("--use_face",      action="store_true",
                    help="Add face engagement context to gating module")
    p.add_argument("--d_face",        type=int, default=4)
    p.add_argument("--use_user",      action="store_true",
                    help="User embedding initialises LSTM state")
    p.add_argument("--d_user",        type=int, default=24)
    p.add_argument("--use_user_film", action="store_true",
                    help="FiLM modulation of spatial-attention output")
    p.add_argument("--horizon_weighted", action="store_true",
                    help="Weight OD loss by w_t = 1 + t/T_H")

    # --- Training ---
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--optimizer", type=str, choices=("adam", "adamw"), default="adam")
    p.add_argument("--eta_min", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=15)

    # --- Infrastructure ---
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--run_dir", type=str, default="runs")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--tag", type=str, default="")

    args = p.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


if __name__ == "__main__":
    train(parse_args())
