"""
=============================================================================
MMVP2 Dataset — Multi-Modal Sliding-Window Loader
=============================================================================

Builds rich training samples on top of three sources of preprocessed data:

  1) Per-(participant, video) trajectory features (REUSED from muse_vp):
        muse_vp/processed_data/<P>/<video_folder>.pt
     Each .pt provides 5 fps tensors:
        head_3d  [N, 3]    head xyz on the unit sphere
        eye_3d   [N, 3]    eye gaze xyz
        offset   [N, 3]    eye-head offset (normalized)
        fixation [N, 2]    (is_fixating, normalized fixation_duration)
        face     [N, 5]    face engagement features
     plus metadata: participant_id, user_id, video_folder, content_name.

  2) Per-video dense SalViT360 saliency (PRODUCED by
     MMVP2/preprocessing/dense_saliency.py):
        MMVP2/processed_data/dense_sal/<content>.pt
        sal_xyzs [N, D_P=512, 4]  (x, y, z, s)

  3) (Optional) Per-video cross-user behavioural prior
        MMVP2/processed_data/behavioral_prior/<content>.pt
        behav_xyzs [N, D_P=512, 4]  (x, y, z, s)

  4) (Optional) Per-video log-mel audio features
        MMVP2/processed_data/audio/<content>.pt
        logmel [N, n_mels=64]

Outputs (per sample):
        past_traj   [T_M, 11]   head(3) + eye(3) + offset(3) + isfix(1) + dwell(1)
        sal_xyz     [T,    D_P, d_in_s]  T = T_M+T_H, d_in_s = 4 or 5
        audio       [T,    n_mels]       (zeros if not available)
        face        [T,    5]            raw face features over the full window
        user_id     []                   torch.long
        e_gaze_3d   [T_H,  3]            last observed eye gaze, replicated -- NO LEAKAGE
        target      [T_H,  3]            future head positions

The class also exposes `samples_by_participant` so the same fair_sample
8:1:1 split used by STAR-VP, MUSE-VP and MMVP can be applied unchanged.
=============================================================================
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset


# ----------------------------------------------------------------------------- #
#  Defaults / paths
# ----------------------------------------------------------------------------- #

REPO_ROOT = Path("/media/user/HDD3/Shini/STAR_VP")
MUSE_VP_PROCESSED   = REPO_ROOT / "muse_vp" / "processed_data"
DENSE_SAL_DIR       = REPO_ROOT / "MMVP2" / "processed_data" / "dense_sal"
BEHAV_PRIOR_DIR     = REPO_ROOT / "MMVP2" / "processed_data" / "behavioral_prior"
AUDIO_DIR           = REPO_ROOT / "MMVP2" / "processed_data" / "audio"

DEFAULT_T_M = 25
DEFAULT_T_H = 25
DEFAULT_N_MELS = 64 #represent audio frequency using 64 mel bands.
DEFAULT_FACE_DIM = 5
DEFAULT_AUX_DIM = 8                   # eye(3) + offset(3) + is_fix(1) + dwell(1)

ALL_PARTICIPANTS = [f"P{i:03d}" for i in range(1, 26)]
FAIR_PARTICIPANTS = ALL_PARTICIPANTS


# ----------------------------------------------------------------------------- #
#  Internal recording struct
# ----------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _Recording:
    participant_id: str
    user_id: int
    video_folder: str
    content_name: str
    head_3d:  torch.Tensor          # [N, 3]
    eye_3d:   torch.Tensor          # [N, 3]
    offset:   torch.Tensor          # [N, 3]
    fixation: torch.Tensor          # [N, 2]
    face:     torch.Tensor          # [N, 5]
    sal_xyzs: torch.Tensor          # [N, D_P, d_in_s]
    """N: number of timesteps.
      D_P: number of saliency tokens per frame.
      d_in_s: number of features per saliency token. 
      n_mels: number of mel bands."""
    audio:    torch.Tensor | None   # [N, n_mels] or None
    length: int


# ----------------------------------------------------------------------------- #
#  Dataset
# ----------------------------------------------------------------------------- #

class MMVP2Dataset(Dataset):
    """
    Multi-modal sliding-window dataset for MMVP2.

    Parameters
    ----------
    muse_vp_dir       : path to muse_vp/processed_data
    dense_sal_dir     : path to MMVP2/processed_data/dense_sal
    behav_prior_dir   : path to MMVP2/processed_data/behavioral_prior, or None to disable
    audio_dir         : path to MMVP2/processed_data/audio, or None to disable
    participants      : list of participant IDs to include (default = all 25)
    T_M, T_H          : past / future window lengths (default 25 / 25)
    use_dense         : if False, the dataset still loads dense_sal but uses
                        only the s channel (so d_in_s is reduced from 4 to 4
                        regardless -- this flag is kept for symmetry but
                        currently has no effect on output dim; the model
                        decides via d_in_s).
    """

    def __init__(
        self,
        muse_vp_dir: str | Path = MUSE_VP_PROCESSED,
        dense_sal_dir: str | Path = DENSE_SAL_DIR,
        behav_prior_dir: str | Path | None = BEHAV_PRIOR_DIR,
        audio_dir: str | Path | None = AUDIO_DIR,
        participants: list[str] | None = None,
        T_M: int = DEFAULT_T_M,
        T_H: int = DEFAULT_T_H,
        sal_pool_factor: int = 1, #reduces saliency grid size for faster training. 1 keeps full size, 2 pools 512 tokens into 128 tokens.
    ):
        super().__init__()
        self.T_M = int(T_M)
        self.T_H = int(T_H)
        self.sal_pool_factor = int(sal_pool_factor)
        self.muse_vp_dir = Path(muse_vp_dir)
        self.dense_sal_dir = Path(dense_sal_dir)
        self.behav_prior_dir = Path(behav_prior_dir) if behav_prior_dir is not None else None
        self.audio_dir = Path(audio_dir) if audio_dir is not None else None

        if not self.muse_vp_dir.exists():
            raise FileNotFoundError(f"muse_vp processed_data dir not found: {self.muse_vp_dir}")
        if not self.dense_sal_dir.exists():
            raise FileNotFoundError(f"dense saliency dir not found: {self.dense_sal_dir}")

        if participants is None:
            participants = ALL_PARTICIPANTS
        self.participants = list(participants)

        # ---- Load all per-video tensors once into memory caches ----
        self._sal_cache: dict[str, torch.Tensor] = {}                # content_name -> [N, D_P, 4 or 5]
        self._audio_cache: dict[str, torch.Tensor | None] = {}       # content_name -> [N, n_mels] or None

        # Load dense visual saliency
        for p in sorted(self.dense_sal_dir.glob("*.pt")):
            d = torch.load(p, weights_only=False, map_location="cpu")
            content = p.stem#gets the filename without .pt
            self._sal_cache[content] = d["sal_xyzs"].float()         # [N, D_P, 4]

        # Optionally augment saliency with behavioural prior as a 5th channel
        self.use_behav = (
            self.behav_prior_dir is not None and self.behav_prior_dir.exists()
        )
        if self.use_behav:
            for content, sal in self._sal_cache.items():
                bp_path = self.behav_prior_dir / f"{content}.pt"
                if not bp_path.exists():
                    continue
                bp = torch.load(bp_path, weights_only=False, map_location="cpu")
                behav = bp["behav_xyzs"].float()                     # [N_bp, D_P, 4]
                # align lengths to min
                n = min(sal.shape[0], behav.shape[0])
                visual_s = sal[:n, :, 3:4]                           # [n, D_P, 1]takes only the visual saliency score.
                behav_s  = behav[:n, :, 3:4]                         # [n, D_P, 1]takes only the behavioral prior score.
                xyz      = sal[:n, :, :3]                             # [n, D_P, 3]takes only the x, y, z coordinates.
                self._sal_cache[content] = torch.cat([xyz, visual_s, behav_s], dim=-1)

        # Optional spatial pooling of saliency tokens (reduces D_P and attention cost)
        # The raw 16×32 = 512-token grid is laid out in row-major order.
        # pool_factor=2 → 8×16 = 128 tokens (16× cheaper attention)
        # pool_factor=4 → 4×8 = 32 tokens (256× cheaper attention)
        if self.sal_pool_factor > 1:
            H_raw, W_raw = 16, 32
            f = self.sal_pool_factor
            H_out = H_raw // f
            W_out = W_raw // f
            for content, sal in self._sal_cache.items():
                # sal: [N, D_P, d]  →  [N, H, W, d]  →  pool  →  [N, H_out*W_out, d]
                N, D_P, d = sal.shape
                if D_P != H_raw * W_raw:
                    # Unexpected size – skip pooling for this video
                    continue
                sal_hwnd = sal.reshape(N, H_raw, W_raw, d)
                # Unfold into (f×f) blocks and mean-pool
                # Use simple strided gather: reshape to [N, H_out, f, W_out, f, d] → mean
                sal_pooled = (
                    sal_hwnd
                    .reshape(N, H_out, f, W_out, f, d)
                    .mean(dim=(2, 4))           # [N, H_out, W_out, d]
                    .reshape(N, H_out * W_out, d)
                )
                self._sal_cache[content] = sal_pooled

        # Optional audio
        if self.audio_dir is not None and self.audio_dir.exists():
            for p in sorted(self.audio_dir.glob("*.pt")):
                d = torch.load(p, weights_only=False, map_location="cpu")
                self._audio_cache[p.stem] = d["logmel"].float()      # [N, n_mels]
        else:
            for content in self._sal_cache:
                self._audio_cache[content] = None

        # ---- Iterate per-(participant, video) tensors and build samples ----
        self.recordings: list[_Recording] = [] #list of _Recording objects. Each object contains the metadata and tensors for a single video. P001 watching video_01,P001 watching video_02P002 watching video_01
        self.samples: list[tuple[int, int]] = [] #list of tuples. Each tuple contains the index of the recording and the timestep boundary. (0, 25) (1, 26) (2, 27) ...
        self.samples_by_participant: dict[str, list[int]] = {} #dict of lists. Each key is a participant ID and the value is a list of sample indices. P001: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] P002: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

        for p_id in self.participants:#loops over each participant,
            p_dir = self.muse_vp_dir / p_id#builds the participant folder path.
            if not p_dir.exists():
                continue
            for rec_path in sorted(p_dir.glob("video_*.pt")):#loads each processed video recording for that participant. video_0.pt, video_1.pt ,video_2.pt. Each file contains head, eye, fixation, face, and metadata for one participant-video pair.
                rec = torch.load(rec_path, weights_only=False, map_location="cpu")
                content = rec.get("content_name")
                if content is None or content not in self._sal_cache:
                    continue
                head_3d  = rec["head_3d"].float()
                eye_3d   = rec["eye_3d"].float()
                offset   = rec["offset"].float()
                fixation = rec["fixation"].float()
                face     = rec["face"].float()
                sal      = self._sal_cache[content]
                audio    = self._audio_cache.get(content)
                user_id  = int(rec.get("user_id", -1))

                lengths = [
                    head_3d.shape[0], eye_3d.shape[0], offset.shape[0],
                    fixation.shape[0], face.shape[0], sal.shape[0],
                ]
                if audio is not None:
                    lengths.append(audio.shape[0])
                n = min(lengths)#takes the shortest common length. This avoids indexing errors.
                if n < (self.T_M + self.T_H):
                    continue

                rec_obj = _Recording(
                    participant_id=str(rec.get("participant_id", p_id)),
                    user_id=user_id,
                    video_folder=str(rec.get("video_folder", rec_path.stem)),
                    content_name=str(content),
                    head_3d=head_3d[:n],
                    eye_3d=eye_3d[:n],
                    offset=offset[:n],
                    fixation=fixation[:n],
                    face=face[:n],
                    sal_xyzs=sal[:n],
                    audio=(audio[:n] if audio is not None else None),
                    length=n,
                )
                rec_idx = len(self.recordings)#stores that full recording in the dataset
                self.recordings.append(rec_obj)
                for t in range(self.T_M, n - self.T_H + 1): #This means every possible time t can become a prediction point.

#For each t, the model will use:

#past:   t - T_M  to  t - 1
#future: t        to  t + T_H - 1
                    sample_idx = len(self.samples)#stores the sample index in the dataset.
                    self.samples.append((rec_idx, t))#stores the sample as:which recording, which prediction 
                    self.samples_by_participant.setdefault(p_id, []).append(sample_idx)

        if not self.samples:
            raise RuntimeError(
                f"No valid sliding windows for T_M={self.T_M}, T_H={self.T_H} under "
                f"{self.muse_vp_dir} (participants={self.participants})."
            )

    # ---------------------------------------------------------------- #
    #  Sample access
    # ---------------------------------------------------------------- #

    def __len__(self) -> int: #If the dataset created 100,000 sliding windows, then len(dataset) is 100,000.
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec_idx, t = self.samples[idx] #Each index points to a specific sliding-window sample.
        rec = self.recordings[rec_idx]

        t0_past, t1_past = t - self.T_M, t
        t0_future, t1_future = t, t + self.T_H

        last_eye = rec.eye_3d[t1_past - 1]                                  # [3] last observed eye gaze.

        # Past 11D trajectory (head + eye + offset + is_fix + dwell)
        past_head     = rec.head_3d[t0_past:t1_past]                        # [T_M, 3]
        past_eye      = rec.eye_3d[t0_past:t1_past]                         # [T_M, 3]
        past_offset   = rec.offset[t0_past:t1_past]                         # [T_M, 3]
        past_fix_full = rec.fixation[t0_past:t1_past]                       # [T_M, 2]
        past_traj = torch.cat(
            [past_head, past_eye, past_offset, past_fix_full], dim=-1
        )                                                                    # [T_M, 11]

        # Saliency window covers past+future (matches STAR-VP / MUSE-VP)
        t0_sal, t1_sal = t0_past, t1_future
        sal_window = rec.sal_xyzs[t0_sal:t1_sal]                            # [T, D_P, 4 or 5]

        # Audio over the same past+future window (zeros if not loaded)
        if rec.audio is not None:
            audio_window = rec.audio[t0_sal:t1_sal]                         # [T, n_mels]
        else:
            audio_window = torch.zeros(
                self.T_M + self.T_H, DEFAULT_N_MELS, dtype=torch.float32
            )

        # Face over the same window
        face_window = rec.face[t0_sal:t1_sal]                               # [T, 5]

        return {
            "past_traj":  past_traj,
            "past_head":  past_head,                                         # convenience
            "sal_xyz":    sal_window,
            "audio":      audio_window,
            "face":       face_window,
            "e_gaze_3d":  last_eye.unsqueeze(0).expand(self.T_H, -1),
            "user_id":    torch.tensor(rec.user_id, dtype=torch.long),
            "target":     rec.head_3d[t0_future:t1_future],
        }

    # ---------------------------------------------------------------- #
    #  Train / val / test split (same logic as MUSE-VP / STAR-VP)
    # ---------------------------------------------------------------- #

    def split_indices(
        self,
        train_ratio: float = 0.8,
        val_ratio:   float = 0.1,
        test_ratio:  float = 0.1,
        seed: int = 42,
    ) -> tuple[list[int], list[int], list[int]]:
        if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6):
            raise ValueError("Split ratios must sum to 1.0")

        rng = random.Random(seed)
        train_idx: list[int] = []
        val_idx:   list[int] = []
        test_idx:  list[int] = []

        for _participant, indices in sorted(self.samples_by_participant.items()):
            idxs = list(indices)
            rng.shuffle(idxs)
            n = len(idxs)
            if n == 1:
                train_idx.extend(idxs); continue
            if n == 2:
                train_idx.append(idxs[0]); val_idx.append(idxs[1]); continue

            n_train = int(n * train_ratio)
            n_val   = int(n * val_ratio)
            n_test  = n - n_train - n_val
            if n_train == 0: n_train = 1
            if n_val   == 0: n_val   = 1
            n_test = n - n_train - n_val
            if n_test <= 0:
                if n_train >= n_val and n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1
                n_test = n - n_train - n_val
            if n_test <= 0:
                n_train = max(1, n - 2)
                n_val   = 1
                n_test  = n - n_train - n_val

            train_idx.extend(idxs[:n_train])
            val_idx.extend(idxs[n_train:n_train + n_val])
            test_idx.extend(idxs[n_train + n_val:])

        if not train_idx: raise RuntimeError("Train split empty")
        if not val_idx:   raise RuntimeError("Val split empty")
        if not test_idx:  raise RuntimeError("Test split empty")
        return train_idx, val_idx, test_idx


# ----------------------------------------------------------------------------- #
#  DataLoader builders
# ----------------------------------------------------------------------------- #

def _make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )
"""one sample:  [past_traj, saliency, audio, face, target]
one batch:   32 samples together """

def create_dataloaders(
    muse_vp_dir: str | Path = MUSE_VP_PROCESSED,
    dense_sal_dir: str | Path = DENSE_SAL_DIR,
    behav_prior_dir: str | Path | None = BEHAV_PRIOR_DIR,
    audio_dir: str | Path | None = AUDIO_DIR,
    batch_size: int = 32,
    num_workers: int = 4,
    T_M: int = DEFAULT_T_M,
    T_H: int = DEFAULT_T_H,
    seed: int = 42,
    return_test: bool = True,
    sal_pool_factor: int = 1,
) -> dict[str, DataLoader]:
    """Build train / val / test DataLoaders following the MUSE-VP / STAR-VP fair split."""
    full_ds = MMVP2Dataset(
        muse_vp_dir=muse_vp_dir,
        dense_sal_dir=dense_sal_dir,
        behav_prior_dir=behav_prior_dir,
        audio_dir=audio_dir,
        T_M=T_M, T_H=T_H,
        sal_pool_factor=sal_pool_factor,
    )
    train_idx, val_idx, test_idx = full_ds.split_indices(seed=seed)

    loaders = {
        "train": _make_loader(Subset(full_ds, train_idx), batch_size, True,  num_workers, True),
        "val":   _make_loader(Subset(full_ds, val_idx),   batch_size, False, num_workers, False),
    }
    if return_test:
        loaders["test"] = _make_loader(
            Subset(full_ds, test_idx), batch_size, False, num_workers, False,
        )
    loaders["full_dataset"] = full_ds   # type: ignore[assignment]
    loaders["splits"] = {                # type: ignore[assignment]
        "train": train_idx, "val": val_idx, "test": test_idx,
    }
    return loaders
