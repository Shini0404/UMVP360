"""
Audio feature extraction for MMVP2.

For each of the 14 360-degree videos under data/wu_mmsys_17/data/, we:
  1. Extract the audio track (ffmpeg) at 16 kHz mono WAV (this matches the
     standard SalViT360-AV / VGGish-style preprocessing — most 360° datasets
     publish stereo audio but the visual saliency network's audio branch
     downmixes to mono before computing log-mel).
  2. Compute log-mel spectrogram features: 64 mel bins, 25 ms window,
     10 ms hop  (following Hershey et al. "CNN Architectures for Large-Scale
     Audio Classification", which the SalViT360-AV audio backbone descends
     from).
  3. Average-pool the spectrogram into 5 fps frames so it aligns 1:1 with
     our head/eye/saliency timeline.

Output per video: [N_5fps, n_mels=64] log-mel features, plus a small
metadata dict.  Total disk: ~1 MB per video, ~14 MB total.

Run
---
    cd /media/user/HDD3/Shini/STAR_VP
    conda activate starvp
    python -m MMVP2.preprocessing.audio_features
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil#Used to check whether ffmpeg exists on the machine
import subprocess#Used to run external commands from Python. Here it runs the ffmpeg command that extracts audio from the .mp4.
import sys
import tempfile #Used to create a temporary .wav file.
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("mmvp2.audio_features")#Creates a named logger for this script. So messages from this file are labeled as coming from mmvp2.audio_features.


# ----------------------------------------------------------------------------- #
#  Config
# ----------------------------------------------------------------------------- #

REPO_ROOT          = Path("/media/user/HDD3/Shini/STAR_VP")
DEFAULT_VIDEO_DIR  = REPO_ROOT / "data" / "wu_mmsys_17" / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "MMVP2" / "processed_data" / "audio"

SAMPLE_RATE = 16_000#16,000 audio samples per second.
N_MELS      = 64#represent audio frequency using 64 mel bands.
WIN_MS      = 25#analyze audio in small windows of 25 milliseconds.
HOP_MS      = 10#after one 25ms window, move forward by 10 milliseconds.
TARGET_FPS  = 5

import re

def _content_name_from_video(fname: str) -> str | None:
    """video_12_LOSC_Football.mp4 -> LOSC_Football"""
    m = re.match(r"video_\d{2}_(.*?)\.mp4$", fname)
    return m.group(1) if m else None


# ----------------------------------------------------------------------------- #
#  ffmpeg + log-mel
# ----------------------------------------------------------------------------- #
#Before doing audio preprocessing, check that ffmpeg exists.
#If not, stop and tell the user how to install it.
def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg not found on PATH. Please install it (sudo apt install ffmpeg)."
        )


def _extract_wav(video_path: Path, sr: int = SAMPLE_RATE) -> Path:
    """Extract mono 16 kHz wav into a temp file and return its path."""
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="mmvp2_audio_")#This creates a temporary file name
    os.close(fd)
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-i", str(video_path),
        "-vn", "-ac", "1",#-vn: ignore video stream, audio only.-ac 1: convert audio to mono, one channel.
        "-ar", str(sr),#-ar 16000: resample to 16 kHz.
        "-f", "wav", tmp,
    ]
    subprocess.run(cmd, check=True)
    return Path(tmp)


def _read_wav(wav_path: Path) -> tuple[np.ndarray, int]:#This opens the WAV file and converts it into a numeric waveform array.
    """Read mono 16-bit PCM wav into a float32 array in [-1, 1]."""
    import wave
    with wave.open(str(wav_path), "rb") as wf:#opens WAV in binary read mode.
        sr     = wf.getframerate()#sample rate, expected 16000.
        nch    = wf.getnchannels()#number of channels, 1 mono or 2 stereo.
        sw     = wf.getsampwidth()#sample width, 2 bytes for 16-bit PCM.        
        nframe = wf.getnframes()#number of frames, total samples in the file.
        raw    = wf.readframes(nframe)# actual audio bytes.
    if sw != 2:
        raise ValueError(f"expected 16-bit PCM, got sampwidth={sw}")
    pcm = np.frombuffer(raw, dtype=np.int16)
    if nch > 1:
        pcm = pcm.reshape(-1, nch).mean(axis=1)
    audio = pcm.astype(np.float32) / 32768.0
    return audio, sr


def _log_mel(
    audio: np.ndarray,
    sr: int,
    n_mels: int = N_MELS,
    win_ms: int = WIN_MS,
    hop_ms: int = HOP_MS,
) -> np.ndarray:
    """Compute log-mel spectrogram via torchaudio (no librosa dep needed)."""
    import torchaudio.transforms as T
    win_length = int(round(sr * win_ms / 1000))#16000 samples/sec * 25/1000 sec = 400 samples
    hop_length = int(round(sr * hop_ms / 1000))#16000 samples/sec * 10/1000 sec = 160 samples
    n_fft = 1 << int(np.ceil(np.log2(win_length)))      # next power of 2 greater than or equal to win_length.

    mel = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=20.0, f_max=sr / 2,#Only use frequencies from 20 Hz up to Nyquist frequency. with 16 kHz sampling, the highest representable frequency is 8 kHz.
        power=2.0,
    )
    x = torch.from_numpy(audio).unsqueeze(0)              # [1, T]
    spec = mel(x).squeeze(0)                              # [n_mels, T_frames]Apply Mel Transform to get the mel spectrogram.
    """ Raw sound energy can have huge range. Some sounds are very quiet, some are very loud.

Log compresses the range and makes it easier for neural networks."""
    log_spec = torch.log(spec + 1e-6)                     # natural log, eps clamp
    return log_spec.numpy().astype(np.float32)            # [n_mels, T_frames]



"""Before this function, _log_mel() gives features every 10 ms.

That is very frequent:

100 audio feature frames per second
But MMVP2 works at:

5 frames per second
So we need to reduce audio from:

100 fps -> 5 fps
That means every model timestep should summarize:

200 ms of audio
because:

1 / 5 fps = 0.2 seconds = 200 ms
log_mel has shape:

[n_mels, T]
Usually:

[64, T] """
def _pool_to_fps(log_mel: np.ndarray, sr: int, hop_ms: int, target_fps: int) -> np.ndarray:
    """
    Average-pool the time axis of a [n_mels, T] log-mel spectrogram into
    one column per video frame at `target_fps` fps.
    """
    n_mels, n_frames = log_mel.shape
    # Each column of log_mel is `hop_ms` ms long. Each output bin is 1/target_fps s.
    samples_per_bin = (1000 / target_fps) / hop_ms       # e.g. 200ms / 10ms = 20
    samples_per_bin = max(1, int(round(samples_per_bin))) #every 20 log-mel columns become 1 MMVP2 audio timestep.
    n_out = n_frames // samples_per_bin
    if n_out == 0:
        # too short, just average everything
        out = log_mel.mean(axis=1, keepdims=True)
        return out.T                                      # [1, n_mels]
    log_mel_trim = log_mel[:, : n_out * samples_per_bin]
    out = log_mel_trim.reshape(n_mels, n_out, samples_per_bin).mean(axis=2)
    return out.T                                          # [n_out_5fps, n_mels]


# ----------------------------------------------------------------------------- #
#  Main
# ----------------------------------------------------------------------------- #

def process_one_video(
    video_path: Path,
    out_path:   Path,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    target_fps: int = TARGET_FPS,
) -> dict:
    wav_path = _extract_wav(video_path, sr)
    try:
        audio, real_sr = _read_wav(wav_path)
        if real_sr != sr:
            logger.warning("sr mismatch %d vs requested %d", real_sr, sr)
        log_mel = _log_mel(audio, real_sr, n_mels=n_mels)            # [n_mels, T_hop]
        feats   = _pool_to_fps(log_mel, real_sr, HOP_MS, target_fps)  # [N_5fps, n_mels]
    finally:
        try: wav_path.unlink()
        except OSError: pass

    # per-feature standardization across this video (zero-mean unit-std)
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-6
    feats_norm = (feats - mu) / sd

    payload = {
        "logmel":          torch.from_numpy(feats_norm.astype(np.float32)),     # [N_5fps, n_mels]
        "logmel_raw":      torch.from_numpy(feats.astype(np.float32)),
        "n_mels":          n_mels,
        "sample_rate":     sr,
        "win_ms":          WIN_MS,
        "hop_ms":          HOP_MS,
        "target_fps":      target_fps,
        "source_video":    video_path.name,
        "num_frames_5fps": int(feats.shape[0]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return {"src": video_path.name, "n_5": int(feats.shape[0])}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir",  type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-fps", type=int,  default=TARGET_FPS)
    parser.add_argument("--n-mels",     type=int,  default=N_MELS)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )

    _ensure_ffmpeg()
    if not args.video_dir.exists():
        raise SystemExit(f"video dir not found: {args.video_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(args.video_dir.glob("video_*.mp4"))
    logger.info("found %d videos in %s", len(videos), args.video_dir)
    if not videos:
        raise SystemExit("no .mp4 files in video dir")

    for vp in videos:
        content = _content_name_from_video(vp.name)
        if content is None:
            logger.warning("skip (cannot parse name): %s", vp.name)
            continue
        out = args.output_dir / f"{content}.pt"
        if out.exists() and not args.overwrite:
            logger.info("[skip] %s already exists", out.name)
            continue
        logger.info("processing %s -> %s", vp.name, out.name)
        info = process_one_video(
            vp, out,
            sr=args.sample_rate,
            n_mels=args.n_mels,
            target_fps=args.target_fps,
        )
        logger.info("[done]  %s : 5fps_frames=%d  size=%.1f KB",
                    content, info["n_5"],
                    os.path.getsize(out) / 1024)

    logger.info("all audio features written to %s", args.output_dir)


if __name__ == "__main__":
    main()
