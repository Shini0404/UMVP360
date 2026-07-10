"""
One-shot orchestrator: dense saliency + audio + behavioural prior.

Run from the repo root in the starvp env:
    cd /media/user/HDD3/Shini/STAR_VP
    conda activate starvp
    python -m MMVP2.preprocessing.prepare_mmvp2_data

The participant-level features (head_3d, eye_3d, offset, fixation, face) are
already produced by muse_vp's data_preprocessing and live under
muse_vp/processed_data/<P>/<video_folder>.pt — MMVP2's dataset reuses those
directly and DOES NOT regenerate them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import dense_saliency
from . import audio_features
from . import behavioral_prior


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-saliency",  action="store_true")
    parser.add_argument("--skip-audio",     action="store_true")
    parser.add_argument("--skip-prior",     action="store_true")
    parser.add_argument("--overwrite",      action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )
    log = logging.getLogger("mmvp2.prepare")

    if not args.skip_saliency:
        log.info("===== STEP 1/3 : dense SalViT360 saliency (16x32) =====")
        argv_s = []
        if args.overwrite:
            argv_s.append("--overwrite")
        dense_saliency.main(argv_s)

    if not args.skip_audio:
        log.info("===== STEP 2/3 : log-mel audio features (5 fps) =====")
        argv_a = []
        if args.overwrite:
            argv_a.append("--overwrite")
        audio_features.main(argv_a)

    if not args.skip_prior:
        log.info("===== STEP 3/3 : cross-user behavioural prior =====")
        argv_b = []
        if args.overwrite:
            argv_b.append("--overwrite")
        behavioral_prior.main(argv_b)

    log.info("ALL DONE.  Outputs under MMVP2/processed_data/")


if __name__ == "__main__":
    main()
