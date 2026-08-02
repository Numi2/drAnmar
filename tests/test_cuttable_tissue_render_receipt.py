# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
GIF_PATH = ROOT / "docs/media/dranmar-cuttable-tissue-curved-cuda.gif"
RECEIPT_PATH = ROOT / "docs/media/dranmar-cuttable-tissue-curved-cuda.json"


def test_real_cuda_render_receipt_matches_encoded_gif():
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert hashlib.sha256(GIF_PATH.read_bytes()).hexdigest() == receipt["gif_sha256"]
    with Image.open(GIF_PATH) as gif:
        assert gif.size == (receipt["width"], receipt["height"])
        assert gif.n_frames == receipt["encoded_frame_count"]
    assert receipt["trajectory_frame_count"] == 64
    assert receipt["trajectory_steps"] == 4000
    assert receipt["warp_device"] == "cuda:0"
    assert receipt["displacement_exaggeration"] == 1.0
    assert receipt["generated_imagery"] is False
