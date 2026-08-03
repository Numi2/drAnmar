# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
GIF_PATH = ROOT / "docs/media/dranmar-cuttable-tissue-curved-cuda.gif"
RECEIPT_PATH = ROOT / "docs/media/dranmar-cuttable-tissue-curved-cuda.json"
MOVING_GIF_PATH = ROOT / "docs/media/dranmar-moving-scalpel-cut-cuda.gif"
MOVING_RECEIPT_PATH = ROOT / "docs/media/dranmar-moving-scalpel-cut-cuda.json"


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


def test_real_moving_scalpel_cuda_render_matches_receipt_and_event_trace():
    receipt = json.loads(MOVING_RECEIPT_PATH.read_text(encoding="utf-8"))
    assert hashlib.sha256(MOVING_GIF_PATH.read_bytes()).hexdigest() == receipt["gif_sha256"]
    with Image.open(MOVING_GIF_PATH) as gif:
        assert gif.size == (receipt["width"], receipt["height"])
        assert gif.n_frames == receipt["encoded_frame_count"]
    assert receipt["trajectory_frame_count"] == 77
    assert receipt["warp_device"] == "cuda:0"
    assert receipt["path_segments"] == 64
    assert receipt["fracture_event_count"] == 1248
    assert receipt["released_pair_count"] == 85
    assert receipt["retained_anchor_node_count"] > 0
    assert receipt["entry_boundary_pair_count"] > 0
    assert receipt["exit_boundary_pair_count"] > 0
    assert receipt["entry_boundary_mean_gap_m"] >= 1.0e-5
    assert receipt["exit_boundary_mean_gap_m"] >= 1.0e-5
    assert receipt["boundary_opening_gates_passed"] is True
    assert receipt["cpu_event_trace_match"] is True
    assert receipt["displacement_exaggeration"] == 1.0
    assert receipt["generated_imagery"] is False
    assert receipt["real_time_transient"] is False
