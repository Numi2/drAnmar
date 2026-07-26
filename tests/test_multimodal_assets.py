from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dr_anmar_asset_registry import discover_asset_units, load_portfolio  # noqa: E402
from dr_anmar_multimodal_assets import (  # noqa: E402
    GENERATIVE_AUTHORITY,
    image_transform_plan,
    load_action_contract,
    load_action_stream,
    safe_action_sample,
    validate_action_stream,
    validate_bundle,
)

BUNDLE_ROOT = ROOT / "assets/dr_anmar/multimodal/cosmos_h_dreams_knot_tying_v1"
BUNDLE_PATH = BUNDLE_ROOT / "asset_bundle.json"
CONTRACT_PATH = BUNDLE_ROOT / "action_contract.json"
STREAM_PATH = BUNDLE_ROOT / "dranmar_action_stream.json"


def test_canonical_multimodal_bundle_is_complete_and_valid() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))

    assert bundle["authority"] == GENERATIVE_AUTHORITY
    assert bundle["pairing"]["source_pairing_status"] == "rejected_unpaired"
    assert bundle["pairing"]["dranmar_control_enabled"] is False
    assert bundle["runtime_contract"]["torch_pickle_loading"] == "disabled"
    assert bundle["runtime_contract"]["generated_success_or_patient_effect_labels_allowed"] is False
    assert validate_bundle(BUNDLE_PATH, ROOT) == ()


def test_external_components_are_content_addressed_and_never_runtime_loaded() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    external = [component for component in bundle["components"] if component["storage"] == "external"]

    assert len(external) == 7
    assert all(len(component["revision"]) == 40 for component in external)
    assert all(len(component["sha256"]) == 64 for component in external)
    torch_artifacts = [
        component
        for component in external
        if component["role"] in {"model_checkpoint", "text_embedding"}
    ]
    assert len(torch_artifacts) == 2
    assert all(component["loading"]["runtime_enabled"] is False for component in torch_artifacts)
    assert all(
        component["loading"]["policy"] == "quarantined_no_deserialization"
        for component in torch_artifacts
    )


def test_action_stream_is_bounded_timed_neutral_and_reproducible() -> None:
    contract = load_action_contract(CONTRACT_PATH)
    stream = load_action_stream(STREAM_PATH)

    assert validate_action_stream(stream, contract) == ()
    assert stream["frame_count"] == len(stream["frames"]) == 161
    assert stream["frames"][0]["action"] == contract["neutral_action"]
    assert stream["frames"][-1]["action"] == contract["neutral_action"]
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_dranmar_multimodal_fixture.py"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )


def test_safe_sampler_interpolates_continuous_axes_and_holds_grippers() -> None:
    contract = load_action_contract(CONTRACT_PATH)
    stream = load_action_stream(STREAM_PATH)

    sample = safe_action_sample(
        stream,
        contract,
        target_timestamp_s=3.475,
        received_monotonic_s=10.0,
        now_monotonic_s=10.1,
    )

    assert sample.status == "interpolated"
    assert sample.action[6] == 1.0
    assert sample.action[13] == 1.0
    assert 0.11 < sample.action[0] <= 0.12


def test_safe_sampler_returns_neutral_stop_for_stale_or_out_of_range_input() -> None:
    contract = load_action_contract(CONTRACT_PATH)
    stream = load_action_stream(STREAM_PATH)
    neutral = tuple(contract["neutral_action"])

    stale = safe_action_sample(
        stream,
        contract,
        target_timestamp_s=4.0,
        received_monotonic_s=10.0,
        now_monotonic_s=10.251,
    )
    outside = safe_action_sample(
        stream,
        contract,
        target_timestamp_s=9.0,
        received_monotonic_s=10.0,
        now_monotonic_s=10.1,
    )

    assert stale.status == "stale_input_neutral_stop"
    assert stale.action == neutral
    assert outside.status == "outside_trajectory_neutral_stop"
    assert outside.action == neutral


def test_invalid_stream_fails_closed_without_sampling_a_command() -> None:
    contract = load_action_contract(CONTRACT_PATH)
    stream = load_action_stream(STREAM_PATH)
    stream["frames"][4]["action"][0] = 99.0

    sample = safe_action_sample(
        stream,
        contract,
        target_timestamp_s=1.0,
        received_monotonic_s=10.0,
        now_monotonic_s=10.1,
    )

    assert sample.status == "invalid_stream_neutral_stop"
    assert sample.action == tuple(contract["neutral_action"])


def test_geometry_plan_preserves_aspect_for_letterbox_and_center_crop() -> None:
    letterbox = image_transform_plan((640, 480), (512, 288), "letterbox")
    crop = image_transform_plan((640, 480), (512, 288), "center_crop")

    assert letterbox["resized_content_size_px"] == [384, 288]
    assert letterbox["padding_left_top_right_bottom_px"] == [64, 0, 64, 0]
    assert crop["resized_content_size_px"] == [512, 384]
    assert crop["crop_left_top_right_bottom_px"] == [0, 48, 0, 48]


def test_bundle_rejects_distorting_resize_and_model_enablement(tmp_path: Path) -> None:
    copied_root = tmp_path / "assets/dr_anmar/multimodal/bundle"
    copied_root.parent.mkdir(parents=True)
    shutil.copytree(BUNDLE_ROOT, copied_root)
    copied_bundle_path = copied_root / "asset_bundle.json"
    copied_bundle = json.loads(copied_bundle_path.read_text(encoding="utf-8"))
    copied_bundle["media_contract"]["model_input_transform"]["mode"] = "stretch"
    copied_bundle["media_contract"]["model_input_transform"]["direct_resize_allowed"] = True
    checkpoint = next(
        component
        for component in copied_bundle["components"]
        if component["role"] == "model_checkpoint"
    )
    checkpoint["loading"]["runtime_enabled"] = True
    copied_bundle_path.write_text(json.dumps(copied_bundle), encoding="utf-8")

    issues = validate_bundle(copied_bundle_path, tmp_path)
    codes = {issue.code for issue in issues}

    assert "distorting_image_transform" in codes
    assert "direct_resize_not_disabled" in codes
    assert "unsafe_model_runtime_enablement" in codes


def test_bundle_is_a_cataloged_non_usd_asset_and_product_capability() -> None:
    units = {unit.asset_id: unit for unit in discover_asset_units(ROOT)}
    asset_id = (
        "dr_anmar_repository:"
        "assets/dr_anmar/multimodal/cosmos_h_dreams_knot_tying_v1"
    )
    portfolio = load_portfolio(ROOT)
    portfolio_ids = {asset["id"] for asset in portfolio["assets"]}

    assert units[asset_id].entrypoints == ("asset_bundle.json",)
    assert "dranmar-cosmos-h-dreams-source-audit-v1" in portfolio_ids


def test_published_json_schemas_are_machine_readable() -> None:
    schema_root = ROOT / "config/schemas"
    names = (
        "dranmar_multimodal_asset_bundle.schema.json",
        "dranmar_action_contract.schema.json",
        "dranmar_timestamped_action_stream.schema.json",
    )
    schemas = [json.loads((schema_root / name).read_text(encoding="utf-8")) for name in names]

    assert all(schema["$schema"].endswith("2020-12/schema") for schema in schemas)
    assert all(schema["type"] == "object" for schema in schemas)
