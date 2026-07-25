#!/usr/bin/env python3
"""Generate the DrAnmar Multimodal Perfusion and Tissue-Viability Robot.

The package is an independently authored, manufacturer-neutral research
system for active intraoperative physiological verification.  A single
vascular-flow and tracer state drives RGB context, NIR/ICG fluorescence,
laser-speckle perfusion, thermal, Doppler, ultrasound, and surface-oxygenation
measurements.  It is not clinically validated and is not approved for patient
care.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw
import trimesh

SCRIPT_PATH = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT_PATH.parent))
from dranmar_asset_authoring import (
    Collider,
    Joint,
    Link,
    Visual,
    annular_sector_mesh,
    available_font,
    box_mesh,
    capsule_axis,
    capsule_between,
    cylinder_axis,
    ellipsoid_mesh,
    export_scene,
    f,
    frame_usda,
    frustum_axis,
    grid_surface_mesh,
    joint_usda,
    link_usda,
    matrix_to_quat_wxyz,
    mesh_bounds,
    mesh_usda,
    nested_over,
    normalize,
    pbr,
    physics_materials_scope,
    quat,
    rotation_matrix,
    rounded_bar_mesh,
    sha256,
    torus_axis,
    transform,
    vec,
    visual_materials_scope,
    wire_path,
    write_checksum,
    write_json,
    zip_tree,
)

VERSION = "0.1.1"
ASSET_NAME = "DrAnmar Multimodal Perfusion and Tissue-Viability Robot"
CATALOG_SUBPATH = Path("Props/SurgicalAssessment/PerfusionViabilityRobot")
ROOT_PRIM = "DrAnmarPerfusionViabilityTool"
STANDALONE_ROOT = "DrAnmarPerfusionViabilityToolStandalone"
PROXY_ROOT = "DrAnmarPerfusionViabilityToolRigidProxy"
TISSUE_ROOT = "DrAnmarPerfusedTissueDemo"
TRACER_ROOT = "DrAnmarICGTracerParticle"
COUPLING_ROOT = "DrAnmarUltrasoundCouplingPad"
OCCLUDER_ROOT = "DrAnmarFlowOccluder"

PACKAGE_ROOT = SCRIPT_PATH.parents[1]
ASSET_ROOT = PACKAGE_ROOT / "assets" / CATALOG_SUBPATH
GLB_ROOT = ASSET_ROOT / "glb"
TEXTURE_ROOT = ASSET_ROOT / "textures"
PREVIEW_ROOT = PACKAGE_ROOT / "previews"
DOCS_ROOT = PACKAGE_ROOT / "docs"
EXAMPLE_ROOT = PACKAGE_ROOT / "examples"
EXTENSION_ROOT = PACKAGE_ROOT / "source/extensions/orbit.surgical.assets"
INTEGRATION_PATH = EXTENSION_ROOT / "orbit/surgical/assets/perfusion_viability_robot.py"
PHYSICS_PROFILE_PATH = PACKAGE_ROOT / "physics_next/surgical-assessment/dranmar-perfusion-viability-v1.json"

WORK_PLANE_Z = 0.205
FRANKA_HAND_EQUIVALENT_ROTATION_DEG = -45.0
REGION_COLS = 6
REGION_ROWS = 4
REGION_COUNT = REGION_COLS * REGION_ROWS
TISSUE_WIDTH_M = 0.180
TISSUE_DEPTH_M = 0.120
TISSUE_THICKNESS_M = 0.008
ARTERIAL_RADIUS_M = 0.00175
VENOUS_RADIUS_M = 0.00210
CAPILLARY_RADIUS_M = 0.00042

COLORS: dict[str, tuple[int, int, int, int]] = {
    "BodyPolymer": (216, 222, 228, 255),
    "DarkPolymer": (17, 22, 28, 255),
    "AccentBlue": (12, 95, 196, 255),
    "AccentCyan": (16, 181, 205, 255),
    "MountMetal": (118, 126, 138, 255),
    "RailMetal": (69, 77, 88, 255),
    "ProbeMetal": (175, 182, 191, 255),
    "SensorGlass": (8, 33, 56, 220),
    "NIRGlass": (24, 120, 66, 225),
    "SpeckleGlass": (145, 14, 18, 225),
    "ThermalGlass": (92, 18, 132, 225),
    "MultispectralGlass": (24, 93, 161, 225),
    "UltrasoundFace": (51, 54, 59, 255),
    "DopplerFace": (232, 226, 208, 255),
    "ContactElastomer": (18, 129, 139, 255),
    "TubeClear": (168, 211, 228, 95),
    "ICGFluid": (35, 222, 106, 145),
    "CouplingGel": (62, 188, 246, 112),
    "IndicatorGreen": (19, 224, 78, 255),
    "IndicatorAmber": (255, 157, 16, 255),
    "IndicatorRed": (239, 17, 27, 255),
    "LaserEmitter": (235, 16, 22, 255),
    "NIRIlluminator": (29, 235, 105, 220),
    "WhiteIlluminator": (244, 247, 250, 255),
    "TissueBase": (204, 127, 113, 255),
    "TissueDeep": (149, 63, 57, 255),
    "Artery": (211, 25, 27, 255),
    "Vein": (34, 74, 185, 255),
    "Capillary": (175, 50, 135, 255),
    "HealthyOverlay": (24, 211, 91, 90),
    "LowFlowOverlay": (43, 100, 213, 115),
    "CongestionOverlay": (118, 51, 181, 120),
    "LeakOverlay": (237, 25, 31, 135),
    "CompressionOverlay": (235, 165, 18, 105),
    "RecoveryOverlay": (38, 224, 116, 105),
    "Occluder": (215, 160, 62, 255),
    "ICGParticle": (32, 255, 121, 195),
    "GuideRed": (244, 29, 29, 255),
    "GuideGreen": (20, 232, 62, 255),
    "GuideBlue": (26, 84, 245, 255),
    "CollisionDebug": (255, 34, 184, 85),
    "RegionGuide": (240, 240, 240, 35),
}

VISUAL_MATERIAL_SPECS = {
    "BodyPolymer": ((0.84, 0.87, 0.90), 0.0, 0.33, 1.0),
    "DarkPolymer": ((0.035, 0.045, 0.058), 0.0, 0.30, 1.0),
    "AccentBlue": ((0.045, 0.34, 0.74), 0.0, 0.27, 1.0),
    "AccentCyan": ((0.04, 0.66, 0.74), 0.0, 0.27, 1.0),
    "MountMetal": ((0.46, 0.50, 0.55), 0.82, 0.23, 1.0),
    "RailMetal": ((0.26, 0.29, 0.34), 0.75, 0.27, 1.0),
    "ProbeMetal": ((0.69, 0.72, 0.76), 0.82, 0.20, 1.0),
    "SensorGlass": ((0.025, 0.12, 0.22), 0.15, 0.08, 0.86),
    "NIRGlass": ((0.05, 0.50, 0.22), 0.12, 0.07, 0.87),
    "SpeckleGlass": ((0.54, 0.04, 0.05), 0.10, 0.08, 0.87),
    "ThermalGlass": ((0.35, 0.04, 0.50), 0.14, 0.09, 0.87),
    "MultispectralGlass": ((0.06, 0.31, 0.62), 0.10, 0.08, 0.87),
    "UltrasoundFace": ((0.12, 0.13, 0.14), 0.0, 0.24, 1.0),
    "DopplerFace": ((0.91, 0.88, 0.80), 0.0, 0.32, 1.0),
    "ContactElastomer": ((0.05, 0.48, 0.52), 0.0, 0.57, 1.0),
    "TubeClear": ((0.64, 0.82, 0.90), 0.0, 0.08, 0.34),
    "ICGFluid": ((0.05, 0.88, 0.32), 0.0, 0.05, 0.55),
    "CouplingGel": ((0.15, 0.68, 0.92), 0.0, 0.04, 0.44),
    "IndicatorGreen": ((0.05, 0.86, 0.28), 0.0, 0.18, 1.0),
    "IndicatorAmber": ((1.0, 0.52, 0.02), 0.0, 0.18, 1.0),
    "IndicatorRed": ((0.93, 0.03, 0.04), 0.0, 0.18, 1.0),
    "LaserEmitter": ((0.93, 0.02, 0.03), 0.0, 0.08, 1.0),
    "NIRIlluminator": ((0.04, 0.86, 0.30), 0.0, 0.08, 0.82),
    "WhiteIlluminator": ((0.96, 0.97, 0.98), 0.0, 0.10, 1.0),
    "TissueBase": ((0.79, 0.47, 0.42), 0.0, 0.50, 1.0),
    "TissueDeep": ((0.55, 0.20, 0.18), 0.0, 0.54, 1.0),
    "Artery": ((0.80, 0.04, 0.04), 0.0, 0.36, 1.0),
    "Vein": ((0.05, 0.17, 0.68), 0.0, 0.38, 1.0),
    "Capillary": ((0.65, 0.10, 0.48), 0.0, 0.40, 1.0),
    "HealthyOverlay": ((0.05, 0.78, 0.28), 0.0, 0.25, 0.34),
    "LowFlowOverlay": ((0.08, 0.30, 0.86), 0.0, 0.25, 0.44),
    "CongestionOverlay": ((0.40, 0.11, 0.70), 0.0, 0.25, 0.46),
    "LeakOverlay": ((0.90, 0.03, 0.05), 0.0, 0.20, 0.52),
    "CompressionOverlay": ((0.91, 0.55, 0.04), 0.0, 0.25, 0.40),
    "RecoveryOverlay": ((0.06, 0.84, 0.36), 0.0, 0.22, 0.38),
    "Occluder": ((0.82, 0.55, 0.16), 0.65, 0.28, 1.0),
    "ICGParticle": ((0.05, 1.0, 0.38), 0.0, 0.02, 0.76),
    "GuideRed": ((0.95, 0.08, 0.08), 0.0, 0.30, 1.0),
    "GuideGreen": ((0.08, 0.90, 0.18), 0.0, 0.30, 1.0),
    "GuideBlue": ((0.08, 0.28, 0.95), 0.0, 0.30, 1.0),
    "CollisionDebug": ((1.0, 0.12, 0.72), 0.0, 0.20, 0.30),
    "RegionGuide": ((0.95, 0.95, 0.95), 0.0, 0.45, 0.14),
}

PHYSICS_MATERIAL_SPECS = {
    "MountPhysics": (0.35, 0.27, 0.02),
    "PolymerPhysics": (0.48, 0.37, 0.03),
    "MetalPhysics": (0.27, 0.20, 0.02),
    "ProbeContactPhysics": (0.68, 0.52, 0.01),
    "TissuePhysics": (0.58, 0.45, 0.00),
    "GelPhysics": (0.22, 0.18, 0.00),
    "VesselPhysics": (0.46, 0.35, 0.00),
}

@dataclass
class ToolBundle:
    links: dict[str, Link]
    joints: list[Joint]
    frames: dict[str, dict[str, object]]
    graph: dict[str, Any]
    tissue_surface: trimesh.Trimesh
    tissue_bulk: trimesh.Trimesh
    region_overlays: dict[str, list[trimesh.Trimesh]]
    vascular_meshes: list[tuple[str, trimesh.Trimesh, str]]
    coupling_pad: trimesh.Trimesh
    occluder: trimesh.Trimesh


def _frame(parent: str, position, role: str, orientation=(1.0,0.0,0.0,0.0)) -> dict[str, object]:
    return {"parent_link": parent, "position": list(position), "orientation_wxyz": list(orientation), "role": role}


def build_graph() -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []
    x_values = np.linspace(-0.075, 0.075, REGION_COLS)
    y_values = np.linspace(-0.045, 0.045, REGION_ROWS)

    for c, x in enumerate(x_values):
        nodes[f"A{c}"] = {"position_m": [float(x), -0.052, -0.0024], "type": "arterial_trunk"}
        nodes[f"V{c}"] = {"position_m": [float(x), 0.052, -0.0020], "type": "venous_trunk"}
    for c in range(REGION_COLS - 1):
        edges.append({"id": f"AT{c}", "from": f"A{c}", "to": f"A{c+1}", "type": "artery", "radius_m": 0.00175 - c*0.00007, "resistance_kpa_s_ml": 0.075 + 0.008*c})
        edges.append({"id": f"VT{c}", "from": f"V{c+1}", "to": f"V{c}", "type": "vein", "radius_m": 0.00210 - c*0.00005, "resistance_kpa_s_ml": 0.040 + 0.004*c})

    for c, x in enumerate(x_values):
        for r, y in enumerate(y_values):
            idx = r * REGION_COLS + c
            ai = f"R{idx:02d}A"
            vo = f"R{idx:02d}V"
            nodes[ai] = {"position_m": [float(x), float(y-0.006), -0.0017], "type": "arteriole", "region": idx}
            nodes[vo] = {"position_m": [float(x), float(y+0.006), -0.0015], "type": "venule", "region": idx}
            edges.append({"id": f"AB{idx:02d}", "from": f"A{c}", "to": ai, "type": "arteriole", "radius_m": 0.00072, "resistance_kpa_s_ml": 0.55 + 0.06*abs(r-1.5)})
            edges.append({"id": f"CP{idx:02d}", "from": ai, "to": vo, "type": "capillary", "radius_m": 0.00042, "resistance_kpa_s_ml": 2.7 + 0.16*((c+r)%3), "region": idx})
            edges.append({"id": f"VB{idx:02d}", "from": vo, "to": f"V{c}", "type": "venule", "radius_m": 0.00082, "resistance_kpa_s_ml": 0.42 + 0.05*abs(r-1.5)})
            regions.append({
                "index": idx,
                "grid": [r, c],
                "center_m": [float(x), float(y), 0.0045],
                "capillary_edge": f"CP{idx:02d}",
                "arteriole_edge": f"AB{idx:02d}",
                "venule_edge": f"VB{idx:02d}",
                "metabolic_consumption_ml_o2_min_100g": 4.0 + 0.35*((r+c)%4),
                "optical_depth": 0.65 + 0.05*r,
                "thermal_capacity_j_k": 2.2 + 0.1*c,
            })

    # Per-condition multipliers are explicit so all sensors read the same state.
    conditions = {
        "healthy": {"edge_multipliers": {}, "region_compression": {}, "leak_edges": {}, "description": "unobstructed arterial inflow and venous outflow"},
        "arterial_occlusion": {"edge_multipliers": {"AT2": 0.015}, "region_compression": {}, "leak_edges": {}, "description": "near-complete central arterial inflow occlusion"},
        "venous_congestion": {"edge_multipliers": {"VT1": 0.035}, "region_compression": {}, "leak_edges": {}, "description": "severe venous outflow restriction"},
        "anastomotic_stenosis": {"edge_multipliers": {"AT2": 0.14}, "region_compression": {}, "leak_edges": {}, "description": "focal arterial reconstruction stenosis with distal hypoperfusion"},
        "branch_leak": {"edge_multipliers": {}, "region_compression": {}, "leak_edges": {"CP14": 0.10}, "description": "regional vascular branch leak and tracer extravasation"},
        "retraction_ischemia": {"edge_multipliers": {}, "region_compression": {"4": 0.18, "5": 0.12, "10": 0.22, "11": 0.16}, "leak_edges": {}, "description": "localized compression from retraction"},
        "dressing_compression": {"edge_multipliers": {}, "region_compression": {"12": 0.28, "13": 0.22, "18": 0.24, "19": 0.18}, "leak_edges": {}, "description": "localized external dressing pressure"},
        "recovered": {"edge_multipliers": {"AT2": 0.92, "VT1": 0.94}, "region_compression": {}, "leak_edges": {}, "description": "post-intervention reperfusion"},
    }
    return {
        "schema": "dr.anmar.perfusion-network.v1",
        "units": {"length": "m", "pressure": "kPa", "flow": "ml_s"},
        "boundary_nodes": {"arterial_inlet": "A0", "venous_outlet": "V0"},
        "reference_pressures_kpa": {"arterial": 13.3, "venous": 1.2},
        "nodes": nodes,
        "edges": edges,
        "regions": regions,
        "conditions": conditions,
    }


def _optical_module(name: str, center: tuple[float,float,float], lens_material: str, labels: tuple[str,...]) -> list[Visual]:
    x,y,z=center
    return [
        Visual(f"{name}Housing", rounded_bar_mesh((0.022,0.018,0.014),(x,y,z),0.003), "DarkPolymer", labels),
        Visual(f"{name}Lens", cylinder_axis(0.0052,0.004,"y",(x,y-0.010,z),sections=36), lens_material, labels),
    ]


def build_tool(graph: dict[str, Any]) -> ToolBundle:
    links: dict[str, Link] = {}
    frames: dict[str, dict[str, object]] = {}

    mount_visuals = [
        Visual("FrankaAdapterPlate", cylinder_axis(0.032,0.012,"z",(0,0,0.006),sections=72), "MountMetal", ("franka_mount",)),
        Visual("QuickReleaseRing", torus_axis(0.0275,0.003,"z",(0,0,0.014),major_sections=72,minor_sections=14), "MountMetal"),
        Visual("MainHousing", ellipsoid_mesh((0.067,0.058,0.040),(0,0,0.058),subdivisions=3), "BodyPolymer", ("perfusion_viability_robot",)),
        Visual("HousingCore", rounded_bar_mesh((0.128,0.100,0.052),(0,0,0.065),0.009), "BodyPolymer"),
        Visual("TurretBearing", torus_axis(0.045,0.004,"z",(0,0,0.108),major_sections=80,minor_sections=16), "RailMetal", ("sensor_turret_bearing",)),
        Visual("ICGReservoirShell", cylinder_axis(0.016,0.044,"y",(-0.038,0.043,0.058),sections=48), "TubeClear", ("icg_reservoir",)),
        Visual("ICGReservoirFill", cylinder_axis(0.013,0.038,"y",(-0.038,0.043,0.058),sections=48), "ICGFluid", ("icg_inventory",)),
        Visual("GelReservoirShell", cylinder_axis(0.015,0.040,"y",(0.000,0.044,0.058),sections=48), "TubeClear", ("ultrasound_gel_reservoir",)),
        Visual("GelReservoirFill", cylinder_axis(0.0125,0.034,"y",(0.000,0.044,0.058),sections=48), "CouplingGel", ("coupling_gel_inventory",)),
        Visual("ComputeModule", rounded_bar_mesh((0.040,0.066,0.028),(0.044,0.0,0.062),0.005), "AccentBlue", ("multimodal_fusion_compute",)),
        Visual("ReadyIndicator", cylinder_axis(0.0045,0.003,"y",(0.044,-0.035,0.065),sections=32), "IndicatorGreen", ("sensor_ready",)),
        Visual("DegradedIndicator", cylinder_axis(0.0045,0.003,"y",(0.044,-0.035,0.065),sections=32), "IndicatorAmber", ("sensor_degraded",)),
        Visual("FaultIndicator", cylinder_axis(0.0045,0.003,"y",(0.044,-0.035,0.065),sections=32), "IndicatorRed", ("sensor_fault",)),
        Visual("ContactGuardHousing", torus_axis(0.052,0.0035,"z",(0,0,0.174),major_sections=84,minor_sections=16), "ContactElastomer", ("atraumatic_contact_guard",)),
    ]
    links["Mount"] = Link("Mount", (0,0,0), mount_visuals,
        [Collider("MountPlateCollider","cylinder",(0,0,0.006),radius=0.032,height=0.012,axis="z",physics_material="MountPhysics"),
         Collider("HousingCollider","box",(0,0,0.065),size=(0.128,0.100,0.052),physics_material="PolymerPhysics")],
        1.10, ("franka_end_effector","physiological_verification_robot"))
    frames.update({
        "panda_link8_mount": _frame("Mount",(0,0,0),"fixed_mount_to_panda_link8"),
        "perfusion_tcp": _frame("Mount",(0,0,WORK_PLANE_Z),"primary_assessment_tcp"),
        "roi_center": _frame("Mount",(0,0,WORK_PLANE_Z),"perfusion_region_of_interest"),
        "contact_guard_reference": _frame("Mount",(0,0,0.174),"atraumatic_guard_center"),
        "count_reference": _frame("Mount",(0,0,0.035),"inventory_count_reference"),
        "handover_reference": _frame("Mount",(0,-0.070,0.065),"handover_reference"),
    })

    turret_visuals = [
        Visual("TurretDisc", cylinder_axis(0.047,0.018,"z",(0,0,0),sections=80), "DarkPolymer", ("sensor_turret",)),
        Visual("TurretAccent", torus_axis(0.040,0.0024,"z",(0,0,0.010),major_sections=80,minor_sections=14), "AccentCyan"),
        Visual("OpticalBridge", rounded_bar_mesh((0.100,0.030,0.024),(0,-0.016,0.028),0.005), "DarkPolymer", ("multimodal_optical_bridge",)),
    ]
    turret_visuals += _optical_module("RGBLeft",(-0.034,-0.032,0.033),"SensorGlass",("rgb_camera","stereo_left"))
    turret_visuals += _optical_module("RGBRight",(0.034,-0.032,0.033),"SensorGlass",("rgb_camera","stereo_right"))
    turret_visuals += _optical_module("NIR",(-0.034,0.008,0.033),"NIRGlass",("nir_camera","icg_fluorescence"))
    turret_visuals += _optical_module("Speckle",(0.0,0.008,0.033),"SpeckleGlass",("laser_speckle_camera",))
    turret_visuals += _optical_module("Thermal",(0.034,0.008,0.033),"ThermalGlass",("thermal_camera",))
    turret_visuals += _optical_module("Multispectral",(0.0,-0.032,0.055),"MultispectralGlass",("surface_oxygenation_camera",))
    turret_visuals += [
        Visual("NIRIlluminationRing", torus_axis(0.020,0.0017,"z",(-0.034,0.008,0.047),major_sections=64,minor_sections=12), "NIRIlluminator", ("nir_illumination",)),
        Visual("WhiteIlluminationRing", torus_axis(0.044,0.0015,"z",(0,0,0.016),major_sections=80,minor_sections=12), "WhiteIlluminator", ("white_light",)),
        Visual("LaserEmitterLeft", cylinder_axis(0.0032,0.006,"y",(-0.013,0.006,0.034),sections=32), "LaserEmitter", ("laser_speckle_emitter",)),
        Visual("LaserEmitterRight", cylinder_axis(0.0032,0.006,"y",(0.013,0.006,0.034),sections=32), "LaserEmitter", ("laser_speckle_emitter",)),
    ]
    links["SensorTurret"] = Link("SensorTurret",(0,0,0.108),turret_visuals,
        [Collider("TurretCollider","cylinder",(0,0,0),radius=0.047,height=0.018,axis="z",physics_material="PolymerPhysics"),
         Collider("OpticalBridgeCollider","box",(0,-0.016,0.028),size=(0.100,0.030,0.024),physics_material="PolymerPhysics")],
        0.38,("multimodal_sensor_turret",))
    frames.update({
        "rgb_left_camera": _frame("SensorTurret",(-0.034,-0.044,0.033),"stereo_rgb_left"),
        "rgb_right_camera": _frame("SensorTurret",(0.034,-0.044,0.033),"stereo_rgb_right"),
        "nir_fluorescence_camera": _frame("SensorTurret",(-0.034,-0.002,0.033),"nir_icg_camera"),
        "speckle_camera": _frame("SensorTurret",(0.0,-0.002,0.033),"laser_speckle_camera"),
        "thermal_camera": _frame("SensorTurret",(0.034,-0.002,0.033),"thermal_camera"),
        "multispectral_camera": _frame("SensorTurret",(0.0,-0.044,0.055),"surface_oxygenation_camera"),
        "optical_scan_reference": _frame("SensorTurret",(0,0,0.070),"registered_optical_scan_origin"),
    })

    filter_visuals = [
        Visual("FilterWheel", cylinder_axis(0.018,0.004,"y",(0,0,0),sections=64), "RailMetal", ("spectral_filter_wheel",)),
    ]
    for i,(mat,label) in enumerate((("WhiteIlluminator","rgb"),("NIRIlluminator","nir"),("SpeckleGlass","speckle"),("MultispectralGlass","multispectral"))):
        a=2*math.pi*i/4
        filter_visuals.append(Visual(f"Filter_{label}",cylinder_axis(0.0042,0.005,"y",(0.010*math.cos(a),0,0.010*math.sin(a)),sections=32),mat,(f"filter_{label}",)))
    links["FilterWheel"] = Link("FilterWheel",(0,-0.003,0.142),filter_visuals,[Collider("FilterWheelCollider","cylinder",(0,0,0),radius=0.018,height=0.004,axis="y",physics_material="MetalPhysics")],0.05,("spectral_filter_wheel",))

    focus_visuals = [
        Visual("FocusRing", torus_axis(0.038,0.0028,"z",(0,0,0),major_sections=72,minor_sections=14), "AccentCyan", ("optical_focus_ring",)),
        Visual("StructuredLightProjector", rounded_bar_mesh((0.024,0.018,0.015),(0.050,0,0.002),0.003), "DarkPolymer", ("structured_light_projector",)),
    ]
    links["OpticalFocus"] = Link("OpticalFocus",(0,0,0.160),focus_visuals,[Collider("FocusRingCollider","cylinder",(0,0,0),radius=0.039,height=0.006,axis="z",physics_material="PolymerPhysics")],0.12,("optical_focus_and_depth_module",))
    frames["structured_light_projector"] = _frame("OpticalFocus",(0.050,0,0.010),"structured_light_projector")
    frames["depth_reference"] = _frame("OpticalFocus",(0,0,0.012),"depth_and_surface_normal_reference")

    mirror_x_mesh = rounded_bar_mesh((0.018,0.014,0.004),(0,0,0),0.002)
    links["SpeckleMirrorX"] = Link("SpeckleMirrorX",(-0.012,0.008,0.170),[Visual("MirrorX",mirror_x_mesh,"ProbeMetal",("speckle_galvo_x",))],[Collider("MirrorXCollider","box",(0,0,0),size=(0.018,0.014,0.004),physics_material="MetalPhysics")],0.018)
    links["SpeckleMirrorY"] = Link("SpeckleMirrorY",(0.012,0.008,0.170),[Visual("MirrorY",mirror_x_mesh,"ProbeMetal",("speckle_galvo_y",))],[Collider("MirrorYCollider","box",(0,0,0),size=(0.018,0.014,0.004),physics_material="MetalPhysics")],0.018)
    frames["speckle_projection_center"] = _frame("SpeckleMirrorY",(0,0,0.012),"laser_speckle_projection_center")

    # Ultrasound arm and probe.
    us_carriage_visuals = [
        Visual("UltrasoundRailBlock",rounded_bar_mesh((0.036,0.040,0.055),(0,0,0),0.005),"AccentBlue",("ultrasound_carriage",)),
        Visual("GelNozzle",frustum_axis(0.0035,0.0012,0.018,"z",(0.018,0,0.035),sections=32),"ProbeMetal",("gel_dispensing_nozzle",)),
    ]
    links["UltrasoundCarriage"] = Link("UltrasoundCarriage",(-0.066,0,0.116),us_carriage_visuals,[Collider("USCarriageCollider","box",(0,0,0),size=(0.036,0.040,0.055),physics_material="PolymerPhysics")],0.22,("robotic_ultrasound_carriage",))
    us_gimbal_visuals = [
        Visual("UltrasoundGimbal",cylinder_axis(0.012,0.034,"y",(0,0,0),sections=48),"RailMetal",("ultrasound_pitch_gimbal",)),
        Visual("UltrasoundProbeBody",rounded_bar_mesh((0.032,0.052,0.038),(0,0,0.030),0.005),"BodyPolymer",("ultrasound_probe",)),
        Visual("UltrasoundArrayFace",rounded_bar_mesh((0.028,0.046,0.006),(0,0,0.052),0.003),"UltrasoundFace",("ultrasound_array_face",)),
    ]
    links["UltrasoundGimbal"] = Link("UltrasoundGimbal",(-0.066,0,0.155),us_gimbal_visuals,[Collider("USProbeCollider","box",(0,0,0.030),size=(0.032,0.052,0.038),physics_material="PolymerPhysics"),Collider("USFaceCollider","box",(0,0,0.052),size=(0.028,0.046,0.006),physics_material="ProbeContactPhysics")],0.19,("robotic_ultrasound_probe",))
    links["UltrasoundCompliance"] = Link("UltrasoundCompliance",(-0.066,0,0.207),[Visual("CompliantFace",rounded_bar_mesh((0.030,0.048,0.004),(0,0,0),0.0025),"ContactElastomer",("ultrasound_compliant_contact",))],[Collider("USComplianceCollider","box",(0,0,0),size=(0.030,0.048,0.004),physics_material="ProbeContactPhysics")],0.045,("ultrasound_force_compliance",))
    links["GelValve"] = Link("GelValve",(-0.048,0,0.151),[Visual("GelPlunger",cylinder_axis(0.004,0.020,"x",(0,0,0),sections=32),"AccentCyan",("gel_metering_plunger",))],[Collider("GelValveCollider","cylinder",(0,0,0),radius=0.004,height=0.020,axis="x",physics_material="PolymerPhysics")],0.016)
    frames.update({
        "ultrasound_probe_face": _frame("UltrasoundCompliance",(0,0,0.003),"ultrasound_contact_face"),
        "ultrasound_probe_axis": _frame("UltrasoundCompliance",(0,0,0.012),"ultrasound_beam_axis"),
        "ultrasound_force_reference": _frame("UltrasoundCompliance",(0,0,0),"ultrasound_contact_force_reference"),
        "gel_dispense_exit": _frame("UltrasoundCarriage",(0.018,0,0.046),"ultrasound_coupling_gel_exit"),
    })

    # Doppler arm and pencil probe.
    links["DopplerCarriage"] = Link("DopplerCarriage",(0.066,0,0.116),[
        Visual("DopplerRailBlock",rounded_bar_mesh((0.034,0.038,0.050),(0,0,0),0.005),"AccentBlue",("doppler_carriage",)),
    ],[Collider("DopplerCarriageCollider","box",(0,0,0),size=(0.034,0.038,0.050),physics_material="PolymerPhysics")],0.18,("doppler_probe_carriage",))
    doppler_visuals = [
        Visual("DopplerGimbal",cylinder_axis(0.010,0.030,"y",(0,0,0),sections=44),"RailMetal",("doppler_pitch_gimbal",)),
        Visual("DopplerProbeBody",capsule_axis(0.008,0.062,"z",(0,0,0.032),sections=28),"BodyPolymer",("doppler_probe",)),
        Visual("DopplerFace",frustum_axis(0.0060,0.0032,0.014,"z",(0,0,0.068),sections=40),"DopplerFace",("doppler_transducer_face",)),
    ]
    links["DopplerGimbal"] = Link("DopplerGimbal",(0.066,0,0.155),doppler_visuals,[Collider("DopplerBodyCollider","cylinder",(0,0,0.032),radius=0.008,height=0.062,axis="z",physics_material="PolymerPhysics"),Collider("DopplerFaceCollider","cylinder",(0,0,0.068),radius=0.004,height=0.012,axis="z",physics_material="ProbeContactPhysics")],0.12,("contact_doppler_probe",))
    frames.update({
        "doppler_probe_tip": _frame("DopplerGimbal",(0,0,0.076),"doppler_contact_tip"),
        "doppler_beam_axis": _frame("DopplerGimbal",(0,0,0.082),"doppler_beam_axis"),
    })

    links["ContactGuard"] = Link("ContactGuard",(0,0,0.174),[
        Visual("GuardRing",torus_axis(0.052,0.004,"z",(0,0,0),major_sections=84,minor_sections=16),"ContactElastomer",("contact_guard",)),
        Visual("GuardFootA",rounded_bar_mesh((0.020,0.010,0.006),(0.052,0,0.004),0.003),"ContactElastomer"),
        Visual("GuardFootB",rounded_bar_mesh((0.020,0.010,0.006),(-0.052,0,0.004),0.003),"ContactElastomer"),
    ],[Collider("GuardRingCollider","cylinder",(0,0,0),radius=0.055,height=0.008,axis="z",physics_material="ProbeContactPhysics")],0.08,("contact_guard","force_reference"))
    frames["contact_guard_force"] = _frame("ContactGuard",(0,0,0),"contact_guard_force_reference")

    joints = [
        Joint("sensor_turret_joint","revolute","Mount","SensorTurret","Z",(0,0,0.108),(0,0,0),-160,160,28,2.0,18),
        Joint("filter_wheel_joint","revolute","SensorTurret","FilterWheel","Y",(0,-0.003,0.034),(0,0,0),0,360,18,1.4,8),
        Joint("optical_focus_joint","prismatic","SensorTurret","OpticalFocus","Z",(0,0,0.052),(0,0,0),0,0.025,1800,65,35),
        Joint("speckle_scan_x_joint","revolute","OpticalFocus","SpeckleMirrorX","Y",(-0.012,0.008,0.010),(0,0,0),-10,10,12,0.8,4),
        Joint("speckle_scan_y_joint","revolute","OpticalFocus","SpeckleMirrorY","X",(0.012,0.008,0.010),(0,0,0),-10,10,12,0.8,4),
        Joint("ultrasound_extension_joint","prismatic","Mount","UltrasoundCarriage","Z",(-0.066,0,0.116),(0,0,0),0,0.075,3200,110,85),
        Joint("ultrasound_pitch_joint","revolute","UltrasoundCarriage","UltrasoundGimbal","Y",(0,0,0.039),(0,0,0),-30,30,700,45,32),
        Joint("ultrasound_compliance_joint","prismatic","UltrasoundGimbal","UltrasoundCompliance","Z",(0,0,0.052),(0,0,0),0,0.010,420,46,22),
        Joint("gel_valve_joint","prismatic","UltrasoundCarriage","GelValve","X",(0.018,0,0.035),(0,0,0),0,0.006,1200,40,18),
        Joint("doppler_extension_joint","prismatic","Mount","DopplerCarriage","Z",(0.066,0,0.116),(0,0,0),0,0.060,2800,100,65),
        Joint("doppler_pitch_joint","revolute","DopplerCarriage","DopplerGimbal","Y",(0,0,0.039),(0,0,0),-35,35,620,38,26),
        Joint("contact_guard_joint","prismatic","Mount","ContactGuard","Z",(0,0,0.174),(0,0,0),0,0.008,360,40,20),
    ]

    # Tissue geometry.
    tissue_surface = grid_surface_mesh(TISSUE_WIDTH_M,TISSUE_DEPTH_M,41,29,z_func=lambda x,y:0.0012*math.sin(20*x)*math.cos(18*y),center=(0,0,0.008))
    tissue_bulk = box_mesh((TISSUE_WIDTH_M,TISSUE_DEPTH_M,TISSUE_THICKNESS_M),(0,0,0.004))
    overlays: dict[str,list[trimesh.Trimesh]] = {k:[] for k in graph["conditions"]}
    region_w=TISSUE_WIDTH_M/REGION_COLS*0.92; region_d=TISSUE_DEPTH_M/REGION_ROWS*0.88
    for region in graph["regions"]:
        x,y,_=region["center_m"]; idx=region["index"]
        patch=rounded_bar_mesh((region_w,region_d,0.00055),(x,y,0.0088),0.003)
        overlays["healthy"].append(patch)
        overlays["recovered"].append(patch.copy())
        if idx in range(3*REGION_COLS,4*REGION_COLS) or idx in range(2*REGION_COLS,3*REGION_COLS):
            overlays["arterial_occlusion"].append(patch.copy())
            overlays["anastomotic_stenosis"].append(patch.copy())
        if idx in (8,9,10,11,14,15,16,17,20,21,22,23):
            overlays["venous_congestion"].append(patch.copy())
        if idx==14:
            overlays["branch_leak"].append(ellipsoid_mesh((0.020,0.016,0.0012),(x,y,0.0092),subdivisions=2))
        if idx in (4,5,10,11): overlays["retraction_ischemia"].append(patch.copy())
        if idx in (12,13,18,19): overlays["dressing_compression"].append(patch.copy())

    vascular_meshes=[]
    nodes=graph["nodes"]
    for edge in graph["edges"]:
        p0=np.asarray(nodes[edge["from"]]["position_m"],dtype=float)
        p1=np.asarray(nodes[edge["to"]]["position_m"],dtype=float)
        # slight arc toward mid-depth to avoid all branches being coplanar.
        mid=(p0+p1)/2; mid[2]-=0.0008
        radius=float(edge["radius_m"])
        mat="Artery" if edge["type"] in ("artery","arteriole") else "Vein" if edge["type"] in ("vein","venule") else "Capillary"
        vascular_meshes.append((edge["id"],wire_path([p0,mid,p1],radius,sections=14 if mat=="Capillary" else 20),mat))

    coupling_pad = rounded_bar_mesh((0.044,0.034,0.003),(0,0,0),0.006)
    occluder = trimesh.util.concatenate([
        torus_axis(0.006,0.0012,"x",(0,0,0),major_sections=48,minor_sections=12),
        rounded_bar_mesh((0.004,0.016,0.006),(0,0.008,0),0.002),
    ])
    return ToolBundle(links,joints,frames,graph,tissue_surface,tissue_bulk,overlays,vascular_meshes,coupling_pad,occluder)


def state_variants(root: str) -> str:
    def vis(path: Sequence[str], value: str) -> str:
        return nested_over(path,[f'token visibility = "{value}"'])
    contrast_full=vis(["Links","Mount","Visuals","ICGReservoirFill"],"inherited")
    contrast_empty=vis(["Links","Mount","Visuals","ICGReservoirFill"],"invisible")
    gel_full=vis(["Links","Mount","Visuals","GelReservoirFill"],"inherited")
    gel_empty=vis(["Links","Mount","Visuals","GelReservoirFill"],"invisible")
    def indicator_state(active: str) -> str:
        return grouped_over(
            ["Links", "Mount", "Visuals"],
            {
                name: [
                    f'token visibility = "{"inherited" if name == active else "invisible"}"'
                ]
                for name in ("ReadyIndicator", "DegradedIndicator", "FaultIndicator")
            },
        )
    ready=indicator_state("ReadyIndicator")
    degraded=indicator_state("DegradedIndicator")
    fault=indicator_state("FaultIndicator")
    return f'''    variantSet "contrast_state" = {{
        "full"
        {{
{contrast_full}
        }}
        "empty"
        {{
{contrast_empty}
        }}
    }}
    variantSet "gel_state" = {{
        "full"
        {{
{gel_full}
        }}
        "empty"
        {{
{gel_empty}
        }}
    }}
    variantSet "sensor_state" = {{
        "ready"
        {{
{ready}
        }}
        "degraded"
        {{
{degraded}
        }}
        "fault"
        {{
{fault}
        }}
    }}'''.replace(f"/{ROOT_PRIM}/",f"/{root}/")


def tool_usda(bundle: ToolBundle, articulation_root: bool) -> str:
    root=STANDALONE_ROOT if articulation_root else ROOT_PRIM
    root_path=f"/{root}"
    schemas='prepend apiSchemas = ["PhysicsArticulationRootAPI"]' if articulation_root else ""
    schema_line=f"    {schemas}\n" if schemas else ""
    links="\n\n".join(link_usda(link,root_path,bundle.frames) for link in bundle.links.values())
    joints="\n\n".join(joint_usda(joint,root_path) for joint in bundle.joints)
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "{ASSET_NAME}: registered RGB, ICG/NIR, laser-speckle, thermal, Doppler, ultrasound, and oxygenation assessment around one perfusion state."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
{schema_line}\
    prepend variantSets = ["contrast_state", "gel_state", "sensor_state"]
    variants = {{
        string contrast_state = "full"
        string gel_state = "full"
        string sensor_state = "ready"
    }}
    customData = {{
        string drAnmarAssetId = "dranmar-perfusion-viability-robot-v1"
        string drAnmarAssetVersion = "{VERSION}"
        bool drAnmarClinicalValidation = false
        bool drAnmarMedicalDevice = false
        string drAnmarStatus = "simulation_training_workcell"
        string drAnmarMount = "replaces_panda_hand_at_panda_link8"
        int drAnmarSensorModalityCount = 8
        string drAnmarSharedStateContract = "all_modalities_read_one_vascular_flow_tracer_compression_and_leak_state"
        string drAnmarSensorModalities = "rgb,nir_icg,laser_speckle,thermal,doppler,ultrasound,surface_oxygenation,depth"
    }}
)
{{
{visual_materials_scope(root,VISUAL_MATERIAL_SPECS)}
{physics_materials_scope(PHYSICS_MATERIAL_SPECS)}
    def Scope "Links"
    {{
{links}
    }}
    def Scope "Joints"
    {{
{joints}
    }}
{state_variants(root)}
}}
'''


def rigid_proxy_usda(bundle: ToolBundle) -> str:
    root=PROXY_ROOT
    visuals=[]
    for link in bundle.links.values():
        T=np.eye(4); T[:3,3]=np.asarray(link.translation,dtype=float)
        for visual in link.visuals:
            mesh=visual.mesh.copy(); mesh.apply_transform(T)
            visuals.append(Visual(f"{link.name}_{visual.name}",mesh,visual.material,visual.labels))
    visual_blocks="\n".join(mesh_usda(v.name,v.mesh,f"/{root}/Looks/{v.material}",v.labels,indent="        ") for v in visuals)
    bmin,bmax=mesh_bounds([v.mesh for v in visuals]); size=bmax-bmin; center=(bmin+bmax)/2
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "Rigid perception and planning proxy for the DrAnmar perfusion and viability robot."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    customData = {{
        string drAnmarAssetId = "dranmar-perfusion-viability-robot-rigid-proxy-v1"
        bool drAnmarClinicalValidation = false
        string drAnmarRepresentation = "rigid_perception_motion_planning_and_synthetic_data_proxy"
    }}
)
{{
    bool physics:rigidBodyEnabled = true
    float physics:mass = 2.55
    point3f physics:centerOfMass = {vec(center)}
    vector3f physics:diagonalInertia = (0.0062, 0.0060, 0.0048)
    quatf physics:principalAxes = (1, 0, 0, 0)
    bool physxRigidBody:enableCCD = true
{visual_materials_scope(root,VISUAL_MATERIAL_SPECS)}
{physics_materials_scope(PHYSICS_MATERIAL_SPECS)}
    def Scope "Visuals"
    {{
{visual_blocks}
    }}
    def Scope "Collisions"
    {{
        def Cube "EnvelopeCollider" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            rel material:binding:physics = </{root}/PhysicsMaterials/PolymerPhysics>
            bool physics:collisionEnabled = true
            float physxCollision:contactOffset = 0.0005
            float physxCollision:restOffset = 0
            double size = 1
            double3 xformOp:translate = {vec(center)}
            double3 xformOp:scale = {vec(size*0.96)}
            uniform token purpose = "guide"
            token visibility = "invisible"
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
    }}
}}
'''


def tissue_materials_scope(root: str) -> str:
    specs={k:v for k,v in VISUAL_MATERIAL_SPECS.items() if k in {"TissueBase","TissueDeep","Artery","Vein","Capillary","HealthyOverlay","LowFlowOverlay","CongestionOverlay","LeakOverlay","CompressionOverlay","RecoveryOverlay","Occluder","RegionGuide"}}
    return visual_materials_scope(root,specs)


def _overlay_blocks(bundle: ToolBundle, root: str) -> tuple[str,dict[str,list[str]]]:
    blocks=[]; paths: dict[str,list[str]]={k:[] for k in bundle.region_overlays}
    material_for={
        "healthy":"HealthyOverlay","recovered":"RecoveryOverlay","arterial_occlusion":"LowFlowOverlay",
        "anastomotic_stenosis":"LowFlowOverlay","venous_congestion":"CongestionOverlay","branch_leak":"LeakOverlay",
        "retraction_ischemia":"CompressionOverlay","dressing_compression":"CompressionOverlay",
    }
    for condition,meshes in bundle.region_overlays.items():
        for index,mesh in enumerate(meshes):
            name=f"{condition.title().replace('_','')}_{index:02d}"
            paths[condition].append(name)
            blocks.append(mesh_usda(name,mesh,f"/{root}/Looks/{material_for[condition]}",("perfusion_overlay",condition),indent="        ",double_sided=True))
    return "\n".join(blocks),paths


def tissue_condition_variants(root: str, overlay_paths: dict[str,list[str]]) -> str:
    all_names=[name for names in overlay_paths.values() for name in names]
    conditions=list(overlay_paths)
    variants=[]
    for condition in conditions:
        visible=set(overlay_paths[condition])
        overlays = grouped_over(
            ["Overlays"],
            {
                name: [
                    f'token visibility = "{"inherited" if name in visible else "invisible"}"'
                ]
                for name in all_names
            },
        )
        markers = grouped_over(
            ["ConditionMarkers"],
            {
                marker: [
                    f'token visibility = "{"inherited" if marker == condition else "invisible"}"'
                ]
                for marker in conditions
            },
        )
        variants.append(f'''        "{condition}"
        {{
{overlays}
{markers}
        }}''')
    return '    variantSet "condition" = {\n'+"\n".join(variants)+'\n    }'


def tissue_usda(bundle: ToolBundle) -> str:
    root=TISSUE_ROOT
    overlay_blocks,overlay_paths=_overlay_blocks(bundle,root)
    vessel_blocks="\n".join(mesh_usda(name,mesh,f"/{root}/Looks/{mat}",(mat.lower(),"vascular_network"),indent="        ") for name,mesh,mat in bundle.vascular_meshes)
    region_frames=[]
    for region in bundle.graph["regions"]:
        region_frames.append(f'''        def Xform "Region_{region['index']:02d}"
        {{
            custom int drAnmar:regionIndex = {region['index']}
            custom string drAnmar:capillaryEdge = "{region['capillary_edge']}"
            double3 xformOp:translate = {vec(region['center_m'])}
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}''')
    # condition markers are physical/visual cues only; the flow model lives in perfusion_network.json.
    marker_parts={
        "healthy": ellipsoid_mesh((0.008,0.008,0.001),(0.082,-0.052,0.010),subdivisions=2),
        "arterial_occlusion": torus_axis(0.0045,0.0012,"x",(0.0,-0.052,-0.0024),major_sections=44,minor_sections=10),
        "venous_congestion": torus_axis(0.0055,0.0014,"x",(-0.040,0.052,-0.0020),major_sections=44,minor_sections=10),
        "anastomotic_stenosis": torus_axis(0.0038,0.0010,"x",(0.0,-0.052,-0.0024),major_sections=44,minor_sections=10),
        "branch_leak": ellipsoid_mesh((0.016,0.012,0.0015),(0.015,0.015,0.0093),subdivisions=2),
        "retraction_ischemia": rounded_bar_mesh((0.056,0.050,0.003),(0.060,0.032,0.012),0.006),
        "dressing_compression": rounded_bar_mesh((0.056,0.050,0.003),(-0.040,0.032,0.012),0.006),
        "recovered": ellipsoid_mesh((0.008,0.008,0.001),(0.082,-0.052,0.010),subdivisions=2),
    }
    marker_material={"healthy":"HealthyOverlay","arterial_occlusion":"Occluder","venous_congestion":"CongestionOverlay","anastomotic_stenosis":"Occluder","branch_leak":"LeakOverlay","retraction_ischemia":"CompressionOverlay","dressing_compression":"CompressionOverlay","recovered":"RecoveryOverlay"}
    marker_blocks=[]
    for condition,mesh in marker_parts.items():
        marker_blocks.append(f'''        def Xform "{condition}"
        {{
{mesh_usda("Marker",mesh,f"/{root}/Looks/{marker_material[condition]}",(condition,"condition_marker"),indent="            ")}
        }}''')
    surface_block=mesh_usda("TissueSurface",bundle.tissue_surface,f"/{root}/Looks/TissueBase",("perfused_tissue_surface","surface_deformable_ready"),indent="    ",double_sided=True)
    bulk_block=mesh_usda("TissueBulkProxy",bundle.tissue_bulk,f"/{root}/Looks/TissueDeep",("tissue_bulk_visual_proxy",),indent="    ")
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "Perfused tissue substrate with one shared vascular graph, 24 territories, controllable obstructions, leakage, compression, and ICG transport state."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend variantSets = ["condition"]
    variants = {{ string condition = "healthy" }}
    customData = {{
        string drAnmarAssetId = "dranmar-perfused-tissue-demo-v1"
        string drAnmarAssetVersion = "{VERSION}"
        bool drAnmarClinicalValidation = false
        int drAnmarPerfusionRegionCount = {REGION_COUNT}
        int drAnmarVascularNodeCount = {len(bundle.graph['nodes'])}
        int drAnmarVascularEdgeCount = {len(bundle.graph['edges'])}
        string drAnmarPerfusionGraph = "./perfusion_network.json"
        string drAnmarDeformableRoute = "runtime_surface_or_volume_deformable_selected_by_host_stack"
    }}
)
{{
{tissue_materials_scope(root)}
    def Scope "Geometry"
    {{
{surface_block}
{bulk_block}
    }}
    def Scope "VascularNetwork"
    {{
{vessel_blocks}
    }}
    def Scope "PerfusionRegions"
    {{
{os.linesep.join(region_frames)}
    }}
    def Scope "Overlays"
    {{
{overlay_blocks}
    }}
    def Scope "ConditionMarkers"
    {{
{os.linesep.join(marker_blocks)}
    }}
{tissue_condition_variants(root,overlay_paths)}
}}
'''


def tracer_usda() -> str:
    root=TRACER_ROOT
    mesh=ellipsoid_mesh((0.00055,0.00055,0.00055),(0,0,0),subdivisions=2)
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "ICG-like tracer particle for synthetic vascular transport visualization."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    customData = {{
        string drAnmarAssetId = "dranmar-icg-tracer-particle-v1"
        bool drAnmarClinicalValidation = false
        string drAnmarRole = "synthetic_tracer_visualization_not_dose_model"
    }}
)
{{
{visual_materials_scope(root,{"ICGParticle":VISUAL_MATERIAL_SPECS["ICGParticle"]})}
{mesh_usda("Visual",mesh,f"/{root}/Looks/ICGParticle",("icg_like_tracer","synthetic_particle"),indent="    ")}
}}
'''


def coupling_pad_usda(bundle: ToolBundle) -> str:
    root=COUPLING_ROOT
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "Ultrasound coupling pad and gel-contact proxy for robotic scanning tasks."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    customData = {{
        string drAnmarAssetId = "dranmar-ultrasound-coupling-pad-v1"
        bool drAnmarClinicalValidation = false
    }}
)
{{
    float physics:mass = 0.006
{visual_materials_scope(root,{"CouplingGel":VISUAL_MATERIAL_SPECS["CouplingGel"]})}
{physics_materials_scope({"GelPhysics":PHYSICS_MATERIAL_SPECS["GelPhysics"]})}
{mesh_usda("Visual",bundle.coupling_pad,f"/{root}/Looks/CouplingGel",("ultrasound_coupling_pad",),indent="    ",double_sided=True)}
    def Cube "Collision" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{root}/PhysicsMaterials/GelPhysics>
        double size = 1
        double3 xformOp:scale = (0.044, 0.034, 0.003)
        uniform token purpose = "guide"
        token visibility = "invisible"
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }}
}}
'''


def occluder_usda(bundle: ToolBundle) -> str:
    root=OCCLUDER_ROOT
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "Category-level flow occluder used to generate controllable perfusion deficits."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    prepend variantSets = ["state"]
    variants = {{ string state = "open" }}
    customData = {{
        string drAnmarAssetId = "dranmar-flow-occluder-v1"
        bool drAnmarClinicalValidation = false
    }}
)
{{
    float physics:mass = 0.004
{visual_materials_scope(root,{"Occluder":VISUAL_MATERIAL_SPECS["Occluder"]})}
{mesh_usda("Visual",bundle.occluder,f"/{root}/Looks/Occluder",("flow_occluder",),indent="    ")}
    variantSet "state" = {{
        "open"
        {{
            custom float drAnmar:conductanceMultiplier = 1
        }}
        "partial"
        {{
            custom float drAnmar:conductanceMultiplier = 0.15
        }}
        "closed"
        {{
            custom float drAnmar:conductanceMultiplier = 0.015
        }}
    }}
}}
'''


def phase_parameters(phase: str) -> dict[str,float]:
    values={
        "inspect": {},
        "rgb": {"focus":0.008},
        "icg": {"turret":-22,"filter":90,"focus":0.012},
        "speckle": {"filter":180,"focus":0.010,"mirror_x":7,"mirror_y":-6},
        "thermal": {"turret":24,"filter":0,"focus":0.006},
        "oxygenation": {"turret":48,"filter":270,"focus":0.010},
        "doppler": {"doppler_extension":0.046,"doppler_pitch":18,"guard":0.003},
        "ultrasound": {"us_extension":0.054,"us_pitch":-12,"us_compliance":0.004,"gel":0.005,"guard":0.004},
        "fused": {"focus":0.010,"doppler_extension":0.030,"us_extension":0.035,"guard":0.003},
    }
    return values.get(phase,{})


def link_transform(link_name: str, phase: str) -> np.ndarray:
    p=phase_parameters(phase); T=np.eye(4)
    # base positions already authored per link.
    # local inspection exports apply direct phase displacements and rotations.
    rotation=np.eye(3); delta=np.zeros(3)
    if link_name=="SensorTurret": rotation=rotation_matrix((0,0,1),math.radians(p.get("turret",0)))
    elif link_name=="FilterWheel": rotation=rotation_matrix((0,1,0),math.radians(p.get("filter",0)))
    elif link_name=="OpticalFocus": delta[2]+=p.get("focus",0)
    elif link_name=="SpeckleMirrorX": rotation=rotation_matrix((0,1,0),math.radians(p.get("mirror_x",0)))
    elif link_name=="SpeckleMirrorY": rotation=rotation_matrix((1,0,0),math.radians(p.get("mirror_y",0)))
    elif link_name=="UltrasoundCarriage": delta[2]+=p.get("us_extension",0)
    elif link_name=="UltrasoundGimbal": delta[2]+=p.get("us_extension",0); rotation=rotation_matrix((0,1,0),math.radians(p.get("us_pitch",0)))
    elif link_name=="UltrasoundCompliance": delta[2]+=p.get("us_extension",0)+p.get("us_compliance",0); rotation=rotation_matrix((0,1,0),math.radians(p.get("us_pitch",0)))
    elif link_name=="GelValve": delta[0]+=p.get("gel",0); delta[2]+=p.get("us_extension",0)
    elif link_name=="DopplerCarriage": delta[2]+=p.get("doppler_extension",0)
    elif link_name=="DopplerGimbal": delta[2]+=p.get("doppler_extension",0); rotation=rotation_matrix((0,1,0),math.radians(p.get("doppler_pitch",0)))
    elif link_name=="ContactGuard": delta[2]+=p.get("guard",0)
    T[:3,:3]=rotation; T[:3,3]=delta
    return T


def world_visual_entries(bundle: ToolBundle, phase: str="inspect") -> list[tuple[str,trimesh.Trimesh,str]]:
    entries=[]
    for link in bundle.links.values():
        base=np.eye(4); base[:3,3]=np.asarray(link.translation,dtype=float)
        motion=link_transform(link.name,phase)
        world=base@motion
        for visual in link.visuals:
            mesh=visual.mesh.copy(); mesh.apply_transform(world)
            entries.append((f"{link.name}_{visual.name}",mesh,visual.material))
    return entries


def collider_mesh(c: Collider) -> trimesh.Trimesh:
    if c.kind=="box": assert c.size is not None; mesh=box_mesh(c.size,c.center)
    elif c.kind=="cylinder": assert c.radius is not None and c.height is not None; mesh=cylinder_axis(c.radius,c.height,c.axis,c.center)
    elif c.kind=="sphere": assert c.radius is not None; mesh=ellipsoid_mesh((c.radius,c.radius,c.radius),c.center,subdivisions=2)
    else: raise ValueError(c.kind)
    R=np.asarray(rotation_matrix((1,0,0),0));
    # Collider primitive orientation is already rare in this asset; apply when non-identity.
    if c.orientation_wxyz!=(1.0,0.0,0.0,0.0):
        w,x,y,z=c.orientation_wxyz
        R=np.asarray([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        mesh=transform(mesh,rotation=R)
    return mesh


def collision_debug_entries(bundle: ToolBundle) -> list[tuple[str,trimesh.Trimesh,str]]:
    entries=world_visual_entries(bundle,"inspect")
    for link in bundle.links.values():
        base=np.eye(4); base[:3,3]=np.asarray(link.translation,dtype=float)
        for c in link.colliders:
            mesh=collider_mesh(c); mesh.apply_transform(base)
            entries.append((f"Collision_{link.name}_{c.name}",mesh,"CollisionDebug"))
    return entries


def axis_entries(bundle: ToolBundle, phase: str="inspect") -> list[tuple[str,trimesh.Trimesh,str]]:
    entries=[]
    for name,data in bundle.frames.items():
        link=bundle.links[data["parent_link"]]
        base=np.eye(4); base[:3,3]=np.asarray(link.translation,dtype=float); world=base@link_transform(link.name,phase)
        origin=np.asarray([*data["position"],1.0]); o=(world@origin)[:3]
        for axis,mat in ((np.array([0.012,0,0]),"GuideRed"),(np.array([0,0.012,0]),"GuideGreen"),(np.array([0,0,0.012]),"GuideBlue")):
            entries.append((f"{name}_{mat}",capsule_between(o,o+axis,0.00042),mat))
    return entries


def region_condition_entries(bundle: ToolBundle, condition: str) -> list[tuple[str,trimesh.Trimesh,str]]:
    entries=[("TissueBulk",bundle.tissue_bulk,"TissueDeep"),("TissueSurface",bundle.tissue_surface,"TissueBase")]
    entries += [(name,mesh,mat) for name,mesh,mat in bundle.vascular_meshes]
    mat={"healthy":"HealthyOverlay","recovered":"RecoveryOverlay","arterial_occlusion":"LowFlowOverlay","anastomotic_stenosis":"LowFlowOverlay","venous_congestion":"CongestionOverlay","branch_leak":"LeakOverlay","retraction_ischemia":"CompressionOverlay","dressing_compression":"CompressionOverlay"}[condition]
    entries += [(f"Overlay_{i}",mesh,mat) for i,mesh in enumerate(bundle.region_overlays[condition])]
    return entries


def tracer_entries(bundle: ToolBundle, stage: str) -> list[tuple[str,trimesh.Trimesh,str]]:
    entries=region_condition_entries(bundle,"healthy")
    fractions={"arrival":0.25,"peak":0.72,"washout":0.92}
    fraction=fractions[stage]
    edges=bundle.graph["edges"]
    count=max(1,int(len(edges)*fraction))
    for edge in edges[:count]:
        p0=np.asarray(bundle.graph["nodes"][edge["from"]]["position_m"],dtype=float)
        p1=np.asarray(bundle.graph["nodes"][edge["to"]]["position_m"],dtype=float)
        for j,t in enumerate(np.linspace(0.1,0.9,3)):
            p=(1-t)*p0+t*p1; p[2]+=0.0008
            entries.append((f"Tracer_{edge['id']}_{j}",ellipsoid_mesh((0.00065,0.00065,0.00065),p,subdivisions=1),"ICGParticle"))
    return entries


def franka_proxy_entries(bundle: ToolBundle, phase: str="fused") -> list[tuple[str,trimesh.Trimesh,str]]:
    entries=[]
    z=0.58
    entries.append(("Base",cylinder_axis(0.085,0.12,"z",(0,0,0.06),sections=56),"DarkPolymer"))
    joints=[np.array([0,0,0.12]),np.array([0,0,0.29]),np.array([0.10,0,0.44]),np.array([0.02,0,0.57]),np.array([-0.08,0,0.68]),np.array([0.0,0,0.78]),np.array([0,0,0.86])]
    for i,(a,b) in enumerate(zip(joints[:-1],joints[1:])):
        entries.append((f"ArmLink{i}",capsule_between(a,b,0.032 if i<3 else 0.027),"BodyPolymer"))
        entries.append((f"Joint{i}",ellipsoid_mesh((0.040,0.040,0.040),a,subdivisions=2),"DarkPolymer"))
    mount=np.array([0,0,0.86])
    for name,mesh,mat in world_visual_entries(bundle,phase):
        moved=mesh.copy(); moved.apply_translation(mount)
        entries.append((f"Tool_{name}",moved,mat))
    return entries


def export_glbs(bundle: ToolBundle) -> list[Path]:
    GLB_ROOT.mkdir(parents=True,exist_ok=True); outputs=[]
    for phase in ("inspect","rgb","icg","speckle","thermal","oxygenation","doppler","ultrasound","fused"):
        path=GLB_ROOT/f"dranmar_perfusion_tool_{phase}.glb"; export_scene(path,world_visual_entries(bundle,phase),COLORS); outputs.append(path)
    for condition in bundle.graph["conditions"]:
        path=GLB_ROOT/f"dranmar_perfused_tissue_{condition}.glb"; export_scene(path,region_condition_entries(bundle,condition),COLORS); outputs.append(path)
    for stage in ("arrival","peak","washout"):
        path=GLB_ROOT/f"dranmar_icg_{stage}.glb"; export_scene(path,tracer_entries(bundle,stage),COLORS); outputs.append(path)
    for name,entries in (
        ("dranmar_perfusion_tool_collision_debug",collision_debug_entries(bundle)),
        ("dranmar_perfusion_tool_frame_debug",world_visual_entries(bundle,"inspect")+axis_entries(bundle,"inspect")),
        ("dranmar_franka_perfusion_assembly",franka_proxy_entries(bundle,"fused")),
    ):
        path=GLB_ROOT/f"{name}.glb"; export_scene(path,entries,COLORS); outputs.append(path)
    return outputs


def generate_textures() -> list[Path]:
    TEXTURE_ROOT.mkdir(parents=True,exist_ok=True)
    outputs=[]
    rng=np.random.default_rng(91572)

    # Polymer base color and roughness detail.
    size=512
    noise=rng.normal(0,1,(size,size))
    noise=(noise-noise.min())/(noise.max()-noise.min())
    base=np.zeros((size,size,3),dtype=np.uint8)
    base[...,0]=np.clip(210+18*(noise-0.5),0,255)
    base[...,1]=np.clip(218+16*(noise-0.5),0,255)
    base[...,2]=np.clip(226+14*(noise-0.5),0,255)
    path=TEXTURE_ROOT/"polymer_basecolor.png"; Image.fromarray(base).save(path); outputs.append(path)

    rough=np.clip(155+35*(noise-0.5),0,255).astype(np.uint8)
    path=TEXTURE_ROOT/"polymer_roughness.png"; Image.fromarray(rough,mode="L").save(path); outputs.append(path)

    # Tissue texture with low-frequency mottling and fine vascular hints.
    yy,xx=np.mgrid[0:size,0:size]
    tissue=np.zeros((size,size,3),dtype=np.float32)
    tissue[...,0]=195+16*np.sin(xx/47)+9*np.sin((xx+yy)/23)
    tissue[...,1]=112+10*np.sin(yy/43)+6*np.sin((xx-yy)/29)
    tissue[...,2]=100+8*np.sin((2*xx+yy)/37)
    tissue += rng.normal(0,4,tissue.shape[:2])[...,None]
    tissue=np.clip(tissue,0,255).astype(np.uint8)
    path=TEXTURE_ROOT/"perfused_tissue_basecolor.png"; Image.fromarray(tissue).save(path); outputs.append(path)

    # Laser speckle pattern.
    speckle=(rng.exponential(scale=44,size=(size,size))).clip(0,255).astype(np.uint8)
    path=TEXTURE_ROOT/"laser_speckle_pattern.png"; Image.fromarray(speckle,mode="L").save(path); outputs.append(path)

    # Ultrasound speckle/attenuation reference.
    depth=np.linspace(1.0,0.18,size)[:,None]
    us=(rng.rayleigh(62,(size,size))*depth).clip(0,255).astype(np.uint8)
    for z in (62,178,338):
        us[max(0,z-2):min(size,z+3),:]=np.clip(us[max(0,z-2):min(size,z+3),:]+65,0,255)
    path=TEXTURE_ROOT/"ultrasound_speckle_reference.png"; Image.fromarray(us,mode="L").save(path); outputs.append(path)

    # False-color palettes as one-dimensional images.
    def gradient(name: str, stops: list[tuple[float,tuple[int,int,int]]]):
        img=np.zeros((32,512,3),dtype=np.uint8)
        for i in range(512):
            t=i/511
            for (ta,ca),(tb,cb) in zip(stops[:-1],stops[1:]):
                if ta<=t<=tb:
                    u=(t-ta)/max(tb-ta,1e-9)
                    color=np.asarray(ca)*(1-u)+np.asarray(cb)*u
                    img[:,i,:]=color.astype(np.uint8);break
        p=TEXTURE_ROOT/name; Image.fromarray(img).save(p); outputs.append(p)
    gradient("icg_palette.png",[(0,(0,0,0)),(0.25,(0,42,18)),(0.55,(0,185,73)),(1,(210,255,225))])
    gradient("thermal_palette.png",[(0,(8,4,38)),(0.25,(63,26,120)),(0.55,(220,54,54)),(0.78,(248,159,41)),(1,(255,252,190))])
    gradient("oxygenation_palette.png",[(0,(29,63,160)),(0.45,(79,129,213)),(0.65,(241,238,196)),(1,(194,31,52))])
    return outputs


def _import_integration_module():
    import importlib.util
    spec=importlib.util.spec_from_file_location("dranmar_perfusion_integration",INTEGRATION_PATH)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load integration module")
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); return module


def _add_mesh(ax, mesh: trimesh.Trimesh, material: str, max_faces: int=900):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    faces=np.asarray(mesh.faces,dtype=int)
    if len(faces)>max_faces:
        step=max(1,len(faces)//max_faces); faces=faces[::step]
    triangles=np.asarray(mesh.vertices)[faces]
    rgba=np.asarray(COLORS.get(material,(180,180,180,255)),dtype=float)/255.0
    poly=Poly3DCollection(triangles,facecolor=rgba,edgecolor=(0,0,0,0.05),linewidth=0.05)
    ax.add_collection3d(poly)


def _configure_3d(ax, title: str, entries: Sequence[tuple[str,trimesh.Trimesh,str]], elev=22, azim=-58):
    for _name,mesh,mat in entries: _add_mesh(ax,mesh,mat)
    mins,maxs=mesh_bounds([m for _,m,_ in entries]); center=(mins+maxs)/2; radius=max(maxs-mins)/2*1.12
    ax.set_xlim(center[0]-radius,center[0]+radius); ax.set_ylim(center[1]-radius,center[1]+radius); ax.set_zlim(center[2]-radius,center[2]+radius)
    ax.view_init(elev=elev,azim=azim); ax.set_title(title,fontsize=12,fontweight="bold"); ax.set_axis_off()


def make_preview(bundle: ToolBundle) -> Path:
    import matplotlib.pyplot as plt
    module=_import_integration_module(); verifier=module.ClosedLoopPerfusionVerifier(bundle.graph)
    conditions=("healthy","arterial_occlusion","venous_congestion","anastomotic_stenosis","recovered")
    scans={c:verifier.scan(c,duration_s=18.0,dt_s=0.15) for c in conditions}
    fig=plt.figure(figsize=(15,9),dpi=150)
    ax=fig.add_subplot(2,3,1,projection="3d")
    _configure_3d(ax,"Multimodal robotic head",world_visual_entries(bundle,"fused"),elev=21,azim=-56)
    titles={"healthy":"Healthy reference","arterial_occlusion":"Arterial inflow occlusion","venous_congestion":"Venous congestion","anastomotic_stenosis":"Anastomotic stenosis","recovered":"Post-intervention recovery"}
    for i,c in enumerate(conditions,start=2):
        ax=fig.add_subplot(2,3,i)
        arr=np.asarray([[r.viability_score for r in scans[c].assessment.regions[j*REGION_COLS:(j+1)*REGION_COLS]] for j in range(REGION_ROWS)])
        im=ax.imshow(arr,vmin=0,vmax=1,origin="lower",interpolation="nearest")
        ax.set_title(titles[c],fontsize=11,fontweight="bold")
        ax.set_xticks(range(REGION_COLS));ax.set_yticks(range(REGION_ROWS));ax.set_xlabel("territory column");ax.set_ylabel("territory row")
        for r in range(REGION_ROWS):
            for col in range(REGION_COLS):
                ax.text(col,r,f"{arr[r,col]:.2f}",ha="center",va="center",fontsize=7,color="white" if arr[r,col]<0.55 else "black")
        fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04,label="fused viability")
    fig.suptitle("DrAnmar Multimodal Perfusion and Tissue-Viability Robot",fontsize=18,fontweight="bold",y=0.98)
    fig.text(0.5,0.015,"One vascular-flow and tracer state drives ICG, speckle, thermal, Doppler, ultrasound, and oxygenation outputs.",ha="center",fontsize=10)
    fig.tight_layout(rect=(0,0.035,1,0.95))
    PREVIEW_ROOT.mkdir(parents=True,exist_ok=True); path=PREVIEW_ROOT/"dranmar_perfusion_viability_robot_preview.png"; fig.savefig(path,bbox_inches="tight");plt.close(fig);return path


def make_full_arm_preview(bundle: ToolBundle) -> Path:
    import matplotlib.pyplot as plt
    entries=franka_proxy_entries(bundle,"fused")
    fig=plt.figure(figsize=(9,9),dpi=150);ax=fig.add_subplot(111,projection="3d")
    _configure_3d(ax,"Franka-mounted physiological verification system",entries,elev=18,azim=-54)
    fig.text(0.5,0.04,"Panda hand replacement at panda_link8 • optical, Doppler, and ultrasound modalities share one registered TCP",ha="center",fontsize=10)
    path=PREVIEW_ROOT/"dranmar_perfusion_viability_robot_full_arm_preview.png";fig.savefig(path,bbox_inches="tight");plt.close(fig);return path


def interaction_frames(bundle: ToolBundle) -> dict[str,Any]:
    return {"schema":"dr.anmar.interaction-frames.v1","asset_id":"dranmar-perfusion-viability-robot-v1","frames":bundle.frames}


def sensor_contract(bundle: ToolBundle) -> dict[str,Any]:
    return {
        "schema":"dr.anmar.multimodal-sensor-contract.v1",
        "modalities":[
            {"id":"stereo_rgb","frames":["rgb_left_camera","rgb_right_camera"],"update_hz":30,"output":"registered_rgb_pair"},
            {"id":"nir_icg","frames":["nir_fluorescence_camera"],"update_hz":30,"output":"fluorescence_intensity_and_time_series"},
            {"id":"laser_speckle","frames":["speckle_camera","speckle_projection_center"],"update_hz":60,"output":"relative_perfusion_index"},
            {"id":"thermal","frames":["thermal_camera"],"update_hz":30,"output":"surface_temperature_c"},
            {"id":"surface_oxygenation","frames":["multispectral_camera"],"update_hz":15,"output":"sto2_fraction"},
            {"id":"depth","frames":["depth_reference","structured_light_projector"],"update_hz":30,"output":"depth_and_normals"},
            {"id":"doppler","frames":["doppler_probe_tip","doppler_beam_axis"],"update_hz":100,"output":"signed_projected_velocity"},
            {"id":"ultrasound","frames":["ultrasound_probe_face","ultrasound_probe_axis"],"update_hz":30,"output":"b_mode_and_color_flow"},
        ],
        "registration":"all_outputs_map_to_24_region_tissue_grid_and_common_world_timestamp",
        "shared_state":"perfusion_network.json plus ICGTracerTransport",
        "runtime_quality_gates":{
            "registration_error_m_max":0.003,
            "timestamp_skew_s_max":0.050,
            "minimum_usable_modalities":3,
            "explicit_abstention":True,
        },
        "consumables":{
            "contrast":"conserved requested_used_remaining ledger",
            "coupling_gel":"conserved requested_used_remaining ledger",
            "empty_state_disables_dependent_measurement":True,
        },
        "fault_states":["ready","degraded","fault"],
        "failed_modality_policy":"exclude failed modalities, renormalize valid weights, and abstain when evidence is insufficient",
        "dynamic_scene_note":"use USD RTX camera route for deforming tissue; Warp ray-caster geometry route is only for compatible static meshes",
    }


def task_contract(bundle: ToolBundle) -> dict[str,Any]:
    return {
        "schema":"dr.anmar.perfusion-viability-task.v1",
        "asset_id":"dranmar-perfusion-viability-robot-v1",
        "phases":["inspect","rgb","icg","speckle","thermal","oxygenation","doppler","ultrasound","fuse","diagnose","intervene","rescan","verify"],
        "conditions":list(bundle.graph["conditions"]),
        "outputs":["arrival_time_s","wash_in_slope_per_s","time_to_peak_s","peak_intensity","washout_slope_per_s","perfusion_asymmetry","nonperfused_fraction","vessel_flow_direction","surface_temperature_c","surface_oxygenation_fraction","sensor_disagreement","confidence","diagnostic_confidence","abstained","usable_modalities","likely_cause","recommended_action"],
        "closed_loop_actions":["remove_or_reposition_occluder_or_clip","release_venous_compression_or_revise_outflow","revise_anastomosis","control_branch_leak","release_retraction_or_reduce_dressing_pressure","no_action"],
        "diagnostic_input_boundary":"inference consumes only registered observable modality maps and temporal ICG metrics; scenario labels and latent flow fields are evaluation-only",
        "intervention_evidence":["reported_displacement_m","reported_lumen_gain_fraction","reported_contact_force_n","reported_seal_fraction","reported_dwell_s"],
        "intervention_rule":"recovery advances continuously from physical evidence; evidence-free success transitions are rejected",
        "success":"post_intervention_scan_improves_global_viability_and_reduces_nonperfused_fraction_without_new_leak_or_sensor_disagreement_fault",
        "intended_use":"simulation_training",
    }


def physics_profile(bundle: ToolBundle) -> dict[str,Any]:
    return {
        "schema":"dr.anmar.perfusion-viability-profile.v1","id":"dranmar-perfusion-viability-robot-v1","version":VERSION,
        "status":"simulation_training_model",
        "tool":{"joint_count":len(bundle.joints),"authored_mass_kg":2.537,"mount":"panda_link8_hand_replacement","work_plane_z_m":WORK_PLANE_Z},
        "tissue":{"width_m":TISSUE_WIDTH_M,"depth_m":TISSUE_DEPTH_M,"thickness_m":TISSUE_THICKNESS_M,"region_count":REGION_COUNT,"vascular_node_count":len(bundle.graph["nodes"]),"vascular_edge_count":len(bundle.graph["edges"])},
        "flow":{"model":"linear_resistive_network_with_boundary_pressure_obstruction_compression_and_leak_sinks","arterial_pressure_kpa_seed":13.3,"venous_pressure_kpa_seed":1.2,"conservation_required":True},
        "tracer":{"model":"edge_cstr_advection_with_region_exchange_and_extravascular_leak_compartments","injection_time_s":1.0,"input_peak_time_s":4.5,"not_a_dose_model":True},
        "modalities":{"rgb":"RTX_camera_context","nir_icg":"shared_tracer_state","laser_speckle":"shared_region_flow","thermal":"perfusion_heat_proxy","doppler":"projected_edge_velocity","ultrasound":"synthetic_b_mode_plus_color_flow_or_i4h_bridge","surface_oxygenation":"delivery_consumption_proxy","depth":"RTX_camera_or_host_depth_route"},
        "native_simulator_evidence":{"host":"numi","gpu":"NVIDIA GeForce RTX 4090","isaac_sim":"6.0.1.0","isaac_lab":"6.1.16","representations":["standalone","franka"],"rendered_registered_cameras":6,"rendered_depth":True,"loaded_arm_sweep":True,"surface_deformable_fixture_attachments":2},
        "evidence_boundary":["no clinical perfusion thresholds","no pharmacokinetic dosing claim","no calibrated optical transport","no calibrated laser speckle decorrelation","no calibrated thermal physiology","no clinical Doppler calibration","no diagnostic ultrasound claim","no physical payload or contact calibration","no patient-care decision support"],
    }


def collider_coverage(bundle: ToolBundle) -> dict[str,Any]:
    report={}
    for link in bundle.links.values():
        if not link.visuals or not link.colliders: continue
        vbmin,vbmax=mesh_bounds([v.mesh for v in link.visuals]); visual=vbmax-vbmin
        collider_meshes=[collider_mesh(c) for c in link.colliders]
        cbmin,cbmax=mesh_bounds(collider_meshes); coll=cbmax-cbmin
        report[link.name]={"visual_envelope_m":[float(x) for x in visual],"collider_envelope_m":[float(x) for x in coll],"axis_coverage_ratio":[float(coll[i]/max(visual[i],1e-9)) for i in range(3)],"collider_count":len(link.colliders)}
    return {"schema":"dr.anmar.collider-coverage.v1","asset_id":"dranmar-perfusion-viability-robot-v1","links":report,"note":"coverage ratios are local-link envelopes; deliberate probe-face insets reduce ghost contact"}


def asset_manifest() -> dict[str,Any]:
    return {
        "schema":"dr.anmar.asset-manifest.v1","asset_id":"dranmar-perfusion-viability-robot-v1","version":VERSION,
        "catalog_subpath":CATALOG_SUBPATH.as_posix(),
        "primary_assets":["dranmar_perfusion_viability_tool_standalone.usda","dranmar_perfusion_viability_tool_payload.usda","dranmar_perfusion_viability_tool_rigid_proxy.usda","dranmar_perfused_tissue_demo.usda","dranmar_icg_tracer_particle.usda","dranmar_ultrasound_coupling_pad.usda","dranmar_flow_occluder.usda"],
        "licence":"Apache-2.0","intended_use":"simulation_training","clinical_validation":False,
    }


def mount_contract() -> dict[str,Any]:
    return {"schema":"dr.anmar.franka-mount.v1","parent_link":"panda_link8","disabled_prims":["panda_hand_joint","panda_hand","panda_finger_joint1","panda_finger_joint2","panda_leftfinger","panda_rightfinger"],"payload_root":"DrAnmarPerfusionViabilityTool","payload_mount_link":"Links/Mount","local_translation_m":[0,0,0],"local_rotation_axis_angle":{"axis":[0,0,1],"degrees":FRANKA_HAND_EQUIVALENT_ROTATION_DEG}}


def example_scene() -> str:
    return '''#!/usr/bin/env python3
"""Minimal scene skeleton for the DrAnmar perfusion and viability robot."""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser); args=parser.parse_args()
app=AppLauncher(args).app
import isaaclab.sim as sim_utils
from orbit.surgical.assets.perfusion_viability_robot import (
    ClosedLoopPerfusionVerifier,
    make_franka_perfusion_viability_robot_cfg,
    phase_targets,
    spawn_tissue_demo,
)

def main():
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120,device=args.device))
    sim_utils.GroundPlaneCfg().func("/World/Ground",sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500).func("/World/Light",sim_utils.DomeLightCfg(intensity=2500))
    robot_cfg=make_franka_perfusion_viability_robot_cfg(prim_path="/World/Robot")
    robot=robot_cfg.class_type(robot_cfg)
    spawn_tissue_demo("/World/PerfusedTissue",condition="anastomotic_stenosis",translation=(0.55,0.0,0.82))
    sim.reset()
    verifier=ClosedLoopPerfusionVerifier()
    result=verifier.scan_intervene_rescan("anastomotic_stenosis",duration_s=18.0,dt_s=0.15)
    print(result["action"],result["viability_gain"])
    while app.is_running():
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim(); sim.step(); robot.update(sim.get_physics_dt())
if __name__=="__main__":
    main(); app.close()
'''


def docs() -> dict[str,str]:
    return {
        "MECHANISM.md":"""# Mechanism\n\nThe end effector replaces the Panda hand and registers optical, Doppler, and ultrasound sensing around one assessment TCP. Twelve driven joints position the spectral filter wheel, optical focus, speckle mirrors, ultrasound probe, Doppler probe, gel valve, and compliant guard. The authored payload mass is 2.537 kg.\n\n`ProbeContactController` couples ultrasound and Doppler acquisition to reported contact force, retracts on overload, and exposes an abort state. The tissue demo can be cooked as a current PhysX surface deformable and retained by two explicit edge-fixture attachments.\n\nThe instrument is category-level and manufacturer-neutral. Dimensions, drive gains, contact settings, masses, and tissue parameters are provisional research values.\n""",
        "SHARED_PHYSIOLOGY_MODEL.md":"""# Shared physiology model\n\nAll synthetic modalities consume one vascular state defined by `perfusion_network.json`. The model solves nodal pressures and edge flows from arterial and venous boundary pressures, edge resistance, obstruction multipliers, regional compression, and leak sinks. The same flow state drives tracer advection, thermal response, oxygenation, Doppler velocity, ultrasound patency, and viability fusion.\n\nCorrective actions change a continuous recovery fraction rather than replacing the scenario label. Flow parameters blend monotonically toward the recovered state and every solve retains the mass-conservation check.\n\nThe network is a reduced-order simulation contract. It is not CFD, pharmacokinetics, or patient-specific physiology.\n""",
        "MULTIMODAL_SENSOR_MODEL.md":"""# Multimodal sensor model\n\nRGB cameras provide scene context. ICG-like fluorescence is generated from graph-transport tracer history. Laser speckle reads normalized regional flow. Thermal output uses a perfusion heat-transfer proxy. Surface oxygenation uses an oxygen-delivery and consumption proxy. Doppler projects solved edge velocity onto the probe beam. Ultrasound can use the supplied synthetic B-mode generator or bridge to the NVIDIA Isaac for Healthcare robotic-ultrasound application using the authored probe pose.\n\nThe estimator receives registered observable maps and temporal ICG metrics only. Scenario labels and latent flow fields are excluded from inference and may be supplied only as evaluation annotations. Failed modalities are removed and remaining weights are renormalized. Registration error, timestamp skew, insufficient modality coverage, or low diagnostic confidence produces an explicit abstention.\n\nContrast and coupling gel use conservative ledgers. Empty consumables disable dependent outputs; `ready`, `degraded`, and `fault` operating states alter measurement validity and confidence rather than only changing visuals.\n""",
        "CLOSED_LOOP_VERIFICATION.md":"""# Closed-loop verification\n\nThe canonical research task is `scan → identify cause → intervene → rescan → verify recovery`. Causes include arterial inflow obstruction, venous outflow obstruction, anastomotic stenosis, branch leakage, retraction ischemia, and dressing compression.\n\nDiagnosis is blind to the authored scenario label. Intervention progress is derived from caller-reported displacement, lumen gain, seal contact, force, and dwell evidence. Evidence-free completion is rejected. The bundled deterministic evidence profile is a test fixture, is identified as such in results, and is not a physical measurement.\n\nNo generated score is a clinical diagnosis or treatment recommendation.\n""",
        "FRANKA_INTEGRATION.md":"""# Franka integration\n\nUse `make_franka_perfusion_viability_robot_cfg()` to load the standard Isaac Lab Franka, deactivate the Panda hand and finger prims, reference the payload, and attach its `Mount` link to `panda_link8`. State variants are selected before physics views initialize.\n\nDynamic tissue uses the USD RTX camera route for image generation. USD camera optical -Z is explicitly rotated onto the authored tissue-facing +Z sensor axis. The CUDA qualification captures nonconstant RGB from all six camera frames and depth from the left camera with one live RTX camera pipeline at a time. Each frame is timestamped; operational fusion must buffer or interpolate to a common time and apply the 50 ms skew gate. The loaded-arm gate then drives the 2.537 kg payload through neutral, left, and right poses. The host may bridge the ultrasound probe pose into the i4h robotic-ultrasound ray-tracing application.\n\nFor low-latency operation, prewarm and reuse one camera/render-product pipeline and its output buffers, then bind or schedule the six registered views serially. Do not construct all six pipelines concurrently. The qualification script intentionally destroys each pipeline before creating the next one to prove cleanup and maximum concurrency of one; that destructive lifecycle is a strong resource gate, not the recommended per-frame production loop.\n""",
    }


def readme() -> str:
    return f'''# {ASSET_NAME} v{VERSION}\n\nDr.Anmar executable simulation-training workcell for registered RGB, NIR/ICG, laser-speckle, thermal, Doppler, ultrasound, depth, and surface-oxygenation assessment workflows.\n\n## Catalog path\n\n`{CATALOG_SUBPATH.as_posix()}`\n\n## Primary contract\n\nOne vascular-flow and tracer state drives the synthetic modality outputs. Diagnostic inference is blind to scenario labels and latent flow fields, carries temporal ICG evidence, excludes failed modalities, and explicitly abstains on insufficient registration, timing, coverage, or confidence. Conserved contrast and gel ledgers, force-coupled probe contact, physical intervention evidence, surface-deformable fixtures, rendered camera/depth evidence, and loaded-Franka native simulator evidence are included.\n\nClinical and real-world evidence are not established.\n'''


def installer_source() -> str:
    return '''#!/usr/bin/env python3
from __future__ import annotations
import json, shutil, sys
from pathlib import Path
PACKAGE_ROOT=Path(__file__).resolve().parents[1]

def copy_contents(src: Path,dst: Path):
    for p in src.rglob("*"):
        if p.is_file():
            q=dst/p.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)

def main():
    if len(sys.argv)!=2: raise SystemExit("usage: install_into_dranmar.py /path/to/drAnmar")
    repo=Path(sys.argv[1]).expanduser().resolve()
    copy_contents(PACKAGE_ROOT/"source",repo/"source")
    copy_contents(PACKAGE_ROOT/"physics_next",repo/"physics_next")
    copy_contents(PACKAGE_ROOT/"docs",repo/"docs")
    copy_contents(PACKAGE_ROOT/"examples",repo/"examples")
    copy_contents(PACKAGE_ROOT/"scripts",repo/"scripts")
    init=repo/"source/extensions/orbit.surgical.assets/orbit/surgical/assets/__init__.py"
    if init.exists():
        text=init.read_text(encoding="utf-8")
        line="from .perfusion_viability_robot import *"
        if line not in text: init.write_text(text.rstrip()+"\\n"+line+"\\n",encoding="utf-8")
    portfolio=repo/"physics_next/dr-anmar-assets.json"
    if portfolio.exists():
        data=json.loads(portfolio.read_text(encoding="utf-8"))
        entry={"id":"dranmar-perfusion-viability-robot-v1","asset":"source/extensions/orbit.surgical.assets/data/Props/SurgicalAssessment/PerfusionViabilityRobot/dranmar_perfusion_viability_tool_standalone.usda","payload_asset":"source/extensions/orbit.surgical.assets/data/Props/SurgicalAssessment/PerfusionViabilityRobot/dranmar_perfusion_viability_tool_payload.usda","auxiliary_asset":"source/extensions/orbit.surgical.assets/data/Props/SurgicalAssessment/PerfusionViabilityRobot/dranmar_perfused_tissue_demo.usda","profile":"physics_next/surgical-assessment/dranmar-perfusion-viability-v1.json","live_behavior":"blind_registered_multimodal_fusion_with_consumable_fault_contact_intervention_and_closed_loop_rescan_contracts","deployment":"enabled_as_training_workcell","product_capability":"executable_training_workcell","training_readiness":"available_for_simulation_training_data_generation_and_evaluation","software_evidence":"repository_verified_asset_task_and_controller_contracts","native_simulator_evidence":"standalone_and_loaded_franka_runs_recorded_on_rtx_4090_isaac_sim_6_0_1","real_world_evidence":"instrumented_multimodal_bench_evidence_not_yet_established","clinical_validation":False}
        assets=[x for x in data.get("assets",[]) if x.get("id")!=entry["id"]];assets.append(entry);data["assets"]=assets
        portfolio.write_text(json.dumps(data,indent=2)+"\\n",encoding="utf-8")
    print(f"Installed into {repo}")
if __name__=="__main__": main()
'''


def sync_extension_data() -> None:
    dst=EXTENSION_ROOT/"data"/CATALOG_SUBPATH
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(ASSET_ROOT,dst)


def build_overlay() -> Path:
    overlay=PACKAGE_ROOT.parent/f"dranmar_perfusion_viability_robot_repo_overlay_v{VERSION}"
    if overlay.exists(): shutil.rmtree(overlay)
    for rel in ("source","physics_next","docs","examples","scripts"):
        src=PACKAGE_ROOT/rel
        if src.exists(): shutil.copytree(src,overlay/rel)
    zip_path=PACKAGE_ROOT.parent/f"dranmar_perfusion_viability_robot_repo_overlay_v{VERSION}.zip"
    if zip_path.exists(): zip_path.unlink()
    zip_tree(overlay,zip_path); shutil.rmtree(overlay); return zip_path


def duplicate_sibling_opinions(text: str) -> list[dict[str,Any]]:
    """Return repeated direct `over` names in the same lexical scope."""

    declaration = re.compile(
        r'^\s*(def|class|over|variantSet|variant)\s+(?:\w+\s+)?"([^"]+)"'
    )
    scopes: list[dict[str,Any]] = []
    pending: tuple[str,str] | None = None
    duplicates = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = declaration.match(line)
        if match:
            pending = (match.group(1), match.group(2))
            if match.group(1) == "over" and scopes:
                name = match.group(2)
                seen = scopes[-1]["over_names"]
                if name in seen:
                    duplicates.append(
                        {
                            "line":line_number,
                            "scope":"/".join(scope["name"] for scope in scopes),
                            "name":name,
                        }
                    )
                seen.add(name)
        opens = line.count("{")
        closes = line.count("}")
        for index in range(opens):
            if index == 0 and pending is not None:
                kind, name = pending
                scopes.append({"kind":kind,"name":name,"over_names":set()})
                pending = None
            else:
                scopes.append({"kind":"block","name":"{}","over_names":set()})
        for _ in range(closes):
            if scopes:
                scopes.pop()
        if opens == 0 and match is None and line.strip():
            pending = None
    return duplicates


def static_report(files: Sequence[Path]) -> dict[str,Any]:
    usd_files=[p for p in files if p.suffix in {".usd",".usda"}]
    results={}
    for path in usd_files:
        text=path.read_text(encoding="utf-8",errors="ignore")
        duplicates=duplicate_sibling_opinions(text)
        checks={
            "balanced_braces":text.count("{")==text.count("}"),
            "flat_quaternion_declarations":not re.search(r"quat[fd]\s+\w+\s*=\s*\([^()]*,\s*\([^()]+\)\)",text),
            "one_line_over_absent":not re.search(r'^\s*over\s+"[^"]+"\s*\{[^}\n]*\}\s*$',text,re.MULTILINE),
            "one_line_custom_data_absent":not re.search(r"customData\s*=\s*\{[^}\n]+\}",text),
            "duplicate_sibling_opinions_absent":not duplicates,
        }
        results[path.relative_to(PACKAGE_ROOT).as_posix()]={
            **checks,
            "duplicate_sibling_opinions":duplicates,
            "status":"pass" if all(checks.values()) else "fail",
        }
    status="pass" if all(item["status"]=="pass" for item in results.values()) else "fail"
    return {"schema":"dr.anmar.static-build-report.v1","asset_id":"dranmar-perfusion-viability-robot-v1","version":VERSION,"status":status,"usd":results,"file_count":len(files)}


def grouped_over(
    path: Sequence[str],
    children: dict[str, Sequence[str]],
    *,
    indent: str = "            ",
) -> str:
    """Author one parent opinion containing multiple child opinions."""

    lines = []
    for depth, name in enumerate(path):
        prefix = indent + "    " * depth
        lines.extend((f'{prefix}over "{name}"', f"{prefix}{{"))
    child_indent = indent + "    " * len(path)
    body_indent = child_indent + "    "
    for name, body_lines in children.items():
        lines.extend((f'{child_indent}over "{name}"', f"{child_indent}{{"))
        lines.extend(f"{body_indent}{line}" for line in body_lines)
        lines.append(f"{child_indent}}}")
    for depth in reversed(range(len(path))):
        prefix = indent + "    " * depth
        lines.append(f"{prefix}}}")
    return "\n".join(lines)


def write_asset_files(bundle: ToolBundle) -> list[Path]:
    ASSET_ROOT.mkdir(parents=True,exist_ok=True);DOCS_ROOT.mkdir(parents=True,exist_ok=True);EXAMPLE_ROOT.mkdir(parents=True,exist_ok=True)
    outputs=[]
    sources={
        "dranmar_perfusion_viability_tool_standalone.usda":tool_usda(bundle,True),
        "dranmar_perfusion_viability_tool_payload.usda":tool_usda(bundle,False),
        "dranmar_perfusion_viability_tool_rigid_proxy.usda":rigid_proxy_usda(bundle),
        "dranmar_perfused_tissue_demo.usda":tissue_usda(bundle),
        "dranmar_icg_tracer_particle.usda":tracer_usda(),
        "dranmar_ultrasound_coupling_pad.usda":coupling_pad_usda(bundle),
        "dranmar_flow_occluder.usda":occluder_usda(bundle),
    }
    for name,text in sources.items():
        path=ASSET_ROOT/name;path.write_text(text,encoding="utf-8");outputs.append(path)
    outputs += [
        write_json(ASSET_ROOT/"perfusion_network.json",bundle.graph),
        write_json(ASSET_ROOT/"interaction_frames.json",interaction_frames(bundle)),
        write_json(ASSET_ROOT/"sensor_modalities.json",sensor_contract(bundle)),
        write_json(ASSET_ROOT/"perfusion_viability_task_contract.json",task_contract(bundle)),
        write_json(ASSET_ROOT/"physics_profile.json",physics_profile(bundle)),
        write_json(PHYSICS_PROFILE_PATH,physics_profile(bundle)),
        write_json(ASSET_ROOT/"collider_coverage.json",collider_coverage(bundle)),
        write_json(ASSET_ROOT/"franka_mount_contract.json",mount_contract()),
        write_json(ASSET_ROOT/"asset_manifest.json",asset_manifest()),
    ]
    (ASSET_ROOT/"README.md").write_text(readme(),encoding="utf-8");outputs.append(ASSET_ROOT/"README.md")
    shutil.copy2("/usr/share/common-licenses/Apache-2.0",ASSET_ROOT/"LICENSE.txt");outputs.append(ASSET_ROOT/"LICENSE.txt")
    for name,text in docs().items():
        path=DOCS_ROOT/name;path.write_text(text,encoding="utf-8");outputs.append(path)
    path=EXAMPLE_ROOT/"franka_perfusion_viability_scene.py";path.write_text(example_scene(),encoding="utf-8");outputs.append(path)
    path=PACKAGE_ROOT/"README.md";path.write_text(readme(),encoding="utf-8");outputs.append(path)
    path=PACKAGE_ROOT/"scripts/install_into_dranmar.py";path.write_text(installer_source(),encoding="utf-8");path.chmod(0o755);outputs.append(path)
    return outputs


def generate() -> dict[str,Any]:
    graph=build_graph();bundle=build_tool(graph)
    files=write_asset_files(bundle)
    files+=generate_textures();files+=export_glbs(bundle)
    files+=[make_preview(bundle),make_full_arm_preview(bundle)]
    sync_extension_data()
    # Include copied extension data and source module in manifest inventory.
    files += [p for p in (EXTENSION_ROOT/"data"/CATALOG_SUBPATH).rglob("*") if p.is_file()]
    files += [INTEGRATION_PATH]
    report=static_report(files)
    if report["status"] != "pass":
        raise RuntimeError("strict static USDA validation failed")
    static_path=PACKAGE_ROOT/"static_build_report.json";write_json(static_path,report);files.append(static_path)
    for cache in PACKAGE_ROOT.rglob("__pycache__"):
        shutil.rmtree(cache)
    for bytecode in PACKAGE_ROOT.rglob("*.pyc"):
        bytecode.unlink()
    # Hash the complete development package, not only generated payloads.
    hashes={
        p.relative_to(PACKAGE_ROOT).as_posix():sha256(p)
        for p in sorted(PACKAGE_ROOT.rglob("*"))
        if p.is_file()
        and p.name != "SHA256SUMS.json"
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
    }
    write_json(PACKAGE_ROOT/"SHA256SUMS.json",hashes)
    dev_zip=PACKAGE_ROOT.parent/f"dranmar_perfusion_viability_robot_v{VERSION}.zip"
    if dev_zip.exists():dev_zip.unlink()
    zip_tree(PACKAGE_ROOT,dev_zip)
    catalog_zip=PACKAGE_ROOT.parent/f"dranmar_perfusion_viability_robot_catalog_v{VERSION}.zip"
    if catalog_zip.exists():catalog_zip.unlink()
    catalog_parent=PACKAGE_ROOT.parent/"_perfusion_catalog_stage"
    if catalog_parent.exists():shutil.rmtree(catalog_parent)
    target=catalog_parent/CATALOG_SUBPATH;target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(ASSET_ROOT,target)
    zip_tree(catalog_parent,catalog_zip);shutil.rmtree(catalog_parent)
    overlay_zip=build_overlay()
    for p in (dev_zip,catalog_zip,overlay_zip):write_checksum(p)
    release={
        "schema":"dr.anmar.release.v1","asset_id":"dranmar-perfusion-viability-robot-v1","version":VERSION,
        "development_package":str(dev_zip),"catalog_package":str(catalog_zip),"repository_overlay":str(overlay_zip),
        "file_count":len(hashes),"primary_usda_count":7,"glb_count":len(list(GLB_ROOT.glob("*.glb"))),
        "vascular_node_count":len(graph["nodes"]),"vascular_edge_count":len(graph["edges"]),"region_count":REGION_COUNT,
        "runtime_validation":"passed_on_numi_rtx4090_isaac_sim_6_0_1_isaac_lab_6_1_16_for_standalone_and_loaded_franka","clinical_validation":False,
    }
    release_path=PACKAGE_ROOT.parent/f"dranmar_perfusion_viability_robot_release_v{VERSION}.json";write_json(release_path,release)
    return {"dev_zip":dev_zip,"catalog_zip":catalog_zip,"overlay_zip":overlay_zip,"release":release_path,"preview":PREVIEW_ROOT/"dranmar_perfusion_viability_robot_preview.png"}


def rebuild_installed_usda() -> list[Path]:
    """Regenerate only canonical USDA files in an installed repository overlay."""

    graph = build_graph()
    bundle = build_tool(graph)
    installed_root = EXTENSION_ROOT / "data" / CATALOG_SUBPATH
    installed_root.mkdir(parents=True, exist_ok=True)
    sources = {
        "dranmar_perfusion_viability_tool_standalone.usda": tool_usda(bundle, True),
        "dranmar_perfusion_viability_tool_payload.usda": tool_usda(bundle, False),
        "dranmar_perfusion_viability_tool_rigid_proxy.usda": rigid_proxy_usda(bundle),
        "dranmar_perfused_tissue_demo.usda": tissue_usda(bundle),
        "dranmar_icg_tracer_particle.usda": tracer_usda(),
        "dranmar_ultrasound_coupling_pad.usda": coupling_pad_usda(bundle),
        "dranmar_flow_occluder.usda": occluder_usda(bundle),
    }
    outputs = []
    for name, text in sources.items():
        path = installed_root / name
        path.write_text(text, encoding="utf-8")
        outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--installed-usda-only",
        action="store_true",
        help="regenerate only the installed extension USDA surface",
    )
    args = parser.parse_args()
    if args.installed_usda_only:
        print(
            json.dumps(
                {"generated": [str(path) for path in rebuild_installed_usda()]},
                indent=2,
            )
        )
        return
    result=generate()
    print(json.dumps({k:str(v) for k,v in result.items()},indent=2))


if __name__=="__main__":
    main()
