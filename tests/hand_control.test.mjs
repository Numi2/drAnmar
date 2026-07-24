// Copyright (c) 2026, Dr.Anmar Project Developers.
// SPDX-License-Identifier: BSD-3-Clause

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  OneEuroFilter,
  adaptiveMotionGain,
  assignHandDetections,
  conditionPoseVector,
  depthOffset,
  handednessToArm,
  median,
  naturalClutchScore,
  normalizedAperture,
  orientationCompensatedPalmScale,
  palmFrame,
  poseOffset,
  predictPoseVector,
  robustCalibrationSample,
  rotationDelta,
  smoothVector,
  trackingQuality,
} from "../web/hand_control.mjs";

function baseHand() {
  const points = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  points[0] = { x: 0.5, y: 0.8, z: 0 };
  points[5] = { x: 0.62, y: 0.58, z: 0 };
  points[9] = { x: 0.5, y: 0.5, z: 0 };
  points[17] = { x: 0.38, y: 0.58, z: 0 };
  points[4] = { x: 0.46, y: 0.42, z: 0 };
  points[8] = { x: 0.54, y: 0.42, z: 0 };
  return points;
}

function rotateZ(basis, angle) {
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return basis.map(([x, y, z]) => [
    cosine * x - sine * y,
    sine * x + cosine * y,
    z,
  ]);
}

test("raw webcam handedness is corrected to physical left and right", () => {
  assert.equal(handednessToArm("Right"), 0);
  assert.equal(handednessToArm("Left"), 1);
  assert.equal(handednessToArm("Left", true), 0);
  assert.equal(handednessToArm("Right", true), 1);
  assert.equal(handednessToArm("Unknown"), null);
});

test("palm frame is finite, normalized, and centered", () => {
  const frame = palmFrame(baseHand());
  assert.ok(frame.scale > 0);
  assert.ok(Math.abs(frame.center.x - 0.5) < 1e-9);
  for (const axis of frame.basis) {
    assert.ok(Math.abs(Math.hypot(...axis) - 1) < 1e-9);
  }
});

test("closed, midpoint, and open calibration map proportionally", () => {
  assert.equal(normalizedAperture(0.1, 0.1, 0.5), 0);
  assert.ok(Math.abs(normalizedAperture(0.3, 0.1, 0.5) - 0.5) < 1e-12);
  assert.equal(normalizedAperture(0.5, 0.1, 0.5), 1);
  assert.equal(normalizedAperture(0.8, 0.1, 0.5), 1);
});

test("closer palm scale advances monotonically and clamps", () => {
  const anchor = { scale: 0.2, center: { x: 0.5, y: 0.5, z: 0 } };
  const near = { scale: 0.24, center: { x: 0.5, y: 0.5, z: -0.01 } };
  const nearer = { scale: 0.30, center: { x: 0.5, y: 0.5, z: -0.02 } };
  assert.ok(depthOffset(anchor, near) > 0);
  assert.ok(depthOffset(anchor, nearer) > depthOffset(anchor, near));
  assert.ok(depthOffset(anchor, { ...nearer, scale: 20 }) <= 0.12);
});

test("palm rotation produces a bounded three-axis rotation vector", () => {
  const basis = palmFrame(baseHand()).basis;
  const delta = rotationDelta(basis, rotateZ(basis, Math.PI / 6));
  assert.ok(Math.abs(Math.hypot(...delta) - Math.PI / 6) < 1e-6);
  const pose = { center: { x: 0.5, y: 0.5, z: 0 }, scale: 0.2, basis };
  const moved = { ...pose, basis: rotateZ(basis, Math.PI) };
  assert.ok(poseOffset(pose, moved).rotation.every(value => Math.abs(value) <= 0.8));
});

test("translation follows displayed lateral/vertical directions and smoothing", () => {
  const basis = palmFrame(baseHand()).basis;
  const anchor = { center: { x: 0.5, y: 0.5, z: 0 }, scale: 0.2, basis };
  const moved = { center: { x: 0.6, y: 0.4, z: 0 }, scale: 0.2, basis };
  const offset = poseOffset(anchor, moved);
  assert.ok(offset.translation[1] > 0);
  assert.ok(offset.translation[2] > 0);
  assert.deepEqual(smoothVector([0, 0], [1, -1], 0.25), [0.25, -0.25]);
});

test("thumb-index aperture remains independent from the natural safety clutch", async () => {
  const source = await readFile(new URL("../web/hand_control.mjs", import.meta.url), "utf8");
  assert.match(source, /landmarks\[4\], landmarks\[8\]/);
  assert.match(source, /Two-of-three consensus/);
  assert.match(source, /Instrument .* frozen · recenter freely/);
  assert.match(source, /curl the resting fingers to move/i);
});

test("natural clutch uses two-of-three finger flexion and adaptive motion gain", () => {
  const extended = baseHand();
  const fingerChains = [
    [9, 10, 11, 12, 0.50],
    [13, 14, 15, 16, 0.44],
    [17, 18, 19, 20, 0.38],
  ];
  for (const [mcp, pip, dip, tip, x] of fingerChains) {
    extended[mcp] = { x, y: 0.58, z: 0 };
    extended[pip] = { x, y: 0.46, z: 0 };
    extended[dip] = { x, y: 0.34, z: 0 };
    extended[tip] = { x, y: 0.22, z: 0 };
  }
  const curled = structuredClone(extended);
  for (const [, pip, dip, tip, x] of fingerChains) {
    curled[pip] = { x, y: 0.48, z: 0 };
    curled[dip] = { x: x + 0.07, y: 0.50, z: 0 };
    curled[tip] = { x: x + 0.08, y: 0.58, z: 0 };
  }
  assert.ok(naturalClutchScore(extended) < 0.1);
  assert.ok(naturalClutchScore(curled) > 0.34);
  assert.ok(adaptiveMotionGain(0.001, true) < adaptiveMotionGain(0.04, true));
  assert.ok(adaptiveMotionGain(0.04, true) < adaptiveMotionGain(0.04, false));
});

test("world/image palm fusion supplies an orientation-compensated depth scale", () => {
  const image = baseHand();
  const world = image.map(point => ({
    x: (point.x - 0.5) * 0.20,
    y: (point.y - 0.5) * 0.20,
    z: point.z,
  }));
  const scale = orientationCompensatedPalmScale(image, world);
  assert.ok(Number.isFinite(scale));
  assert.ok(scale > 0);
});

test("robust calibration rejects motion and ignores isolated outliers", () => {
  const stable = Array.from({ length: 23 }, (_, index) => 0.2 + (index % 3 - 1) * 0.001);
  stable.push(0.9);
  const sample = robustCalibrationSample(stable);
  assert.ok(Math.abs(sample.value - 0.2) < 0.002);
  assert.ok(sample.stability > 0.8);
  assert.equal(median([9, 1, 3, 2]), 2.5);
  assert.throws(
    () => robustCalibrationSample(Array.from({ length: 12 }, (_, index) => 0.1 + index * 0.03)),
    /moved too much/,
  );
});

test("spatial continuity prevents one-frame handedness label swaps", () => {
  const pose = x => ({
    center: { x, y: 0.5, z: 0 },
    scale: 0.2,
    geometryQuality: 1,
    confidence: 0.95,
  });
  const previous = new Map([[0, pose(0.30)], [1, pose(0.70)]]);
  const assigned = assignHandDetections([
    { proposedArm: 1, labelConfidence: 0.9, pose: pose(0.31) },
    { proposedArm: 0, labelConfidence: 0.9, pose: pose(0.69) },
  ], previous);
  assert.equal(assigned.get(0).center.x, 0.31);
  assert.equal(assigned.get(1).center.x, 0.69);
});

test("tracking quality gates implausible jumps and edge-of-frame geometry", () => {
  const stable = {
    center: { x: 0.5, y: 0.5, z: 0 },
    scale: 0.2,
    geometryQuality: 1,
    confidence: 0.94,
  };
  assert.ok(trackingQuality(stable, stable, 1 / 30) > 0.9);
  assert.equal(
    trackingQuality({ ...stable, center: { x: 0.95, y: 0.5, z: 0 } }, stable, 0.01),
    0,
  );
  assert.throws(() => {
    const points = baseHand();
    points[5] = { ...points[9] };
    points[17] = { ...points[9] };
    palmFrame(points);
  }, /degenerate/);
});

test("adaptive filter suppresses rest jitter without hiding deliberate motion", () => {
  const filter = new OneEuroFilter({ minCutoff: 1.7, beta: 0.22 });
  const raw = Array.from({ length: 30 }, (_, index) => (index % 2 ? 0.02 : -0.02));
  const filtered = raw.map((value, index) => filter.filter(value, index / 30));
  const rawVariation = raw.slice(1).reduce((sum, value, index) => sum + Math.abs(value - raw[index]), 0);
  const filteredVariation = filtered.slice(1).reduce(
    (sum, value, index) => sum + Math.abs(value - filtered[index]),
    0,
  );
  assert.ok(filteredVariation < rawVariation * 0.45);
  const step = filter.filter(0.4, 1.01);
  assert.ok(step > 0.08);
});

test("short-horizon prediction and pose conditioning stay safely bounded", () => {
  const predicted = predictPoseVector(
    [0, 0, 0, 0, 0, 0],
    [0.02, -0.02, 0.01, 0.2, -0.2, 0.1],
    0.01,
  );
  assert.ok(Math.abs(predicted[0] - 0.02) <= 0.0040001);
  assert.ok(Math.abs(predicted[3] - 0.2) <= 0.0350001);
  assert.deepEqual(conditionPoseVector([0.0001, 0, 0, 0.001, 0, 0]), [0, 0, 0, 0, 0, 0]);
});

test("vision inference is synchronized to decoded video frames", async () => {
  const source = await readFile(new URL("../web/hand_control.mjs", import.meta.url), "utf8");
  assert.match(source, /requestVideoFrameCallback/);
  assert.match(source, /metadata\.mediaTime/);
  assert.match(source, /MIN_INFERENCE_INTERVAL_MS/);
  assert.match(source, /generation !== this\.startGeneration/);
  assert.match(source, /stream\.getTracks\(\)\.forEach\(track => track\.stop\(\)\)/);
});
