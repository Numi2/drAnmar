// Copyright (c) 2026, Dr.Anmar Project Developers.
// SPDX-License-Identifier: BSD-3-Clause

import assert from "node:assert/strict";
import test from "node:test";
import {
  OneEuroFilter,
  adaptiveCalibrationProfile,
  adaptiveMotionGain,
  assignHandDetections,
  conditionPoseVector,
  depthOffset,
  downwardPointingClutchScore,
  handednessToArm,
  longRangeTranslation,
  median,
  normalizedAperture,
  orientationCompensatedPalmScale,
  palmFrame,
  poseOffset,
  predictPoseVector,
  robustCalibrationSample,
  rotationDelta,
  smoothVector,
  tableReachProgress,
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

function pointDownHand({ restingFingersOpen = false } = {}) {
  const points = baseHand();
  points[5] = { x: 0.62, y: 0.58, z: 0 };
  points[6] = { x: 0.60, y: 0.68, z: 0 };
  points[7] = { x: 0.58, y: 0.78, z: 0 };
  points[8] = { x: 0.56, y: 0.90, z: 0 };
  const restingChains = [
    [9, 10, 11, 12, 0.50],
    [13, 14, 15, 16, 0.44],
    [17, 18, 19, 20, 0.38],
  ];
  for (const [mcp, pip, dip, tip, x] of restingChains) {
    points[mcp] = { x, y: 0.58, z: 0 };
    if (restingFingersOpen) {
      points[pip] = { x, y: 0.68, z: 0 };
      points[dip] = { x, y: 0.79, z: 0 };
      points[tip] = { x, y: 0.90, z: 0 };
    } else {
      points[pip] = { x, y: 0.66, z: 0 };
      points[dip] = { x: x + 0.08, y: 0.62, z: 0 };
      points[tip] = { x: x + 0.08, y: 0.55, z: 0 };
    }
  }
  return points;
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

test("point-down clutch rejects an open palm and expands deliberate motion", () => {
  const pointing = pointDownHand();
  const openPalm = pointDownHand({ restingFingersOpen: true });
  const relaxed = structuredClone(pointing);
  relaxed[6] = { x: 0.60, y: 0.52, z: 0 };
  relaxed[7] = { x: 0.58, y: 0.46, z: 0 };
  relaxed[8] = { x: 0.56, y: 0.40, z: 0 };
  assert.ok(downwardPointingClutchScore(pointing) > 0.72);
  assert.ok(downwardPointingClutchScore(openPalm) < 0.48);
  assert.ok(downwardPointingClutchScore(relaxed) < 0.48);
  assert.ok(adaptiveMotionGain(0.001, true) < adaptiveMotionGain(0.04, true));
  assert.ok(adaptiveMotionGain(0.04, true) < adaptiveMotionGain(0.04, false));
  const mapped = longRangeTranslation([0.02, 0.04, -0.04], 1, 0, 1);
  assert.ok(mapped[1] > 0.04);
  assert.ok(mapped[2] < -0.04);
});

test("fingertip reaching the bottom guide maps to the safe table endpoint", () => {
  assert.equal(tableReachProgress(0.52, 0.52), 0);
  assert.ok(Math.abs(tableReachProgress(0.72, 0.52) - 0.5) < 1e-12);
  assert.equal(tableReachProgress(0.92, 0.52), 1);
  assert.deepEqual(
    longRangeTranslation([0, 0, 0], 1, 1, 1),
    [0, 0, -0.12],
  );
  assert.ok(longRangeTranslation([1, -1, 1], 2, 0, 1).every(value => Math.abs(value) <= 0.12));
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

test("automatic calibration starts safely without a setup wizard", () => {
  const profile = adaptiveCalibrationProfile([
    { pinchRatio: 0.42, depthScale: 0.21, quality: 0.95 },
  ]);
  assert.equal(profile.closeRatio, 0.10);
  assert.equal(profile.openRatio, 0.50);
  assert.equal(profile.neutralScale, 0.21);
  assert.ok(profile.openRatio - profile.closeRatio >= 0.16);
});

test("automatic calibration learns a natural pinch span and rejects outliers", () => {
  const samples = [
    ...Array.from({ length: 18 }, (_, index) => ({
      pinchRatio: 0.15 + (index % 3 - 1) * 0.004,
      depthScale: 0.22 + (index % 2 ? 0.001 : -0.001),
      quality: 0.96,
    })),
    ...Array.from({ length: 18 }, (_, index) => ({
      pinchRatio: 0.46 + (index % 3 - 1) * 0.004,
      depthScale: 0.22 + (index % 2 ? 0.001 : -0.001),
      quality: 0.96,
    })),
    { pinchRatio: 0.88, depthScale: 3, quality: 0.10 },
  ];
  let profile = adaptiveCalibrationProfile(samples);
  for (let pass = 0; pass < 5; pass += 1) {
    profile = adaptiveCalibrationProfile(samples, profile);
  }
  assert.ok(profile.closeRatio > 0.12 && profile.closeRatio < 0.18);
  assert.ok(profile.openRatio > 0.42 && profile.openRatio < 0.50);
  assert.ok(Math.abs(profile.neutralScale - 0.22) < 0.003);
  assert.ok(profile.stability > 0.90);
});

test("automatic calibration preserves a usable span during one-sided pinching", () => {
  const previous = {
    closeRatio: 0.10,
    openRatio: 0.50,
    neutralScale: 0.20,
    stability: 1,
  };
  const closedHold = Array.from({ length: 24 }, () => ({
    pinchRatio: 0.17,
    depthScale: 0.20,
    quality: 0.94,
  }));
  const profile = adaptiveCalibrationProfile(closedHold, previous);
  assert.ok(profile.closeRatio > previous.closeRatio);
  assert.ok(profile.openRatio - profile.closeRatio >= 0.16);
  assert.equal(profile.openRatio, previous.openRatio);
});

test("spatial continuity prevents a low-confidence one-frame label swap", () => {
  const pose = x => ({
    center: { x, y: 0.5, z: 0 },
    scale: 0.2,
    geometryQuality: 1,
    confidence: 0.95,
  });
  const previous = new Map([[0, pose(0.30)], [1, pose(0.70)]]);
  const assigned = assignHandDetections([
    { proposedArm: 1, labelConfidence: 0.55, pose: pose(0.31) },
    { proposedArm: 0, labelConfidence: 0.55, pose: pose(0.69) },
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
