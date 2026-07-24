// Copyright (c) 2026, Dr.Anmar Project Developers.
// SPDX-License-Identifier: BSD-3-Clause

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  depthOffset,
  handednessToArm,
  normalizedAperture,
  palmFrame,
  poseOffset,
  rotationDelta,
  smoothVector,
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
  assert.ok(depthOffset(anchor, near) < 0);
  assert.ok(depthOffset(anchor, nearer) < depthOffset(anchor, near));
  assert.ok(depthOffset(anchor, { ...nearer, scale: 20 }) >= -0.12);
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

test("only thumb and index are intentional gesture controls", async () => {
  const source = await readFile(new URL("../web/hand_control.mjs", import.meta.url), "utf8");
  assert.match(source, /landmarks\[4\], landmarks\[8\]/);
  assert.doesNotMatch(source, /middle.*curl|ring.*curl|pinky.*curl/i);
  assert.match(source, /Engage tracked/);
  assert.match(source, /Freeze both/);
});
