// Copyright (c) 2026, Dr.Anmar Project Developers.
// SPDX-License-Identifier: BSD-3-Clause

let landmarker = null;

function serializableResult(result) {
  const points = hands => (hands || []).map(hand => hand.map(point => ({
    x: Number(point.x),
    y: Number(point.y),
    z: Number(point.z || 0),
  })));
  return {
    landmarks: points(result.landmarks),
    worldLandmarks: points(result.worldLandmarks),
    handednesses: (result.handednesses || []).map(categories => categories.map(category => ({
      categoryName: category.categoryName,
      score: Number(category.score || 0),
    }))),
  };
}

async function initialize(assetBaseUrl) {
  const base = new URL(assetBaseUrl);
  const visionModule = await import(new URL("vision_bundle.mjs", base).href);
  const files = await visionModule.FilesetResolver.forVisionTasks(
    new URL("wasm", base).href,
  );
  const options = {
    baseOptions: {
      modelAssetPath: new URL("hand_landmarker.task", base).href,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.58,
    minHandPresenceConfidence: 0.58,
    minTrackingConfidence: 0.62,
  };
  try {
    landmarker = await visionModule.HandLandmarker.createFromOptions(files, options);
  } catch (_gpuError) {
    options.baseOptions.delegate = "CPU";
    landmarker = await visionModule.HandLandmarker.createFromOptions(files, options);
  }
}

self.onmessage = async event => {
  const message = event.data || {};
  if (message.type === "init") {
    try {
      await initialize(message.assetBaseUrl);
      self.postMessage({ type: "ready" });
    } catch (error) {
      self.postMessage({ type: "error", message: String(error?.message || error) });
    }
    return;
  }
  if (message.type === "close") {
    landmarker?.close?.();
    landmarker = null;
    self.close();
    return;
  }
  if (message.type !== "frame" || !message.bitmap || !landmarker) return;

  const startedAt = performance.now();
  try {
    const result = landmarker.detectForVideo(message.bitmap, message.timestampMs);
    message.bitmap.close?.();
    self.postMessage({
      type: "result",
      frameTimeMs: message.frameTimeMs,
      inferenceMs: performance.now() - startedAt,
      result: serializableResult(result),
    });
  } catch (error) {
    message.bitmap.close?.();
    self.postMessage({ type: "error", message: String(error?.message || error) });
  }
};
