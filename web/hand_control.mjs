// Copyright (c) 2026, Dr.Anmar Project Developers.
// SPDX-License-Identifier: BSD-3-Clause

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

const CALIBRATION_SAMPLE_COUNT = 24;
const MIN_TRACKING_QUALITY = 0.60;
const MIN_INFERENCE_INTERVAL_MS = 24;
const PREDICTION_HORIZON_S = 0.025;
const MIN_PREDICTION_HORIZON_S = 0.012;
const MAX_PREDICTION_HORIZON_S = 0.045;
const LOCAL_WEBCAM_ORIGIN = "http://127.0.0.1:12360";
const CLUTCH_ENGAGE_SCORE = 0.72;
const CLUTCH_RELEASE_SCORE = 0.48;
const CLUTCH_ENGAGE_HOLD_S = 0.18;
const TABLE_CONTACT_LINE_Y = 0.92;
const MAX_TRANSLATION_M = 0.12;
const MAX_ROTATION_RAD = 0.80;
const IDENTITY_CONFIDENCE_LOCK = 0.78;
const IDENTITY_AMBIGUITY_MARGIN = 0.12;
const AUTO_CALIBRATION_WINDOW = 48;
const AUTO_CALIBRATION_MIN_SPAN = 0.16;
const AUTO_CALIBRATION_PERSIST_INTERVAL_MS = 1000;
const DEFAULT_CLOSE_RATIO = 0.10;
const DEFAULT_OPEN_RATIO = 0.50;

export const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));

export function isLoopbackHostname(hostname) {
  const normalized = String(hostname || "").toLowerCase().replace(/^\[|\]$/g, "");
  return normalized === "localhost"
    || normalized === "::1"
    || normalized.startsWith("127.");
}

export function localWebcamTarget() {
  const target = new URL("/workstation/", LOCAL_WEBCAM_ORIGIN);
  target.searchParams.set("webcam", "1");
  return target.href;
}

export function monotonicMediaPipeTimestamp(candidateMs, previousMs = -1) {
  const candidate = Number.isFinite(Number(candidateMs))
    ? Math.floor(Number(candidateMs))
    : 0;
  const previous = Number.isFinite(Number(previousMs))
    ? Math.floor(Number(previousMs))
    : -1;
  return Math.max(0, candidate, previous + 1);
}

export function handednessToArm(label, inputMirrored = false) {
  // MediaPipe handedness assumes selfie-mirrored input. detectForVideo receives
  // the raw camera frame here, while only the preview is CSS-mirrored.
  const physicalLabel = inputMirrored
    ? label
    : label === "Left"
      ? "Right"
      : label === "Right"
        ? "Left"
        : label;
  if (physicalLabel === "Left") return 0;
  if (physicalLabel === "Right") return 1;
  return null;
}

export function distance3(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function subtract(a, b) {
  return [a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0)];
}

function normalize(vector, label = "vector") {
  const length = Math.hypot(...vector);
  if (!Number.isFinite(length) || length < 1e-7) throw new Error(`${label} is degenerate`);
  return vector.map(value => value / length);
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot(a, b) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

export function palmFrame(landmarks) {
  if (!landmarks || landmarks.length < 21) throw new Error("A complete 21-point hand is required");
  if (!landmarks.every(point => [point.x, point.y, point.z || 0].every(Number.isFinite))) {
    throw new Error("Hand landmarks must be finite");
  }
  const wrist = landmarks[0];
  const indexMcp = landmarks[5];
  const middleMcp = landmarks[9];
  const pinkyMcp = landmarks[17];
  const across = normalize(subtract(indexMcp, pinkyMcp), "palm width");
  const forwardSeed = normalize(subtract(middleMcp, wrist), "palm length");
  const rawNormal = cross(across, forwardSeed);
  const orthogonality = Math.hypot(...rawNormal);
  if (orthogonality < 0.08) throw new Error("Palm frame is too edge-on or degenerate");
  const normal = normalize(rawNormal, "palm normal");
  const forward = normalize(cross(normal, across), "palm forward");
  const centerPoints = [wrist, indexMcp, middleMcp, pinkyMcp];
  const center = centerPoints.reduce(
    (sum, point) => ({
      x: sum.x + point.x / centerPoints.length,
      y: sum.y + point.y / centerPoints.length,
      z: sum.z + (point.z || 0) / centerPoints.length,
    }),
    { x: 0, y: 0, z: 0 },
  );
  const scale = (
    distance3(wrist, middleMcp)
    + distance3(indexMcp, pinkyMcp)
  ) * 0.5;
  if (!Number.isFinite(scale) || scale < 1e-4) throw new Error("Palm scale is invalid");
  const geometryQuality = clamp((orthogonality - 0.08) / 0.55, 0, 1);
  return { center, scale, basis: [across, forward, normal], geometryQuality };
}

function multiplyMatrix(a, b) {
  return a.map(row => b[0].map((_, column) => dot(row, b.map(item => item[column]))));
}

function transpose(matrix) {
  return matrix[0].map((_, column) => matrix.map(row => row[column]));
}

export function rotationDelta(anchorBasis, currentBasis) {
  const anchorMatrix = transpose(anchorBasis);
  const currentMatrix = transpose(currentBasis);
  const rotation = multiplyMatrix(currentMatrix, transpose(anchorMatrix));
  const cosine = clamp((rotation[0][0] + rotation[1][1] + rotation[2][2] - 1) * 0.5, -1, 1);
  const angle = Math.acos(cosine);
  if (angle < 1e-6) return [0, 0, 0];
  if (Math.PI - angle < 1e-4) {
    const signedRoot = (value, signHint) => (
      Math.sign(signHint || 1) * Math.sqrt(Math.max(0, value))
    );
    const axis = normalize([
      Math.sqrt(Math.max(0, (rotation[0][0] + 1) * 0.5)),
      signedRoot((rotation[1][1] + 1) * 0.5, rotation[0][1]),
      signedRoot((rotation[2][2] + 1) * 0.5, rotation[0][2]),
    ], "half-turn axis");
    return axis.map(value => value * angle);
  }
  const denominator = 2 * Math.sin(angle);
  const axis = [
    (rotation[2][1] - rotation[1][2]) / denominator,
    (rotation[0][2] - rotation[2][0]) / denominator,
    (rotation[1][0] - rotation[0][1]) / denominator,
  ];
  return axis.map(value => value * angle);
}

export function normalizedAperture(pinchRatio, closeRatio, openRatio) {
  const span = Number(openRatio) - Number(closeRatio);
  if (!Number.isFinite(span) || span <= 0.01) throw new Error("Open calibration must be wider than closed calibration");
  return clamp((Number(pinchRatio) - Number(closeRatio)) / span, 0, 1);
}

export function depthOffset(anchorPose, currentPose, gain = 1) {
  const anchorDepthScale = Number(anchorPose.depthScale || anchorPose.scale);
  const currentDepthScale = Number(currentPose.depthScale || currentPose.scale);
  if (
    !Number.isFinite(anchorDepthScale)
    || !Number.isFinite(currentDepthScale)
    || anchorDepthScale <= 0
    || currentDepthScale <= 0
  ) {
    return 0;
  }
  // Image/world landmark agreement produces an orientation-compensated
  // inverse-depth signal: a larger projected scale means the hand is closer.
  return clamp(Math.log(currentDepthScale / anchorDepthScale) * 0.10 * gain, -0.12, 0.12);
}

export function poseOffset(anchorPose, currentPose, gain = 1) {
  const translation = [
    depthOffset(anchorPose, currentPose, gain),
    clamp((currentPose.center.x - anchorPose.center.x) * 0.18 * gain, -0.12, 0.12),
    clamp((anchorPose.center.y - currentPose.center.y) * 0.15 * gain, -0.12, 0.12),
  ];
  const cameraRotation = rotationDelta(anchorPose.basis, currentPose.basis);
  // MediaPipe world axes are camera right, camera down, and camera depth.
  // Transmit the same semantic order as translation: forward, right, up.
  const rotation = [
    cameraRotation[2],
    cameraRotation[0],
    -cameraRotation[1],
  ].map(value => clamp(value * gain, -0.8, 0.8));
  return { translation, rotation };
}

function cosineBetween(a, b) {
  const aLength = Math.hypot(...a);
  const bLength = Math.hypot(...b);
  if (aLength < 1e-7 || bLength < 1e-7) return 1;
  return clamp(dot(a, b) / (aLength * bLength), -1, 1);
}

export function fingerFlexionScore(landmarks, indices) {
  if (!landmarks || landmarks.length < 21 || indices.length !== 4) return 0;
  const [mcp, pip, dip, tip] = indices.map(index => landmarks[index]);
  const proximal = subtract(pip, mcp);
  const middle = subtract(dip, pip);
  const distal = subtract(tip, dip);
  const meanDirectionCosine = (
    cosineBetween(proximal, middle)
    + cosineBetween(middle, distal)
  ) * 0.5;
  return clamp((1 - meanDirectionCosine) / 0.70, 0, 1);
}

export function downwardPointingClutchScore(imageLandmarks, worldLandmarks = null) {
  if (!imageLandmarks || imageLandmarks.length < 21) return 0;
  const geometryLandmarks = worldLandmarks?.length === 21 ? worldLandmarks : imageLandmarks;
  const indexMcp = imageLandmarks[5];
  const indexTip = imageLandmarks[8];
  const indexDirection = subtract(indexTip, indexMcp);
  const indexLength = Math.hypot(indexDirection[0], indexDirection[1]);
  if (!Number.isFinite(indexLength) || indexLength < 1e-6) return 0;

  let imageScale = 0;
  try {
    imageScale = palmFrame(imageLandmarks).scale;
  } catch (_error) {
    return 0;
  }
  const indexFlexion = fingerFlexionScore(geometryLandmarks, [5, 6, 7, 8]);
  const restingFingerCurl = [
    fingerFlexionScore(geometryLandmarks, [9, 10, 11, 12]),
    fingerFlexionScore(geometryLandmarks, [13, 14, 15, 16]),
    fingerFlexionScore(geometryLandmarks, [17, 18, 19, 20]),
  ].sort((left, right) => right - left)[1] || 0;
  const restingTipY = median([12, 16, 20].map(index => imageLandmarks[index].y));
  const directionScore = clamp((indexDirection[1] / indexLength - 0.42) / 0.46, 0, 1);
  const extensionScore = clamp((0.40 - indexFlexion) / 0.30, 0, 1);
  const lengthScore = clamp((indexLength / Math.max(imageScale, 1e-6) - 0.65) / 0.70, 0, 1);
  const tipLeadScore = clamp(
    ((indexTip.y - restingTipY) / Math.max(imageScale, 1e-6) - 0.10) / 0.75,
    0,
    1,
  );
  const restingFingerScore = clamp((restingFingerCurl - 0.16) / 0.38, 0, 1);
  const pointingCore = (
    0.36 * directionScore
    + 0.25 * extensionScore
    + 0.18 * lengthScore
    + 0.21 * tipLeadScore
  );
  // Tucked resting fingers distinguish an intentional one-finger point from
  // an open palm, while retaining tolerance for one partially occluded finger.
  return clamp(pointingCore * (0.48 + 0.52 * restingFingerScore), 0, 1);
}

function smoothStep01(value) {
  const bounded = clamp(Number(value) || 0, 0, 1);
  return bounded * bounded * (3 - 2 * bounded);
}

export function tableReachProgress(
  indexTipY,
  anchorIndexTipY,
  contactLineY = TABLE_CONTACT_LINE_Y,
) {
  const tip = Number(indexTipY);
  const anchor = Number(anchorIndexTipY);
  const contact = Number(contactLineY);
  if (![tip, anchor, contact].every(Number.isFinite) || contact <= anchor + 1e-4) return 0;
  return smoothStep01((tip - anchor) / (contact - anchor));
}

export function longRangeTranslation(
  translation,
  motionGain,
  tableProgress = 0,
  takeUp = 1,
) {
  const gain = Math.max(0, Number(motionGain) || 0) * clamp(Number(takeUp) || 0, 0, 1);
  const axisReach = [0.92, 1.08, 1.18];
  const mapped = translation.map((value, index) => Number(value) * gain * axisReach[index]);
  // Preserve ordinary camera-up motion until the fingertip deliberately moves
  // toward the table guide. The old min() formulation suppressed every upward
  // command even when table reach was zero. Blending preserves full XYZ control
  // and still lands exactly on the bounded table endpoint at 100% reach.
  const tableBlend = smoothStep01(tableProgress) * clamp(Number(takeUp) || 0, 0, 1);
  mapped[2] = mapped[2] * (1 - tableBlend) - MAX_TRANSLATION_M * tableBlend;
  return mapped.map(value => clamp(value, -MAX_TRANSLATION_M, MAX_TRANSLATION_M));
}

export function orientationCompensatedPalmScale(imageLandmarks, worldLandmarks) {
  if (
    !imageLandmarks
    || !worldLandmarks
    || imageLandmarks.length < 21
    || worldLandmarks.length < 21
  ) {
    return palmFrame(imageLandmarks).scale;
  }
  const pairs = [[0, 5], [0, 9], [0, 17], [5, 9], [9, 13], [13, 17], [5, 17]];
  const ratios = [];
  for (const [from, to] of pairs) {
    const imageDistance = Math.hypot(
      imageLandmarks[to].x - imageLandmarks[from].x,
      imageLandmarks[to].y - imageLandmarks[from].y,
    );
    const worldProjectedDistance = Math.hypot(
      worldLandmarks[to].x - worldLandmarks[from].x,
      worldLandmarks[to].y - worldLandmarks[from].y,
    );
    if (
      Number.isFinite(imageDistance)
      && Number.isFinite(worldProjectedDistance)
      && imageDistance > 1e-5
      && worldProjectedDistance > 1e-5
    ) {
      ratios.push(imageDistance / worldProjectedDistance);
    }
  }
  return ratios.length >= 3 ? median(ratios) : palmFrame(imageLandmarks).scale;
}

export function adaptiveMotionGain(offsetMagnitudeM, precision = true) {
  const normalized = clamp((Number(offsetMagnitudeM) - 0.0015) / 0.028, 0, 1);
  const smooth = normalized * normalized * (3 - 2 * normalized);
  return precision
    ? 0.26 + 0.86 * smooth
    : 0.42 + 1.18 * smooth;
}

export function smoothVector(previous, current, alpha = 0.35) {
  return current.map((value, index) => previous[index] + alpha * (value - previous[index]));
}

export function median(values) {
  const finite = values.map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  if (!finite.length) throw new Error("At least one finite sample is required");
  const middle = Math.floor(finite.length / 2);
  return finite.length % 2 ? finite[middle] : (finite[middle - 1] + finite[middle]) * 0.5;
}

export function robustCalibrationSample(values, minimumSamples = 12) {
  const finite = values.map(Number).filter(Number.isFinite);
  if (finite.length < minimumSamples) throw new Error(`Hold steady for at least ${minimumSamples} samples`);
  const value = median(finite);
  const mad = median(finite.map(sample => Math.abs(sample - value)));
  const relativeMad = mad / Math.max(Math.abs(value), 1e-6);
  if (relativeMad > 0.075) throw new Error("Hand moved too much during calibration");
  return {
    value,
    mad,
    samples: finite.length,
    stability: clamp(1 - relativeMad / 0.075, 0, 1),
  };
}

export function adaptiveCalibrationProfile(samples, previous = null) {
  const valid = (samples || []).filter(sample => (
    Number.isFinite(Number(sample?.pinchRatio))
    && Number.isFinite(Number(sample?.depthScale))
    && Number(sample.depthScale) > 0
    && Number(sample.pinchRatio) >= 0.03
    && Number(sample.pinchRatio) <= 0.90
    && Number(sample.quality ?? 1) >= MIN_TRACKING_QUALITY
  ));
  if (!valid.length) {
    return previous
      ? { ...previous }
      : {
          closeRatio: DEFAULT_CLOSE_RATIO,
          openRatio: DEFAULT_OPEN_RATIO,
          neutralScale: 1,
          stability: 0,
        };
  }

  const pinches = valid.map(sample => Number(sample.pinchRatio)).sort((a, b) => a - b);
  const depths = valid.map(sample => Number(sample.depthScale));
  const pinchMedian = median(pinches);
  const depthMedian = median(depths);
  const depthMad = median(depths.map(value => Math.abs(value - depthMedian)));
  const priorClose = Number.isFinite(Number(previous?.closeRatio))
    ? Number(previous.closeRatio)
    : DEFAULT_CLOSE_RATIO;
  const priorOpen = Number.isFinite(Number(previous?.openRatio))
    ? Number(previous.openRatio)
    : DEFAULT_OPEN_RATIO;
  const priorNeutral = Number.isFinite(Number(previous?.neutralScale)) && Number(previous.neutralScale) > 0
    ? Number(previous.neutralScale)
    : depthMedian;
  const midpoint = (priorClose + priorOpen) * 0.5;
  let targetClose = priorClose;
  let targetOpen = priorOpen;

  if (valid.length >= 6) {
    const lower = pinches[Math.floor((pinches.length - 1) * 0.20)];
    const upper = pinches[Math.ceil((pinches.length - 1) * 0.80)];
    if (upper - lower >= AUTO_CALIBRATION_MIN_SPAN * 0.72) {
      targetClose = lower;
      targetOpen = upper;
    } else if (pinchMedian <= midpoint - 0.035) {
      targetClose = pinchMedian;
    } else if (pinchMedian >= midpoint + 0.035) {
      targetOpen = pinchMedian;
    }
  }

  const adaptation = previous ? 0.22 : 0.55;
  let closeRatio = clamp(
    priorClose + adaptation * (targetClose - priorClose),
    0.03,
    0.62,
  );
  let openRatio = clamp(
    priorOpen + adaptation * (targetOpen - priorOpen),
    0.18,
    0.90,
  );
  if (openRatio - closeRatio < AUTO_CALIBRATION_MIN_SPAN) {
    const center = (openRatio + closeRatio) * 0.5;
    closeRatio = clamp(center - AUTO_CALIBRATION_MIN_SPAN * 0.5, 0.03, 0.62);
    openRatio = clamp(closeRatio + AUTO_CALIBRATION_MIN_SPAN, 0.18, 0.90);
  }

  const neutralAdaptation = previous ? 0.08 : 1;
  const neutralScale = priorNeutral + neutralAdaptation * (depthMedian - priorNeutral);
  const relativeDepthMad = depthMad / Math.max(depthMedian, 1e-6);
  return {
    closeRatio,
    openRatio,
    neutralScale,
    stability: clamp(1 - relativeDepthMad / 0.10, 0, 1),
  };
}

function smoothingAlpha(cutoff, dt) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

export class OneEuroFilter {
  constructor({ minCutoff = 1.7, beta = 0.22, derivativeCutoff = 1.0 } = {}) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.derivativeCutoff = derivativeCutoff;
    this.reset();
  }

  reset(value = null, timestampSeconds = null) {
    this.value = value;
    this.rawValue = value;
    this.derivative = 0;
    this.timestampSeconds = timestampSeconds;
  }

  filter(value, timestampSeconds) {
    const current = Number(value);
    const timestamp = Number(timestampSeconds);
    if (!Number.isFinite(current) || !Number.isFinite(timestamp)) throw new Error("Filter input must be finite");
    if (this.value === null || this.timestampSeconds === null || timestamp <= this.timestampSeconds) {
      this.reset(current, timestamp);
      return current;
    }
    const dt = clamp(timestamp - this.timestampSeconds, 1 / 240, 0.2);
    const rawDerivative = (current - this.rawValue) / dt;
    const derivativeAlpha = smoothingAlpha(this.derivativeCutoff, dt);
    this.derivative += derivativeAlpha * (rawDerivative - this.derivative);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.derivative);
    const alpha = smoothingAlpha(cutoff, dt);
    this.value += alpha * (current - this.value);
    this.rawValue = current;
    this.timestampSeconds = timestamp;
    return this.value;
  }
}

export class OneEuroVectorFilter {
  constructor(dimensions, options = {}) {
    this.filters = Array.from({ length: dimensions }, () => new OneEuroFilter(options));
  }

  reset(values = null, timestampSeconds = null) {
    this.filters.forEach((filter, index) => filter.reset(values?.[index] ?? null, timestampSeconds));
  }

  filter(values, timestampSeconds) {
    if (values.length !== this.filters.length) throw new Error("Filter dimension mismatch");
    return values.map((value, index) => this.filters[index].filter(value, timestampSeconds));
  }
}

export function predictPoseVector(
  previous,
  current,
  dt,
  horizonSeconds = PREDICTION_HORIZON_S,
) {
  if (!previous || previous.length !== current.length || !Number.isFinite(dt) || dt <= 0 || dt > 0.2) {
    return [...current];
  }
  const leadLimits = current.map((_, index) => index < 3 ? 0.004 : 0.035);
  return current.map((value, index) => {
    const lead = clamp((value - previous[index]) * horizonSeconds / dt, -leadLimits[index], leadLimits[index]);
    return value + lead;
  });
}

export function conditionPoseVector(
  values,
  {
    translationDeadband = 0.00045,
    rotationDeadband = 0.006,
  } = {},
) {
  return values.map((value, index) => {
    const deadband = index < 3 ? translationDeadband : rotationDeadband;
    if (Math.abs(value) <= deadband) return 0;
    return Math.sign(value) * (Math.abs(value) - deadband);
  });
}

export function landmarkGeometryQuality(landmarks) {
  if (!landmarks || landmarks.length < 21) return 0;
  let frame;
  try {
    frame = palmFrame(landmarks);
  } catch (_error) {
    return 0;
  }
  const scale = Math.max(frame.scale, 1e-6);
  const boneRatios = HAND_CONNECTIONS.map(
    ([from, to]) => distance3(landmarks[from], landmarks[to]) / scale,
  );
  const plausibleFraction = boneRatios.filter(
    ratio => Number.isFinite(ratio) && ratio >= 0.035 && ratio <= 1.65,
  ).length / boneRatios.length;
  const palmSegments = [
    distance3(landmarks[0], landmarks[5]),
    distance3(landmarks[0], landmarks[9]),
    distance3(landmarks[0], landmarks[17]),
    distance3(landmarks[5], landmarks[17]),
  ].map(value => value / scale);
  const palmSpread = Math.max(...palmSegments) - Math.min(...palmSegments);
  const consistency = clamp((1.8 - palmSpread) / 1.5, 0, 1);
  return clamp(plausibleFraction * consistency, 0, 1);
}

export function trackingQuality(pose, previousPose = null, dt = 1 / 30) {
  if (!pose || !Number.isFinite(pose.scale) || pose.scale <= 0) return 0;
  // MediaPipe's category score is handedness certainty, not landmark accuracy.
  // Pose acceptance therefore uses geometric and temporal evidence while
  // handedness confidence is reserved for identity assignment.
  const landmarkQuality = clamp(
    Number(pose.landmarkQuality ?? pose.confidence) || 0,
    0,
    1,
  );
  const geometry = clamp(Number(pose.geometryQuality) || 0, 0, 1);
  const margin = Math.min(
    pose.center.x,
    1 - pose.center.x,
    pose.center.y,
    1 - pose.center.y,
  );
  const boundsQuality = clamp((margin - 0.005) / 0.08, 0, 1);
  let motionQuality = 1;
  if (previousPose && Number.isFinite(dt) && dt > 0) {
    const centerVelocity = distance3(pose.center, previousPose.center) / dt;
    const scaleVelocity = Math.abs(Math.log(pose.scale / previousPose.scale)) / dt;
    motionQuality = Math.min(
      clamp((4.0 - centerVelocity) / 2.0, 0, 1),
      clamp((6.0 - scaleVelocity) / 3.0, 0, 1),
    );
  }
  return clamp(Math.min(landmarkQuality, geometry, boundsQuality) * motionQuality, 0, 1);
}

export function assignHandDetectionsDetailed(detections, previousPoses = new Map()) {
  const usable = detections
    .filter(item => item?.pose && (item.proposedArm === 0 || item.proposedArm === 1))
    .slice(0, 2);
  const poses = new Map();
  const ambiguousArms = new Set();
  if (!usable.length) return { poses, ambiguousArms };

  if (usable.length === 1) {
    const detection = usable[0];
    const proposed = detection.proposedArm;
    const other = proposed === 0 ? 1 : 0;
    const proposedPrevious = previousPoses.get(proposed);
    const otherPrevious = previousPoses.get(other);
    const proposedDistance = proposedPrevious
      ? distance3(detection.pose.center, proposedPrevious.center)
      : Infinity;
    const otherDistance = otherPrevious
      ? distance3(detection.pose.center, otherPrevious.center)
      : Infinity;
    const identityConfidence = clamp(Number(detection.labelConfidence) || 0, 0, 1);

    // Never hand an engaged instrument to the other physical hand merely
    // because an occlusion left one detection near the other hand's last pose.
    // A high-confidence label keeps ownership; a low-confidence conflict
    // freezes both identities until an unambiguous frame arrives.
    if (
      proposedPrevious
      && otherPrevious
      && identityConfidence < IDENTITY_CONFIDENCE_LOCK
      && otherDistance + 0.06 < proposedDistance
    ) {
      ambiguousArms.add(0);
      ambiguousArms.add(1);
      return { poses, ambiguousArms };
    }
    poses.set(proposed, detection.pose);
    return { poses, ambiguousArms };
  }

  const assignments = [[0, 1], [1, 0]];
  const candidates = [];
  for (const arms of assignments) {
    let cost = 0;
    for (let index = 0; index < usable.length; index += 1) {
      const detection = usable[index];
      const arm = arms[index];
      const labelConfidence = clamp(Number(detection.labelConfidence) || 0, 0, 1);
      if (detection.proposedArm !== arm) cost += 1.35 * labelConfidence;
      const previous = previousPoses.get(arm);
      if (previous) cost += 2.4 * distance3(detection.pose.center, previous.center);
    }
    candidates.push({ cost, arms });
  }
  candidates.sort((left, right) => left.cost - right.cost);
  const [best, second] = candidates;
  if (second && second.cost - best.cost < IDENTITY_AMBIGUITY_MARGIN) {
    ambiguousArms.add(0);
    ambiguousArms.add(1);
    return { poses, ambiguousArms };
  }
  best.arms.forEach((arm, index) => {
    const detection = usable[index];
    if (
      arm !== detection.proposedArm
      && Number(detection.labelConfidence) >= IDENTITY_CONFIDENCE_LOCK
    ) {
      ambiguousArms.add(arm);
      return;
    }
    poses.set(arm, detection.pose);
  });
  for (const arm of ambiguousArms) poses.delete(arm);
  return { poses, ambiguousArms };
}

export function assignHandDetections(detections, previousPoses = new Map()) {
  return assignHandDetectionsDetailed(detections, previousPoses).poses;
}

function numericCalibration(calibration) {
  if (!calibration || typeof calibration !== "object") return null;
  const hands = {};
  for (const [arm, values] of Object.entries(calibration.hands || {})) {
    const close = Number(values.closeRatio);
    const open = Number(values.openRatio);
    const neutralScale = Number(values.neutralScale);
    if (![close, open, neutralScale].every(Number.isFinite) || open <= close + 0.01 || neutralScale <= 0) {
      continue;
    }
    hands[arm] = {
      closeRatio: close,
      openRatio: open,
      neutralScale,
      stability: clamp(Number(values.stability ?? 1), 0, 1),
    };
  }
  return Object.keys(hands).length ? { version: 2, hands } : null;
}

function createInterface() {
  const style = document.createElement("style");
  style.textContent = `
    .hand-control-dock{position:absolute;z-index:32;left:12px;top:12px;display:flex;align-items:center;pointer-events:auto}
    .hand-control-launch{display:flex;align-items:center;gap:7px;min-height:34px!important;padding:0 10px!important;border:1px solid #52747d!important;border-radius:9px!important;background:#101a1fdd!important;color:#e6eef0!important;box-shadow:0 8px 26px #0008!important;backdrop-filter:blur(10px);font-size:10px!important}
    .hand-control-launch:hover{border-color:#8bc6cd!important;background:#1c2a30ee!important}.hand-control-launch .hand-camera-dot{width:7px;height:7px;border-radius:50%;background:#718087;box-shadow:0 0 0 3px #71808722}
    .hand-control-launch.state-active{border-color:#79c8a2!important;background:#162922ee!important;color:#effff7!important}.hand-control-launch.state-active .hand-camera-dot{background:#79c8a2;box-shadow:0 0 10px #79c8a299}
    .hand-control-launch kbd{height:18px;min-width:30px;padding:0 4px;font-size:8px}
    .hand-panel{box-sizing:border-box;container-type:inline-size;position:fixed;z-index:70;right:18px;bottom:18px;width:min(420px,calc(100vw - 24px));min-width:min(300px,calc(100vw - 12px));min-height:min(260px,calc(100vh - 12px));max-width:calc(100vw - 12px);max-height:calc(100vh - 12px);resize:none;overflow:auto;border:1px solid #2f5968;border-radius:14px;background:#07151df2;color:#e9f8fa;box-shadow:0 18px 52px #000b;backdrop-filter:blur(16px);padding:12px}
    .hand-panel.hidden{display:none}.hand-head{display:flex;align-items:flex-start;gap:8px;cursor:grab;touch-action:none;user-select:none}.hand-head.dragging{cursor:grabbing}.hand-head h2{margin:1px 0 4px;color:#e9f8fa}.hand-head p{margin:0;color:#88a6b2;font-size:11px}.hand-head .spacer{flex:1}.hand-head-actions{display:flex;gap:3px;flex-wrap:nowrap;justify-content:flex-end}.hand-head button{min-height:25px;padding:3px 6px;cursor:pointer;touch-action:auto;font-size:8px}.hand-head button.hand-head-compact{min-height:21px;padding:2px 5px;border-color:#294650;background:#0a171d;color:#9ab4bc;font-size:7px}.hand-head button.hand-head-compact:hover{border-color:#4b7580;color:#d7eef2}.hand-head button.engaged{background:#2cd2e8;color:#031014;border-color:#2cd2e8}
    .hand-video-wrap{position:relative;margin-top:12px;aspect-ratio:16/9;border-radius:11px;overflow:hidden;background:#020608;border:1px solid #24404d}.hand-video-wrap video,.hand-video-wrap canvas{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform:scaleX(-1)}.hand-video-wrap canvas{z-index:1;transform:none;pointer-events:none}
    .hand-banner{box-sizing:border-box;position:absolute;z-index:3;left:10px;top:10px;max-width:62%;padding:6px 9px;border-radius:7px;background:#07151ddd;border:1px solid #315766;color:#b7d5dc;font:700 10px/1.35 ui-monospace,SFMono-Regular;white-space:normal}.hand-banner.good{color:#42e49b;border-color:#32725e}.hand-banner.warn{color:#ffba93;border-color:#82513d}
    .hand-metrics{position:absolute;z-index:3;right:10px;top:10px;display:flex;gap:5px;flex-wrap:wrap;justify-content:flex-end;max-width:72%}.hand-metrics span{padding:5px 7px;border-radius:6px;background:#07151ddd;border:1px solid #264653;color:#9db6bf;font:700 9px ui-monospace,SFMono-Regular}
    .hand-surface-guide{position:absolute;z-index:2;left:0;right:0;bottom:0;height:8%;display:flex;align-items:flex-end;justify-content:space-between;padding:0 10px 7px;box-sizing:border-box;border-top:1px dashed #66828b99;background:linear-gradient(180deg,transparent,#07151d99);color:#91aab2;pointer-events:none;transition:border-color 140ms ease,background 140ms ease,color 140ms ease,box-shadow 140ms ease}.hand-surface-guide span,.hand-surface-guide b{padding:3px 6px;border-radius:5px;background:#07151ddd;font:800 8px/1 ui-monospace,SFMono-Regular;letter-spacing:.04em}.hand-surface-guide b{color:#c5d6da}.hand-surface-guide.pointing{border-color:#2cd2e8cc;color:#7fefff;background:linear-gradient(180deg,transparent,#08313a99);box-shadow:0 -14px 28px #2cd2e811}.hand-surface-guide.engaged{border-color:#42e49b;color:#8ef0bd;background:linear-gradient(180deg,transparent,#0a352799);box-shadow:0 -22px 38px #42e49b16}.hand-surface-guide.contact{border-top-style:solid;border-color:#f4d27a;color:#ffe8a8;background:linear-gradient(180deg,transparent,#4b371699);box-shadow:0 -32px 48px #e5b83b24}.hand-surface-guide.contact b{color:#ffe8a8}
    .hand-banner,.hand-metrics,.hand-surface-guide{display:none!important}
    .hand-cards{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:8px}.hand-card{padding:8px;border:1px solid #24404d;border-radius:9px;background:#091920}.hand-card.inactive{opacity:.62}.hand-card.quality-hold{border-color:#82513d;background:#1b1412}.hand-card header{height:auto;padding:0;border:0;background:none}.hand-card b{font-size:11px}.hand-card .track{margin-left:auto;color:#ff8a90;font:800 8px ui-monospace}.hand-card.tracked .track{color:#42e49b}.hand-card dl{display:grid;grid-template-columns:auto 1fr;margin:7px 0 0;gap:3px 7px;font:9px ui-monospace,SFMono-Regular}.hand-card dt{color:#71929d}.hand-card dd{margin:0;text-align:right}.hand-card button{width:100%;margin-top:7px;min-height:30px}
    .hand-advanced.hidden{display:none}
    .hand-privacy{margin:10px 2px 0;color:#6f909b;font-size:10px}
    .hand-resize-handle{position:sticky;z-index:4;right:0;bottom:0;float:right;width:28px!important;min-width:28px!important;height:28px!important;min-height:28px!important;margin:-22px -8px -8px 0!important;padding:0!important;border:0!important;border-radius:8px 0 8px 0!important;background:linear-gradient(135deg,transparent 45%,#315766 46% 53%,transparent 54% 62%,#8bc6cd 63% 70%,transparent 71%)!important;box-shadow:none!important;cursor:nwse-resize!important;touch-action:none}
    .hand-resize-handle:focus-visible{outline:2px solid #8bc6cd;outline-offset:-2px}.hand-resize-handle:hover{background:linear-gradient(135deg,transparent 42%,#52747d 43% 51%,transparent 52% 60%,#b9f6ff 61% 70%,transparent 71%)!important}
    @container(max-width:350px){.hand-cards{grid-template-columns:1fr}.hand-head p{font-size:10px}.hand-metrics{top:auto;bottom:7px;left:7px;right:7px;max-width:none;justify-content:flex-start}}
    @media(max-width:640px){.hand-control-dock{left:8px;top:8px}.hand-control-launch{min-height:32px!important;padding:0 8px!important}.hand-panel{right:6px;bottom:6px;width:calc(100vw - 12px)}}
  `;
  document.head.append(style);

  const launchDock = document.createElement("div");
  launchDock.className = "hand-control-dock";
  const launch = document.createElement("button");
  launch.id = "handControlLaunch";
  launch.className = "hand-control-launch";
  launch.dataset.shortcut = "CAM";
  launch.setAttribute("aria-controls", "handControlPanel");
  launch.setAttribute("aria-expanded", "false");
  launch.setAttribute("aria-label", "Open webcam hand control");
  launch.innerHTML = '<span class="hand-camera-dot" aria-hidden="true"></span><span>Webcam view</span><kbd>CAM</kbd>';
  launchDock.append(launch);
  const operativeView = document.querySelector("#cameraView, .view");
  if (operativeView) operativeView.append(launchDock);
  else document.body.append(launchDock);

  const panel = document.createElement("section");
  panel.id = "handControlPanel";
  panel.className = "hand-panel hidden";
  panel.setAttribute("aria-label", "Webcam hand control");
  panel.innerHTML = `
    <div class="hand-head"><div><h2>Webcam control</h2></div><div class="spacer"></div><div class="hand-head-actions"><button id="handClose" data-shortcut="CAM">Close</button></div></div>
    <div class="hand-video-wrap"><video id="handVideo" playsinline muted></video><canvas id="handOverlay"></canvas><div id="handBanner" class="hand-banner">Camera off</div><div class="hand-metrics"><span id="handRate">0 Hz</span><span id="handInference">— ms vision</span><span id="handLatency">— ms loop</span></div><div id="handSurfaceGuide" class="hand-surface-guide"><span>POINT INDEX ↓</span><b>TABLE REACH</b></div></div>
    <div id="handAdvanced" class="hand-advanced hidden">
      <div class="hand-cards">
        <article id="handCard0" class="hand-card"><header><b>Left hand · Instrument 1</b><span class="track">NOT TRACKED</span></header><dl><dt>Safety</dt><dd data-field="safety">Frozen</dd><dt>Clutch</dt><dd data-field="clutch">Point index ↓</dd><dt>Table reach</dt><dd data-field="table">0%</dd><dt>XYZ mm</dt><dd data-field="xyz">0 · 0 · 0</dd><dt>RPY °</dt><dd data-field="rpy">0 · 0 · 0</dd><dt>Gripper</dt><dd data-field="gripper">—</dd><dt>Signal quality</dt><dd data-field="confidence">—</dd></dl><button data-hand-arm="0" data-shortcut="L CAM">Engage left</button></article>
        <article id="handCard1" class="hand-card"><header><b>Right hand · Instrument 2</b><span class="track">NOT TRACKED</span></header><dl><dt>Safety</dt><dd data-field="safety">Frozen</dd><dt>Clutch</dt><dd data-field="clutch">Point index ↓</dd><dt>Table reach</dt><dd data-field="table">0%</dd><dt>XYZ mm</dt><dd data-field="xyz">0 · 0 · 0</dd><dt>RPY °</dt><dd data-field="rpy">0 · 0 · 0</dd><dt>Gripper</dt><dd data-field="gripper">—</dd><dt>Signal quality</dt><dd data-field="confidence">—</dd></dl><button data-hand-arm="1" data-shortcut="R CAM">Engage right</button></article>
      </div>
      <p class="hand-privacy">Only calibrated numeric pose commands leave this browser. Webcam frames and raw landmarks are never uploaded or recorded. Single-camera depth is relative, not metric or clinical-grade.</p>
    </div>
  `;
  document.body.append(panel);
  return { launch, panel };
}

class HandController {
  constructor() {
    const ui = createInterface();
    this.launch = ui.launch;
    this.panel = ui.panel;
    this.video = this.panel.querySelector("#handVideo");
    this.canvas = this.panel.querySelector("#handOverlay");
    this.context = this.canvas.getContext("2d");
    this.banner = this.panel.querySelector("#handBanner");
    this.stream = null;
    this.landmarker = null;
    this.visionWorker = null;
    this.visionFramePending = false;
    this.running = false;
    this.starting = false;
    this.startGeneration = 0;
    this.serverEnabled = false;
    this.reacquire = [true, true];
    this.serverSafety = ["disabled", "disabled"];
    this.inFlight = false;
    this.suspendTransmission = false;
    this.queuedPayload = null;
    this.transportGeneration = 0;
    this.activeRequestController = null;
    this.safetyStopPromise = null;
    this.sequenceBase = Date.now() * 1000;
    this.sequenceCounter = 0;
    this.lastInferenceAt = 0;
    this.lastMediaPipeTimestampMs = -1;
    this.visionRecoveryAttempted = false;
    this.lastStatusAt = 0;
    this.lastFrameAt = null;
    this.frameRate = 0;
    this.inferenceMs = 0;
    this.roundTripMs = 0;
    this.serverTransportAgeMs = null;
    this.serverApplyAgeMs = null;
    this.transportDrops = 0;
    this.videoFrameRequest = null;
    this.poses = new Map();
    this.poseDiagnostics = new Map();
    this.identityAmbiguousArms = new Set();
    this.anchors = new Map();
    this.engaged = [false, false];
    this.controlMode = "single";
    this.primaryArm = null;
    this.singleHandTried = false;
    this.singleHandTriedAt = null;
    this.secondHandVisibleSince = null;
    this.automaticSecondHandSuppressed = false;
    this.automaticSecondHandStarting = false;
    this.secondHandAdmissionArmed = false;
    this.secondHandPointSince = null;
    this.provisionalCalibration = [null, null];
    this.autoCalibrationSamples = [[], []];
    this.lastCalibrationPersistAt = [0, 0];
    this.autoEngagePending = false;
    this.autoEngageTimer = null;
    this.clutchReadySince = [null, null];
    this.clutchEngagePending = [false, false];
    this.cameraControlFrameRevision = null;
    this.precision = true;
    this.poseFilters = [
      new OneEuroVectorFilter(6),
      new OneEuroVectorFilter(6),
    ];
    this.apertureFilters = [
      new OneEuroFilter({ minCutoff: 2.4, beta: 0.18 }),
      new OneEuroFilter({ minCutoff: 2.4, beta: 0.18 }),
    ];
    this.rawOffsets = [null, null];
    this.motionDiagnostics = [
      { tremor: 0, speed: 0, quality: 0, velocity: [0, 0, 0, 0, 0, 0] },
      { tremor: 0, speed: 0, quality: 0, velocity: [0, 0, 0, 0, 0, 0] },
    ];
    this.tableReach = [0, 0];
    this.currentCommands = [this.emptyCommand(0), this.emptyCommand(1)];
    this.lastAperture = [1, 1];
    this.calibration = null;
    this.calibrationActive = false;
    this.calibrationAutomatic = false;
    this.calibrationStep = 0;
    this.calibrationDraft = { hands: {} };
    this.calibrationTargetArms = [];
    this.calibrationCandidateSignature = "";
    this.calibrationCapture = null;
    this.calibrationReadySince = null;
    this.calibrationStageStartedAt = 0;
    this.panelGeometryKey = "drAnmar.handPanelGeometry.v3";
    this.panelGeometryReady = false;
    this.panelDrag = null;
    this.panelResize = null;
    this.panelResizeObserver = null;
    this.handlePanelResizeMove = event => this.movePanelResize(event);
    this.handlePanelResizeEnd = event => this.endPanelResize(event);
    this.controlsExpanded = false;
    this.operatorId = this.resolveOperatorId();
    this.bind();
  }

  resolveOperatorId() {
    const query = new URLSearchParams(location.search).get("operator");
    if (query) return query;
    // Standalone localhost tabs must share the same operator lease. A
    // per-tab session identity lets an older camera tab indefinitely lock a
    // newly opened one out of motion even though both belong to this browser.
    let value = localStorage.getItem("drAnmarHandOperatorId")
      || sessionStorage.getItem("drAnmarOperatorId");
    if (!value) {
      value = `browser-${crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36)}`;
    }
    localStorage.setItem("drAnmarHandOperatorId", value);
    sessionStorage.setItem("drAnmarOperatorId", value);
    return value;
  }

  bind() {
    this.launch.addEventListener("pointerdown", event => event.stopPropagation());
    this.launch.addEventListener("click", () => this.open());
    this.panel.querySelector("#handClose").addEventListener("click", () => this.close());
    const dragHandle = this.panel.querySelector(".hand-head");
    dragHandle.addEventListener("pointerdown", event => this.startPanelDrag(event));
    dragHandle.addEventListener("pointermove", event => this.movePanel(event));
    dragHandle.addEventListener("pointerup", event => this.endPanelDrag(event));
    dragHandle.addEventListener("pointercancel", event => this.endPanelDrag(event));
    this.handleWindowResize = () => this.constrainPanelToViewport(true);
    window.addEventListener("resize", this.handleWindowResize);
    if (typeof ResizeObserver === "function") {
      this.panelResizeObserver = new ResizeObserver(() => {
        if (this.panelGeometryReady && !this.panelDrag && !this.panelResize) {
          this.constrainPanelToViewport(true);
        }
      });
      this.panelResizeObserver.observe(this.panel);
    }
    this.panel.querySelectorAll("[data-hand-arm]").forEach(button => {
      button.addEventListener("click", () => {
        this.cancelAutomaticEngage();
        this.toggleArm(Number(button.dataset.handArm));
      });
    });
    window.addEventListener("pagehide", () => this.dispose(), { once: true });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) this.hardFreeze("page_hidden");
    });
    this.renderModeControl();
  }

  preparePanelGeometry() {
    if (this.panelGeometryReady) {
      this.constrainPanelToViewport();
      return;
    }
    const rect = this.panel.getBoundingClientRect();
    let stored = null;
    try {
      stored = JSON.parse(localStorage.getItem(this.panelGeometryKey) || "null");
    } catch (_error) {}
    const viewportWidth = Math.max(1, window.innerWidth);
    const viewportHeight = Math.max(1, window.innerHeight);
    const maximumWidth = Math.max(1, viewportWidth - 12);
    const maximumHeight = Math.max(1, viewportHeight - 12);
    const minimumWidth = Math.min(300, maximumWidth);
    const minimumHeight = Math.min(260, maximumHeight);
    const width = clamp(Number(stored?.width) || rect.width, minimumWidth, maximumWidth);
    const height = clamp(Number(stored?.height) || rect.height, minimumHeight, maximumHeight);
    const left = clamp(
      Number.isFinite(Number(stored?.left)) ? Number(stored.left) : rect.left,
      6,
      Math.max(6, viewportWidth - width - 6),
    );
    const top = clamp(
      Number.isFinite(Number(stored?.top)) ? Number(stored.top) : rect.top,
      6,
      Math.max(6, viewportHeight - height - 6),
    );
    Object.assign(this.panel.style, {
      left: `${left}px`,
      top: `${top}px`,
      right: "auto",
      bottom: "auto",
      width: `${width}px`,
      height: `${height}px`,
    });
    this.panelGeometryReady = true;
    this.savePanelGeometry();
  }

  startPanelDrag(event) {
    if (event.button !== 0 || event.target.closest("button, input, select, textarea, a")) return;
    this.preparePanelGeometry();
    const rect = this.panel.getBoundingClientRect();
    this.panelDrag = {
      pointerId: event.pointerId,
      pointerX: event.clientX,
      pointerY: event.clientY,
      left: rect.left,
      top: rect.top,
      width: rect.width,
      height: rect.height,
    };
    event.currentTarget.setPointerCapture?.(event.pointerId);
    event.currentTarget.classList.add("dragging");
    event.preventDefault();
  }

  movePanel(event) {
    const drag = this.panelDrag;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const left = clamp(
      drag.left + event.clientX - drag.pointerX,
      6,
      Math.max(6, window.innerWidth - drag.width - 6),
    );
    const top = clamp(
      drag.top + event.clientY - drag.pointerY,
      6,
      Math.max(6, window.innerHeight - drag.height - 6),
    );
    this.panel.style.left = `${left}px`;
    this.panel.style.top = `${top}px`;
    event.preventDefault();
  }

  endPanelDrag(event) {
    if (!this.panelDrag || this.panelDrag.pointerId !== event.pointerId) return;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
    event.currentTarget.classList.remove("dragging");
    this.panelDrag = null;
    this.savePanelGeometry();
  }

  startPanelResize(event) {
    if (event.button !== 0) return;
    this.preparePanelGeometry();
    const rect = this.panel.getBoundingClientRect();
    this.panelResize = {
      pointerId: event.pointerId,
      pointerX: event.clientX,
      pointerY: event.clientY,
      width: rect.width,
      height: rect.height,
      handle: event.currentTarget,
    };
    try {
      event.currentTarget.setPointerCapture?.(event.pointerId);
    } catch (_error) {}
    window.addEventListener("pointermove", this.handlePanelResizeMove);
    window.addEventListener("pointerup", this.handlePanelResizeEnd);
    window.addEventListener("pointercancel", this.handlePanelResizeEnd);
    event.preventDefault();
    event.stopPropagation();
  }

  movePanelResize(event) {
    const resize = this.panelResize;
    if (!resize || resize.pointerId !== event.pointerId) return;
    this.setPanelSize(
      resize.width + event.clientX - resize.pointerX,
      resize.height + event.clientY - resize.pointerY,
    );
    event.preventDefault();
  }

  endPanelResize(event) {
    if (!this.panelResize || this.panelResize.pointerId !== event.pointerId) return;
    try {
      this.panelResize.handle?.releasePointerCapture?.(event.pointerId);
    } catch (_error) {}
    this.panelResize = null;
    window.removeEventListener("pointermove", this.handlePanelResizeMove);
    window.removeEventListener("pointerup", this.handlePanelResizeEnd);
    window.removeEventListener("pointercancel", this.handlePanelResizeEnd);
    this.constrainPanelToViewport(true);
  }

  resizePanelFromKeyboard(event) {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
    this.preparePanelGeometry();
    const rect = this.panel.getBoundingClientRect();
    const step = event.shiftKey ? 64 : 24;
    const width = rect.width + (event.key === "ArrowRight" ? step : event.key === "ArrowLeft" ? -step : 0);
    const height = rect.height + (event.key === "ArrowDown" ? step : event.key === "ArrowUp" ? -step : 0);
    this.setPanelSize(width, height);
    this.constrainPanelToViewport(true);
    event.preventDefault();
  }

  setPanelSize(width, height) {
    const maximumWidth = Math.max(1, window.innerWidth - 12);
    const maximumHeight = Math.max(1, window.innerHeight - 12);
    const minimumWidth = Math.min(300, maximumWidth);
    const minimumHeight = Math.min(260, maximumHeight);
    this.panel.style.width = `${clamp(width, minimumWidth, maximumWidth)}px`;
    this.panel.style.height = `${clamp(height, minimumHeight, maximumHeight)}px`;
  }

  constrainPanelToViewport(save = false) {
    if (!this.panelGeometryReady || this.panel.classList.contains("hidden")) return;
    const rect = this.panel.getBoundingClientRect();
    const maximumWidth = Math.max(1, window.innerWidth - 12);
    const maximumHeight = Math.max(1, window.innerHeight - 12);
    const width = Math.min(rect.width, maximumWidth);
    const height = Math.min(rect.height, maximumHeight);
    const left = clamp(rect.left, 6, Math.max(6, window.innerWidth - width - 6));
    const top = clamp(rect.top, 6, Math.max(6, window.innerHeight - height - 6));
    if (width !== rect.width) this.panel.style.width = `${width}px`;
    if (height !== rect.height) this.panel.style.height = `${height}px`;
    this.panel.style.left = `${left}px`;
    this.panel.style.top = `${top}px`;
    if (save) this.savePanelGeometry();
  }

  savePanelGeometry() {
    if (!this.panelGeometryReady || this.panel.classList.contains("hidden")) return;
    const rect = this.panel.getBoundingClientRect();
    try {
      localStorage.setItem(this.panelGeometryKey, JSON.stringify({
        left: Math.round(rect.left),
        top: Math.round(rect.top),
        width: Math.round(rect.width),
        height: Math.round(rect.height),
      }));
    } catch (_error) {}
  }

  resetPanelGeometry() {
    this.toggleAdvancedControls(false);
    try {
      localStorage.removeItem(this.panelGeometryKey);
    } catch (_error) {}
    this.panelGeometryReady = false;
    Object.assign(this.panel.style, {
      left: "",
      top: "",
      right: "",
      bottom: "",
      width: "",
      height: "",
    });
    this.preparePanelGeometry();
    this.setBanner("Webcam window fitted to the operative view", "good");
  }

  toggleAdvancedControls(force = null) {
    this.controlsExpanded = force === null ? !this.controlsExpanded : Boolean(force);
    const advanced = this.panel.querySelector("#handAdvanced");
    const button = this.panel.querySelector("#handControlsToggle");
    advanced.classList.toggle("hidden", !this.controlsExpanded);
    button.textContent = this.controlsExpanded ? "Hide controls" : "Controls";
    button.setAttribute("aria-expanded", String(this.controlsExpanded));
    button.classList.toggle("engaged", this.controlsExpanded);
    if (this.panelGeometryReady) {
      this.constrainPanelToViewport(true);
    }
  }

  isArmEnabled(arm) {
    return this.controlMode === "dual" || this.primaryArm === arm;
  }

  bestTrackedArm() {
    if (this.primaryArm !== null && this.poses.has(this.primaryArm)) return this.primaryArm;
    const candidates = [0, 1].filter(arm => this.poses.has(arm));
    candidates.sort(
      (left, right) => (this.poses.get(right)?.quality || 0) - (this.poses.get(left)?.quality || 0),
    );
    return candidates[0] ?? null;
  }

  selectPrimaryArm(arm) {
    if (arm !== 0 && arm !== 1) return;
    this.primaryArm = arm;
    for (const otherArm of [0, 1]) {
      if (otherArm === arm) continue;
      this.engaged[otherArm] = false;
      this.anchors.delete(otherArm);
      this.resetMotionFilter(otherArm);
      this.clutchReadySince[otherArm] = null;
    }
    this.renderModeControl();
  }

  renderModeControl() {
    this.panel.dataset.controlMode = this.controlMode;
  }

  toggleControlMode({ automatic = false } = {}) {
    this.cancelAutomaticEngage();
    this.automaticSecondHandStarting = false;
    if (this.controlMode === "dual") {
      this.controlMode = "single";
      this.automaticSecondHandSuppressed = !automatic;
      this.secondHandVisibleSince = null;
      this.secondHandAdmissionArmed = false;
      this.secondHandPointSince = null;
      this.singleHandTriedAt = performance.now() / 1000;
      const selectedArm = this.bestTrackedArm() ?? this.primaryArm ?? 0;
      this.selectPrimaryArm(selectedArm);
      this.freezeAll(false);
      this.autoEngagePending = false;
      this.setBanner("One-hand mode · point the index finger down to move", "good");
      this.renderCards();
      return;
    }
    if (!this.singleHandTried || this.primaryArm === null) {
      this.setBanner("Try one-hand control first", "warn");
      return;
    }
    this.automaticSecondHandSuppressed = false;
    this.controlMode = "dual";
    this.freezeAll(false);
    this.renderModeControl();
    this.autoEngagePending = false;
    this.setBanner("Two-hand mode ready", "good");
    this.renderCards();
  }

  maybeAddSecondHandAutomatically(timestampSeconds) {
    if (
      this.controlMode !== "single"
      || !this.singleHandTried
      || this.primaryArm === null
      || this.calibrationActive
      || this.automaticSecondHandSuppressed
      || this.automaticSecondHandStarting
    ) {
      this.secondHandVisibleSince = null;
      this.secondHandAdmissionArmed = false;
      this.secondHandPointSince = null;
      return;
    }
    const secondArm = this.primaryArm === 0 ? 1 : 0;
    const secondPose = this.poses.get(secondArm);
    const triedAt = this.singleHandTriedAt ?? timestampSeconds;
    const eligibleAt = triedAt + 3.0;
    if (!secondPose) {
      this.secondHandVisibleSince = null;
      this.secondHandAdmissionArmed = false;
      this.secondHandPointSince = null;
      if (timestampSeconds >= eligibleAt) {
        this.setBanner(
          `One-hand active · show an open ${secondArm === 0 ? "left" : "right"} hand to add it`,
          "good",
        );
      }
      return;
    }
    if (timestampSeconds < eligibleAt) {
      this.secondHandVisibleSince = null;
      this.setBanner(
        `One-hand active · try it for ${(eligibleAt - timestampSeconds).toFixed(1)} s`,
        "good",
      );
      return;
    }
    if (!this.secondHandAdmissionArmed) {
      if (secondPose.clutchScore > CLUTCH_RELEASE_SCORE) {
        this.secondHandVisibleSince = null;
        this.setBanner(
          `Open the ${secondArm === 0 ? "left" : "right"} hand once to arm Instrument ${secondArm + 1}`,
          "warn",
        );
        return;
      }
      if (this.secondHandVisibleSince === null) this.secondHandVisibleSince = timestampSeconds;
      if (timestampSeconds - this.secondHandVisibleSince >= 0.45) {
        this.secondHandAdmissionArmed = true;
        this.secondHandPointSince = null;
      }
    }
    if (!this.secondHandAdmissionArmed) {
      this.setBanner(
        `Hold the open ${secondArm === 0 ? "left" : "right"} hand steady`,
        "warn",
      );
      return;
    }
    if (secondPose.clutchScore < CLUTCH_ENGAGE_SCORE) {
      this.secondHandPointSince = null;
      this.setBanner(
        `Instrument ${secondArm + 1} ready · point its index finger down to add it`,
        "good",
      );
      return;
    }
    if (this.secondHandPointSince === null) this.secondHandPointSince = timestampSeconds;
    const pointHoldSeconds = timestampSeconds - this.secondHandPointSince;
    if (pointHoldSeconds >= CLUTCH_ENGAGE_HOLD_S) {
      this.automaticSecondHandStarting = true;
      this.secondHandVisibleSince = null;
      this.secondHandAdmissionArmed = false;
      this.secondHandPointSince = null;
      this.toggleControlMode({ automatic: true });
    }
  }

  emptyCommand(arm) {
    return {
      arm,
      tracked: false,
      motion_engaged: false,
      translation_offset_m: [0, 0, 0],
      rotation_vector_rad: [0, 0, 0],
      aperture_normalized: this?.lastAperture?.[arm] ?? 1,
      confidence: 0,
    };
  }

  async request(path, body, { signal = null, keepalive = false } = {}) {
    const response = await fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json", "x-dr-anmar-operator": this.operatorId },
      body: JSON.stringify(body),
      signal,
      keepalive,
    });
    let data = {};
    try { data = await response.json(); } catch (_error) {}
    if (!response.ok) throw new Error(data.detail || "Hand-control request failed");
    return data;
  }

  setBanner(message, mode = "") {
    this.banner.textContent = message;
    this.banner.className = `hand-banner ${mode}`.trim();
  }

  async open() {
    this.panel.classList.remove("hidden");
    this.launch.setAttribute("aria-expanded", "true");
    this.launch.setAttribute("aria-label", "Webcam hand control is open");
    this.preparePanelGeometry();
    if (!this.running) await this.start();
  }

  async initializeVisionWorker() {
    if (typeof Worker !== "function" || typeof createImageBitmap !== "function") return false;
    const worker = new Worker("./hand-control-worker.mjs", { type: "module" });
    try {
      await new Promise((resolve, reject) => {
        const timeout = window.setTimeout(
          () => reject(new Error("Vision worker startup timed out")),
          12_000,
        );
        worker.onmessage = event => {
          if (event.data?.type === "ready") {
            window.clearTimeout(timeout);
            resolve();
          } else if (event.data?.type === "error") {
            window.clearTimeout(timeout);
            reject(new Error(event.data.message || "Vision worker failed"));
          }
        };
        worker.onerror = event => {
          window.clearTimeout(timeout);
          reject(new Error(event.message || "Vision worker failed"));
        };
        worker.postMessage({
          type: "init",
          assetBaseUrl: new URL("./hand-control-assets/", location.href).href,
        });
      });
    } catch (error) {
      worker.terminate();
      throw error;
    }
    worker.onmessage = event => this.handleWorkerVisionMessage(event.data);
    worker.onerror = event => this.handleVisionError(
      new Error(event.message || "Vision worker stopped"),
    );
    this.visionWorker = worker;
    return true;
  }

  handleWorkerVisionMessage(message) {
    if (!message || message.type !== "result") {
      if (message?.type === "error") {
        this.visionFramePending = false;
        this.handleVisionError(new Error(message.message || "Vision worker failed"));
      }
      return;
    }
    this.visionFramePending = false;
    if (!this.running) return;
    const time = Number(message.frameTimeMs);
    this.lastCaptureEpochMs = performance.timeOrigin + time;
    this.visionRecoveryAttempted = false;
    this.inferenceMs = this.ewma(this.inferenceMs, Number(message.inferenceMs) || 0, 0.18);
    if (this.lastFrameAt !== null && time > this.lastFrameAt) {
      this.frameRate = this.ewma(this.frameRate, 1000 / (time - this.lastFrameAt), 0.15);
    }
    this.lastFrameAt = time;
    try {
      this.processResult(message.result, time / 1000);
      this.collectCalibrationSamples(time);
      this.draw();
      this.transmit();
      this.renderMetrics();
    } catch (error) {
      this.handleVisionError(error);
    }
  }

  submitWorkerFrame(time, frameTimestampMs) {
    if (!this.visionWorker || this.visionFramePending) return;
    this.visionFramePending = true;
    createImageBitmap(this.video)
      .then(bitmap => {
        if (!this.running || !this.visionWorker) {
          bitmap.close?.();
          this.visionFramePending = false;
          return;
        }
        this.visionWorker.postMessage(
          {
            type: "frame",
            bitmap,
            timestampMs: frameTimestampMs,
            frameTimeMs: time,
          },
          [bitmap],
        );
      })
      .catch(error => {
        this.visionFramePending = false;
        this.handleVisionError(error);
      });
  }

  async start() {
    if (this.running || this.starting) return;
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      if (!isLoopbackHostname(location.hostname)) {
        this.setBanner("Switching to the private local camera connection…", "warn");
        const target = localWebcamTarget();
        window.setTimeout(() => {
          try {
            window.top.location.assign(target);
          } catch (_error) {
            window.location.assign(target);
          }
        }, 120);
      } else {
        this.setBanner("Camera access is blocked in this browser. Allow camera access and reload.", "warn");
      }
      return;
    }
    const generation = ++this.startGeneration;
    this.starting = true;
    this.setBanner("Requesting camera…");
    try {
      const assetStatus = await fetch("./api/hand-control/assets").then(response => response.json());
      if (generation !== this.startGeneration) return;
      if (!assetStatus.ready) throw new Error("Pinned MediaPipe assets are not installed");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280, min: 640 },
          height: { ideal: 720, min: 360 },
          frameRate: { ideal: 60, min: 30 },
        },
        audio: false,
      });
      if (generation !== this.startGeneration) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }
      this.stream = stream;
      this.video.srcObject = this.stream;
      await this.video.play();
      if (generation !== this.startGeneration) return;
      const settings = this.stream.getVideoTracks()[0]?.getSettings?.() || {};
      // v5 continuously learns a robust physical pinch span without a setup
      // wizard and stores the orientation-compensated neutral palm scale.
      this.calibrationKey = `drAnmar.handCalibration.v5:${settings.deviceId || "default"}`;
      const storedCalibration = localStorage.getItem(this.calibrationKey);
      try {
        this.calibration = numericCalibration(JSON.parse(storedCalibration || "null"));
      } catch (_error) {
        localStorage.removeItem(this.calibrationKey);
        this.calibration = null;
      }
      this.provisionalCalibration = [0, 1].map(arm => {
        const saved = this.calibration?.hands?.[arm];
        return saved ? { ...saved } : null;
      });
      this.autoCalibrationSamples = [[], []];
      try {
        const workerReady = await this.initializeVisionWorker();
        if (!workerReady) throw new Error("Vision worker is unavailable");
      } catch (_workerError) {
        const visionModule = await import("./hand-control-assets/vision_bundle.mjs");
        if (generation !== this.startGeneration) return;
        const files = await visionModule.FilesetResolver.forVisionTasks(
          new URL("./hand-control-assets/wasm", location.href).href,
        );
        if (generation !== this.startGeneration) return;
        const options = {
          baseOptions: {
            modelAssetPath: new URL("./hand-control-assets/hand_landmarker.task", location.href).href,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.58,
          minHandPresenceConfidence: 0.58,
          minTrackingConfidence: 0.62,
        };
        try {
          this.landmarker = await visionModule.HandLandmarker.createFromOptions(files, options);
        } catch (_gpuError) {
          options.baseOptions.delegate = "CPU";
          this.landmarker = await visionModule.HandLandmarker.createFromOptions(files, options);
        }
      }
      if (generation !== this.startGeneration) {
        this.visionWorker?.terminate();
        this.visionWorker = null;
        this.landmarker?.close?.();
        this.landmarker = null;
        this.stopMedia();
        return;
      }
      this.running = true;
      this.lastInferenceAt = 0;
      this.lastMediaPipeTimestampMs = -1;
      this.lastFrameAt = null;
      this.launch.classList.add("state-active");
      this.setBanner("Tracking hands · motion frozen", "good");
      await this.setServerEnabled(true, "camera_started");
      // A stable tracked hand should reach control directly. Saved profiles
      // are reused and new operators are calibrated continuously in motion.
      this.calibrationActive = false;
      this.calibrationAutomatic = false;
      this.calibrationCapture = null;
      this.autoEngagePending = true;
      this.scheduleFrame();
    } catch (error) {
      if (generation === this.startGeneration) {
        this.setBanner(error.message, "warn");
        this.stopMedia();
      }
    } finally {
      if (generation === this.startGeneration) this.starting = false;
    }
  }

  scheduleFrame() {
    if (!this.running) return;
    if (typeof this.video.requestVideoFrameCallback === "function") {
      this.usingVideoFrameCallback = true;
      this.videoFrameRequest = this.video.requestVideoFrameCallback(
        (time, metadata) => this.frame(time, metadata),
      );
    } else {
      this.usingVideoFrameCallback = false;
      this.videoFrameRequest = requestAnimationFrame(time => this.frame(time, null));
    }
  }

  async setServerEnabled(enabled, reason = enabled ? "operator_enable" : "operator_freeze") {
    try {
      if (enabled && this.safetyStopPromise) {
        await this.safetyStopPromise;
        this.safetyStopPromise = null;
      }
      const response = await this.request("./api/teleop/hands/control", { enabled, reason });
      this.updateServerSnapshot(response.hand_teleop);
      if (!this.serverEnabled) this.freezeAll(false);
      return this.serverEnabled;
    } catch (error) {
      this.serverEnabled = false;
      this.setBanner(error.message, "warn");
      return false;
    }
  }

  beginCalibration({ automatic = true, targetArms = [] } = {}) {
    this.cancelAutomaticEngage();
    this.freezeAll(false);
    this.calibrationActive = true;
    this.calibrationAutomatic = automatic;
    this.calibrationStep = 0;
    this.calibrationDraft = { hands: {} };
    this.calibrationTargetArms = [...targetArms].filter(arm => arm === 0 || arm === 1).slice(0, 1);
    this.calibrationCandidateSignature = "";
    this.calibrationCapture = null;
    this.calibrationReadySince = null;
    this.renderCalibration();
  }

  renderCalibration() {
    this.calibrationStageStartedAt = performance.now();
    this.setBanner(`${this.calibrationStagePrompt()} · waiting for stability`, "warn");
  }

  calibrationStagePrompt(arms = this.calibrationTargetArms) {
    const subject = arms[0] === 0
        ? "Left hand"
        : arms[0] === 1
          ? "Right hand"
          : null;
    return [
      subject ? `1/3 · ${subject} selected · hold neutral` : "1/3 · Show one hand at neutral",
      subject ? `2/3 · ${subject}: touch thumb + index` : "2/3 · Touch thumb + index",
      subject ? `3/3 · ${subject}: open thumb + index` : "3/3 · Open thumb + index",
    ][this.calibrationStep] || "Calibration";
  }

  captureCalibration(arms = null) {
    if (!this.poses.size) {
      this.setBanner("Show at least one hand before capturing", "warn");
      return;
    }
    const candidates = (arms || [...this.poses.keys()])
      .filter(arm => this.poses.get(arm)?.quality >= MIN_TRACKING_QUALITY);
    candidates.sort(
      (left, right) => (this.poses.get(right)?.quality || 0) - (this.poses.get(left)?.quality || 0),
    );
    const visibleArms = this.calibrationTargetArms.length
      ? candidates
      : candidates.slice(0, 1);
    if (!visibleArms.length) {
      this.setBanner("Hold one clearly tracked hand in view", "warn");
      return;
    }
    if (!this.calibrationTargetArms.length) {
      this.calibrationTargetArms = [...visibleArms];
    }
    const captureArms = this.calibrationTargetArms.filter(arm => visibleArms.includes(arm));
    if (captureArms.length !== this.calibrationTargetArms.length) {
      this.setBanner(`${this.calibrationStagePrompt()} · keep the detected hand in view`, "warn");
      return;
    }
    this.calibrationCapture = {
      step: this.calibrationStep,
      startedAt: performance.now(),
      arms: captureArms,
      samples: { 0: [], 1: [] },
    };
    this.setBanner(`${this.calibrationStagePrompt()} · sampling 0%`, "warn");
  }

  collectCalibrationSamples(timestampMs) {
    const capture = this.calibrationCapture;
    if (!capture) {
      this.maybeStartAutomaticCalibrationCapture(timestampMs);
      return;
    }
    for (const arm of capture.arms) {
      const pose = this.poses.get(arm);
      if (!pose) continue;
      if (pose.quality < MIN_TRACKING_QUALITY) continue;
      const value = capture.step === 0 ? pose.depthScale : pose.pinchRatio;
      if (capture.samples[arm].length < CALIBRATION_SAMPLE_COUNT) capture.samples[arm].push(value);
    }
    const visibleCounts = capture.arms.map(arm => capture.samples[arm].length);
    const progressCount = visibleCounts.length ? Math.min(...visibleCounts) : 0;
    this.setBanner(
      `${this.calibrationStagePrompt()} · sampling ${Math.round(100 * progressCount / CALIBRATION_SAMPLE_COUNT)}%`,
      "warn",
    );
    const complete = visibleCounts.length && visibleCounts.every(count => count >= CALIBRATION_SAMPLE_COUNT);
    if (complete || timestampMs - capture.startedAt > 1600) this.finishCalibrationCapture();
  }

  maybeStartAutomaticCalibrationCapture(timestampMs) {
    if (!this.calibrationActive || !this.calibrationAutomatic || this.calibrationCapture) return;
    if (timestampMs - this.calibrationStageStartedAt < 900) return;
    const visibleArms = [0, 1].filter(arm => this.poses.get(arm)?.quality >= MIN_TRACKING_QUALITY);
    visibleArms.sort(
      (left, right) => (this.poses.get(right)?.quality || 0) - (this.poses.get(left)?.quality || 0),
    );
    const targetArms = this.calibrationTargetArms.length
      ? this.calibrationTargetArms
      : visibleArms.slice(0, 1);
    const signature = targetArms.join(",");
    if (signature !== this.calibrationCandidateSignature) {
      this.calibrationCandidateSignature = signature;
      this.calibrationReadySince = null;
    }
    if (!targetArms.length) {
      this.setBanner(this.calibrationStagePrompt(), "warn");
      return;
    }
    const poses = targetArms.map(arm => this.poses.get(arm));
    let ready = poses.every(pose => pose && pose.quality >= MIN_TRACKING_QUALITY);
    if (ready && this.calibrationStep === 1) {
      ready = poses.every(pose => pose.pinchRatio <= 0.24);
    }
    if (ready && this.calibrationStep === 2) {
      ready = targetArms.every((arm, index) => {
        const closed = this.calibrationDraft.hands[arm]?.closeRatio;
        const pose = poses[index];
        return Number.isFinite(closed) && pose.pinchRatio >= closed + 0.18;
      });
    }
    if (!ready) {
      this.calibrationReadySince = null;
      if (this.calibrationTargetArms.length) {
        this.setBanner(`${this.calibrationStagePrompt()} · hold the pose in view`, "warn");
      }
      return;
    }
    if (this.calibrationReadySince === null) this.calibrationReadySince = timestampMs;
    const stableForMs = timestampMs - this.calibrationReadySince;
    const remainingMs = Math.max(0, 650 - stableForMs);
    const stagePrompt = this.calibrationStagePrompt(targetArms);
    this.setBanner(
      remainingMs
        ? `${stagePrompt} · hold ${(remainingMs / 1000).toFixed(1)} s`
        : `${stagePrompt} · capturing automatically`,
      remainingMs ? "warn" : "good",
    );
    if (stableForMs >= 650) {
      this.calibrationReadySince = null;
      if (!this.calibrationTargetArms.length) {
        this.calibrationTargetArms = [...targetArms];
      }
      this.captureCalibration(targetArms);
    }
  }

  finishCalibrationCapture() {
    const capture = this.calibrationCapture;
    this.calibrationCapture = null;
    let accepted = 0;
    try {
      for (const arm of capture.arms) {
        if (capture.samples[arm].length < 12) continue;
        const sample = robustCalibrationSample(capture.samples[arm]);
        const draft = this.calibrationDraft.hands[arm] || {};
        if (capture.step === 0) draft.neutralScale = sample.value;
        if (capture.step === 1) draft.closeRatio = sample.value;
        if (capture.step === 2) draft.openRatio = sample.value;
        draft.stability = Math.min(draft.stability ?? 1, sample.stability);
        this.calibrationDraft.hands[arm] = draft;
        accepted += 1;
      }
      if (this.calibrationAutomatic && accepted !== capture.arms.length) {
        throw new Error("Keep the detected hand visible through the complete sample");
      }
      if (!accepted) throw new Error("No stable hand sample was captured");
    } catch (error) {
      this.setBanner(`${error.message} · capture again`, "warn");
      this.renderCalibration();
      return;
    }
    if (this.calibrationStep < 2) {
      this.calibrationStep += 1;
      this.calibrationReadySince = null;
      this.renderCalibration();
      return;
    }
    const calibration = numericCalibration({
      version: 2,
      hands: {
        ...(this.calibration?.hands || {}),
        ...this.calibrationDraft.hands,
      },
    });
    if (!calibration) {
      this.setBanner("Calibration span is too small · try again", "warn");
      this.beginCalibration({ targetArms: this.calibrationTargetArms });
      return;
    }
    this.calibration = calibration;
    this.calibrationActive = false;
    this.calibrationAutomatic = false;
    for (const arm of Object.keys(calibration.hands).map(Number)) {
      this.provisionalCalibration[arm] = null;
    }
    if (this.controlMode === "single" && this.calibrationTargetArms.length) {
      this.selectPrimaryArm(this.calibrationTargetArms[0]);
    }
    localStorage.setItem(this.calibrationKey, JSON.stringify(calibration));
    this.autoEngagePending = false;
    this.setBanner("Calibration saved · point the index finger down to move", "good");
  }

  poseFromLandmarks(landmarks, worldLandmarks, identityConfidence) {
    const hasWorldLandmarks = worldLandmarks?.length === 21;
    const pose = palmFrame(hasWorldLandmarks ? worldLandmarks : landmarks);
    const imagePose = palmFrame(landmarks);
    pose.center.x = imagePose.center.x;
    pose.center.y = imagePose.center.y;
    pose.center.z = imagePose.center.z;
    pose.scale = imagePose.scale;
    pose.geometryQuality = Math.min(pose.geometryQuality, imagePose.geometryQuality);
    pose.depthScale = orientationCompensatedPalmScale(
      landmarks,
      hasWorldLandmarks ? worldLandmarks : null,
    );
    pose.pinchRatio = distance3(landmarks[4], landmarks[8]) / Math.max(imagePose.scale, 1e-6);
    pose.clutchScore = downwardPointingClutchScore(
      landmarks,
      hasWorldLandmarks ? worldLandmarks : null,
    );
    pose.indexTipY = landmarks[8].y;
    pose.landmarks = landmarks;
    pose.identityConfidence = clamp(Number(identityConfidence) || 0, 0, 1);
    pose.landmarkQuality = landmarkGeometryQuality(landmarks);
    pose.confidence = pose.landmarkQuality;
    return pose;
  }

  frame(time, metadata = null) {
    if (!this.running) return;
    this.lastCaptureEpochMs = performance.timeOrigin + time;
    const frameTimestampCandidateMs = Number.isFinite(metadata?.mediaTime)
      ? metadata.mediaTime * 1000
      : time;
    if (time - this.lastInferenceAt >= MIN_INFERENCE_INTERVAL_MS && this.video.readyState >= 2) {
      this.lastInferenceAt = time;
      const frameTimestampMs = monotonicMediaPipeTimestamp(
        frameTimestampCandidateMs,
        this.lastMediaPipeTimestampMs,
      );
      this.lastMediaPipeTimestampMs = frameTimestampMs;
      if (this.visionWorker) {
        this.submitWorkerFrame(time, frameTimestampMs);
      } else {
        const inferenceStarted = performance.now();
        try {
          const result = this.landmarker.detectForVideo(this.video, frameTimestampMs);
          this.visionRecoveryAttempted = false;
          this.inferenceMs = this.ewma(this.inferenceMs, performance.now() - inferenceStarted, 0.18);
          if (this.lastFrameAt !== null && time > this.lastFrameAt) {
            this.frameRate = this.ewma(this.frameRate, 1000 / (time - this.lastFrameAt), 0.15);
          }
          this.lastFrameAt = time;
          this.processResult(result, time / 1000);
          this.collectCalibrationSamples(time);
          this.draw();
          this.transmit();
          this.renderMetrics();
        } catch (error) {
          this.handleVisionError(error);
        }
      }
    }
    if (time - this.lastStatusAt > 800) {
      this.lastStatusAt = time;
      this.pollStatus();
    }
    this.scheduleFrame();
  }

  handleVisionError(error) {
    this.hardFreeze("vision_error");
    const message = String(error?.message || error || "Unknown vision error");
    const graphTimingFailure = /Packet timestamp mismatch|CalculatorGraph::Run|WaitUntilIdle/i.test(message);
    if (graphTimingFailure && !this.visionRecoveryAttempted) {
      this.visionRecoveryAttempted = true;
      this.setBanner("Vision timing reset · restarting safely…", "warn");
      this.stopMedia();
      window.setTimeout(() => this.start(), 150);
      return;
    }
    if (graphTimingFailure) {
      this.stopMedia();
      this.setBanner("Vision safety hold · camera stopped; press Start camera to retry.", "warn");
      return;
    }
    this.setBanner(`Vision safety hold · ${message}`, "warn");
  }

  ewma(previous, current, alpha) {
    return previous > 0 ? previous + alpha * (current - previous) : current;
  }

  processResult(result, timestampSeconds) {
    const detections = [];
    (result.landmarks || []).forEach((landmarks, index) => {
      const category = result.handednesses?.[index]?.[0];
      // detectForVideo sees the raw, unmirrored camera frame. MediaPipe's
      // handedness classifier assumes mirrored selfie input, so convert its
      // label here; the CSS mirror affects only what the operator sees.
      const proposedArm = handednessToArm(category?.categoryName, false);
      if (proposedArm === null || proposedArm > 1) return;
      try {
        detections.push({
          proposedArm,
          labelConfidence: Number(category?.score || 0),
          pose: this.poseFromLandmarks(
            landmarks,
            result.worldLandmarks?.[index],
            Number(category?.score || 0),
          ),
        });
      } catch (_error) {}
    });
    const assignment = assignHandDetectionsDetailed(detections, this.poses);
    const assigned = assignment.poses;
    this.identityAmbiguousArms = assignment.ambiguousArms;
    const next = new Map();
    const diagnostics = new Map();
    const dt = this.lastPoseTimestamp
      ? clamp(timestampSeconds - this.lastPoseTimestamp, 1 / 240, 0.2)
      : 1 / 30;
    for (const [arm, pose] of assigned) {
      const previousPose = this.poses.get(arm);
      pose.centerVelocity = previousPose
        ? distance3(pose.center, previousPose.center) / Math.max(dt, 1 / 240)
        : 0;
      const quality = trackingQuality(pose, previousPose, dt);
      pose.quality = quality;
      pose.timestampSeconds = timestampSeconds;
      diagnostics.set(arm, {
        quality,
        state: quality >= MIN_TRACKING_QUALITY ? "ready" : "quality hold",
        pose,
      });
      if (quality >= MIN_TRACKING_QUALITY) next.set(arm, pose);
      if (quality >= MIN_TRACKING_QUALITY) {
        this.observeAutomaticCalibration(arm, pose, timestampSeconds);
      }
    }
    for (let arm = 0; arm < 2; arm += 1) {
      if (!next.has(arm) && this.poses.has(arm)) {
        this.engaged[arm] = false;
        this.anchors.delete(arm);
        this.resetMotionFilter(arm);
      }
    }
    this.lastPoseTimestamp = timestampSeconds;
    this.poses = next;
    this.poseDiagnostics = diagnostics;
    if (this.identityAmbiguousArms.size) {
      this.setBanner("Hand identity ambiguous · motion frozen until hands separate", "warn");
    }
    this.updatePointDownClutch(timestampSeconds);
    this.updateCommands(timestampSeconds);
    this.renderCards();
    this.maybeAddSecondHandAutomatically(timestampSeconds);
  }

  updatePointDownClutch(_timestampSeconds) {
    if (this.calibrationActive) return;
    if (this.controlMode === "single" && this.primaryArm === null) {
      const arm = this.bestTrackedArm();
      if (arm !== null) this.selectPrimaryArm(arm);
    }
    const trackedArms = [0, 1].filter(
      arm => this.poses.has(arm) && this.isArmEnabled(arm),
    );
    if (!trackedArms.length || trackedArms.some(arm => this.engaged[arm])) return;
    this.autoEngagePending = true;
    this.scheduleAutomaticEngage();
  }

  updateCommands(timestampSeconds) {
    for (const arm of [0, 1]) this.currentCommands[arm] = this.buildCommandForArm(arm, timestampSeconds);
  }

  buildCommandForArm(arm, timestampSeconds) {
    const pose = this.poses.get(arm);
    if (!pose || !this.isArmEnabled(arm)) {
      return this.emptyCommand(arm);
    }
    const calibration = this.controlCalibration(arm, pose);
    let aperture = this.lastAperture[arm];
    if (!this.calibrationActive) {
      try {
        const rawAperture = normalizedAperture(pose.pinchRatio, calibration.closeRatio, calibration.openRatio);
        aperture = clamp(this.apertureFilters[arm].filter(rawAperture, timestampSeconds), 0, 1);
        this.lastAperture[arm] = aperture;
      } catch (_error) {}
    }
    const speedButton = document.querySelector(
      `[data-hand-speed-arm="${arm}"].active`,
    );
    const speed = Number(speedButton?.dataset.handSpeed || 1);
    let offset = { translation: [0, 0, 0], rotation: [0, 0, 0] };
    const anchor = this.anchors.get(arm);
    if (this.engaged[arm] && anchor) {
      const calibrationGain = clamp(
        calibration.neutralScale / Math.max(anchor.depthScale || anchor.scale, 1e-6),
        0.7,
        1.3,
      );
      const rawOffset = poseOffset(anchor, pose, 1);
      const surfaceProgress = tableReachProgress(pose.indexTipY, anchor.indexTipY);
      this.tableReach[arm] = surfaceProgress;
      const translationMagnitude = Math.hypot(...rawOffset.translation);
      const rotationMagnitude = Math.hypot(...rawOffset.rotation);
      const translationGain = (
        adaptiveMotionGain(translationMagnitude, this.precision)
        * speed
        * calibrationGain
        * (0.82 + 0.28 * smoothStep01((pose.centerVelocity || 0) / 0.75))
      );
      const rotationGain = (
        adaptiveMotionGain(rotationMagnitude * 0.035, this.precision)
        * speed
      );
      const takeUpProgress = clamp(
        (timestampSeconds - Number(anchor.engagedAt || timestampSeconds)) / 0.22,
        0,
        1,
      );
      const takeUp = takeUpProgress * takeUpProgress * (3 - 2 * takeUpProgress);
      offset = {
        translation: longRangeTranslation(
          rawOffset.translation,
          translationGain,
          surfaceProgress,
          takeUp,
        ),
        rotation: rawOffset.rotation.map(value => value * rotationGain * takeUp),
      };
      const raw = [...offset.translation, ...offset.rotation];
      const previous = this.rawOffsets[arm];
      const sampleDt = previous
        ? clamp(timestampSeconds - previous.timestampSeconds, 1 / 240, 0.2)
        : 1 / Math.max(this.frameRate || 30, 1);
      const velocity = previous
        ? raw.map((value, index) => (value - previous.value[index]) / sampleDt)
        : [0, 0, 0, 0, 0, 0];
      const diagnostic = this.motionDiagnostics[arm];
      const translationSpeed = Math.hypot(...velocity.slice(0, 3));
      const rotationSpeed = Math.hypot(...velocity.slice(3, 6));
      const velocityInnovation = Math.hypot(
        ...velocity.map((value, index) => value - diagnostic.velocity[index]),
      );
      const deliberateMotion = clamp(
        translationSpeed / 0.16 + rotationSpeed / 2.4,
        0,
        1,
      );
      const tremorSample = clamp(
        (velocityInnovation * sampleDt) / 0.018,
        0,
        1,
      ) * (1 - 0.72 * deliberateMotion);
      diagnostic.tremor = diagnostic.tremor
        ? diagnostic.tremor + 0.14 * (tremorSample - diagnostic.tremor)
        : tremorSample;
      diagnostic.speed = translationSpeed + rotationSpeed * 0.035;
      diagnostic.quality = pose.quality;
      diagnostic.velocity = velocity;
      const latencyPredictionS = clamp(
        (this.inferenceMs + 0.5 * this.roundTripMs) / 1000,
        MIN_PREDICTION_HORIZON_S,
        MAX_PREDICTION_HORIZON_S,
      ) * clamp(0.35 + 0.65 * pose.quality, 0.35, 1);
      const predicted = predictPoseVector(
        previous?.value,
        raw,
        sampleDt,
        latencyPredictionS,
      );
      this.rawOffsets[arm] = { value: raw, timestampSeconds };
      const conditioned = conditionPoseVector(predicted, {
        translationDeadband: (
          0.00030
          + (1 - pose.quality) * 0.0010
          + diagnostic.tremor * 0.00075
        ),
        rotationDeadband: (
          0.0045
          + (1 - pose.quality) * 0.012
          + diagnostic.tremor * 0.008
        ),
      });
      const filtered = this.poseFilters[arm].filter(conditioned, timestampSeconds);
      offset = {
        translation: filtered.slice(0, 3).map(
          value => clamp(value, -MAX_TRANSLATION_M, MAX_TRANSLATION_M),
        ),
        rotation: filtered.slice(3, 6).map(
          value => clamp(value, -MAX_ROTATION_RAD, MAX_ROTATION_RAD),
        ),
      };
    } else {
      this.tableReach[arm] = 0;
    }
    return {
      arm,
      tracked: true,
      motion_engaged: Boolean(this.serverEnabled && this.engaged[arm] && anchor),
      translation_offset_m: offset.translation,
      rotation_vector_rad: offset.rotation,
      aperture_normalized: aperture,
      confidence: clamp(pose.quality, 0, 1),
    };
  }

  controlCalibration(arm, pose) {
    const calibrated = this.provisionalCalibration[arm] || this.calibration?.hands?.[arm];
    if (calibrated) return calibrated;
    this.provisionalCalibration[arm] = adaptiveCalibrationProfile([{
      pinchRatio: pose.pinchRatio,
      depthScale: pose.depthScale,
      quality: pose.quality,
    }]);
    return this.provisionalCalibration[arm];
  }

  observeAutomaticCalibration(arm, pose, timestampSeconds) {
    const samples = this.autoCalibrationSamples[arm];
    samples.push({
      pinchRatio: pose.pinchRatio,
      depthScale: pose.depthScale,
      quality: pose.quality,
    });
    if (samples.length > AUTO_CALIBRATION_WINDOW) {
      samples.splice(0, samples.length - AUTO_CALIBRATION_WINDOW);
    }
    const previous = this.provisionalCalibration[arm] || this.calibration?.hands?.[arm] || null;
    const profile = adaptiveCalibrationProfile(samples, previous);
    if (this.engaged[arm] && previous) profile.neutralScale = previous.neutralScale;
    this.provisionalCalibration[arm] = profile;

    const timestampMs = Number(timestampSeconds) * 1000;
    if (
      !this.calibrationKey
      || timestampMs - this.lastCalibrationPersistAt[arm] < AUTO_CALIBRATION_PERSIST_INTERVAL_MS
    ) {
      return;
    }
    this.lastCalibrationPersistAt[arm] = timestampMs;
    this.calibration = {
      version: 2,
      hands: {
        ...(this.calibration?.hands || {}),
        [arm]: { ...profile },
      },
    };
    try {
      localStorage.setItem(this.calibrationKey, JSON.stringify(this.calibration));
    } catch (_error) {}
  }

  commandForArm(arm) {
    const command = this.currentCommands[arm] || this.emptyCommand(arm);
    return {
      ...command,
      translation_offset_m: [...command.translation_offset_m],
      rotation_vector_rad: [...command.rotation_vector_rad],
    };
  }

  resetMotionFilter(arm) {
    this.poseFilters[arm].reset();
    this.rawOffsets[arm] = null;
    this.motionDiagnostics[arm] = {
      tremor: 0,
      speed: 0,
      quality: 0,
      velocity: [0, 0, 0, 0, 0, 0],
    };
  }

  renderMetrics() {
    this.panel.querySelector("#handRate").textContent = `${Math.round(this.frameRate)} Hz`;
    this.panel.querySelector("#handInference").textContent =
      `${this.inferenceMs ? Math.round(this.inferenceMs) : "—"} ms vision`;
    const applied = Number.isFinite(this.serverApplyAgeMs)
      ? ` · ${Math.round(this.serverApplyAgeMs)} ms apply`
      : "";
    this.panel.querySelector("#handLatency").textContent =
      `${this.roundTripMs ? Math.round(this.roundTripMs) : "—"} ms net${applied}`;
  }

  async transmit(forceFrozen = false) {
    if (!this.running || this.suspendTransmission) return;
    const hands = [0, 1].map(arm => {
      const command = this.commandForArm(arm);
      if (forceFrozen) {
        command.motion_engaged = false;
        command.translation_offset_m = [0, 0, 0];
        command.rotation_vector_rad = [0, 0, 0];
      }
      return command;
    });
    const payload = {
      sequence: Math.floor(this.sequenceBase + this.sequenceCounter++),
      hands,
      captured_at_ms: Number(this.lastCaptureEpochMs) || null,
      inference_ms: Number(this.inferenceMs) || null,
      client_sent_at_ms: null,
      transport_drops: this.transportDrops,
    };
    if (this.queuedPayload) this.transportDrops += 1;
    this.queuedPayload = payload;
    if (this.inFlight) return;
    const generation = this.transportGeneration;
    this.inFlight = true;
    while (
      this.queuedPayload
      && this.running
      && generation === this.transportGeneration
    ) {
      const next = this.queuedPayload;
      this.queuedPayload = null;
      next.client_sent_at_ms = Date.now();
      next.transport_drops = this.transportDrops;
      const controller = new AbortController();
      this.activeRequestController = controller;
      try {
        const sentAt = performance.now();
        const response = await this.request(
          "./api/teleop/hands",
          next,
          { signal: controller.signal },
        );
        this.roundTripMs = this.ewma(this.roundTripMs, performance.now() - sentAt, 0.20);
        this.updateServerSnapshot(response.hand_teleop);
      } catch (error) {
        if (
          generation === this.transportGeneration
          && error?.name !== "AbortError"
        ) {
          this.hardFreeze("transport_error");
          this.setBanner(`Control link stopped · ${error.message}`, "warn");
        }
        break;
      } finally {
        if (this.activeRequestController === controller) {
          this.activeRequestController = null;
        }
      }
    }
    this.inFlight = false;
  }

  updateServerSnapshot(snapshot) {
    if (!snapshot) return;
    this.serverEnabled = Boolean(snapshot.enabled);
    this.serverTransportAgeMs = Number.isFinite(snapshot.transport?.latest_age_ms)
      ? snapshot.transport.latest_age_ms
      : null;
    this.serverApplyAgeMs = Number.isFinite(snapshot.transport?.applied_age_ms)
      ? snapshot.transport.applied_age_ms
      : null;
    const controlFrameRevision = snapshot.control_frame?.revision;
    if (
      Number.isInteger(controlFrameRevision)
      && this.cameraControlFrameRevision !== null
      && controlFrameRevision !== this.cameraControlFrameRevision
      && this.engaged.some(Boolean)
    ) {
      this.hardFreeze("operative_view_changed");
      this.setBanner("Operative view changed · relax, then point down to re-anchor", "warn");
    }
    if (Number.isInteger(controlFrameRevision)) {
      this.cameraControlFrameRevision = controlFrameRevision;
    }
    for (const arm of [0, 1]) {
      const armState = snapshot.arms?.find(item => item.arm === arm);
      if (armState) {
        this.reacquire[arm] = Boolean(armState.reacquire_unclutched);
        this.serverSafety[arm] = armState.safety_state || (armState.stale ? "watchdog" : "ready");
      }
    }
  }

  async sendFrozenFor(arms) {
    this.suspendTransmission = true;
    this.queuedPayload = null;
    for (let attempts = 0; this.inFlight && attempts < 100; attempts += 1) {
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    try {
      const frozen = new Set(arms);
      const hands = [0, 1].map(arm => {
        const command = this.commandForArm(arm);
        if (frozen.has(arm)) {
          command.motion_engaged = false;
          command.translation_offset_m = [0, 0, 0];
          command.rotation_vector_rad = [0, 0, 0];
        }
        return command;
      });
      const response = await this.request("./api/teleop/hands", {
        sequence: Math.floor(this.sequenceBase + this.sequenceCounter++),
        hands,
      });
      this.updateServerSnapshot(response.hand_teleop);
    } finally {
      this.suspendTransmission = false;
    }
  }

  async engageArm(arm, { automatic = false } = {}) {
    const pose = this.poses.get(arm);
    if (!pose) {
      this.setBanner("Track that hand first", "warn");
      return;
    }
    if (this.controlMode === "single" && this.primaryArm !== arm) {
      this.selectPrimaryArm(arm);
    }
    if (!this.serverEnabled && !(await this.setServerEnabled(true))) return;
    if (this.reacquire[arm]) {
      this.engaged[arm] = false;
      await this.sendFrozenFor([arm]);
      if (this.reacquire[arm]) {
        this.setBanner(
          automatic
            ? "Relax the pointing finger once, then point down again"
            : "Hold that hand still and click Engage again",
          "warn",
        );
        return;
      }
    }
    this.anchors.set(arm, {
      center: { ...pose.center },
      scale: pose.scale,
      depthScale: pose.depthScale,
      basis: pose.basis.map(axis => [...axis]),
      indexTipY: pose.indexTipY,
      engagedAt: pose.timestampSeconds,
    });
    this.resetMotionFilter(arm);
    this.poseFilters[arm].reset([0, 0, 0, 0, 0, 0], pose.timestampSeconds);
    this.engaged[arm] = true;
    if (this.controlMode === "single") {
      if (!this.singleHandTried) this.singleHandTriedAt = pose.timestampSeconds;
      this.singleHandTried = true;
      this.renderModeControl();
    }
    this.setBanner(`Instrument ${arm + 1} long-range clutch engaged`, "good");
    this.renderCards();
  }

  freezeArm(arm, { automatic = false } = {}) {
    this.engaged[arm] = false;
    this.anchors.delete(arm);
    this.tableReach[arm] = 0;
    this.clutchReadySince[arm] = null;
    this.resetMotionFilter(arm);
    this.currentCommands[arm] = {
      ...this.commandForArm(arm),
      motion_engaged: false,
      translation_offset_m: [0, 0, 0],
      rotation_vector_rad: [0, 0, 0],
    };
    this.sendFrozenFor([arm]).catch(error => this.setBanner(error.message, "warn"));
    if (!automatic) this.cancelAutomaticEngage();
    this.renderCards();
  }

  toggleArm(arm) {
    if (this.engaged[arm]) this.freezeArm(arm);
    else this.engageArm(arm);
  }

  async engageAll({ automatic = false } = {}) {
    if (!this.serverEnabled && !(await this.setServerEnabled(true))) {
      this.autoEngagePending = automatic;
      return;
    }
    const visibleArms = [0, 1].filter(arm => this.poses.has(arm));
    if (this.controlMode === "single" && this.primaryArm === null) {
      const selectedArm = this.bestTrackedArm();
      if (selectedArm !== null) this.selectPrimaryArm(selectedArm);
    }
    const trackedArms = visibleArms.filter(arm => this.isArmEnabled(arm));
    if (!trackedArms.length) {
      this.autoEngagePending = automatic;
      this.setBanner(
        this.controlMode === "single" && this.primaryArm !== null
          ? `Show your ${this.primaryArm === 0 ? "left" : "right"} hand to engage`
          : "Show at least one hand before engaging",
        "warn",
      );
      return;
    }
    const needsReacquire = trackedArms.filter(arm => this.reacquire[arm]);
    if (needsReacquire.length) await this.sendFrozenFor(needsReacquire);
    let engagedCount = 0;
    for (const arm of trackedArms) {
      const pose = this.poses.get(arm);
      if (!pose || this.reacquire[arm]) continue;
      this.anchors.set(arm, {
        center: { ...pose.center },
        scale: pose.scale,
        depthScale: pose.depthScale,
        basis: pose.basis.map(axis => [...axis]),
        indexTipY: pose.indexTipY,
        engagedAt: pose.timestampSeconds,
      });
      this.resetMotionFilter(arm);
      this.poseFilters[arm].reset([0, 0, 0, 0, 0, 0], pose.timestampSeconds);
      this.engaged[arm] = true;
      engagedCount += 1;
    }
    if (!engagedCount) {
      this.autoEngagePending = automatic;
      this.setBanner("Hold steady while the motion safety latch rearms", "warn");
      return;
    }
    if (this.controlMode === "single") {
      if (!this.singleHandTried) {
        this.singleHandTriedAt = this.poses.get(trackedArms[0])?.timestampSeconds
          ?? performance.now() / 1000;
      }
      this.singleHandTried = true;
      this.renderModeControl();
    }
    const provisional = trackedArms.some(arm => !this.calibration?.hands?.[arm]);
    this.setBanner(
      provisional
        ? "Motion engaged · recalibrate for an exact personal jaw span"
        : automatic
          ? this.controlMode === "dual"
            ? "Stable tracking · both instruments engaged"
            : "Stable tracking · one-hand control engaged"
          : this.controlMode === "dual"
            ? "Both tracked instruments engaged"
            : "Selected instrument engaged",
      provisional ? "warn" : "good",
    );
    this.renderCards();
  }

  cancelAutomaticEngage() {
    this.autoEngagePending = false;
    if (this.autoEngageTimer !== null) {
      window.clearTimeout(this.autoEngageTimer);
      this.autoEngageTimer = null;
    }
  }

  scheduleAutomaticEngage() {
    if (!this.autoEngagePending || !this.running || this.calibrationCapture) return;
    if (!this.poses.size) {
      if (this.autoEngageTimer !== null) {
        window.clearTimeout(this.autoEngageTimer);
        this.autoEngageTimer = null;
      }
      return;
    }
    if (this.autoEngageTimer !== null) return;
    this.autoEngageTimer = window.setTimeout(async () => {
      this.autoEngageTimer = null;
      if (!this.autoEngagePending || !this.running || !this.poses.size) return;
      this.autoEngagePending = false;
      await this.engageAll({ automatic: true });
    }, 180);
  }

  freezeAll(transmit = true) {
    this.engaged = [false, false];
    this.anchors.clear();
    for (const arm of [0, 1]) {
      this.clutchReadySince[arm] = null;
      this.tableReach[arm] = 0;
      this.resetMotionFilter(arm);
      this.currentCommands[arm] = {
        ...this.commandForArm(arm),
        motion_engaged: false,
        translation_offset_m: [0, 0, 0],
        rotation_vector_rad: [0, 0, 0],
      };
    }
    if (transmit) this.transmit(true);
    this.setBanner(this.running ? "Motion frozen · recenter freely" : "Camera off");
    this.renderCards();
  }

  hardFreeze(reason = "operator_freeze") {
    this.cancelAutomaticEngage();
    this.transportGeneration += 1;
    this.queuedPayload = null;
    this.activeRequestController?.abort();
    this.activeRequestController = null;
    this.inFlight = false;
    this.freezeAll(false);
    this.serverEnabled = false;
    this.reacquire = [true, true];
    this.serverSafety = [reason, reason];
    const body = JSON.stringify({ enabled: false, reason });
    this.safetyStopPromise = fetch("./api/teleop/hands/control", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-dr-anmar-operator": this.operatorId,
      },
      body,
      keepalive: true,
    })
      .then(async response => {
        if (!response.ok) throw new Error("Server stop was not acknowledged");
        const data = await response.json();
        this.updateServerSnapshot(data.hand_teleop);
        return true;
      })
      .catch(() => false);
    this.setBanner("All hand control frozen · point down to re-arm", "warn");
    this.renderCards();
    return this.safetyStopPromise;
  }

  renderCards() {
    this.renderModeControl();
    for (const arm of [0, 1]) {
      const card = this.panel.querySelector(`#handCard${arm}`);
      const pose = this.poses.get(arm);
      const diagnostic = this.poseDiagnostics.get(arm);
      const command = this.commandForArm(arm);
      const enabled = this.isArmEnabled(arm);
      card.classList.toggle("tracked", Boolean(pose) && enabled);
      card.classList.toggle("inactive", !enabled);
      card.classList.toggle("quality-hold", !pose && Boolean(diagnostic));
      card.querySelector(".track").textContent = pose
        ? enabled ? "TRACKED" : "AVAILABLE"
        : diagnostic
          ? "QUALITY HOLD"
          : "NOT TRACKED";
      const safety = !enabled
        ? "Single-hand mode"
        : this.identityAmbiguousArms.has(arm)
          ? "Identity lock"
        : !pose && diagnostic
        ? "Vision quality hold"
        : this.reacquire[arm]
          ? "Recenter required"
          : this.engaged[arm]
            ? "Motion enabled"
            : this.serverSafety[arm]?.replaceAll("_", " ") || "Frozen";
      card.querySelector('[data-field="safety"]').textContent = safety;
      const clutchScore = clamp(Number(pose?.clutchScore) || 0, 0, 1);
      card.querySelector('[data-field="clutch"]').textContent = !enabled
        ? "Inactive"
        : this.engaged[arm]
          ? "Long-range on"
          : clutchScore >= CLUTCH_ENGAGE_SCORE
            ? "Hold point ↓"
            : "Point index ↓";
      card.querySelector('[data-field="table"]').textContent =
        `${Math.round(this.tableReach[arm] * 100)}%`;
      card.querySelector('[data-field="xyz"]').textContent = command.translation_offset_m.map(value => Math.round(value * 1000)).join(" · ");
      card.querySelector('[data-field="rpy"]').textContent = command.rotation_vector_rad.map(value => Math.round(value * 180 / Math.PI)).join(" · ");
      card.querySelector('[data-field="gripper"]').textContent = !enabled
        ? "Not active"
        : pose
          ? `${Math.round(command.aperture_normalized * 100)}%`
          : "Held";
      const quality = pose?.quality ?? diagnostic?.quality;
      const tremor = this.motionDiagnostics[arm]?.tremor;
      card.querySelector('[data-field="confidence"]').textContent =
        Number.isFinite(quality)
          ? `${Math.round(quality * 100)}% · ${
            Number.isFinite(tremor) && tremor > 0.34 ? "stabilizing" : "steady"
          }`
          : "—";
      const button = card.querySelector("[data-hand-arm]");
      button.textContent = !enabled
        ? `Use ${arm ? "right" : "left"} only`
        : this.engaged[arm]
          ? `Freeze ${arm ? "right" : "left"}`
          : `Engage ${arm ? "right" : "left"}`;
      button.classList.toggle("engaged", this.engaged[arm]);
    }
    this.renderSurfaceGuide();
  }

  renderSurfaceGuide() {
    const guide = this.panel.querySelector("#handSurfaceGuide");
    if (!guide) return;
    const candidates = [0, 1]
      .filter(arm => this.isArmEnabled(arm) && this.poses.has(arm))
      .sort((left, right) => Number(this.engaged[right]) - Number(this.engaged[left]));
    const arm = candidates[0];
    const pose = arm === undefined ? null : this.poses.get(arm);
    const score = clamp(Number(pose?.clutchScore) || 0, 0, 1);
    const progress = arm === undefined ? 0 : this.tableReach[arm];
    const pointing = score >= CLUTCH_ENGAGE_SCORE;
    const engaged = arm !== undefined && this.engaged[arm];
    const contact = engaged && progress >= 0.92;
    guide.classList.toggle("pointing", pointing);
    guide.classList.toggle("engaged", engaged);
    guide.classList.toggle("contact", contact);
    guide.querySelector("span").textContent = contact
      ? "TIP AT FLOOR · HOLD STEADY"
      : engaged
        ? `LONG RANGE · ${Math.round(progress * 100)}% DOWN`
        : pointing
          ? "HOLD POINT ↓ TO ENGAGE"
          : "POINT INDEX ↓";
    guide.querySelector("b").textContent = contact ? "TABLE TARGET" : "TABLE REACH";
  }

  draw() {
    const width = this.video.videoWidth || 960;
    const height = this.video.videoHeight || 540;
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.context.clearRect(0, 0, width, height);
    const displayPoses = new Map(
      [...this.poseDiagnostics].map(([arm, diagnostic]) => [arm, diagnostic.pose]),
    );
    for (const [arm, pose] of this.poses) displayPoses.set(arm, pose);
    for (const [arm, pose] of displayPoses) {
      const points = pose.landmarks.map(point => ({ x: (1 - point.x) * width, y: point.y * height }));
      const accepted = this.poses.has(arm);
      this.context.strokeStyle = accepted
        ? this.engaged[arm] ? "#42e49b" : "#2cd2e8"
        : "#ff956d";
      this.context.fillStyle = this.context.strokeStyle;
      this.context.lineWidth = Math.max(2, width / 500);
      for (const [from, to] of HAND_CONNECTIONS) {
        this.context.beginPath();
        this.context.moveTo(points[from].x, points[from].y);
        this.context.lineTo(points[to].x, points[to].y);
        this.context.stroke();
      }
      points.forEach((point, index) => {
        this.context.beginPath();
        this.context.arc(point.x, point.y, index === 8 ? 8 : index === 4 ? 6 : 3, 0, Math.PI * 2);
        this.context.fill();
      });
      if (accepted) {
        const thumb = points[4];
        const index = points[8];
        this.context.lineWidth = Math.max(5, width / 220);
        this.context.strokeStyle = this.engaged[arm] ? "#42e49b" : "#2cd2e8";
        this.context.beginPath();
        this.context.moveTo(thumb.x, thumb.y);
        this.context.lineTo(index.x, index.y);
        this.context.stroke();
      }
    }
  }

  async pollStatus() {
    try {
      const response = await fetch("./api/status/live", { cache: "no-store" });
      if (!response.ok) return;
      const status = await response.json();
      const wasEnabled = this.serverEnabled;
      this.updateServerSnapshot(status.hand_teleop);
      if (wasEnabled && status.hand_teleop && !status.hand_teleop.enabled) {
        this.freezeAll(false);
        this.setBanner("Manual takeover · relax, then point down to re-enable", "warn");
      }
    } catch (_error) {}
  }

  stopMedia() {
    this.cancelAutomaticEngage();
    this.startGeneration += 1;
    this.starting = false;
    this.running = false;
    if (this.videoFrameRequest !== null) {
      if (this.usingVideoFrameCallback && typeof this.video.cancelVideoFrameCallback === "function") {
        this.video.cancelVideoFrameCallback(this.videoFrameRequest);
      } else {
        cancelAnimationFrame(this.videoFrameRequest);
      }
      this.videoFrameRequest = null;
    }
    this.stream?.getTracks().forEach(track => track.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.visionWorker?.postMessage({ type: "close" });
    this.visionWorker?.terminate();
    this.visionWorker = null;
    this.visionFramePending = false;
    this.landmarker?.close?.();
    this.landmarker = null;
    this.lastMediaPipeTimestampMs = -1;
    this.launch.classList.remove("state-active");
  }

  async close() {
    this.savePanelGeometry();
    await this.hardFreeze("panel_closed");
    this.stopMedia();
    this.panel.classList.add("hidden");
    this.launch.setAttribute("aria-expanded", "false");
    this.launch.setAttribute("aria-label", "Open webcam hand control");
  }

  dispose() {
    this.savePanelGeometry();
    this.panelResizeObserver?.disconnect();
    window.removeEventListener("pointermove", this.handlePanelResizeMove);
    window.removeEventListener("pointerup", this.handlePanelResizeEnd);
    window.removeEventListener("pointercancel", this.handlePanelResizeEnd);
    window.removeEventListener("resize", this.handleWindowResize);
    this.startGeneration += 1;
    this.starting = false;
    this.running = false;
    this.transportGeneration += 1;
    this.queuedPayload = null;
    this.activeRequestController?.abort();
    this.activeRequestController = null;
    this.stream?.getTracks().forEach(track => track.stop());
    this.visionWorker?.terminate();
    this.visionWorker = null;
    this.visionFramePending = false;
    this.landmarker?.close?.();
    this.landmarker = null;
    const body = JSON.stringify({ enabled: false, reason: "page_unloaded" });
    fetch("./api/teleop/hands/control", {
      method: "POST",
      headers: { "content-type": "application/json", "x-dr-anmar-operator": this.operatorId },
      body,
      keepalive: true,
    }).catch(() => {});
  }
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  const initialize = () => {
    if (!document.getElementById("handControlLaunch")) {
      window.drAnmarHandController = new HandController();
      const query = new URLSearchParams(location.search);
      if (query.get("webcam") === "1") {
        query.delete("webcam");
        const remainingQuery = query.toString();
        const cleanUrl = `${location.pathname}${remainingQuery ? `?${remainingQuery}` : ""}${location.hash}`;
        history.replaceState(null, "", cleanUrl);
        window.drAnmarHandController.open();
      }
    }
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize, { once: true });
  else initialize();
}
