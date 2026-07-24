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

export const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));

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
  const scaleDepth = -Math.log(currentPose.scale / anchorPose.scale) * 0.12;
  const landmarkDepth = (currentPose.center.z - anchorPose.center.z) * 0.04;
  return clamp((scaleDepth + landmarkDepth) * gain, -0.12, 0.12);
}

export function poseOffset(anchorPose, currentPose, gain = 1) {
  const translation = [
    depthOffset(anchorPose, currentPose, gain),
    clamp((currentPose.center.x - anchorPose.center.x) * 0.18 * gain, -0.12, 0.12),
    clamp((anchorPose.center.y - currentPose.center.y) * 0.15 * gain, -0.12, 0.12),
  ];
  const rotation = rotationDelta(anchorPose.basis, currentPose.basis)
    .map(value => clamp(value * gain, -0.8, 0.8));
  return { translation, rotation };
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

export function conditionPoseVector(values) {
  return values.map((value, index) => {
    const deadband = index < 3 ? 0.00045 : 0.006;
    if (Math.abs(value) <= deadband) return 0;
    return Math.sign(value) * (Math.abs(value) - deadband);
  });
}

export function trackingQuality(pose, previousPose = null, dt = 1 / 30) {
  if (!pose || !Number.isFinite(pose.scale) || pose.scale <= 0) return 0;
  const confidence = clamp(Number(pose.confidence) || 0, 0, 1);
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
  return clamp(Math.min(confidence, geometry, boundsQuality) * motionQuality, 0, 1);
}

export function assignHandDetections(detections, previousPoses = new Map()) {
  const usable = detections.filter(item => item?.pose && (item.proposedArm === 0 || item.proposedArm === 1));
  if (!usable.length) return new Map();
  const assignments = usable.length === 1
    ? [[0], [1]]
    : [[0, 1], [1, 0]];
  let best = null;
  for (const arms of assignments) {
    let cost = 0;
    for (let index = 0; index < usable.length; index += 1) {
      const detection = usable[index];
      const arm = arms[index];
      const labelConfidence = clamp(Number(detection.labelConfidence) || 0, 0, 1);
      if (detection.proposedArm !== arm) cost += 0.80 * labelConfidence;
      const previous = previousPoses.get(arm);
      if (previous) cost += 3.2 * distance3(detection.pose.center, previous.center);
    }
    if (!best || cost < best.cost) best = { cost, arms };
  }
  return new Map(best.arms.map((arm, index) => [arm, usable[index].pose]));
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
    .hand-control-launch{background:#123440!important;border-color:#2cd2e8!important;color:#eaffff!important}
    .hand-control-launch.state-active{background:#2cd2e8!important;color:#031014!important}
    .hand-panel{position:fixed;z-index:70;right:24px;bottom:24px;width:min(620px,calc(100vw - 32px));max-height:calc(100vh - 96px);overflow:auto;border:1px solid #2f5968;border-radius:16px;background:#07151df2;color:#e9f8fa;box-shadow:0 24px 70px #000b;backdrop-filter:blur(16px);padding:15px}
    .hand-panel.hidden{display:none}.hand-head{display:flex;align-items:flex-start;gap:12px}.hand-head h2{margin:1px 0 4px;color:#e9f8fa}.hand-head p{margin:0;color:#88a6b2;font-size:12px}.hand-head .spacer{flex:1}
    .hand-video-wrap{position:relative;margin-top:12px;aspect-ratio:16/9;border-radius:11px;overflow:hidden;background:#020608;border:1px solid #24404d}.hand-video-wrap video,.hand-video-wrap canvas{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform:scaleX(-1)}.hand-video-wrap canvas{transform:none;pointer-events:none}
    .hand-banner{position:absolute;left:10px;top:10px;padding:6px 9px;border-radius:7px;background:#07151ddd;border:1px solid #315766;color:#b7d5dc;font:700 10px ui-monospace,SFMono-Regular}.hand-banner.good{color:#42e49b;border-color:#32725e}.hand-banner.warn{color:#ffba93;border-color:#82513d}
    .hand-metrics{position:absolute;right:10px;top:10px;display:flex;gap:5px;flex-wrap:wrap;justify-content:flex-end;max-width:72%}.hand-metrics span{padding:5px 7px;border-radius:6px;background:#07151ddd;border:1px solid #264653;color:#9db6bf;font:700 9px ui-monospace,SFMono-Regular}
    .hand-actions{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:7px;margin-top:10px}.hand-actions button{min-height:40px}.hand-actions .engaged{background:#2cd2e8;color:#031014;border-color:#2cd2e8}
    .hand-cards{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}.hand-card{padding:10px;border:1px solid #24404d;border-radius:10px;background:#091920}.hand-card.quality-hold{border-color:#82513d;background:#1b1412}.hand-card header{height:auto;padding:0;border:0;background:none}.hand-card b{font-size:12px}.hand-card .track{margin-left:auto;color:#ff8a90;font:800 9px ui-monospace}.hand-card.tracked .track{color:#42e49b}.hand-card dl{display:grid;grid-template-columns:auto 1fr;margin:8px 0 0;gap:4px 8px;font:10px ui-monospace,SFMono-Regular}.hand-card dt{color:#71929d}.hand-card dd{margin:0;text-align:right}.hand-card button{width:100%;margin-top:8px;min-height:32px}
    .hand-calibration{margin-top:10px;padding:10px;border:1px solid #7a693d;border-radius:10px;background:#241f12}.hand-calibration.hidden{display:none}.hand-calibration b{display:block;margin-bottom:4px}.hand-calibration p{margin:0 0 8px;color:#d8cda9;font-size:12px}.hand-calibration progress{width:100%;height:7px;margin:2px 0 8px;accent-color:#2cd2e8}.hand-privacy{margin:10px 2px 0;color:#6f909b;font-size:10px}
    @media(max-width:640px){.hand-panel{right:8px;bottom:8px;width:calc(100vw - 16px)}.hand-cards{grid-template-columns:1fr}.hand-actions{grid-template-columns:1fr 1fr}.hand-actions button:first-child{grid-column:1/-1}.hand-metrics{top:auto;bottom:9px;left:9px;right:9px;max-width:none;justify-content:flex-start}}
  `;
  document.head.append(style);

  const launch = document.createElement("button");
  launch.id = "handControlLaunch";
  launch.className = "hand-control-launch";
  launch.dataset.shortcut = "CAM";
  launch.innerHTML = "Hand control <kbd>CAM</kbd>";
  const stopRow = document.querySelector(".control-stop-row");
  if (stopRow) stopRow.prepend(launch);
  else document.body.append(launch);

  const panel = document.createElement("section");
  panel.id = "handControlPanel";
  panel.className = "hand-panel hidden";
  panel.setAttribute("aria-label", "Webcam hand control");
  panel.innerHTML = `
    <div class="hand-head"><div><h2>Two-finger surgical control</h2><p>Left hand → Instrument 1 · Right hand → Instrument 2</p></div><div class="spacer"></div><button id="handClose" data-shortcut="CAM">Close</button></div>
    <div class="hand-video-wrap"><video id="handVideo" playsinline muted></video><canvas id="handOverlay"></canvas><div id="handBanner" class="hand-banner">Camera off</div><div class="hand-metrics"><span id="handRate">0 Hz</span><span id="handInference">— ms vision</span><span id="handLatency">— ms loop</span></div></div>
    <div class="hand-actions"><button id="handStart" data-shortcut="CAM">Start camera</button><button id="handFreezeAll" data-shortcut="FREEZE">Freeze both</button><button id="handEngageAll" data-shortcut="ENGAGE">Engage tracked</button><button id="handPrecision" class="engaged" data-shortcut="PRECISION">Precision on</button><button id="handRecalibrate" data-shortcut="CAL">Recalibrate</button></div>
    <div class="hand-cards">
      <article id="handCard0" class="hand-card"><header><b>Left hand · Instrument 1</b><span class="track">NOT TRACKED</span></header><dl><dt>Safety</dt><dd data-field="safety">Frozen</dd><dt>Clutch</dt><dd data-field="clutch">Frozen</dd><dt>XYZ mm</dt><dd data-field="xyz">0 · 0 · 0</dd><dt>RPY °</dt><dd data-field="rpy">0 · 0 · 0</dd><dt>Gripper</dt><dd data-field="gripper">—</dd><dt>Signal quality</dt><dd data-field="confidence">—</dd></dl><button data-hand-arm="0" data-shortcut="L CAM">Engage left</button></article>
      <article id="handCard1" class="hand-card"><header><b>Right hand · Instrument 2</b><span class="track">NOT TRACKED</span></header><dl><dt>Safety</dt><dd data-field="safety">Frozen</dd><dt>Clutch</dt><dd data-field="clutch">Frozen</dd><dt>XYZ mm</dt><dd data-field="xyz">0 · 0 · 0</dd><dt>RPY °</dt><dd data-field="rpy">0 · 0 · 0</dd><dt>Gripper</dt><dd data-field="gripper">—</dd><dt>Signal quality</dt><dd data-field="confidence">—</dd></dl><button data-hand-arm="1" data-shortcut="R CAM">Engage right</button></article>
    </div>
    <div id="handCalibration" class="hand-calibration hidden"><b id="handCalibrationTitle">Camera calibration</b><p id="handCalibrationText"></p><progress id="handCalibrationProgress" max="${CALIBRATION_SAMPLE_COUNT}" value="0"></progress><button id="handCalibrationCapture" data-shortcut="CAL">Capture stable sample</button></div>
    <p class="hand-privacy">Only calibrated numeric pose commands leave this browser. Webcam frames and raw landmarks are never uploaded or recorded. Single-camera depth is relative, not metric or clinical-grade.</p>
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
    this.running = false;
    this.starting = false;
    this.startGeneration = 0;
    this.serverEnabled = false;
    this.reacquire = [true, true];
    this.serverSafety = ["disabled", "disabled"];
    this.inFlight = false;
    this.suspendTransmission = false;
    this.queuedPayload = null;
    this.sequenceBase = Date.now() * 1000;
    this.sequenceCounter = 0;
    this.lastInferenceAt = 0;
    this.lastStatusAt = 0;
    this.lastFrameAt = null;
    this.frameRate = 0;
    this.inferenceMs = 0;
    this.roundTripMs = 0;
    this.transportDrops = 0;
    this.videoFrameRequest = null;
    this.poses = new Map();
    this.poseDiagnostics = new Map();
    this.anchors = new Map();
    this.engaged = [false, false];
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
    this.currentCommands = [this.emptyCommand(0), this.emptyCommand(1)];
    this.lastAperture = [1, 1];
    this.calibration = null;
    this.calibrationStep = 0;
    this.calibrationDraft = { hands: {} };
    this.calibrationCapture = null;
    this.operatorId = this.resolveOperatorId();
    this.bind();
  }

  resolveOperatorId() {
    const query = new URLSearchParams(location.search).get("operator");
    if (query) return query;
    let value = sessionStorage.getItem("drAnmarOperatorId");
    if (!value) {
      value = `browser-${crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36)}`;
      sessionStorage.setItem("drAnmarOperatorId", value);
    }
    return value;
  }

  bind() {
    this.launch.addEventListener("click", () => this.open());
    this.panel.querySelector("#handClose").addEventListener("click", () => this.close());
    this.panel.querySelector("#handStart").addEventListener("click", () => this.start());
    this.panel.querySelector("#handFreezeAll").addEventListener("click", () => this.freezeAll());
    this.panel.querySelector("#handEngageAll").addEventListener("click", () => this.engageAll());
    this.panel.querySelector("#handPrecision").addEventListener("click", () => {
      this.precision = !this.precision;
      const button = this.panel.querySelector("#handPrecision");
      button.textContent = this.precision ? "Precision on" : "Precision off";
      button.classList.toggle("engaged", this.precision);
      this.freezeAll();
    });
    this.panel.querySelector("#handRecalibrate").addEventListener("click", () => this.beginCalibration());
    this.panel.querySelectorAll("[data-hand-arm]").forEach(button => {
      button.addEventListener("click", () => this.toggleArm(Number(button.dataset.handArm)));
    });
    this.panel.querySelector("#handCalibrationCapture").addEventListener("click", () => this.captureCalibration());
    window.addEventListener("pagehide", () => this.dispose(), { once: true });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) this.freezeAll(false);
    });
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

  async request(path, body) {
    const response = await fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json", "x-dr-anmar-operator": this.operatorId },
      body: JSON.stringify(body),
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
    if (!this.running) await this.start();
  }

  async start() {
    if (this.running || this.starting) return;
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      this.setBanner("HTTPS is required for webcam access", "warn");
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
        video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 540 } },
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
      this.calibrationKey = `drAnmar.handCalibration.v2:${settings.deviceId || "default"}`;
      const storedCalibration = localStorage.getItem(this.calibrationKey)
        || localStorage.getItem(`drAnmar.handCalibration.v1:${settings.deviceId || "default"}`);
      this.calibration = numericCalibration(JSON.parse(storedCalibration || "null"));
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
        minHandDetectionConfidence: 0.55,
        minHandPresenceConfidence: 0.55,
        minTrackingConfidence: 0.55,
      };
      try {
        this.landmarker = await visionModule.HandLandmarker.createFromOptions(files, options);
      } catch (_gpuError) {
        options.baseOptions.delegate = "CPU";
        this.landmarker = await visionModule.HandLandmarker.createFromOptions(files, options);
      }
      if (generation !== this.startGeneration) {
        this.landmarker?.close?.();
        this.landmarker = null;
        this.stopMedia();
        return;
      }
      this.running = true;
      this.launch.classList.add("state-active");
      this.panel.querySelector("#handStart").textContent = "Camera active";
      this.setBanner("Tracking hands · motion frozen", "good");
      await this.setServerEnabled(true);
      if (!this.calibration) this.beginCalibration();
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

  async setServerEnabled(enabled) {
    try {
      const response = await this.request("./api/teleop/hands/control", { enabled });
      this.updateServerSnapshot(response.hand_teleop);
      if (!this.serverEnabled) this.freezeAll(false);
      return this.serverEnabled;
    } catch (error) {
      this.serverEnabled = false;
      this.setBanner(error.message, "warn");
      return false;
    }
  }

  beginCalibration() {
    this.freezeAll(false);
    this.calibrationStep = 0;
    this.calibrationDraft = { hands: {} };
    this.calibrationCapture = null;
    this.panel.querySelector("#handCalibration").classList.remove("hidden");
    this.renderCalibration();
  }

  renderCalibration() {
    const instructions = [
      ["1 · Neutral pose", "Hold both hands comfortably in the center with motion frozen."],
      ["2 · Fully closed", "Touch only each thumb and index fingertip together."],
      ["3 · Fully open", "Open each thumb and index finger to a comfortable maximum."],
    ];
    const [title, text] = instructions[this.calibrationStep] || instructions[0];
    this.panel.querySelector("#handCalibrationTitle").textContent = title;
    this.panel.querySelector("#handCalibrationText").textContent = text;
    const progress = this.panel.querySelector("#handCalibrationProgress");
    progress.value = 0;
    const button = this.panel.querySelector("#handCalibrationCapture");
    button.disabled = false;
    button.textContent = this.calibrationStep === 2 ? "Sample open and finish" : "Capture stable sample";
  }

  captureCalibration() {
    if (!this.poses.size) {
      this.setBanner("Show at least one hand before capturing", "warn");
      return;
    }
    this.calibrationCapture = {
      step: this.calibrationStep,
      startedAt: performance.now(),
      samples: { 0: [], 1: [] },
    };
    const button = this.panel.querySelector("#handCalibrationCapture");
    button.disabled = true;
    button.textContent = "Hold steady · sampling 0%";
    this.setBanner("Sampling stable hand geometry…");
  }

  collectCalibrationSamples(timestampMs) {
    const capture = this.calibrationCapture;
    if (!capture) return;
    for (const [arm, pose] of this.poses) {
      if (pose.quality < MIN_TRACKING_QUALITY) continue;
      const value = capture.step === 0 ? pose.scale : pose.pinchRatio;
      if (capture.samples[arm].length < CALIBRATION_SAMPLE_COUNT) capture.samples[arm].push(value);
    }
    const visibleCounts = [...this.poses.keys()].map(arm => capture.samples[arm].length);
    const progressCount = visibleCounts.length ? Math.min(...visibleCounts) : 0;
    this.panel.querySelector("#handCalibrationProgress").value = progressCount;
    this.panel.querySelector("#handCalibrationCapture").textContent =
      `Hold steady · sampling ${Math.round(100 * progressCount / CALIBRATION_SAMPLE_COUNT)}%`;
    const complete = visibleCounts.length && visibleCounts.every(count => count >= CALIBRATION_SAMPLE_COUNT);
    if (complete || timestampMs - capture.startedAt > 1600) this.finishCalibrationCapture();
  }

  finishCalibrationCapture() {
    const capture = this.calibrationCapture;
    this.calibrationCapture = null;
    let accepted = 0;
    try {
      for (const arm of [0, 1]) {
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
      if (!accepted) throw new Error("No stable hand sample was captured");
    } catch (error) {
      this.setBanner(`${error.message} · capture again`, "warn");
      this.renderCalibration();
      return;
    }
    if (this.calibrationStep < 2) {
      this.calibrationStep += 1;
      this.renderCalibration();
      this.setBanner("Sample accepted · continue calibration", "good");
      return;
    }
    const calibration = numericCalibration({ version: 2, hands: this.calibrationDraft.hands });
    if (!calibration) {
      this.setBanner("Calibration span is too small · try again", "warn");
      this.beginCalibration();
      return;
    }
    this.calibration = calibration;
    localStorage.setItem(this.calibrationKey, JSON.stringify(calibration));
    this.panel.querySelector("#handCalibration").classList.add("hidden");
    this.setBanner("Stable calibration saved · motion frozen", "good");
  }

  poseFromLandmarks(landmarks, worldLandmarks, confidence) {
    const pose = palmFrame(worldLandmarks?.length === 21 ? worldLandmarks : landmarks);
    const imagePose = palmFrame(landmarks);
    pose.center.x = imagePose.center.x;
    pose.center.y = imagePose.center.y;
    pose.center.z = imagePose.center.z;
    pose.scale = imagePose.scale;
    pose.geometryQuality = Math.min(pose.geometryQuality, imagePose.geometryQuality);
    pose.pinchRatio = distance3(landmarks[4], landmarks[8]) / Math.max(imagePose.scale, 1e-6);
    pose.landmarks = landmarks;
    pose.confidence = confidence;
    return pose;
  }

  frame(time, metadata = null) {
    if (!this.running) return;
    const frameTimestampMs = Number.isFinite(metadata?.mediaTime)
      ? metadata.mediaTime * 1000
      : time;
    if (time - this.lastInferenceAt >= MIN_INFERENCE_INTERVAL_MS && this.video.readyState >= 2) {
      const inferenceStarted = performance.now();
      this.lastInferenceAt = time;
      try {
        const result = this.landmarker.detectForVideo(this.video, frameTimestampMs);
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
        this.freezeAll();
        this.setBanner(`Vision safety hold · ${error.message}`, "warn");
      }
    }
    if (time - this.lastStatusAt > 800) {
      this.lastStatusAt = time;
      this.pollStatus();
    }
    this.scheduleFrame();
  }

  ewma(previous, current, alpha) {
    return previous > 0 ? previous + alpha * (current - previous) : current;
  }

  processResult(result, timestampSeconds) {
    const detections = [];
    (result.landmarks || []).forEach((landmarks, index) => {
      const category = result.handednesses?.[index]?.[0];
      const proposedArm = handednessToArm(category?.categoryName);
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
    const assigned = assignHandDetections(detections, this.poses);
    const next = new Map();
    const diagnostics = new Map();
    const dt = this.lastPoseTimestamp
      ? clamp(timestampSeconds - this.lastPoseTimestamp, 1 / 240, 0.2)
      : 1 / 30;
    for (const [arm, pose] of assigned) {
      const quality = trackingQuality(pose, this.poses.get(arm), dt);
      pose.quality = quality;
      pose.timestampSeconds = timestampSeconds;
      diagnostics.set(arm, {
        quality,
        state: quality >= MIN_TRACKING_QUALITY ? "ready" : "quality hold",
        pose,
      });
      if (quality >= MIN_TRACKING_QUALITY) next.set(arm, pose);
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
    this.updateCommands(timestampSeconds);
    this.renderCards();
  }

  updateCommands(timestampSeconds) {
    for (const arm of [0, 1]) this.currentCommands[arm] = this.buildCommandForArm(arm, timestampSeconds);
  }

  buildCommandForArm(arm, timestampSeconds) {
    const pose = this.poses.get(arm);
    const calibration = this.calibration?.hands?.[arm];
    if (!pose || !calibration) {
      return this.emptyCommand(arm);
    }
    let aperture = this.lastAperture[arm];
    try {
      const rawAperture = normalizedAperture(pose.pinchRatio, calibration.closeRatio, calibration.openRatio);
      aperture = clamp(this.apertureFilters[arm].filter(rawAperture, timestampSeconds), 0, 1);
      this.lastAperture[arm] = aperture;
    } catch (_error) {}
    const speedButton = document.querySelector(
      `[data-hand-speed-arm="${arm}"].active`,
    );
    const speed = Number(speedButton?.dataset.handSpeed || 1) * (this.precision ? 0.62 : 1);
    let offset = { translation: [0, 0, 0], rotation: [0, 0, 0] };
    const anchor = this.anchors.get(arm);
    if (this.engaged[arm] && anchor) {
      const calibrationGain = clamp(calibration.neutralScale / anchor.scale, 0.7, 1.3);
      offset = poseOffset(anchor, pose, speed * calibrationGain);
      const raw = [...offset.translation, ...offset.rotation];
      const previous = this.rawOffsets[arm];
      const predicted = predictPoseVector(
        previous?.value,
        raw,
        previous ? timestampSeconds - previous.timestampSeconds : 0,
      );
      this.rawOffsets[arm] = { value: raw, timestampSeconds };
      const conditioned = conditionPoseVector(predicted);
      const filtered = this.poseFilters[arm].filter(conditioned, timestampSeconds);
      offset = {
        translation: filtered.slice(0, 3),
        rotation: filtered.slice(3, 6),
      };
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
  }

  renderMetrics() {
    this.panel.querySelector("#handRate").textContent = `${Math.round(this.frameRate)} Hz`;
    this.panel.querySelector("#handInference").textContent =
      `${this.inferenceMs ? Math.round(this.inferenceMs) : "—"} ms vision`;
    this.panel.querySelector("#handLatency").textContent =
      `${this.roundTripMs ? Math.round(this.roundTripMs) : "—"} ms loop`;
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
    };
    if (this.queuedPayload) this.transportDrops += 1;
    this.queuedPayload = payload;
    if (this.inFlight) return;
    this.inFlight = true;
    while (this.queuedPayload && this.running) {
      const next = this.queuedPayload;
      this.queuedPayload = null;
      try {
        const sentAt = performance.now();
        const response = await this.request("./api/teleop/hands", next);
        this.roundTripMs = this.ewma(this.roundTripMs, performance.now() - sentAt, 0.20);
        this.updateServerSnapshot(response.hand_teleop);
      } catch (error) {
        this.freezeAll(false);
        this.setBanner(error.message, "warn");
      }
    }
    this.inFlight = false;
  }

  updateServerSnapshot(snapshot) {
    if (!snapshot) return;
    this.serverEnabled = Boolean(snapshot.enabled);
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

  async engageArm(arm) {
    const pose = this.poses.get(arm);
    if (!pose || !this.calibration?.hands?.[arm]) {
      this.setBanner("Track and calibrate that hand first", "warn");
      return;
    }
    if (!this.serverEnabled && !(await this.setServerEnabled(true))) return;
    if (this.reacquire[arm]) {
      this.engaged[arm] = false;
      await this.sendFrozenFor([arm]);
      if (this.reacquire[arm]) {
        this.setBanner("Hold that hand still and click Engage again", "warn");
        return;
      }
    }
    this.anchors.set(arm, {
      center: { ...pose.center },
      scale: pose.scale,
      basis: pose.basis.map(axis => [...axis]),
    });
    this.resetMotionFilter(arm);
    this.poseFilters[arm].reset([0, 0, 0, 0, 0, 0], pose.timestampSeconds);
    this.engaged[arm] = true;
    this.setBanner(`Instrument ${arm + 1} engaged`, "good");
    this.renderCards();
  }

  freezeArm(arm) {
    this.engaged[arm] = false;
    this.anchors.delete(arm);
    this.resetMotionFilter(arm);
    this.currentCommands[arm] = {
      ...this.commandForArm(arm),
      motion_engaged: false,
      translation_offset_m: [0, 0, 0],
      rotation_vector_rad: [0, 0, 0],
    };
    this.sendFrozenFor([arm]).catch(error => this.setBanner(error.message, "warn"));
    this.renderCards();
  }

  toggleArm(arm) {
    if (this.engaged[arm]) this.freezeArm(arm);
    else this.engageArm(arm);
  }

  async engageAll() {
    if (!this.serverEnabled && !(await this.setServerEnabled(true))) return;
    const trackedArms = [0, 1].filter(
      arm => this.poses.has(arm) && this.calibration?.hands?.[arm],
    );
    const needsReacquire = trackedArms.filter(arm => this.reacquire[arm]);
    if (needsReacquire.length) await this.sendFrozenFor(needsReacquire);
    for (const arm of [0, 1]) {
      const pose = this.poses.get(arm);
      if (!pose || !this.calibration?.hands?.[arm] || this.reacquire[arm]) continue;
      this.anchors.set(arm, {
        center: { ...pose.center },
        scale: pose.scale,
        basis: pose.basis.map(axis => [...axis]),
      });
      this.resetMotionFilter(arm);
      this.poseFilters[arm].reset([0, 0, 0, 0, 0, 0], pose.timestampSeconds);
      this.engaged[arm] = true;
    }
    this.setBanner("Tracked instruments engaged", "good");
    this.renderCards();
  }

  freezeAll(transmit = true) {
    this.engaged = [false, false];
    this.anchors.clear();
    for (const arm of [0, 1]) {
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

  renderCards() {
    for (const arm of [0, 1]) {
      const card = this.panel.querySelector(`#handCard${arm}`);
      const pose = this.poses.get(arm);
      const diagnostic = this.poseDiagnostics.get(arm);
      const command = this.commandForArm(arm);
      card.classList.toggle("tracked", Boolean(pose));
      card.classList.toggle("quality-hold", !pose && Boolean(diagnostic));
      card.querySelector(".track").textContent = pose
        ? "TRACKED"
        : diagnostic
          ? "QUALITY HOLD"
          : "NOT TRACKED";
      const safety = !pose && diagnostic
        ? "Vision quality hold"
        : this.reacquire[arm]
          ? "Recenter required"
          : this.engaged[arm]
            ? "Motion enabled"
            : this.serverSafety[arm]?.replaceAll("_", " ") || "Frozen";
      card.querySelector('[data-field="safety"]').textContent = safety;
      card.querySelector('[data-field="clutch"]').textContent = this.engaged[arm] ? "Engaged" : "Frozen";
      card.querySelector('[data-field="xyz"]').textContent = command.translation_offset_m.map(value => Math.round(value * 1000)).join(" · ");
      card.querySelector('[data-field="rpy"]').textContent = command.rotation_vector_rad.map(value => Math.round(value * 180 / Math.PI)).join(" · ");
      card.querySelector('[data-field="gripper"]').textContent = pose ? `${Math.round(command.aperture_normalized * 100)}%` : "Held";
      const quality = pose?.quality ?? diagnostic?.quality;
      card.querySelector('[data-field="confidence"]').textContent =
        Number.isFinite(quality) ? `${Math.round(quality * 100)}%` : "—";
      const button = card.querySelector("[data-hand-arm]");
      button.textContent = this.engaged[arm] ? `Freeze ${arm ? "right" : "left"}` : `Engage ${arm ? "right" : "left"}`;
      button.classList.toggle("engaged", this.engaged[arm]);
    }
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
        ? arm === 0 ? "#2cd2e8" : "#42e49b"
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
        this.context.arc(point.x, point.y, index === 4 || index === 8 ? 6 : 3, 0, Math.PI * 2);
        this.context.fill();
      });
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
        this.setBanner("Manual takeover · click Engage to re-enable", "warn");
      }
    } catch (_error) {}
  }

  stopMedia() {
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
    this.landmarker?.close?.();
    this.landmarker = null;
    this.launch.classList.remove("state-active");
  }

  async close() {
    this.freezeAll(false);
    await this.setServerEnabled(false);
    this.stopMedia();
    this.panel.classList.add("hidden");
  }

  dispose() {
    this.startGeneration += 1;
    this.starting = false;
    this.running = false;
    this.queuedPayload = null;
    this.stream?.getTracks().forEach(track => track.stop());
    const body = JSON.stringify({ enabled: false });
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
    if (!document.getElementById("handControlLaunch")) window.drAnmarHandController = new HandController();
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize, { once: true });
  else initialize();
}
