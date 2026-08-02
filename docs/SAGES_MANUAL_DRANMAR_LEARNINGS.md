# SAGES Manual -> Dr.Anmar surgical-training learnings

Status: research curriculum synthesis

Source: *The SAGES Manual, Volume 1: Basic Laparoscopy and Endoscopy*, third
edition, edited by Nathaniel J. Soper and Carol E.H. Scott-Conner, Springer,
2012. This document is a transformative training synthesis of the supplied
book, not a reproduction of it.

The detailed chapter-level extraction is maintained in the
[knowledge ledger](SAGES_MANUAL_KNOWLEDGE_LEDGER.md). This document is the
implementation-facing curriculum derived from that ledger.

## Executive translation

The book should become a **fundamentals-to-procedure curriculum** for Dr.Anmar,
not a collection of scripted procedures or an end-to-end “autonomous surgeon.”
Its central contribution is a disciplined order of work:

```text
prepare the room and instruments
-> establish safe access and working space
-> orient the camera and maintain exposure
-> identify anatomy and uncertainty
-> approach tissue with controlled contact
-> perform a bounded intervention
-> verify the physical effect
-> detect failure, stop, recover, or hand over
-> document what happened and what remains uncertain
```

In Dr.Anmar, the policy can request bounded motion and intervention intent. It
cannot write `bleeding_controlled`, `repair_complete`, `perfusion_restored`, or
`patient_stable`. Isaac/PhysX produces contact, geometry, attachment, flow,
pressure, and sensor evidence; the patient-effect layer turns that evidence
into modeled patient state; learning consumes the resulting transition.

This gives the book a precise Dr.Anmar interpretation:

| Book idea | Dr.Anmar implementation |
| --- | --- |
| FLS/FES fundamentals before clinical work | Proficiency-gated simulator skills before procedure graphs |
| Efficiency and precision | Time/path economy plus geometric/contact precision, never speed alone |
| Setup and troubleshooting | Versioned room/tool preflight and observable fault-injection drills |
| Ergonomics and workflow | Camera, monitor, tool, cable, collision, and workload state in the task contract |
| Safe access and working space | Access-state transitions, visibility, protected structures, and abort/re-entry |
| Hemostasis and tissue approximation | Contact/energy/attachment/flow/pressure evidence with retained-effect verification |
| Diagnostic laparoscopy and endoscopy | Observe-identify-target-verify tasks with uncertainty and abstention |
| Procedure chapters | Bounded skill graphs, not monolithic procedure policies |
| Documentation | Transition-aligned episode records with provenance, evidence, and unresolved uncertainty |

## What Dr.Anmar should learn from the book

### 1. Make fundamentals explicit and measurable

The manual describes FLS as discipline-independent knowledge plus five manual
skills: peg transfer, precision cutting, ligating-loop placement, and both
extracorporeal and intracorporeal suturing. It describes FES as technology,
patient preparation, sedation, upper/lower endoscopy, hemostasis, tissue
removal, enteral access, and endoscopic therapy, with navigation, loop
reduction, retroflexion, mucosal evaluation, and targeting as hands-on skills.

For Dr.Anmar, these become reusable simulator primitives rather than a one-off
exam:

1. camera orientation and horizon stabilization;
2. single-tool and dual-tool relative task-space control;
3. approach, grasp, hold, release, and re-entry;
4. precision cutting along a declared boundary;
5. atraumatic traction and exposure;
6. targeting and verification under partial visibility;
7. extracorporeal and intracorporeal knot/closure mechanics;
8. endoscope navigation, loop management, retroflexion, inspection, and target
   acquisition; and
9. structured fault recognition, neutral stop, recovery, and handover.

The existing Dr.Anmar path remains the motor foundation:

```text
DrAnmar-Reach-PSM-IK-Rel-v0
-> dual PSM reach
-> block lift
-> needle lift
-> block handover
-> needle handover
-> force-gated left-span needle entry
-> curved subsurface drive and right-span top exit
-> opposite-PSM acquisition and receiver-only pull-through
```

That path qualifies control and contact infrastructure. It should feed the
first surgical layer below; it should not be labeled laparoscopic competence.
For the tissue-interaction stages, the simulator owns entry, exit, embedded
arc, custody, and complete-clearance events. A policy may command only bounded
relative motion and jaw intent. Contact by either PSM shaft or wrist is a hard
failure, and a visually plausible needle pose without the matching physics
receipt is not a completed bite.

### 2. Treat setup as a learned safety skill

The setup chapter is valuable because it treats preparation, equipment
availability, monitor placement, cable paths, insufflation, suction,
electrosurgery, lighting, recording, and troubleshooting as part of the
operation. Dr.Anmar should therefore expose setup state to the trainee and
policy rather than silently constructing a perfect room.

Required setup drills:

- identify the planned target quadrant and choose camera/tool approach frames;
- place monitors and tool stations without blocking the operator or the
  emergency path;
- verify camera, light, insufflation, suction/irrigation, energy, tool, and
  recording channels;
- diagnose no-image, poor-image, weak insufflation, blocked suction, missing
  tool, invalid sensor, and energy-system faults;
- continue only when the relevant channel is valid; otherwise abstain or call
  for assistance; and
- preserve a preflight receipt containing room, tool, asset, solver, sensor,
  and task-contract revisions.

The simulator should inject faults before and during a task. A policy that
finishes only in a perfect room has not learned the book’s equipment lesson.

### 3. Make ergonomics computational

The ergonomics chapters are not just advice about posture. They connect
monitor placement, instrument handles, foot pedals, cables, lighting,
distractions, interruptions, workflow, and mental workload to error and
fatigue. In a robot simulator the equivalent variables are:

- camera horizon, image centering, magnification, and field-of-view stability;
- instrument approach angle, triangulation, reachability, and collision margin;
- tool velocity, acceleration, jerk, and dwell near tissue;
- cable/tube collision and access to the neutral-stop path;
- monitor and sensor latency, dropped frames, and validity flags;
- interruptions, alarm load, operator handover, and decision time; and
- asymmetric or degraded visibility that requires repositioning instead of
  forceful probing.

These belong in observations and evaluation. A clean trajectory with poor
visibility, excessive reorientation, or unsafe approach geometry is not a
high-quality demonstration.

### 4. Teach access as a state transition, not a teleport

The access chapters make entry, insufflation, trocar/cannula placement,
extraperitoneal space, previous surgery, and access complications explicit.
Dr.Anmar should represent access as a sequence with visible evidence:

```text
closed abdomen
-> entry candidate
-> confirmed working space
-> port/tool established
-> usable visualization
-> procedure-ready
```

Each transition needs a success condition, a failure condition, and a recovery
route. Examples include loss of insufflation, abdominal-wall bleeding, poor
visibility, suspected visceral injury, major-vessel proximity, adhesions, and
an inaccessible target. The correct learned behavior may be to stop, improve
exposure, choose another access route, or request takeover.

This maps naturally onto the dynamic abdominal patient’s access-state and
wound-state contracts. It must not be reduced to a boolean `access_open` flag
set by the action handler.

### 5. Teach contact before cutting

The hemostasis and tissue-approximation chapters repeatedly make the same
operational point: identify tissue and vessels, expose them, select the
appropriate device, apply controlled contact, and verify the effect. More
force, more energy, more traction, or more compression is not automatically
better.

Dr.Anmar skill primitives:

| Skill | Intent | Environment-owned evidence |
| --- | --- | --- |
| Atraumatic exposure | Capture, lift, spread, and hold tissue | Bilateral contact, force symmetry, visibility, deformation, protected-structure clearance |
| Hemostasis | Compress, clip, seal, patch, irrigate, or suction | Contact dwell, placement, force/pressure, flow reduction, attachment, distal perfusion, overload |
| Safe dissection | Separate target from protected structure | Tool path, tissue-plane geometry, contact mode, clearance, continuity, unintended injury |
| Tissue approximation | Align, bite, approximate, knot/staple, and release | Bite geometry, gap, retention, integrity, leak/pressure hold, strain, patency |
| Seal and divide | Center, compress, seal, verify, then divide | Jaw contact, tissue inclusion, energy dose/spread, seal state, cut topology, residual bleeding |
| Perfusion assessment | Scan, identify, intervene, rescan, verify | Registered sensor frames, flow/oxygenation/thermal evidence, validity, recovery or abstention |

These directly extend the current Dr.Anmar assets: Atraumatic Exposure,
Adaptive Hemostasis, SafePlane Dissection, Adaptive Seal and Divide, Adaptive
Anastomosis, and Perfusion Viability. The book supplies the surgical reasoning;
the asset contracts must supply the mechanics and evidence.

### 6. Separate observation from intervention

Diagnostic laparoscopy, biopsy, cholangiography, laparoscopic ultrasound, and
flexible endoscopy are training opportunities for a policy to decide that it
does not yet know enough.

Every diagnostic task should have:

```text
observe -> localize -> classify -> state confidence/uncertainty
-> choose target or abstain -> sample/intervene if authorized
-> verify the new state -> record evidence
```

The policy must not infer that a target is safe merely because a visual label
is present. The simulator should provide counterexamples with occlusion,
anatomic variation, poor contrast, imaging artifacts, unexpected pathology,
and instrument-induced distortion.

### 7. Use procedure chapters as skill graphs

The manual’s operation chapters should not become 20 separate end-to-end
policies. They are a library of reusable subskills and decision points.

For example, a cholecystectomy-inspired training graph can be represented as:

```text
setup and access
-> establish exposure
-> identify relevant anatomy
-> confirm the intended structure
-> bounded dissection
-> clip/seal/ligate or abstain
-> divide only after interlock conditions
-> inspect for bleeding/injury
-> verify flow/patency when selected
-> release, close, document, recover
```

The same graph pattern can support diagnostic laparoscopy, bowel handling,
gastrostomy, appendectomy, colostomy, hernia repair, resection, and endoscopic
procedures. Procedure-specific anatomy and effect models remain separate from
the reusable control, monitoring, and recovery layers.

## Dr.Anmar curriculum

### Stage 0 - knowledge, visual recognition, and scenario judgment

Book basis: FLS/FES didactics, equipment, access, hemostasis, complications,
and documentation (Chapters 1, 3, 5, 9-13, 17-19, 37-40).

Train:

- name instruments, energy modes, access components, sensors, and failure
  states;
- distinguish target tissue, protected tissue, vessel, lumen, artifact, and
  uncertain view;
- choose between continue, reposition, verify, stop, and handover;
- interpret synchronized images and post-physics evidence; and
- produce a short, structured procedural record.

Promotion gate: scenario accuracy plus correct abstention on ambiguous and
unsafe cases. No motion policy is needed to pass this stage.

### Stage 1 - camera and motor foundations

Book basis: FLS manual skills and FES navigation/targeting.

Train:

- horizon stabilization and camera centering;
- relative tool-tip positioning and orientation;
- peg transfer and precision placement;
- precision cutting along a visual boundary;
- target acquisition under a moving camera; and
- instrument exchange and neutral positioning.

Promotion gate: held-out target poses, bounded path length, low overshoot,
stable view, no workspace violations, and repeatable success across seeds.

### Stage 2 - exposure and atraumatic contact

Book basis: ergonomics, access, retraction, tissue handling, and diagnostic
laparoscopy.

Train in the dynamic abdominal patient:

- establish a view without excessive contact;
- grasp and release at permitted tissue zones;
- distribute bilateral traction and hold a requested exposure;
- preserve protected-structure clearance;
- recover from slip, lost view, excessive force, and port conflict; and
- release and confirm that tissue and patient state return to an acceptable
  condition.

Primary Dr.Anmar asset: Atraumatic Exposure. Supporting assets: dynamic
abdominal patient, endoscope/camera telemetry, suction and irrigation.

Promotion gate: exposure quality, force symmetry, tissue-damage budget,
visibility dwell, release recovery, and no hard safety violations.

### Stage 3 - dissection and bounded tissue separation

Book basis: access, energy, hemostasis, diagnostic visualization, and
procedure-specific dissection chapters.

Train:

- approach a declared tissue plane;
- use blunt, spreading, hydrodissection, scissors, or energy only when the
  contract permits it;
- maintain a protected-structure no-go zone;
- detect bleeding, loss of plane, thermal spread, or uncertain anatomy;
- stop and re-expose instead of forcing a difficult separation; and
- verify continuity and absence of unintended division.

Primary Dr.Anmar asset: SafePlane Dissection. Supporting assets: Wound
Preparation, Adaptive Seal and Divide, Perfusion Viability, dynamic patient.

Promotion gate: plane adherence, clearance, damage, thermal budget, bleeding
outcome, recovery detection, and held-out anatomy variants.

### Stage 4 - hemostasis and tissue approximation

Book basis: Chapters 9-12 and the cholecystectomy complication chapter.

Train separate bounded skills before composing them:

1. compress a bleeding source with measured bilateral contact;
2. place a clip or ligating device only with complete target visualization;
3. apply a seal/energy device with tissue inclusion and thermal constraints;
4. use suction/irrigation to restore the field without hiding evidence;
5. place a suture, knot, staple, patch, or approximation device;
6. verify retention, integrity, pressure/leak behavior, and distal perfusion;
7. release and test whether the modeled benefit persists; and
8. escalate when the source cannot be safely controlled.

Primary Dr.Anmar assets: Adaptive Hemostasis and Adaptive Anastomosis.
Supporting assets: Adaptive Seal and Divide, Closure Robot, fluid/pressure
effects, perfusion viability.

Promotion gate: benefit must be produced by post-physics evidence and persist
under release/perturbation tests. A reward or self-reported `verified` flag is
not sufficient.

### Stage 5 - diagnostic and verification skills

Book basis: diagnostic laparoscopy, biopsy, cholangiography, laparoscopic
ultrasound, and endoscopic inspection.

Train:

- inspect systematically rather than only chasing a salient target;
- maintain a landmark/coverage record;
- distinguish observation from diagnosis and diagnosis from intervention;
- report confidence and missing views;
- target a biopsy or measurement without injuring adjacent structures;
- compare pre- and post-intervention state; and
- abstain when evidence is incomplete or contradictory.

Primary Dr.Anmar asset: Perfusion Viability and multimodal study surfaces,
with diagnostic procedure rooms layered over the dynamic patient.

Promotion gate: coverage, localization, calibration, uncertainty reporting,
false-positive/false-negative strata, and safe target acquisition.

### Stage 6 - flexible endoscopy

Book basis: Chapters 37-46.

Build a separate flexible-endoscopy track rather than treating the laparoscope
as the same kinematic problem:

- equipment care, leak checks, image and channel troubleshooting;
- insertion, tip deflection, torque, navigation, and loop reduction;
- retroflexion and controlled target approach;
- mucosal inspection and documentation;
- sedation/recovery as an observable room-state model, not a hidden reset;
- upper and lower GI diagnostic flows;
- tissue removal, enteral access, hemostasis, and therapy as bounded skills;
- perforation, bleeding, incomplete examination, and recovery drills.

Candidate Dr.Anmar task IDs:

```text
DrAnmar-Endoscope-Navigate-v0
DrAnmar-Endoscope-Target-v0
DrAnmar-Endoscope-Inspect-Document-v0
DrAnmar-Endoscope-Intervene-Verify-v0
```

Promotion gate: navigation efficiency, mucosal coverage, target precision,
image validity, loop/torque management, complication detection, and safe
withdrawal or handover.

### Stage 7 - integrated procedure graph

Only after the subskills pass individually should Dr.Anmar compose them into a
procedure graph. The first integrated demonstrations should be intentionally
small:

1. access -> exposure -> inspect -> neutral stop;
2. exposure -> controlled bleeding -> hemostasis -> release -> verify;
3. exposure -> safe-plane dissection -> perfusion scan -> recovery;
4. tissue alignment -> approximation -> leak/pressure test -> document; and
5. inspect -> target -> bounded intervention -> complication -> takeover.

The initial policy objective is not “finish the operation.” It is:

```text
select the correct bounded skill
-> execute or abstain
-> verify the patient effect
-> recover or hand back control
```

## Skill-card contract

Every book-derived Dr.Anmar task should be authored with this compact contract:

```yaml
task_id: DrAnmar-<skill>-v0
entry_state: observable patient, room, tool, and access preconditions
intent: bounded intervention request; never a patient-outcome write
observations:
  - synchronized camera/RGB-D/endoscope views and validity
  - robot pose, velocity, action age, tool identity, and workspace margin
  - post-physics contact, force, separation, deformation, attachment
  - relevant flow, pressure, perfusion, leak, integrity, and damage state
actions:
  - relative task-space target or short bounded action chunk
  - tool/mode selection and explicit neutral-stop request
safety:
  - workspace, speed, force, thermal, protected-structure, and uncertainty limits
success: environment-owned post-physics outcome with declared dwell
failure: hard event or patient-state deterioration; not merely low reward
recovery: abstain, neutral stop, re-expose, irrigate/suction, re-enter, or handover
evidence: immutable revision-bound receipt and transition-aligned episode record
```

## Evaluation and promotion

Use the book’s efficiency/precision idea, but extend it to contact-rich patient
effects. A candidate advances only when all applicable gates pass:

| Gate | Required evidence |
| --- | --- |
| Contract | Units, frames, timing, reset, sensor validity, action bounds, and authority boundaries |
| Competence | Frozen checkpoint on held-out seeds and anatomy/tool variants |
| Precision | Target geometry, path length, view stability, bite/placement accuracy |
| Physical behavior | Contact, force, slip, tissue response, energy, release, attachment, flow, and pressure |
| Patient effect | Benefit/harm transition computed by the environment; no action-authored outcome |
| Safety | No hard workspace, protected-structure, overload, thermal, or uncertainty violations |
| Recovery | Detection, abstention, neutral stop, re-entry, recovery, and operator takeover |
| Documentation | Complete event timeline, evidence hashes, unresolved uncertainty, and intervention provenance |
| Efficiency | Time and motion budget after safety and competence pass |
| Transfer boundary | Claims limited to the qualified simulator/phantom/research scope |

Metrics to report per episode:

- completion and abstention rate;
- time to stable target view and time to verified effect;
- path length, motion smoothness, tool exchanges, and action clipping;
- force magnitude, force asymmetry, contact dwell, tissue deformation, and
  protected-structure clearance;
- energy dose, thermal spread, bleeding/flow, pressure, leak, perfusion,
  attachment, repair integrity, and tissue damage;
- visibility loss, sensor dropouts, uncertainty, and invalid-action rate;
- failure detection latency, recovery success, takeover timing, and residual
  patient harm; and
- performance stratified by anatomy, visibility, pathology, tool, fault, and
  perturbation rather than one aggregate mean.

Reward remains a diagnostic signal. Hard harm and safety events are constraints,
not penalties that a policy can trade against faster completion.

## Recommended implementation order

1. Add a machine-readable SAGES-derived skill registry beside the current
   learning-path contract. Register source chapters, entry/exit conditions,
   sensors, effect providers, and promotion gates.
2. Implement camera/orientation, setup-fault, and atraumatic-exposure tasks on
   the existing PSM and dynamic-patient paths.
3. Connect hemostasis and approximation tasks to native post-physics evidence,
   including release persistence, flow/pressure, damage, and perfusion checks.
4. Add diagnostic inspection/verification tasks with coverage and uncertainty
   telemetry before training intervention policies.
5. Add flexible-endoscopy navigation and targeting as a separate kinematic
   family with its own action and sensor contract.
6. Build complete expert demonstrations with approach, contact, effect,
   release, recovery, and documentation; do not train from phase labels alone.
7. Compare scripted/controller, recurrent behavior cloning, action chunking,
   diffusion, and bounded residual PPO under the same frozen contract.
8. Promote only frozen checkpoints that pass held-out competence, physical,
   safety, recovery, and evidence gates.

The shortest credible route is warm-started and hierarchical: controller or
demonstration first, bounded policy residual second, recovery and robustness
third, procedure composition last.

## Chapter-to-Dr.Anmar coverage map

| Source chapters | Book domain | Dr.Anmar learning surface |
| --- | --- | --- |
| 1-2 | FLS/FES and maintenance of certification | Knowledge scenarios, proficiency gates, GOALS/GAGES-inspired rubrics |
| 3-4 | Equipment, troubleshooting, ergonomics, workflow | Room preflight, fault injection, camera/tool ergonomics, workload and interruption |
| 5-8 | Access, workspace, single-site, hand-assisted methods | Access-state transitions, port geometry, exposure, re-entry and recovery |
| 9-12 | Energy, hemostatic adjuncts, tissue approximation, devices | Hemostasis, seal/divide, suturing, closure, retained-effect verification |
| 13-16 | Documentation, special situations, robotics | Provenance-rich event logs, uncertainty, robot authority and takeover |
| 17-19 | Emergency and diagnostic laparoscopy, biopsy/staging | Observe-localize-target-abstain-intervene-verify tasks |
| 20-25 | Cholecystectomy, complications, biliary imaging/exploration | Anatomy confirmation, protected structures, flow/patency, complication recovery |
| 26-33 | Gastric, bowel, appendix, colon, and hernia procedures | Reusable exposure, dissection, division, approximation, leak, and inspection skills |
| 34-36 | Pediatric MIS and complications | Separate anatomy/scale/physiology variants; do not mix into adult promotion averages |
| 37-40 | Flexible endoscope fundamentals and documentation | Navigation, channel/tool validity, handling, recovery, image/evidence records |
| 41-43 | Upper GI endoscopy, feeding access, capsule enteroscopy | Diagnostic targeting, access placement, inspection, retrieval and verification |
| 44-46 | Sigmoidoscopy, colonoscopy, therapeutic colonoscopy | Navigation, mucosal coverage, therapy, bleeding/perforation detection and handover |
| Appendix | SAGES guidelines, privileging, training, checklists, and education resources | Source provenance, supervision, modernization, and promotion dependencies |

Detailed source ranges: Chapters 1-16, pp. 3-205; Chapters 17-19, pp. 207-254;
Chapters 20-25, pp. 255-344; Chapters 26-33, pp. 345-442; Chapters 34-36,
pp. 443-496; Chapters 37-40, pp. 497-538; Chapters 41-43, pp. 539-580; and
Chapters 44-46, pp. 581-626; Appendix, pp. 627-629.

## Boundary and update rule

This manual is a 2012 educational reference. It is useful for skill structure,
workflow, equipment reasoning, and failure-oriented training, but it is not a
current clinical guideline, device manual, patient-specific protocol, or
evidence of clinical efficacy. Current standards, local supervision, updated
device behavior, and independent biomechanical/clinical validation remain
separate work.

For Dr.Anmar, the honest claim is:

> SAGES-derived surgical fundamentals are now organized as measurable,
> physics-grounded simulation skills with explicit outcome authority, recovery,
> and evidence gates.

That claim is stronger and more useful than claiming that the simulator has
learned to perform a real operation.
