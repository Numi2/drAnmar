# SAGES Manual knowledge ledger for Dr.Anmar

This is the chapter-level extraction behind
[SAGES_MANUAL_DRANMAR_LEARNINGS.md](SAGES_MANUAL_DRANMAR_LEARNINGS.md). It
preserves the book's operative knowledge as training-relevant rules, failure
modes, and simulator requirements without copying the book.

Source: *The SAGES Manual, Volume 1: Basic Laparoscopy and Endoscopy*, third
edition, Soper and Scott-Conner, Springer, 2012, supplied as
`978-1-4614-2344-7.pdf`.

The source is an educational reference from 2012. Specific clinical choices,
device behavior, medication, pressure, and screening values below are source
knowledge to be represented as parameterized simulation scenarios. They are
not current clinical protocols and must not be promoted to real-patient rules.

## Extraction rule

Each chapter is represented by three labeled sections, with its decision and
failure boundary made explicit inside the source-knowledge bullets and probes:

1. the knowledge the chapter teaches;
2. the Dr.Anmar state/action/effect translation; and
3. a training probe that can distinguish competence from a successful video.

The repeated pattern across the book is:

```text
preparation -> exposure -> identification -> controlled action
-> verification -> complication surveillance -> recovery/documentation
```

## Cross-book rules

### Rule A - basic technical proficiency is a gate, not an autonomy claim

FLS and FES are designed to establish basic knowledge and technical skill
before clinical work. Their tests reward efficiency and precision, while
GOALS/GAGES provide structured global assessments. Passing a fundamentals
program does not make someone a competent surgeon or endoscopist.

Dr.Anmar consequence: register a skill as qualified only after the exact task
contract, held-out competence, physical outcomes, safety, and recovery gates
pass. Keep the claim at simulator-skill level.

### Rule B - the field of view is a safety state

Laparoscopic and endoscopic maneuvers are repeatedly conditioned on seeing the
lumen, target, adjacent anatomy, device jaws, and result. Blind advancement,
blind clipping, blind energy, and blind needle passage are failure states.

Dr.Anmar consequence: `view_valid`, target visibility, protected-structure
visibility, and evidence freshness are hard preconditions for action. A camera
policy must be allowed to withdraw, clean, irrigate, reorient, or abstain.

### Rule C - geometry and force are coupled

Triangulation, approach angle, port position, tissue traction, needle geometry,
instrument leverage, endoscope torque, and working distance determine the force
needed and the injury risk.

Dr.Anmar consequence: record approach frames, tool separation, axis alignment,
contact force, speed, dwell, deformation, and clearance together. Do not learn
force behavior from scalar reward alone.

### Rule D - effect verification must be independent of intent

The book uses direct inspection, flow, contrast, leak tests, visible anatomy,
pressure, and postoperative signs to verify results. The operator's belief that
a clip, knot, tube, or repair is correct is not the result.

Dr.Anmar consequence: patient effects come only from environment-owned
post-physics evidence. Every benefit has a persistence test and every damage
path remains observable.

### Rule E - complications are part of the skill

Nearly every procedure chapter has a prevention, recognition, and management
section. Early exposure, accurate identification, timely control, additional
access, conversion, and escalation are treated as technical decisions, not
embarrassing exceptions.

Dr.Anmar consequence: training data must include failures, interventions, and
handover. An agent that completes nominal cases but cannot detect a failed seal,
leak, injury, poor view, or uncontrolled bleeding has not learned the chapter.

### Rule F - documentation closes the loop

FLS/FES assessment, endoscopic documentation, image/video capture, nursing
flowsheets, findings, interventions, and complications all create a durable
record for continuity, accountability, quality improvement, and research.

Dr.Anmar consequence: an episode is incomplete until it contains a causal event
timeline, source/revision hashes, sensor validity, intervention intent,
post-physics evidence, outcome transition, uncertainty, and recovery path.

## Part I - basic laparoscopy and endoscopy: general principles

### Chapter 1 - Fundamentals of Laparoscopic Surgery and Endoscopic Surgery

Source: pp. 3-13.

Knowledge extracted:

- FLS is discipline-independent and combines preoperative, intraoperative,
  basic-procedure, postoperative/complication, and manual-skills teaching.
- The FLS manual tasks are peg transfer, precision cutting, ligating-loop
  placement, extracorporeal knot tying, and intracorporeal knot tying. They are
  scored with task-specific time/precision standards.
- FES extends the same model to technology, patient preparation, sedation,
  upper/lower endoscopy, anatomy/pathology, hemostasis, tissue removal,
  enteral access, and endoscopic therapies.
- FES manual skills include navigation, tip deflection, torque, loop reduction,
  retroflexion, mucosal evaluation, and targeting.
- GOALS and GAGES convert expert judgment into anchored global assessments.
- Simulator practice is valuable because deliberate, proficiency-based practice
  can accelerate technical learning, but fundamentals assessment remains a
  safety baseline rather than a complete operative qualification.

Dr.Anmar translation:

- Create separate state-based and vision-based foundational tasks before any
  procedure graph.
- Preserve time, precision, view stability, and tissue/contact safety as
  separate metrics.
- Add a structured global rubric to every local task: depth perception,
  bimanual dexterity, efficiency, tissue handling, autonomy/decision making,
  and recovery.

Training probes:

- Hold out target layouts, camera offsets, tool variants, and visual clutter.
- Require the policy to stop when the target or protected structure is not
  visible.
- Test the same skill with no-op, scripted-controller, demonstration, and
  learned-residual baselines.

### Chapter 2 - Maintenance of Certification

Source: pp. 15-19.

Knowledge extracted:

- Certification is not a one-time exam; it is continuous learning and
  reassessment.
- The chapter frames surgical competence around patient care, medical
  knowledge, practice-based learning and improvement, interpersonal/
  communication skills, professionalism, and systems-based practice.
- Continuing education, self-assessment, case/procedure logs, quality
  improvement, and periodic reassessment are complementary rather than
  interchangeable.
- The underlying safety idea is to detect knowledge drift and practice gaps
  after initial training.

Dr.Anmar translation:

- Treat checkpoint promotion as provisional and revision-bound.
- Maintain a longitudinal skill log with task exposure, held-out results,
  errors, recovery, human interventions, and time since last assessment.
- Schedule challenge cases after policy changes, asset changes, solver changes,
  sensor changes, or long periods without use.

Training probes:

- Re-evaluate a previously qualified checkpoint after changing visibility,
  tissue properties, latency, tool geometry, or patient variant.
- Require the report to show which competencies were not tested rather than
  presenting one aggregate score.

### Chapter 3 - Equipment Setup and Troubleshooting

Source: pp. 21-43.

Knowledge extracted:

- Room layout, doors, outlets, table orientation, monitors, assistant position,
  anesthesia access, C-arm path, and cable routing should be planned before the
  patient arrives.
- A laparoscopic setup includes table, monitors, suction/irrigation,
  electrosurgical unit and return electrode, light source, insufflator,
  camera/processor, recording, trocars, graspers, dissectors, needle holders,
  scissors, retractors, clips, staplers, loops, sutures, and backup supplies.
- Preflight checks include gas supply, table tilt, straps/footboard, Foley or
  nasogastric needs, power, insufflator alarm, irrigation/suction, recording,
  electrosurgical settings, instrument jaws, trocar seals, and Veress/Hasson
  function.
- Troubleshooting follows a diagnosis path: inspect for damage, verify power and
  connections, read the error state, isolate the subsystem, and substitute a
  known-good component when needed.
- Common faults include low/lost insufflation, excessive pressure, poor or
  excessive light, no image, poor image, fogging, flicker, blur, weak suction,
  poor grounding, energy return-electrode alarms, and damaged insulation.
- Thermal injury and operating-room fire are distinct hazards. Unintended RF
  burns can arise from insulation failure, direct coupling, capacitive coupling,
  current concentration, tissue sticking, or a hot light cord; alcohol vapors
  and energized equipment create fire risk.

Dr.Anmar translation:

- Build an environment-owned room preflight state rather than spawning a
  perfect tool chain.
- Give the trainee a fault code, symptom, and available recovery options. Do
  not expose the hidden answer.
- Model channel validity independently: camera, light, insufflation, suction,
  irrigation, energy, recording, force/contact, and patient monitor.

Training probes:

- Inject one fault before entry and one fault after tissue contact.
- Measure time to diagnosis, unsafe actions while a fault is active, neutral
  stop latency, and whether the policy resumes only after a valid receipt.
- Record thermal dose and unintended-contact state even when the task fails.

### Chapter 4 - Ergonomics in Operating Room Design

Source: pp. 45-59.

Knowledge extracted:

- MIS creates a human-machine interface in which the operator works through a
  two-dimensional display, fixed ports, long instruments, and indirect motion.
- Visual display quality, eye strain, visual attention, information complexity,
  mental workload, interruptions, and distractions influence safety.
- Surgical workflow has physical, instrument, and cognitive components.
  Instrument exchanges, staff movement, interruptions, and interruptions from
  equipment can be observed and quantified.
- Monitor location, table height, handle design, cables/tubes, foot pedals,
  lighting, and access around the patient are ergonomic and safety variables.
- Integrated OR design should centralize control, manage information flow,
  protect the patient and staff, and remain flexible and expandable.
- Education is needed because the existence of an ergonomic guideline does not
  mean it is recognized or followed.

Dr.Anmar translation:

- Add view centering, horizon, magnification, latency, action age, and visual
  clutter to the observation contract.
- Track workspace margin, tool crossing, collisions, cable/tube entanglement,
  handover reachability, and neutral-stop accessibility.
- Add interruptions and delayed/dropped frames as challenge strata, not only
  random pixel noise.

Training probes:

- Compare two trajectories with equal task success but different view loss,
  tool crossing, jerk, reorientation, and unnecessary instrument exchanges.
- Inject an interruption during a high-risk step and score pause, state
  reconstruction, and safe resumption.

### Chapter 5 - Access to Abdomen

Source: pp. 61-77.

Knowledge extracted:

- Access equipment includes the insufflator, Veress needle, trocars, optical
  entry, and open Hasson cannula.
- Closed access requires controlled entry, confirmation of intraperitoneal
  position, insufflation behavior, and careful trocar placement; open access
  exposes the fascia and enters under direct control.
- The Veress confirmation logic uses aspiration, saline flow, re-aspiration,
  free flow into the cavity, pressure/flow behavior, and visualization after
  insufflation. No single reassuring sign is a substitute for observation.
- Trocar entry should be directed and controlled; blunt or failed safety
  mechanisms must not be forced.
- Alternative sites and open entry become important when the umbilicus is
  unsafe, prior surgery suggests adhesions, or local anatomy is abnormal.
- Complications include abdominal-wall bleeding, visceral injury, and major
  vascular injury. Recognition should lead to direct inspection, temporary
  control, repair or conversion rather than continued routine work.

Dr.Anmar translation:

- Implement an access state machine with `candidate`, `confirmed`, `unsafe`,
  `working_space`, and `procedure_ready` states.
- Expose pressure, flow, needle trajectory, tissue contact, organ proximity,
  and port seal as post-physics evidence.
- Treat suspected entry injury as a terminal/recovery event, not as a reward
  penalty that can be outweighed by later completion.

Training probes:

- Present normal entry, preperitoneal placement, bowel contact, vascular
  proximity, loss of insufflation, and trocar-site bleeding.
- Require a policy to choose between repositioning, open access, additional
  inspection, controlled compression, and conversion.

### Chapter 6 - Generation of Working Space: Extraperitoneal Approaches

Source: pp. 79-85.

Knowledge extracted:

- The preperitoneal, retroperitoneal, and subperitoneal spaces have different
  anatomy and risks; the aorta and vena cava are key retroperitoneal hazards.
- Access options include open entry, a lens-tipped trocar, and closed Veress
  entry with gas-assisted dissection.
- Dissection may use an operating laparoscope, blunt instruments, a balloon
  dissector, or a structural balloon trocar. Additional trocars should be
  placed under direct vision after the plane exists.
- The space is maintained by CO2 insufflation, abdominal-wall lifting,
  structural trocars, or peritoneal retraction; the route changes by operation.
- Peritoneal penetration, gas leak, venous bleeding, prior dissection, prior
  intra-abdominal surgery, and obesity can collapse or distort the plane.
- Conversion to a transabdominal route is an acceptable response when the
  intended space cannot be safely maintained.

Dr.Anmar translation:

- Model a plane as a bounded deformable region with boundaries and leakage,
  rather than as an empty hidden volume.
- Train plane identification, controlled balloon expansion, leak detection,
  and re-entry/conversion.

Training probes:

- Vary plane thickness, prior dissection, adhesions, gas leakage, and vascular
  proximity. Score whether the tool remains in the intended plane.

### Chapter 7 - Single-Site Access Surgery

Source: pp. 87-97.

Knowledge extracted:

- Single-site surgery has uncertain indications and depends heavily on patient
  selection, procedure steps, habitus, physiology, and team experience.
- Access can use multiple ports through one incision, a commercial multiport
  system, or a single-incision platform; the approach changes instrument
  geometry and camera behavior.
- The central technical problem is loss of triangulation: instrument crossing,
  in-line vision, external trocar collision, limited retraction, and awkward
  port angles.
- Solutions include articulating instruments, flexible/long telescopes,
  crossed-hand techniques, sutures or internal retractors, and conversion to
  standard laparoscopy when safety or progress is compromised.
- Patient safety outweighs cosmetic access benefits; the technique remains
  technology- and procedure-dependent.

Dr.Anmar translation:

- Create a constrained-geometry variant that changes only access topology and
  preserves the same tissue/effect contract.
- Measure the cost of reduced triangulation and the decision to convert.

Training probes:

- Compare single-site and multiport access with the same target and protected
  anatomy. Inject instrument collision, in-line vision, poor retraction, and
  stalled progress; require conversion when safety or exposure is lost.

### Chapter 8 - Hand-Assisted Laparoscopic Surgery

Source: pp. 99-103.

Knowledge extracted:

- Hand-assisted laparoscopy can restore tactile sensation, depth cues, direct
  retraction, extraction, and rapid control of difficult bleeding.
- It is useful for large specimens, complex dissection, difficult exposure,
  required tactile feedback, or a laparoscopic case approaching conversion.
- Tradeoffs include a larger incision, loss of pneumoperitoneum, hand-port
  crowding, poorer ergonomics, and interference with instruments.
- Port location should preserve triangulation and may be chosen to facilitate
  later open conversion. The hand port should be used with the same atraumatic
  tissue principles as instruments.

Dr.Anmar translation:

- Treat a hand port or proxy haptic interface as a separate authority and
  sensing mode, not as an invisible increase in policy capability.
- Add the hand's contact force, occupancy, visual occlusion, and conversion
  readiness to the episode record.

Training probes:

- Vary specimen size, bleeding, hand-port crowding, pneumoperitoneum loss, and
  visual occlusion. Require atraumatic retraction, explicit hand-state logging,
  and a conversion decision when control cannot be maintained.

### Chapter 9 - Laparoscopic Hemostasis: Energy Sources

Source: pp. 105-119.

Knowledge extracted:

- Tissue effect depends on temperature, heating rate, tissue composition,
  contact area, current density, impedance, power, activation time, and cooling.
- The chapter gives illustrative thermal transitions: collagen changes near
  45 C, irreversible protein denaturation near 60 C, carbonization near 80 C,
  vaporization around 90-100 C, and oxidation above 125 C. These values are
  source-era teaching values, not Dr.Anmar calibration.
- Electrosurgery concepts include circuit, current, voltage, resistance,
  impedance, capacitance, power, active/return electrodes, and current density.
- Monopolar cut, coagulation, and blended waveforms differ in continuity,
  frequency, voltage, tissue effect, speed, and hemostasis. Bipolar vessel
  sealing confines current between jaws and can combine grasping, dissection,
  hemostasis, and division. Radiofrequency and argon-enhanced systems add
  other effect profiles.
- Ultrasonic shears combine compression and friction; cutting and coagulating
  modes trade speed, tissue effect, hemostasis, and thermal spread.
- Electrical complications include current concentration, insulation failure,
  direct coupling, capacitive coupling, tissue sticking, incomplete
  coagulation, and burns away from the visible target. Ultrasonic systems can
  still transmit heat and injure adjacent tissue.

Dr.Anmar translation:

- Treat energy as a measured intervention with waveform/mode, jaw contact,
  tissue inclusion, activation time, temperature field, thermal spread,
  current path, and return-electrode state.
- Make energy safety a hard constraint. Do not allow a policy to trade an
  adjacent burn for faster completion.

Training probes:

- Use identical geometry with different tissue hydration, thickness, contact
  area, activation duration, and adjacent protected structures.
- Test insulation failure, direct coupling, capacitive coupling, incomplete
  seal, tissue sticking, and no-return-electrode alarms.

### Chapter 10 - Laparoscopic Hemostasis: Hemostatic Products and Adjuncts

Source: pp. 121-127.

Knowledge extracted:

- Prevention starts with careful dissection and identification of vascular
  anatomy; adjuncts supplement rather than replace exposure and control.
- Mechanical choices include titanium or polymer clips, vascular staplers,
  pretied loops, and sutures. Clips require complete visualization and correct
  placement; polymer locking systems have device-specific safety considerations.
- Staplers require correct tissue thickness, jaw placement, compression, and
  complete visualization of both distal ends. Misfire or incomplete firing is a
  distinct failure state.
- Sutures and knots can control bleeding but require controlled traction and
  practice; sawing or avulsion can worsen tissue injury.
- Tissue adjuncts include fibrin glues, gelatin agents, fibrinogen/thrombin
  fleece, oxidized cellulose, and microfibrillar collagen. Their usefulness
  depends on the bleeding pattern, moisture, pressure, and whether a
  functional vascular target is present.
- Active bleeding management follows exposure, suction/irrigation, direct
  pressure, precise source identification, clip/suture/adjunct selection, and
  conversion when safe control is not possible.

Dr.Anmar translation:

- Each hemostatic action needs a device-specific interlock: target isolated,
  jaws visible, protected structures clear, tissue included, and placement
  verified.
- The effect provider must distinguish temporary compression, retained clip or
  patch, thermal seal, and uncontrolled bleeding.

Training probes:

- Present oozing, a named vessel, diffuse parenchymal bleeding, obscured view,
  clip migration, incomplete stapler firing, and failed seal.
- Score time to exposure, amount of suction/irrigation, repeated attempts,
  residual flow, distal perfusion, and recovery/handover.

### Chapter 11 - Principles of Tissue Approximation

Source: pp. 129-142.

Knowledge extracted:

- Laparoscopic suturing is difficult because the surgeon sees a magnified,
  indirect 2D image, works through fixed ports, lacks direct tactile feedback,
  and must coordinate a camera with two instruments.
- Triangulation and port geometry determine whether needle driving is possible;
  the camera, target, needle, and instruments should be arranged to create a
  useful working triangle.
- Needle-holder design, needle curvature, needle point, suture material,
  trocar length, jaw grip, and instrument orientation affect tissue trauma and
  driving force.
- Needle positioning can use needle-trailing or suture-trailing approaches;
  the needle should be presented at an angle that follows the intended arc
  rather than pushed straight through tissue.
- Interrupted sutures are easier to control and can use a slip-knot sequence;
  continuous sutures are faster but require greater control. Suture selection
  depends on tissue and intended persistence.
- Hollow-viscus anastomosis requires attention to lumen alignment, wall
  thickness, tension, spacing, perfusion, and leak prevention. End-to-end,
  side-to-side, and other configurations have different geometry.
- Intracorporeal and extracorporeal knot tying are deliberate motor skills;
  formal simulation and many hours of practice are emphasized.

Dr.Anmar translation:

- Represent the needle as a rigid body with a safe bite corridor, tissue
  penetration event, suture tension, knot state, and retained attachment.
- Separate approach, needle pickup, reorientation, bite, pull-through, knot,
  cut, and release as inspectable subskills.
- Make the bite route explicit: enter the declared tissue span, keep the curved
  arc subsurface, re-emerge through the top of the opposite span, transfer the
  exposed arc to the second instrument, and prove receiver-only clearance.
- Keep tissue collision authoritative for every instrument link. Needle-only
  puncture permission must never make a PSM shaft or wrist ghost through the
  tissue, and the environment—not policy phase labels—owns entry, exit,
  custody, and clearance receipts.
- Include 2D projection, camera handoff, port geometry, and suture slack in the
  task state.

Training probes:

- Evaluate needle-tip path, bite symmetry, tissue gap, force/traction, suture
  tension, knot integrity, leak/pressure hold, and needle recovery after a
  dropped or poorly oriented needle.

### Chapter 12 - Other Devices for Tissue Approximation

Source: pp. 143-161.

Knowledge extracted:

- Staplers substitute for suturing only when the tissue is correctly selected,
  perfused, positioned, and fully contained between the jaws.
- Stapler choice depends on tissue thickness/staple height, linear or circular
  design, articulating or straight jaws, cartridge length, handle reach, and
  cutting versus noncutting function.
- Before firing, the operator should rehearse controls, check tissue inclusion,
  avoid adjacent structures, understand compression time, and inspect the
  staple line. Misfire management is a planned skill.
- Tissue fasteners and clips require suitable port size, direction, target
  exposure, and verification that the distal jaws are free before firing.
- Ligating loops are pre-tied devices that require a grasper to control the
  target while the loop is seated, tightened, and checked.
- Tissue glues vary in indication, preparation, storage, delivery, and
  internal/external suitability. Gas-driven spray can create embolic risk and
  glue should not substitute for an unrecognized active source.
- Experimental anastomotic rings and laser welding illustrate why a new device
  needs separate evidence and should not inherit the safety of sutures.

Dr.Anmar translation:

- Build a typed device state machine: loaded, introduced, jaws visible, target
  included, safety interlock satisfied, fired, retained, failed, or misfired.
- Verify approximation by topology, gap, attachment, leak, pressure hold,
  patency, and tissue viability rather than by the commanded fire event.

Training probes:

- Inject wrong cartridge, partial tissue capture, adjacent structure capture,
  incomplete fire, misfire, clip jaw obstruction, loop slip, glue delivery
  failure, and post-release leak.

### Chapter 13 - Documentation

Source: pp. 163-171.

Knowledge extracted:

- Video imaging is part of technical performance, not only archiving. Camera,
  processor, signal path, display, recording, and storage quality determine
  what can be reviewed.
- Documentation should preserve the observed anatomy, procedure steps,
  interventions, findings, and complications in a way that supports review and
  continuity.
- Image/video systems have evolving signal, recording-media, and quality
  considerations; a recording can fail even when the procedure itself did not.

Dr.Anmar translation:

- Synchronize camera frames with robot pose, contact, tool mode, sensor
  validity, patient effect, and operator intervention.
- Detect missing or stale frames and record them as evidence limitations.
- Keep complete episodes rather than phase labels or highlight clips.

Training probes:

- Drop frames, alter latency, corrupt a channel, or desynchronize video and
  force. Require the system to mark the interval invalid rather than infer a
  safe outcome.

### Chapter 14 - Laparoscopy During Pregnancy

Source: pp. 173-181.

Knowledge extracted:

- Indication, timing, maternal/fetal condition, positioning, venous-thrombo-
  embolism prevention, access, insufflation pressure, and intraoperative
  monitoring must be considered together.
- Port placement changes with uterine size and gestational stage; the entry
  route should avoid the uterus and use direct visualization when appropriate.
- Positioning and pneumoperitoneum can affect maternal hemodynamics and fetal
  physiology; monitoring and anesthesia coordination are part of the operation.
- The chapter discusses procedures such as appendectomy and cholecystectomy in
  pregnancy and highlights modified access and exposure rather than a wholly
  different operation.

Dr.Anmar translation:

- Treat pregnancy as a distinct patient-physiology and geometry variant, not a
  random visual texture.
- Parameterize safe pressure, positioning, uterine clearance, monitoring, and
  abort criteria. Do not hard-code the chapter's source-era values as universal
  limits.

Training probes:

- Vary gestational geometry, access site, insufflation tolerance, visibility,
  and physiological response. Score whether the system adapts or escalates.

### Chapter 15 - Previous Abdominal Surgery

Source: pp. 183-189.

Knowledge extracted:

- Prior operations alter access risk, port placement, adhesions, anatomy,
  dissection planes, and the likelihood of bowel or abdominal-wall injury.
- Preoperative assessment should include prior incisions, operative reports,
  expected adhesions, and the intended procedure.
- Entry may be moved away from scars or suspected adhesions; open access or a
  different route may be safer.
- Adhesiolysis should restore working space while limiting unnecessary injury;
  bleeding and enterotomy are important failure modes.
- The surgeon should convert when safe progress is not possible or anatomy
  cannot be reliably identified.

Dr.Anmar translation:

- Generate patient variants from revision-bound adhesion maps and prior-port
  histories.
- Put the prior-surgery context in the observation and require a changed access
  plan, not merely a harder policy score.

Training probes:

- Compare clean, single-operation, and multiply operated abdomens. Measure
  entry injury, adhesiolysis path, tissue damage, working-space recovery, and
  conversion decisions.

### Chapter 16 - Robotics in Laparoscopic and Thoracoscopic Surgery

Source: pp. 191-205.

Knowledge extracted:

- The chapter distinguishes image-guided systems, computer-enhanced
  telesurgery, and systems described as true robotics, while reviewing current
  cardiac, general, urologic, gynecologic, and thoracic applications.
- Advantages include stable vision, motion scaling, tremor filtering, and
  articulated instruments; limitations include cost, setup, lack of or limited
  tactile feedback, training requirements, communication/latency, and the need
  for credentialing.
- Robotic capability does not remove the need for anatomy, exposure,
  complication management, or conversion planning.
- Emerging technologies should be evaluated by patient safety and outcomes,
  not novelty or mechanical sophistication.

Dr.Anmar translation:

- Keep high-level intent, local skill policy, safety shield, deterministic
  controller, and patient-effect monitor separate.
- Measure transport latency, action scaling, controller clipping, takeover,
  haptic/visual uncertainty, and operator override.

Training probes:

- Run the same skill with direct teleoperation, delayed teleoperation, learned
  residual, and autonomous proposal-plus-safety-shield modes.
- Introduce network delay, camera drift, tool saturation, and forced handover.

## Part II - diagnostic laparoscopy and biopsy

### Chapter 17 - Emergency Laparoscopy

Source: pp. 207-213.

Knowledge extracted:

- Emergency laparoscopy can address selected abdominal pain, obstruction,
  peritonitis, abscess/drainage, and trauma questions, but hemodynamic
  instability or clear major injury may require immediate open exploration.
- Diagnostic exploration should be systematic, include bowel manipulation and
  all relevant spaces, and should not be considered complete merely because a
  focal abnormality was found.
- A 30-degree and 0-degree scope, multiple monitors, and additional trocars
  may be needed for full inspection and manipulation.
- In trauma, mechanism, hemodynamics, imaging, peritoneal violation, diaphragm
  injury, bowel injury, and the possibility of occult thoracic injury influence
  the choice of laparoscopy versus laparotomy.
- Complications include hypothermia, pneumothorax, acidosis/cardiac effects,
  missed injury, and delayed conversion.

Dr.Anmar translation:

- Build an emergency task with time pressure but hard uncertainty and
  instability terminals.
- Separate exploration completeness from target detection and require explicit
  conversion/handover when the simulator cannot establish safety.

Training probes:

- Use stable pain, obstruction, peritonitis, blunt trauma, stab wound, and
  suspected diaphragm injury variants. Score coverage, missed injury, time to
  escalation, and unnecessary manipulation.

### Chapter 18 - Elective Diagnostic Laparoscopy and Cancer Staging

Source: pp. 215-229.

Knowledge extracted:

- Diagnostic/staging laparoscopy complements CT, ultrasound, MRI, and other
  imaging; it is used to inspect peritoneal disease, ascites, liver lesions,
  organ surfaces, and resectability questions.
- Techniques include systematic inspection, peritoneal lavage/cytology,
  directed biopsy with forceps or core needle, fine-needle aspiration, and
  laparoscopic ultrasound.
- Biopsy should be clean and representative without crushing tissue, and
  hemostasis should be prepared before sampling vascular or liver lesions.
- Staging plans differ for esophageal, gastric, liver, and pancreatic tumors;
  the procedure is a decision-support examination, not a visual checklist
  detached from treatment planning.
- LUS can find small lesions, inspect the biliary tree and vessels, define
  pancreatic/periampullary disease, and guide biopsy.

Dr.Anmar translation:

- Create an inspection-coverage map with landmarks, lesions, uncertainty,
  sample identity, sample site, and post-biopsy bleeding state.
- Treat the specimen as a tracked object with provenance and tissue integrity.

Training probes:

- Present small, low-contrast, multiple, or misleading lesions. Require a
  systematic survey, confidence report, safe biopsy, and complete sample
  record.

### Chapter 19 - Lymph Node Biopsy, Dissection, and Staging Laparoscopy

Source: pp. 231-253.

Knowledge extracted:

- Indications depend on tumor site, staging need, accessible node location,
  prior imaging, and whether less invasive sampling is adequate.
- Preparation can include bowel/bladder preparation, antibiotics, thrombo-
  embolism prevention, and a position selected for the node basin.
- Port and instrument choices vary for retrogastric, para-aortic, iliac,
  pelvic, mediastinal, and other dissections. Ultrasound and energy devices are
  adjuncts.
- Node dissection requires identification of major vessels, ureters, nerves,
  bowel, and lymphatic channels; specimen orientation and hemostasis matter.
- Complications include vascular/organ injury, bleeding, lymphatic leak,
  chylous collection, nerve injury, and incomplete staging.

Dr.Anmar translation:

- Make node sampling a constrained target-acquisition task with a protected
  anatomy graph, sample-quality state, and lymph-flow outcome.
- Add staged difficulty by basin and exposure rather than by arbitrary reward.

Training probes:

- Place target nodes beside vessels, ureters, nerves, bowel, or lymphatic
  channels. Require representative orientation, hemostasis, and a stop or
  alternate route when the target cannot be isolated safely.

## Part III - laparoscopic cholecystectomy and common duct exploration

### Chapter 20 - Laparoscopic Cholecystectomy

Source: pp. 255-264.

Knowledge extracted:

- The chapter covers symptomatic/complicated gallstone disease, selected
  acalculous disease, dyskinesia, polyps, and relative contraindications.
- Position, reverse Trendelenburg/right-side-up exposure, monitor placement,
  stomach emptying, and four-port geometry support the operation.
- The critical view of safety requires clearing the peritoneal coverings and
  fat, identifying the relevant structures, and seeing only two structures
  entering the gallbladder with the cystic plate exposed before division.
- The cystic duct and artery are clipped and divided only after identification;
  the gallbladder is dissected from the liver bed, inspected for hemostasis, and
  extracted directly or in a retrieval bag.
- Difficult cases include a tense/inflamed gallbladder, contracted or large
  stone-filled gallbladder, indurated Calot triangle, and dense adhesions.
  Decompression, suction/irrigation, cholangiography, dissection close to the
  gallbladder, subtotal/fundus-first approaches, or conversion may be used.

Dr.Anmar translation:

- Implement `critical_view_confirmed` as an environment-owned geometric
  predicate requiring target structures, cystic plate, and protected anatomy
  evidence; it cannot be an action parameter.
- Create a cholecystectomy skill graph with reversible checkpoints before clip,
  divide, bed dissection, and extraction.

Training probes:

- Normal anatomy, short/wide duct, inflammation, anomalous anatomy, obscured
  view, bleeding, and failed critical-view attempts. Score abstention and
  conversion as successful safety behavior where appropriate.

### Chapter 21 - Laparoscopic Cholecystectomy: Avoiding Complications

Source: pp. 265-272.

Knowledge extracted:

- Important complications are hemorrhage, common-bile-duct injury, bile leak,
  gallbladder perforation, bowel injury, and retained common-duct stone.
- Risk rises with inflammation, short or wide ducts, prior upper-abdominal
  surgery, obesity, anatomic variation, cirrhosis, fistula, malignancy, and
  experience.
- Bleeding requires source identification; blind clipping in Calot's triangle
  can injure the right hepatic artery or bile ducts. Lens cleaning, suction,
  additional access, temporary compression, precise isolation, and low
  threshold for laparotomy are emphasized.
- Bile-duct injury is prevented by clear anatomy and selective imaging; bile
  leaks require recognition, drainage/flow assessment, and escalation.
- Gallbladder perforation and stone spillage require controlled suction,
  retrieval, irrigation, and awareness of retained material.

Dr.Anmar translation:

- Build complication variants around the exact pre-failure state rather than
  just random terminal labels.
- Give the outcome monitor bleeding rate, duct continuity, bile flow/leak,
  stone/specimen inventory, and visibility state.

Training probes:

- Inject obscured Calot anatomy, active bleeding, duct variation, bile leak,
  bowel contact, and spilled stones. Score exposure, no-blind-clipping,
  retrieval, verification, and conversion or handover.

### Chapter 22 - Cholangiography

Source: pp. 273-289.

Knowledge extracted:

- Intraoperative cholangiography is used selectively for unclear anatomy,
  suspected duct injury, duct dilation, stones/sludge, prior altered anatomy,
  or other pre/intraoperative risk signals.
- The cystic duct is dissected, cannulated with an appropriate catheter,
  connected to saline/contrast, and injected while the ductal system is
  visualized. Catheter depth, seal, flow, pressure, and contrast volume affect
  interpretability.
- The examination should identify the cystic/common/proximal ducts and distal
  passage; overfilling, poor positioning, sphincter behavior, or incomplete
  proximal filling can mislead interpretation.
- If cannulation fails, alternative imaging or direct intervention may be
  required rather than forceful catheterization.
- Complications include duct injury, biliovenous reflux under pressure,
  pancreatitis, and contrast reaction.

Dr.Anmar translation:

- Model the catheter, duct lumen, contrast particles, pressure, seal, and image
  coverage as linked evidence. A contrast command alone cannot verify anatomy.
- Train interpretation under underfill, overfill, artifact, short duct,
  abnormal junction, and incomplete proximal filling.

Training probes:

- Present a usable study, failed cannulation, leak at the catheter seal,
  underfill, overfill, biliovenous reflux, and incomplete proximal filling.
  Require confidence calibration and an alternative imaging or escalation path.

### Chapter 23 - Laparoscopic Ultrasound of the Biliary Tree

Source: pp. 291-310.

Knowledge extracted:

- LUS is presented as a radiation-free method for finding common-duct stones,
  defining anatomy, evaluating masses, and complementing cholangiography.
- The operator learns to follow the common bile duct from the hepatic confluence
  toward the ampulla and use the hepatic artery, portal vein, gallbladder,
  liver, pancreas, and duodenum as landmarks.
- Stones create echogenic interfaces and posterior shadowing; Doppler helps
  distinguish vascular flow from non-flowing ducts.
- The duodenum can act as an acoustic window when lightly compressed to displace
  air; fat and poor contact limit penetration. Probe angle, pressure, and
  transverse/longitudinal scanning matter.
- Variants and other pathology may be found in the liver, pancreas,
  gallbladder, nodes, and ducts. Novice recognition requires a learning curve.

Dr.Anmar translation:

- Add registered probe pose, contact pressure, acoustic coupling, Doppler
  validity, scan plane, landmark coverage, and artifact state to the sensor
  contract.
- Make `anatomy_confirmed` depend on a completed scan path and multiple
  landmarks, not a single image classification.

Training probes:

- Vary air, fat, poor coupling, probe pressure, stone shadowing, and vessels
  near the duct. Require a multi-landmark sweep, valid Doppler interpretation,
  and abstention when the acoustic window is inadequate.

### Chapter 24 - Laparoscopic Common Bile Duct Exploration: Transcystic Duct
Approach

Source: pp. 311-330.

Knowledge extracted:

- Transcystic exploration is selected based on duct anatomy, stone burden,
  cystic-duct access, and the ability to clear the duct without a
  choledochotomy.
- The equipment chain includes a cholangiography/choledochoscopy system,
  guidewire/catheter, baskets or extraction tools, irrigation, and imaging.
- The procedure requires exposure, cystic-duct opening/cannulation, guidewire
  or scope control, stone localization, extraction, and confirmation of duct
  clearance and drainage.
- Difficult anatomy, impacted stones, duct injury, failed cannulation,
  retained stones, and high-pressure irrigation are explicit failure modes.

Dr.Anmar translation:

- Build a tool-chain task in which each action changes the physical state of
  the catheter, wire, stone, duct, and flow field.
- Verify success through stone inventory, duct continuity, flow, pressure,
  residual obstruction, and absence of injury.

Training probes:

- Vary cystic-duct access, stone burden, impacted distal stones, failed
  cannulation, duct diameter, and irrigation pressure. Require clearance and
  drainage evidence, not a visually plausible basket motion.

### Chapter 25 - Laparoscopic Common Bile Duct Exploration via Choledochotomy

Source: pp. 331-344.

Knowledge extracted:

- Choledochotomy is a separate route used when transcystic access is unsuitable
  or duct exploration requires a larger/direct opening.
- The key steps are exposure and identification of the duct, controlled
  incision, choledochoscopic or instrument exploration, stone retrieval,
  irrigation/confirmation, and closure or drainage appropriate to the case.
- Duct wall, vascular structures, instrumentation angle, closure integrity,
  and postoperative leak/patency must be protected.
- Conversion or an alternative drainage/reconstruction route is appropriate
  when the duct cannot be safely cleared or closed.

Dr.Anmar translation:

- Represent incision topology, duct-wall continuity, stone/object state,
  irrigation pressure, closure integrity, leak, and patency as separate effect
  variables.

Training probes:

- Vary stone burden, distal obstruction, duct diameter, wall quality, and
  closure quality. Require post-repair leak and flow verification.

## Part IV - basic laparoscopic gastric surgery

### Chapter 26 - Laparoscopic Gastrostomy

Source: pp. 345-352.

Knowledge extracted:

- Laparoscopic gastrostomy provides feeding or decompression when PEG is not
  possible or when direct visual selection and fixation are important.
- Site choice must confirm stomach rather than colon, avoid tension, reach the
  abdominal wall comfortably, and preserve blood supply.
- A simple gastrostomy uses gastric insufflation, T-fasteners, a central needle
  and guidewire, serial dilation, tube placement, balloon inflation, and
  approximation of stomach to the abdominal wall.
- A Janeway-style mucosa-lined tube uses a stapled gastric fold and requires
  adequate lumen, blood supply, stoma maturation, and leak testing.
- Complications include gastric leakage, perforation from tension/cautery,
  and stoma necrosis from a narrow or poorly perfused tube.

Dr.Anmar translation:

- Model organ-to-wall apposition, T-fastener attachment, tube lumen, balloon,
  pressure, flow, leak, and tissue perfusion as one coupled but inspectable
  process.
- Require dye/contrast or flow-based verification before declaring a tube safe.

Training probes:

- Vary stomach-versus-colon exposure, off-axis needle entry, excessive tension,
  poor perfusion, and post-release leak. Require safe-site confirmation,
  apposition, flow, and a recovery path.

### Chapter 27 - Laparoscopic Plication of Perforated Ulcer

Source: pp. 353-360.

Knowledge extracted:

- The operation is built around controlled exploration, lavage, exposure of the
  perforation, assessment of size/location/cause, and closure with a patch or
  plication when technically suitable.
- If the liver has sealed the defect, the seal should not be disrupted before
  exploration and contamination control are planned.
- Fibrin and contamination are removed with suction/irrigation; an extra
  retractor may be required for exposure.
- Large, poorly defined, posterior, or suspicious perforations may be unsafe
  to plicate laparoscopically and should trigger conversion or another repair.
- Repair is followed by leak testing and contamination surveillance.

Dr.Anmar translation:

- Add contamination particles, fluid balance, defect topology, patch/suture
  contact, leak, and lavage evidence to the patient effect model.
- Treat blind closure or uncontrolled lavage as a failure even if the defect is
  visually covered.

Training probes:

- Present a sealed perforation, exposed anterior defect, large or posterior
  defect, unclear tissue, and uncontrolled contamination. Require exploration,
  controlled lavage, repair or conversion, and a leak test.

## Part V - small intestine, appendix, and colon

### Chapter 28 - Small Bowel Resection, Enterolysis, and Enteroenterostomy

Source: pp. 361-378.

Knowledge extracted:

- The chapter combines exposure, bowel handling, adhesiolysis, segmental
  resection, mesenteric control, anastomosis, and postoperative leak/bleeding
  recognition.
- Adhesiolysis is often the most dangerous step in a previously operated or
  tumor-involved abdomen. Correct plane, sharp dissection, traction direction,
  and conversion judgment matter.
- Bowel viability, mesenteric vessels, orientation, lumen alignment, tension,
  blood supply, and preservation of the remaining bowel are central.
- Anastomosis may use stapled or sutured methods; the closure must be complete,
  patent, perfused, and free of twist or leak.
- Enterotomy, mesenteric bleeding, postoperative hemorrhage, obstruction,
  anastomotic leak, and partial-thickness injury are major failure modes.

Dr.Anmar translation:

- Make bowel segments, mesentery, vessels, lumen, contents, and continuity
  explicit topology objects.
- A policy must verify both local repair and global continuity/patency; a
  partial-thickness injury may be more dangerous than its appearance suggests.

Training probes:

- Clean bowel, adhesions, inflammatory mass, ischemic segment, hidden enterotomy,
  stapled gap, and twisted anastomosis. Score tissue preservation and recovery.

### Chapter 29 - Laparoscopic Placement of Jejunostomy Tube

Source: pp. 379-388.

Knowledge extracted:

- Jejunostomy is used when the proximal GI route is unavailable or unsafe but
  the intestine can receive enteral nutrition, including after major upper-GI
  reconstruction or selected trauma/pancreatitis cases.
- The target is found near the ligament of Treitz and a convenient jejunal loop
  is brought to the abdominal wall without kinking.
- Four anchoring sutures are placed in a diamond, a needle and guidewire enter
  the lumen, serial dilators create the tract, and the tube is advanced under
  direct vision.
- The balloon should be snug but not overinflated; fixation is tied with
  laparoscopic visualization, and flow is tested with saline or dye.
- Complications include posterior-wall perforation, obstruction from an
  overinflated balloon, leakage from poor fixation/injury, and dislodgment.

Dr.Anmar translation:

- Model the lumen, posterior wall, wire, dilator, tube, balloon, abdominal-wall
  attachment, kink, flow, and leak as separate evidence channels.
- Make safe insertion depend on tip depth and posterior-wall visibility.

Training probes:

- Vary the jejunal loop, anchor spacing, off-axis needle, posterior-wall
  contact, balloon volume, kink, dislodgment, and dye leak. Require tip-depth,
  fixation, flow, and no-perforation evidence.

### Chapter 30 - Laparoscopic Appendectomy

Source: pp. 389-402.

Knowledge extracted:

- The operation includes access and port placement, exploration, appendix
  identification and mobilization, mesoappendix control, base control, division,
  specimen retrieval, and inspection for contamination/bleeding.
- Port locations adapt to the appendix position, body habitus, prior surgery,
  pregnancy, and need for triangulation.
- The appendix may be retrocecal, pelvic, inflamed, perforated, or adherent;
  mobilization and protected structure awareness are required.
- Mesoappendix vessels can be divided with clips, energy, stapler, or other
  bounded methods. The stump must be secure, viable, and non-leaking.
- Complications include trocar injury, bleeding, stump leak, abscess, wound
  infection, bowel injury, retained specimen, and missed alternate pathology.

Dr.Anmar translation:

- Make appendix identification a landmark/coverage task, not a segmentation
  shortcut. Require cecum, base, terminal ileum, and surrounding bowel context.
- Track stump integrity, contamination, specimen identity, and retrieval.

Training probes:

- Include retrocecal, pelvic, perforated, adherent, and friable appendices plus
  hidden stump bleeding and a dropped specimen. Require cecal context, secure
  stump, contamination control, and retrieval or handover.

### Chapter 31 - Laparoscopic Colostomy

Source: pp. 403-412.

Knowledge extracted:

- Colostomy creation requires preselected skin marking, mobilization of the
  appropriate colon, adequate length without tension, mesenteric orientation,
  vascular preservation, and a safe abdominal-wall route.
- The site should be reachable and visible, away from folds/scars/pressure
  points, and positioned so the stoma can function and be managed.
- The bowel is brought through the wall without twist or excessive constriction,
  and the stoma is matured with viable mucosa.
- Failure modes include ischemia, retraction, stenosis, parastomal hernia,
  obstruction, contamination, and incorrect segment/orientation.

Dr.Anmar translation:

- Create a stoma-placement task with skin topology, bowel path, mesenteric
  orientation, perfusion, aperture, tension, and output flow.
- Verify function after release, not only physical exteriorization.

Training probes:

- Vary marked and unmarked sites, short mesentery, bowel twist, tight aperture,
  poor perfusion, and early obstruction. Require a reachable viable stoma with
  correct orientation and a post-release output check.

### Chapter 32 - Laparoscopic Inguinal Hernia Repair: TAPP and TEP

Source: pp. 413-430.

Knowledge extracted:

- TAPP enters the peritoneal cavity and opens the peritoneum; TEP develops a
  preperitoneal plane. Both require knowledge of the myopectineal orifice,
  inferior epigastric vessels, cord structures, pubic/Cooper anatomy, and
  danger zones.
- Exposure and dissection must create a broad plane, reduce the sac, preserve
  cord/vas/vessels, and provide adequate mesh overlap.
- Mesh fixation can use tacks, glue, or no fixation depending on the technique;
  tacks near the triangle of pain or vascular danger can injure nerves or
  vessels.
- Peritoneal tears in TEP can lose working space and allow bowel herniation or
  mesh exposure; closure, decompression, or conversion to TAPP are possible.
- Complications include vascular, bladder, bowel, nerve, cord/testicular,
  chronic-pain, mesh, seroma, recurrence, and urinary complications.

Dr.Anmar translation:

- Build a no-go anatomy graph and a plane-maintenance state; mesh success
  requires coverage, overlap, fixation/adhesion, and absence of exposed bowel.
- Score protected-structure clearance and chronic-effect proxies, not only
  immediate defect coverage.

Training probes:

- Vary TAPP/TEP plane quality, peritoneal tears, sac reduction, cord structures,
  triangle-of-pain/vascular danger zones, mesh overlap, and fixation choice.
  Require plane recovery or conversion and reject unsafe coverage.

### Chapter 33 - Laparoscopic Repair of Ventral Hernia

Source: pp. 431-442.

Knowledge extracted:

- Patient selection balances reduced wound morbidity against the limitations of
  bridging mesh and abdominal-wall function; defect size, age, body habitus,
  and activity matter.
- Incarcerated hernias require controlled reduction without bowel injury.
  Multiply operated abdomens demand careful adhesiolysis and prior-mesh review.
- Suprapubic repair requires bladder localization; subxiphoid repair requires
  falciform mobilization and protection of the pericardium/diaphragm.
- The repair requires defect measurement, adequate mesh overlap, correct
  orientation, fixation, and inspection of the entire abdominal wall.
- Complications include enterotomy, seroma, mesh infection, recurrence,
  fixation pain, bowel injury, and missed defects.

Dr.Anmar translation:

- Make a large-area coverage task in which mesh placement is an object-to-wall
  registration problem with overlap, tension, fixation, and protected anatomy.
- Use a defect/mesh topology receipt and test for edge lift, migration, and
  adjacent-bowel contact.

Training probes:

- Include incarcerated contents, adhesions, missed defects, suprapubic bladder,
  subxiphoid/falciform hazards, enterotomy, mesh edge lift, and bowel contact.
  Require reduction and inspection before declaring coverage complete.

## Part VI - hernia repair

## Part VII - pediatric laparoscopy and endoscopy

### Chapter 34 - Pediatric Minimally Invasive Surgery: General Considerations

Source: pp. 443-448.

Knowledge extracted:

- Pediatric MIS magnifies the importance of ergonomics because the cavity,
  abdominal wall, and working distances are smaller.
- A baseball analogy is used for layout: camera at home plate, target at second
  base, working ports at first and third, and monitors behind the target.
- Port diameter, instrument length, stapler size, scope image quality, and
  access technique must match the child's size and procedure.
- Children have thinner abdominal walls and smaller safety margins; Veress and
  trocar entry require controlled traction, direct observation, and caution.
- Insufflation tolerance is weight- and physiology-dependent; closure of ports
  that would be left open in adults is more important in children.

Dr.Anmar translation:

- Build a pediatric scale/physiology variant with explicit weight, cavity depth,
  port diameter, pressure tolerance, and closure contracts.
- Never pool pediatric and adult scores without stratification.

Training probes:

- Stratify body size and cavity depth while varying port geometry, wall injury,
  insufflation response, and closure needs. Require an adapted layout and
  pressure/entry policy rather than an adult policy scaled by coordinates.

### Chapter 35 - Pediatric Minimally Invasive Surgery: Specific Procedures

Source: pp. 449-478.

Knowledge extracted:

- Pediatric appendectomy uses small-port access, suction/irrigation, mesoappendix
  control, stump security, and specimen handling; perforation changes the
  contamination and abscess risk.
- Pediatric cholecystectomy follows the adult anatomy but adapts ports and
  instruments; cystic duct imaging may be needed when anatomy is unclear.
- Splenectomy emphasizes splenic hilar control, short gastric division,
  accessory spleen search, specimen retrieval, and postsplenectomy infection
  prevention.
- Pediatric adrenalectomy, contralateral inguinal exploration, pyloromyotomy,
  fundoplication, undescended-testis exploration, Hirschsprung pull-through,
  Meckel diverticulum, and intussusception each demonstrate a different
  combination of small anatomy, viability, reconstruction, and conversion.
- Pyloromyotomy requires complete muscle separation without mucosal injury;
  fundoplication requires wrap geometry without obstruction or torsion.
- Pull-through and bowel procedures require biopsy/landmark confirmation,
  mesenteric preservation, pelvic dissection, and anastomosis.
- Thoracoscopy for empyema illustrates access, irrigation, decortication,
  drainage, lung protection, and the need to recognize a nonresponsive case.

Dr.Anmar translation:

- Use pediatric chapters as a family of scale-conditioned tasks, not one
  generic small-body model.
- Separate access, tissue-plane, viability, and reconstruction tasks so that
  errors can be diagnosed.

Training probes:

- Rotate appendectomy, cholecystectomy, splenectomy, pyloromyotomy,
  fundoplication, pull-through, Meckel/intussusception, and thoracoscopy
  variants. Require procedure-specific landmarks, viability, reconstruction,
  and conversion evidence.

### Chapter 36 - Complications in Pediatric MIS

Source: pp. 479-496.

Knowledge extracted:

- Veress and trocar injuries are more dangerous because the cavity and wall are
  small; immediate inspection after access is a required safety routine.
- Trocar-site hernias, abdominal-wall hemorrhage, CO2 crepitus, hypercarbia,
  abdominal compartment syndrome, and tension pneumothorax are important
  general complications.
- Insufflation pressure must be adapted to weight and physiologic response;
  desufflation, lower pressure, or abandoning MIS may be required.
- Procedure-specific complications include appendiceal stump leak/abscess,
  bile-duct injury/leak, splenic and pancreatic/gastric/colon injury,
  specimen spill, postsplenectomy sepsis, pyloromyotomy mucosal tear or
  incomplete myotomy, fundoplication volvulus/tear/perforation, and injury in
  pull-through/intussusception/empyema cases.
- The pattern is consistent: detect early, inspect, repair or drain, and
  convert when the minimally invasive route cannot provide reliable control.

Dr.Anmar translation:

- Create a pediatric physiology monitor with pressure, CO2, ventilation,
  hemodynamics, wall injury, and recovery state.
- Promote only policies that recognize both mechanical and physiologic failure.

Training probes:

- Inject trocar injury, hypercarbia, crepitus, compartment physiology,
  pneumothorax, stump leak, mucosal tear, wrap injury, or specimen spill.
  Require immediate detection, stabilization, inspection, and escalation.

## Part VIII - flexible endoscopy: general principles

### Chapter 37 - Flexible Endoscopes: Characteristics, Troubleshooting, and Equipment Care

Source: pp. 497-507.

Knowledge extracted:

- Fiberoptic scopes transmit images through coherent fiber bundles and show
  characteristic resolution/fiber-break artifacts; videoendoscopes use a CCD
  chip and processor for improved display and shared viewing.
- Narrow-band/multiband imaging, chromoendoscopy, endomicroscopy, and
  endoscopic ultrasound change the information content and require separate
  interpretation contracts.
- Scope categories differ in length, diameter, view direction, channel size,
  and intended anatomy: gastroscopes, enteroscopes, duodenoscopes,
  choledochoscopes, echoendoscopes, colonoscopes, sigmoidoscopes, and NOTES
  systems.
- Working channels support instruments, suction, air, and water; controls map
  to up/down and right/left tip deflection, torque, shaft stiffness, and in
  some scopes an elevator.
- Illumination, air/water, suction, valves, cables, water bottle, connectors,
  processor, and recording must be tested before use.
- Endoscopes are fragile; common problems include sticky valves, no water,
  loose connections, leaks, poor image, light issues, broken fibers, and
  damaged channels. Cleaning and reprocessing are equipment-safety functions.

Dr.Anmar translation:

- Represent endoscope type, channel topology, tip control, view direction,
  image mode, optical artifact, and equipment validity explicitly.
- Train failure diagnosis and safe withdrawal before therapeutic tasks.

Training probes:

- Swap fiberoptic and video scopes and inject broken fibers, poor light,
  channel leaks, sticky valves, failed water, loose connections, and stale
  recording. Require diagnosis, safe withdrawal, and a valid-equipment receipt.

### Chapter 38 - Endoscopy Handling

Source: pp. 509-523.

Knowledge extracted:

- The room should provide monitors, oxygen, suction, noninvasive monitoring,
  airway access, enough space, a travel cart, and a trained assistant.
- Upper and lower endoscopy use different patient/endoscopist positions; the
  monitor should remain in direct line of sight and monitoring/airway access
  must not be blocked.
- The left hand controls the headpiece, large wheel, air/water, suction, and
  image buttons; the right hand advances, withdraws, torques, stiffens, and
  accesses the working channel.
- The large wheel's apparent monitor motion is inverse to the physical tip
  motion; the suction port is at a predictable screen position and can capture
  mucosa if the tip is poorly oriented.
- Basic maneuvers include controlled insertion, lumen-in-view advancement,
  gentle tip deflection, torque, shaft stiffening, retroflexion, shortening a
  looped scope, and controlled withdrawal with circumferential inspection.
- Sharp tip angulation and blind push can create paradoxical motion, loss of
  the lumen, mucosal trauma, or perforation.

Dr.Anmar translation:

- Do not reuse the laparoscopic action ABI for endoscopy. Use action primitives
  for insertion length, up/down, left/right, torque, stiffness, insufflation,
  irrigation, suction, channel tool, and retroflexion.
- Make the lumen, wall, scope curve, loop state, tip velocity, and pressure
  first-class observations.

Training probes:

- Test inverse-control understanding, suction orientation, loop formation,
  retroflexion, paradoxical motion, withdrawal inspection, and a required
  withdrawal when the lumen is lost.

### Chapter 39 - Monitoring, Sedation, and Recovery

Source: pp. 525-529.

Knowledge extracted:

- Pre-procedure assessment includes organ disease, airway/sleep-apnea risk,
  medications, allergies, prior sedation response, last meal, substance use,
  vital signs, airway anatomy, and physical-status class.
- Emergency airway/cardiac equipment and trained assistance must be available.
- During endoscopy, heart rate, blood pressure, respiratory rate, oxygen
  saturation, and consciousness are monitored repeatedly; oxygen saturation
  alone does not exclude hypercapnia.
- Moderate sedation aims for meaningful response without airway support;
  deeper sedation may require different expertise and monitoring.
- Benzodiazepines, opioids, propofol, topical anesthetics, and reversal agents
  have different onset, duration, respiratory, circulatory, and recovery
  behavior. Reversal can wear off before the original sedative effect.
- Recovery requires continued observation of ventilation, oxygenation,
  circulation, and consciousness before discharge.

Dr.Anmar translation:

- Add patient monitoring and sedation state to endoscopic tasks. A procedure is
  not successful if the model's patient is physiologically unstable.
- Make recovery a state machine with alarms, airway support, reversal, delay,
  escalation, and discharge readiness.

Training probes:

- Inject hypoventilation, hypercapnia with normal oxygen saturation, hypotension,
  oversedation, and delayed recurrence after reversal. Score detection and
  response, not just endoscope completion.

### Chapter 40 - Flexible Endoscopy: Principles of Documentation

Source: pp. 531-535.

Knowledge extracted:

- Documentation supports continuity, accountability, quality assurance,
  research, and comparison with future examinations.
- Consent is a process that records patient/team, purpose, benefits,
  alternatives, risks, consequences, and agreement.
- Nursing records include timeout, identification, antibiotics, sedation,
  oxygenation, vital signs, and chronological events.
- The procedure note includes diagnoses, operator/team, anesthesia and doses,
  scope/instruments, indication, extent of examination, landmarks, adequacy
  of visualization/preparation, interventions, and diagnostic impression.
- Images and video should be linked to findings, interventions, and landmarks;
  capsule/virtual modalities have different evidence and storage paths.

Dr.Anmar translation:

- Make documentation a required output of every episode, not an afterthought.
- Store landmark coverage, image validity, intervention coordinates, instrument
  identity, medication/sedation state, vital transitions, and unresolved views.

Training probes:

- Drop a landmark image, desynchronize an intervention timestamp, omit a dose,
  or leave a blind region. Require the system to mark the record incomplete and
  hand over the unresolved evidence rather than fabricate continuity.

## Part IX - flexible upper GI endoscopy

### Chapter 41 - Diagnostic Upper Gastrointestinal Endoscopy

Source: pp. 539-556.

Knowledge extracted:

- EGD evaluates symptoms, surveillance, bleeding, varices, ulcers,
  malabsorption, and postoperative anatomy; therapeutic uses include bleeding
  control, foreign-body removal, polyp removal, dilation, feeding/drainage,
  variceal therapy, and palliation.
- Preparation includes fasting, consent, medication/anticoagulation review,
  monitoring, airway access, topical anesthesia, and gastric aspiration/lavage
  when retained contents or blood are expected.
- Safe insertion is midline and visually guided through the pharynx and upper
  esophageal sphincter; gentle swallowing and withdrawal are safer than force.
- Advancement through the esophagus, stomach, pylorus, duodenal bulb, and
  second/third duodenum uses lumen-in-view, torque, small deflections,
  straightening, and controlled insufflation.
- Complete inspection includes antegrade and retroflexed views of the cardia,
  fundus, incisura, stomach, duodenal bulb, and esophagus during withdrawal.
- Postoperative stomachs require prior operative/anatomy review and awareness
  of blind pouches, altered outlets, anastomoses, and complications.

Dr.Anmar translation:

- Build an upper-GI landmark graph with lumen validity, scope pose, wall
  contact, insufflation, suction, secretions/blood, and inspection coverage.
- Make the action planner choose withdraw/reorient/clean/aspirate rather than
  push into a lost lumen.

Training probes:

- Normal anatomy, retained blood, difficult upper sphincter, pylorospasm,
  postoperative anatomy, blind pouch, and retroflexion. Require landmarks in
  the record before success.

### Chapter 42 - Percutaneous Endoscopic Feeding Tube Placement

Source: pp. 557-570.

Knowledge extracted:

- PEG and related feeding access require suitable anatomy, fasting, sedation
  and monitoring, infection prevention, gastric insufflation, and safe site
  selection.
- Transillumination, one-to-one finger indentation, and especially the
  safe-tract needle test are used to exclude interposed bowel or other organs.
- Pull and push PEG methods require guidewire control, snare/traction, direct
  visualization, correct internal bumper engagement, and an external bumper
  that is snug but not tight. An introducer/Seldinger route is used when oral
  passage is difficult.
- Jejunal extensions and direct percutaneous endoscopic jejunostomy require
  pyloric/jejunal navigation, long-scope reach, fixation, and sometimes
  fluoroscopic confirmation.
- Complications include early dislodgment with gastric leakage/peritonitis,
  buried bumper, clogging, peritubal infection, pneumoperitoneum, bleeding,
  and inadvertent visceral injury.

Dr.Anmar translation:

- Treat safe-tract and apposition as hard insertion interlocks. The tube cannot
  create its own safe target.
- Verify internal position with direct visualization, contrast/flow, balloon
  state, wall apposition, and absence of leak.

Training probes:

- Interposed colon, poor transillumination, off-axis needle, lost guidewire,
  overly tight bumper, tube dislodgment, clogged tube, and leak after release.

### Chapter 43 - Capsule Enteroscopy

Source: pp. 571-580.

Knowledge extracted:

- Capsule endoscopy uses an ingestible camera to image mucosa through the small
  bowel, transmitting and storing images for later review.
- Its strengths are noninvasive coverage and access to bowel beyond routine
  upper/lower scopes; limitations include lack of steering, inability to biopsy
  or treat, variable transit, incomplete coverage, and retention risk.
- Patient selection, preparation, image review, localization, and recognition
  of obstruction/retention are part of the examination.
- Capsule findings require interpretation in temporal/spatial context and may
  require a different procedure for confirmation or therapy.

Dr.Anmar translation:

- Build a passive-sensor task with no direct action authority and explicit
  coverage, frame-quality, transit, localization, and retention states.
- Train the planner to escalate an image finding to a controllable diagnostic
  or therapeutic skill rather than pretend the capsule can intervene.

Training probes:

- Vary rapid transit, incomplete coverage, blur, retention, suspected lesion,
  and uncertain localization. Require a coverage report, retention escalation,
  and referral to a controllable follow-up procedure when intervention is needed.

## Part X - flexible lower GI endoscopy

### Chapter 44 - Flexible Sigmoidoscopy

Source: pp. 581-596.

Knowledge extracted:

- Flexible sigmoidoscopy is a limited distal-colon examination used for selected
  symptoms, bleeding, screening strategies, and local therapy; the required
  extent depends on age, risk, symptoms, and findings.
- Insertion methods include direct advancement, deflection/torque, and
  dither-torque to accordionize a fixed sigmoid. The lumen must remain visible;
  withdrawal to regain the lumen is safer than probing a suspected
  diverticulum.
- Withdrawal requires deliberate circumferential mucosal inspection; routine
  retroflexion is not always appropriate in a low-compliance rectum.
- Therapy includes biopsy, selected small-polyp removal, anastomotic-stricture
  dilation after diagnosis, and foreign-body retrieval. Energy may be unsafe
  when bowel preparation leaves combustible gas.
- Diverticulosis, pelvic adhesions, pain, bleeding, perforation, and inadequate
  cleaning can terminate or complicate the examination. Nonproductive effort
  should lead to anesthesia, laparoscopy, or another approach.

Dr.Anmar translation:

- Create an endoscope curriculum in which lumen visibility, loop shape, wall
  contact, torque, scope shortening, and withdrawal inspection are explicit.
- Teach a graded escalation: reorient, withdraw, reposition patient, use
  assistance, stop, or convert.

Training probes:

- Fixed sigmoid, diverticular false lumen, adhesion, foreign body, stricture,
  incomplete prep, and pain/physiology challenge. Score false-lumen entry and
  unnecessary force.

### Chapter 45 - Diagnostic Colonoscopy

Source: pp. 597-610.

Knowledge extracted:

- Colonoscopy provides complete mucosal inspection, lesion detection,
  localization, biopsy/resection, hemorrhage evaluation, inflammatory-disease
  assessment, and intraoperative lesion localization.
- Preparation, consent, sedation, room setup, monitoring, and patient position
  are part of the procedure.
- Advancement uses lubrication, minimal adequate insufflation, torque,
  deflection, withdrawal when the lumen is lost, and loop reduction. Pushing
  through a loop increases discomfort and perforation risk.
- The sigmoid, descending colon, splenic flexure, transverse colon, hepatic
  flexure, ascending colon, cecum, and terminal ileum each offer landmarks and
  failure patterns. External pressure and patient repositioning can help.
- Cecal completion requires landmark evidence such as appendiceal orifice,
  ileocecal valve, haustral convergence, or terminal-ileum view. Withdrawal
  must inspect circumferentially and include the low rectum/retroflexed view.
- Postsurgical anatomy includes anastomoses, blind pouches, stomas, and altered
  loop geometry; these must be identified before advancement.

Dr.Anmar translation:

- Use a coverage graph and landmark receipt rather than a binary `reached_cecum`
  signal.
- Measure loop energy, push/pull/torque sequence, wall contact, insufflation,
  mucosal coverage, withdrawal time, and missed regions.

Training probes:

- Include alpha loops, paradoxical movement, sharp flexures, diverticula,
  altered anastomoses, colostomy entry, and poor preparation. Require explicit
  acknowledgement of incomplete examination.

### Chapter 46 - Therapeutic Colonoscopy and Its Complications

Source: pp. 611-626.

Knowledge extracted:

- Therapy includes cold biopsy, hot forceps, snare polypectomy, endoscopic
  mucosal resection, specimen retrieval, decompression for pseudo-obstruction,
  volvulus reduction, and internal-hemorrhoid band ligation.
- Polyp size, shape, stalk, surface, and suspicion determine the tool and
  resection plan. Larger sessile lesions may require submucosal lift and
  staged/en-bloc/piecemeal decisions; suspicious lesions should be localized
  and documented.
- Retrieval is part of the intervention. Lost specimens are a procedural
  failure, not a clerical detail.
- Decompression requires minimal insufflation, suction/irrigation, mucosal
  viability inspection, serial response monitoring, and readiness for surgery.
  Sigmoid volvulus can be reduced endoscopically in selected cases; necrotic
  mucosa or failure mandates urgent surgery.
- Hemorrhoid banding requires target selection, suction into the cylinder, and
  a safe distance above the dentate line; pain, bleeding, thrombosis, and pelvic
  sepsis are complications.
- Colonoscopy complications include bleeding, perforation, infection, missed
  diagnosis, lost specimens, pain, and delayed recognition. Prevention depends
  on controlled manipulation, adequate visualization, inspection of blind
  spots, and timely escalation.

Dr.Anmar translation:

- Model resection as tissue inclusion -> energy/mechanical action -> specimen
  separation -> retrieval -> defect inspection -> bleeding/perforation
  surveillance.
- Model decompression as a patient-effect task with pressure, volume, flow,
  mucosal viability, and serial abdominal-state outcomes.

Training probes:

- Small and large polyps, sessile lesion, residual tissue, immediate/delayed
  bleeding, perforation, lost specimen, pseudo-obstruction, viable/nonviable
  volvulus, and painful/low ligation. Require recovery or handover.

## Appendix - SAGES reference and education map

Source: pp. 627-629.

Knowledge extracted:

- The appendix points readers to SAGES clinical/practice guidelines,
  privileging guidance, training guidance, troubleshooting charts, MIS and
  surgical-safety checklists, patient-information materials, educational
  videos, FLS, Grand Rounds, Pearls, and procedure collections.
- Its deeper lesson is governance: technical learning sits inside current
  guidelines, institutional privileges, supervised education, checklists,
  patient communication, and continuing review.
- The list is explicitly partial and dated to October 27, 2011. It is a map of
  resource categories, not a current clinical rule set or a set of permanent
  URLs.

Dr.Anmar translation:

- Make every scenario carry a source revision, applicable training level,
  privilege/supervision requirement, checklist version, and modernization
  status.
- Add resource provenance to demonstrations, rubrics, and promotion receipts;
  never treat a book citation or a passing simulator score as authorization for
  real-patient practice.

Training probes:

- Present a stale guideline, missing checklist, changed device warning,
  unprivileged operator, or incomplete consent record. Require the system to
  flag the dependency, stop promotion, and request current supervised review.

## Final Dr.Anmar knowledge contract

The book's complete training value can be implemented as five linked layers:

1. **Room and equipment layer** - setup, checks, ergonomics, monitoring,
   channel validity, and failure diagnosis.
2. **Access and exposure layer** - entry, working-space creation, camera,
   triangulation, retraction, insufflation, and protected anatomy.
3. **Interaction layer** - grasp, traction, dissection, energy, biopsy,
   approximation, clipping, stapling, suturing, tube placement, and endoscope
   navigation.
4. **Effect layer** - bleeding/flow, tissue damage, thermal field, leak,
   pressure, perfusion, patency, attachment, continuity, specimen identity,
   physiology, and recovery.
5. **Evidence layer** - landmark coverage, synchronized video/state, causal
   events, uncertainty, documentation, intervention provenance, and promotion
   receipts.

The minimum environment-owned evidence for a book-derived skill is:

```yaml
room: setup, equipment, channel validity, monitoring, faults
geometry: tool/scope pose, target, port/access topology, protected structures
interaction: contact, force, velocity, dwell, deformation, energy, pressure
state: lumen/continuity, attachment, leak, flow, perfusion, tissue damage
decision: intent, uncertainty, abstain/continue, recovery, handover
record: timestamps, landmarks, images, actions, outcomes, revisions, hashes
```

The minimum policy action is bounded intent and relative motion. The policy
does not own patient outcomes. The minimum promotion evidence is held-out
competence plus physical behavior, safety, recovery, and an immutable record.

## Source limitations and modernization backlog

The book is broad and technically useful, but it is not a current replacement
for specialty guidance. Before any Dr.Anmar scenario is labeled current or
clinically meaningful, separately update:

- device-specific warnings and energy systems;
- endoscopy cleaning/reprocessing and infection-control standards;
- sedation, anesthesia, monitoring, and rescue standards;
- pregnancy and pediatric physiology;
- cancer screening, staging, and therapeutic thresholds;
- hernia mesh, fixation, and reconstruction practice;
- current robotic, imaging, AI, and simulation evidence; and
- calibrated tissue, force, thermal, flow, and physiological parameters.

The correct final claim is that the book has been converted into a
chapter-traceable, physics-grounded simulation curriculum. It is not a claim
of clinical competence, calibrated biomechanics, or safe physical surgical
autonomy.
