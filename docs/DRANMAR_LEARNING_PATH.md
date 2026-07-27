# DrAnmar Learning Path

The DrAnmar Learning Path begins with one deliberately small skill: move a
single PSM tool tip to a commanded pose. It promotes a policy only after
measured success is repeatable, then adds coordination and physical contact in
controlled stages.

```mermaid
flowchart LR
    A["Single-tool pose control"] --> B["Dual-tool pose control"]
    B --> C["Block lift"]
    C --> D["Needle lift"]
    D --> E["Block handover"]
    E --> F["Needle handover"]
```

Stage 6 ends the rigid-object foundation. The next frontier is deformable
tissue, beginning with the canonical
[Needle-Ready Tissue Unit](DRANMAR_NEEDLE_READY_TISSUE.md). Its first task is
retained-needle approach to a sampled tissue entry frame; puncture, tract,
thread passage, and tissue damage remain blocked until topology and physical
force gates pass.

The versioned contract is
[`config/dranmar_learning_path.json`](../config/dranmar_learning_path.json).
Every task also has a stable `DrAnmar-*` Gym ID, so datasets, checkpoints, and
benchmark evidence do not depend on an inherited task name.

## First training target

Start with:

```text
DrAnmar-Reach-PSM-IK-Rel-v0
```

This stage is small enough to expose infrastructure and reward defects quickly:

- one PSM and one stable target per episode;
- bounded relative Cartesian actions;
- direct target-relative position and axis-angle orientation in the policy
  observation;
- normalized actor and critic observations;
- bounded coarse and fine pose rewards;
- an explicit position-and-orientation success envelope;
- immediate episode completion on success; and
- direct success measurement from Isaac Lab's `success` termination tensor.

It is a control qualification task, not a surgical-skill claim.

## Efficient training contract

Stages 1 and 2 do not spend PPO samples rediscovering relative Cartesian
controllers already encoded by the action space. Stage 1 uses one analytic
relative-IK base and Stage 2 uses one base controller per PSM. Their residual
networks start at zero, so the first frozen policies reproduce the controllers
exactly. PPO is reserved for bounded single-arm corrections or dual-arm
coordination residuals only when deterministic held-out evaluation misses the
stage gate.

The Stage 2 dual policy observes position and axis-angle orientation error for
both tool tips. Its 12-dimensional action is assembled from two independent
six-dimensional controller outputs before the learned coordination residual is
applied. A versioned 0.25 dual-loop gain bounds the physical IK delta after
action clipping. The observation offsets, controller scales, initialization
method, held-out seeds, and promotion threshold are versioned in the
learning-path contract.

Stage 2 is deliberately collision-disabled free-space control qualification.
It proves simultaneous two-tool kinematics without allowing collision impulses
to contaminate the controller measurement. It does not establish collision
avoidance or contact competence. Stage 3 turns contact back on and requires
measured contact, stability, force, slip, and drop evidence.

Stage 3 uses an analytic approach-grasp-lift base with a bounded learned
residual. The block begins at a collision-clear 15 mm root height, the
commanded target is 8 cm, and success
requires the object to remain above 6 cm for ten consecutive 50 Hz control
steps. Every one of those steps must also satisfy bilateral PhysX contact,
goal-pose, object-speed, and force-termination constraints. This 0.2-second
dwell prevents a single contact impulse or initial-state coincidence from
being counted as a lift. The force limits are versioned simulator research
envelopes, not clinically calibrated tissue limits.

Stage 3 preserves orientation qualification; the corrected identity
quaternion makes that contract physically possible at reset. Its analytic base
approaches from 20 mm above the block's contact-calibrated grasp frame and
closes within 5 mm of that waypoint. A parallel 1,200-environment Isaac
Lab sweep compared six offsets from the same reset distribution and counted
only the first terminal outcome per world. It selected
`(0.0, 0.0, -0.0014)` metres from the authored root: 167 of its 200 assigned
worlds reached strict success, tied for the best measured rate while producing
the lowest mean object angular speed. The closed-mesh volume centroid was
rejected because the PSM tool-tip frame is not the jaw collision center;
commanding that centroid eliminated completed lifts.
Two subsequent shared-distribution 1,200-environment sweeps isolated jaw
closure timing. A 5 mm close radius achieved 185 of 200 strict successes on
each of two disjoint world shards; the fine sweep found a stable 4.6–5.0 mm
plateau, while 5.4 mm fell to 45 of 200. The controller therefore uses the
repeatable 5 mm boundary and requires a full-world evaluation before promotion.
Two lateral-alignment sweeps then found a repeatable 4.75–5.5 mm plateau and
selected its 5 mm center; both tighter 4.5 mm and wider 5.75 mm transitions
collapsed below 40% on their assigned shards.
Cartesian commands are capped at 0.1 during the final 20 mm approach. During
bilateral-contact carry, lateral translation remains capped at 0.1 while
vertical recovery is capped at 0.18, corresponding to at most 1.0 mm lateral
and 1.8 mm vertical per 50 Hz command. A full-population 1,200-environment
first-outcome sweep measured 997 strict successes at the 0.18 vertical limit,
up from 870 at 0.1, with no hard termination. The maximum observed object
contact force was 2.70 N: above the versioned 1 N soft penalty but below the
unchanged 5 N hard termination, so held-out force evidence is still required.
The carry controller also exposes a bounded vertical target-offset challenger.
Its versioned default remains zero until a complete 1,200-world first-outcome
comparison earns promotion; the actual 8 cm goal, 15 mm pose tolerance,
ten-step dwell, and force terminations remain unchanged during that comparison.
The controller lifts vertically until the object is within 40 mm of target
height before allowing simultaneous lateral goal tracking. A full-population
1,200-environment first-outcome sweep measured 1,023 strict successes at this
clearance, up from 997 at 20 mm, with no hard termination. Once physics-owned
object height rises above 18 mm, carry and gripper closure remain latched until
the object drops below that threshold, preventing contact-sensor flicker from
restarting the approach phase. Held-out play evidence separately records the
longest consecutive bilateral jaw-contact loss after the object rose above
30 mm, with 2-, 5-, and 10-control-step cohorts split by terminal success and
failure. This retention diagnostic never contributes reward or success credit;
it distinguishes transient contact-manifold flicker from mid-air slips that
would otherwise appear only as timeouts. Stage 4 requires its own
contact-calibrated needle grasp frame before promotion. Its primary
achievement is physical needle pickup: both native jaw contacts must exceed
0.01 N while the needle root remains above 6 cm for ten consecutive 50 Hz
control steps. This short dwell rejects a one-frame collision spike; it does
not require the needle to settle at an arbitrary commanded pose. The unchanged
5 N object-force and 2 N protected-surface hard terminations still veto
success. Goal position, orientation, and motion stability remain recorded as
transport-quality diagnostics for physical handover, but they do not gate
needle-pickup completion or contribute goal-tracking reward in Stage 4.
The composed-needle arc fraction `0.40` is the current pickup-qualified grasp
frame. It produced 1,101 of 1,200 sustained pickups on seed 17 and 1,127 of
1,200 on held-out seed 2361, with zero hard failures in both populations. The
combined 2,228/2,400 rate is 92.83%; the maximum measured needle force was
0.99 N and no protected-surface force was observed. This was a checkpoint-free
analytic controller sweep: its evidence records `checkpoint: null`. The
available learned single-lift checkpoint is not transferred into handover.

Stage 6 certifies physical ownership transfer rather than final
precision-placement quality. At reset, the tool tip with the shorter physical
path to the needle is latched as the giver; the other arm is the receiver.
The giver must establish native bilateral needle contact in at least three of
five control steps and lift the needle at least 10 mm above its episode-start
support height. The receiver must then establish its own three-of-five
bilateral contact window while the giver still owns the needle. The giver
releases, and the receiver retains the elevated needle for ten 50 Hz control
steps (0.2 seconds). One missing receiver-contact frame is allowed only when
the needle remains elevated and preserves the receiver-relative acquisition
offset. While native bilateral receiver contact remains present, that physical
contact is sufficient; needle-center motion within the closed jaws is not a
failure. Commanded closure cannot advance a phase without native filtered
contact. A drop, premature giver release, receiver loss during retention, 5 N
needle-force violation, 2 N unintended contact violation, or timeout is
failure.

Final target position, needle orientation, post-handover retreat, predefined
arc regions, jaw-retreat distance, long dual-grasp dwell, perfect RCM motion,
and zero incidental sub-limit contact are not Stage 6 success requirements.
They remain diagnostics and later curriculum objectives. A receiver recovery
pose is likewise diagnostic only.

The analytic seed controller executes the task in three explicit physical
segments: the closer arm picks up the needle, transports it into the shared
workspace while the receiver remains still, and presents it for acquisition.
Once the receiver acquires the needle, the giver opens without retreating
only when current receiver bilateral contact is present, and it does not
retreat until native giver contact is gone. The receiver freezes its wrist
position and orientation to avoid pulling or twisting the grasp during
release. Release is recognized
only when the giver is commanded open and its native contact is physically
gone, so a thin-needle contact flicker cannot start the retention clock early.
The receiver waits during pickup and transport; it approaches only after the
giver reaches the shared presentation point. On first native needle contact,
it stops the gross approach and uses only the bounded jaw-centering correction
while closing, avoiding a push through the giver's grasp. The two PSM roots
are 10 cm apart; the current
seed moves the needle 3.5 cm from the giver toward the receiver, keeping both
instruments inside comfortable reach. Reaching that authored exchange point
organizes the demonstration but is not part of the success predicate. The
receiver begins its approach within 5 mm of that point; wider entry allowed
the receiver to disrupt an incompletely stabilized transport.
Both receiver translation and wrist orientation remain locked until this
presentation gate is met while the giver has live bilateral needle contact.
The receiver stops its approach on first native contact; if giver custody
flickers away first, the receiver waits while the closest arm regraspes.
The giver begins lifting as soon as current native bilateral contact is live,
while the three-of-five contact window remains the only authority that advances
the physical pickup phase. Once that filtered window latches phase 1, the
bounded lift continues through shorter-than-debounce contact-sensor flicker.
Three consecutive loss steps plus loss of object following declare a real
pickup failure and transfer control to safe recovery, so transient thin-needle
contact noise no longer sends the tool back toward approach. Recovery stops
presentation, commands both grippers open, returns the giver to the 20 mm
pregrasp above the live needle pose, and then reuses the same analytic pickup
primitive. Episodes permit at most three total pickup attempts. A recovered
transfer remains a valid physical success, but evidence reports first-attempt
and recovered successes separately; entering recovery earns no reward or phase
credit. The qualified `0.40` arc offset is an object-relative grasp frame:
after a slip, both the grasp point and recovery offset follow the settled
needle's table-plane yaw rather than its reset orientation. Roll and pitch
never rotate the offset below the support surface; the vertical grasp offset
and proven identity tool posture remain unchanged.
Recovery does not return to approach until the needle is within 5 mm of
support with linear speed below 0.05 m/s and angular speed below 5 rad/s.
One physics-owned post-slip context bit is appended after the existing action
observation. It leaves first-attempt geometry unchanged and activates the
rotated arc frame only for retries. This recovery path intentionally starts
from the analytic base with zero residual and does not claim compatibility
with prior learned-policy observation shapes.
The first handover pickup and recovered re-lift both use a `0.010` vertical
action limit before the 10 mm clearance gate, then return to the `0.015`
loaded transport limit. The checkpoint-free Stage 4 standalone authority of
`0.18` was explicitly rejected for fallen-needle recovery: a 1,200-environment
seed-17 challenge produced 547 successes, exhausted all three attempts in 110
environments, and recovered only one success. A fallen and reoriented needle
does not inherit the reset-aligned Stage 4 dynamics contract. The normal
handover pickup uses `0.010` while presentation retains `0.015`. The slower
pickup reduced needle acceleration
and mid-air loss without reducing final presentation authority. Controlled
full-population challenges also tested `0.0105`, `0.0125`, `0.015`, `0.03`,
`0.06`, `0.12`, `0.18`, and `0.24`; the response was not monotonic, so only
complete terminal outcomes qualify a setting. The faster Stage 4
standalone-pickup authority did not transfer safely to the coupled handover:
it increased drops and reduced physical transfer success. The parameters
remain separately controllable for causal checkpoint evaluation, but
standalone pickup throughput is not treated as evidence for bimanual transport
stability.

After the 10 mm pickup gate, lateral presentation authority follows a
minimum-jerk ramp over the next 10 mm of vertical clearance. This removes the
one-frame transition from zero lateral motion to the full `0.06` action limit
without reducing final presentation authority. The ramp is stateless, uses
measured object height, does not alter rewards or success criteria, and is
recorded in runtime evidence.

The `0.010` live-contact pickup plus 10 mm lateral ramp achieved 696 of 1,200
strict terminal successes (58.00%) on seed 17 and 661 of 1,200 (55.08%) on
held-out seed 2361. Both runs used the same source-locked checkpoint and
unchanged drop, retention, 5 N object-force, and 2 N protected-contact gates.
Seed 17 recorded two protected-contact terminations; held-out seed 2361
recorded none. This is the current throughput champion and training baseline,
not a passed promotion gate, a claim of clinical validation, or task
convergence.

The giver aligns its wrist to the identity tool frame before descending from
the 20 mm pregrasp and may close only within `0.035` rad of that frame.
Orientation alignment stops on first native contact, preventing the controller
from twisting an established thin-needle grasp. This pregrasp gate is a motion
condition only; physical contact, clearance, force, transfer, and retention
remain the outcome authorities.

Stage 6 no longer starts PPO from a random 14-action policy. The exact
closest-arm/contact-aware sequence is the deterministic policy base. The
promoted v24 checkpoint learns only bounded receiver XYZ correction during
qualified receiver approach; pickup, vertical lift, presentation, wrist motion,
and both grippers remain analytic. The residual action limit is 0.01 with
0.005 active exploration standard deviation.

An isolated v25 challenger froze the promoted receiver mean, shared features,
and observation normalization while allowing PPO gradients and exploration
only on giver XY output rows. It improved the 600-environment screen but
regressed retained success at 2,000 environments from 46.95% to 44.50%, so the
checkpoint is rejected and recorded rather than promoted. Intermediate phase
transitions remain diagnostics with zero positive weight. Only a retained
terminal handover earns positive task credit, so partial pickup progress cannot
offset a later hard physical failure.

During filtered phase 1 the giver may continue bounded lift through fewer than
three consecutive live-contact-loss frames. Phase 2 transport still requires
live native bilateral custody. Before receiver acquisition, safe recovery
requires both three consecutive control frames without live giver custody and
more than 5 mm separation from the giver-relative acquisition offset, or an
actual return to the support surface after lift. This prevents
contact-manifold flicker from reversing the lift while preserving a physical
loss trigger. Failure on the third pickup attempt terminates as
`pickup_attempts_exhausted`; post-acquisition drop and retention failures
remain immediate failures.
The analytic base cannot grant physical success or bypass the held-out
promotion gate.

The PSM foundation profile owns the physical jaw contract: a 0.07 radian
symmetric close target and 0.15 N·m actuator effort limit. Physical-parameter
challengers cannot be sharded within one replicated PhysX scene, so the sweep
runner requires one repeated value across all 1,200 environments and records
the applied environment override. Close-target or effort promotion must improve
the unchanged first-terminal success gate without increasing hard force
terminations. The 0.15 N·m effort setting achieved 4,379 strict successes in
4,800 first episodes across seed 17 and held-out seeds 2361, 4099, and 7919.
All four source-locked populations exceeded 90%, with zero hard termination and
a maximum measured object-contact force of 3.67 N below the unchanged 5 N
limit. This qualifies block lifting in simulation; other PSM task families
still require their own regression evidence. These simulator parameters are
not clinically calibrated jaw forces.

Stage 3 residual PPO cannot modify the proven approach, gripper, or orientation
commands. It receives only a bounded `(x, y, z)` correction after physics-owned
contact or lifted-object height has latched carry mode. The residual action
limit is `0.03`; its Gaussian exploration standard deviation is fixed at
`0.01`, and samples are applied only to carry translation. This preserves the
analytic approach, wrist orientation, and binary gripper during both training
and serving while allowing PPO to reduce transport timeouts.

Long visual diagnostics are written in bounded video chunks so recording does
not retain an entire multi-episode rollout in RAM. Single-environment evidence
records each episode's inclusive start and terminal frames plus its physics
termination outcome. This allows strict timeout and hard-failure clips to be
cut from the original recording without relabeling episodes by visual judgment.

Isaac Lab frontier asset rotations use `(x, y, z, w)` quaternion order. Lift
objects therefore use the explicit identity `(0, 0, 0, 1)`. This prevents the
legacy `(w, x, y, z)` identity from becoming a 180-degree X rotation, which
would create a π-radian goal error before the first action. In that identity
orientation, the composed and scaled block collision mesh reaches 14.613 mm
below its root; a 15 mm root height clears the approximately 0.2 mm table top.
The needle begins at a collision-clear 1 mm support height based on its scaled
mesh bounds.

The shared RSL-RL configuration uses separate actor and critic models, action
clipping, numerical checks, observation normalization, compact ELU networks,
adaptive-KL PPO, and task-specific rollout lengths. PhysX scenes use Fabric
cloning, visual markers are disabled during headless training, and target
commands do not resample partway through an episode.

The qualified Gilgamesh RTX 4090 training count is 1,200 environments. Use it
directly for the current learning path:

```bash
DR_ANMAR_TRUST_REQUESTED_NUM_ENVS=1 \
./dr_anmar_learning.sh train \
  DrAnmar-Reach-PSM-IK-Rel-v0 \
  1200
```

The override is explicit because the general launcher retains conservative
live-memory fitting for unknown machines and concurrent workloads. Evidence
records both the requested/actual count and whether the qualified-count
override was used.

The launcher requires at least 1,024 MiB of free GPU memory and 4,096 MiB of
available system RAM by default. It measures both resources immediately before
launch and caps parallel worlds from 8 up to 1,024 using the stricter live
allowance. The requested count remains in the evidence bundle alongside the
actual fitted count, total and available RAM, free VRAM, process peak memory,
and Torch peak GPU allocation.

Kit startup shares Torch's active CUDA context to avoid allocating a redundant
primary context while other GPU services are active. PhysX scene creation then
uses its own thread-safe context. Together with Fabric cloning, this lets
training survive a busy workstation and scale back up when memory becomes
available without silently falling back to CPU.

Override `DR_ANMAR_MIN_FREE_GPU_MIB` or
`DR_ANMAR_MIN_AVAILABLE_SYSTEM_MIB` only for a deliberately qualified lower
memory configuration. Set `DR_ANMAR_TRUST_REQUESTED_NUM_ENVS=1` only for an
environment count already qualified on the exact machine and task family.

## Commands

Configure the active Linux runtime when it is not in the default DrAnmar
workstation location:

```bash
export DR_ANMAR_ISAACLAB_ROOT=/absolute/path/to/IsaacLab
export DR_ANMAR_ISAAC_PYTHON=/absolute/path/to/isaac-python
export OMNI_KIT_ACCEPT_EULA=YES
```

Set the EULA variable only after accepting NVIDIA's license terms.

Then:

```bash
# Pure source and contract validation
./dr_anmar_learning.sh validate

# Confirm frontier runtime registration
./dr_anmar_learning.sh list

# Inspect native observation/action shapes, initial geometry, contacts, and
# termination terms before spending samples on a new task
DR_ANMAR_TRUST_REQUESTED_NUM_ENVS=1 \
./dr_anmar_learning.sh probe \
  DrAnmar-Lift-Block-PSM-IK-Rel-v0 \
  1200 10 \
  output/dranmar-learning/probe/lift-block-psm-1200-v1

# Create one native environment and complete one PPO iteration
./dr_anmar_learning.sh smoke

# Measure ten iterations without claiming convergence
./dr_anmar_learning.sh benchmark

# Initialize and validate the analytic-base residual Stage 1 policy
./dr_anmar_learning.sh pretrain

# Initialize and validate Stage 2 at the qualified count
DR_ANMAR_TRUST_REQUESTED_NUM_ENVS=1 \
./dr_anmar_learning.sh pretrain \
  DrAnmar-Reach-Dual-PSM-IK-Rel-v0 \
  1200 32 500 \
  output/dranmar-learning/pretrain/reach-dual-psm-1200-v1

# PPO refinement when deterministic held-out evaluation still misses its gate
./dr_anmar_learning.sh train

# Evaluate a frozen checkpoint
./dr_anmar_learning.sh play /absolute/path/to/model.pt
```

The benchmark and play commands emit typed runtime, learning, hardware, version,
resource, and success bundles under `output/dranmar-learning/`. They are the
primary experiment records; console output alone is not evidence.

Play qualification counts exactly the first terminal outcome from every
parallel environment. Environments that succeed early may reset and contribute
to diagnostic all-episode totals, but those later episodes cannot increase the
promotion success rate. Lift evidence also classifies every first episode by
the last achieved physical stage: bilateral contact, minimum height, goal
position, instantaneous qualified state, or sustained dwell. Target-distance
strata distinguish controller limitations from an unfavorable reset mix.

## Promotion gates

Each stage has an explicit success threshold and must pass all held-out seeds in
the manifest. Promotion requires:

1. a frozen checkpoint and cryptographic hash;
2. the task ID, source revision, runtime versions, GPU, seed, and environment
   count;
3. seeded play bundles for every held-out seed, using one first terminal
   outcome per environment;
4. success rate and stage-stratified failure distribution, not reward alone;
5. contact and force qualification for lift and handover; and
6. a fresh benchmark if the robot, asset, physics, reward, observation, or
   runtime contract changes.

Stage 6 additionally requires a same-seed, same-population analytic baseline
from the same source revision. A checkpoint must reach the absolute success
gate, must not trail that baseline, and must strictly reduce the rate of
three-step mid-air giver contact loss unless the baseline rate is already zero.
The complete first-terminal population is required, object-drop, excessive
object-force, and protected-surface-force terminations must all be zero, and
the evidence must prove the XY-only residual, analytic vertical authority,
disabled receiver residual, and 0.01 residual scale. A final or periodic PPO
checkpoint is never promoted merely because it was saved.

This structure follows residual learning's intended division between a
competent nominal controller and a small learned correction, keeps hard
constraints separate from task reward, and leaves recovery under its own
bounded controller. The primary method references are [Residual Reinforcement
Learning for Robot Control](https://arxiv.org/abs/1812.03201), [Constrained
Policy Optimization](https://proceedings.mlr.press/v70/achiam17a), and
[Recovery RL](https://arxiv.org/abs/2010.15920).

For handover play bundles, the stage-stratified distribution records the
maximum physical phase reached before each environment's first terminal
outcome: giver contact, 10 mm lift, receiver acquisition, retained transfer,
or success. This separates failed pickup from failed transfer without changing
the success predicate.

Early stopping counts completed and successful episodes directly from Isaac
Lab's termination manager. It reduces wasted compute after the exact episode
success rate has remained above its threshold for the declared window. It does
not replace held-out evaluation.

## Evidence boundary

The Learning Path establishes reproducible simulator training and evaluation
contracts. Reach completion establishes pose-control performance in the named
simulation. Lift and handover completion additionally require simulator-owned
bilateral contacts, stable object motion, bounded force, and release/ownership
transitions.

These results are not clinical validation, physics calibration, a medical
device approval, or permission to control physical surgical hardware or provide
patient care.
