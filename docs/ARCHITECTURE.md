# Language tasks on the official OpenArm v2 / Genesis scene

The native viewer owns the only live simulation. FastAPI is a command, progress, and result broker;
it does not instantiate an environment. The same `TaskExecutor` is used by the native owner,
headless verification scripts, and the asynchronous execution wrapper.

## Language and grounding

`planning/agent_planner.py` asks a configured provider for a `TaskProgram`, not motor commands.
The bounded Pydantic schema in `planning/task_program.py` has ten operations: pick, handover,
place, approach, move, open, close, home, throw, pour. Each action can name an object, acting arm, receiving
arm, semantic destination, or a 1–30cm relative motion. Extra fields and arbitrary code are rejected.
There are at most 24 intentions and 144 compiled skills.

The provider sees current object positions, the contact-derived occupancy of both hands, and a
compact discourse context. The compiler resolves `it`, held:left/right, and spatial references
against this snapshot. It simulates occupancy over the complete program before execution.
A second pick by another hand becomes a handover. A place automatically acquires an unheld
object, while an already held object is transported without opening the gripper again.

A hand cannot hold two objects or receive an object while occupied. A low-level lift must follow
a grasp, a release must name the held object, and transport requires a hold. Unsupported objects
and targets, invalid distances, and unresolved references fail before motion.

Japanese/English clauses covered by `planning/language_fallback.py` provide an explicit fallback.
When the entire request has an exact parse, the model cannot silently change its ordered object,
arm, receiver, or destination trace. Invalid provider programs receive one bounded repair attempt.
Provider failures contain categories rather than exception strings that could reveal credentials.
Free paraphrases require a functioning language model; the fallback does not guess the first task.
The native entry point defaults `OPENARM_FAST_LANGUAGE` to `1` after loading
`.env`. When the entire instruction already compiles against the observed world,
it returns that exact program immediately as `deterministic_fallback`, with no
model name or round trip. Unrecognised wording still uses the model. Setting this
flag to `0` retains model confirmation for multi-action exact requests; the same
signature/quantity/object checks apply. This cuts avoidable planning latency,
independently of the native physics/motion speed improvements below.

## Local model adaptation without training

The existing local `qwen3-vl:4b` was inspected through Ollama's `/api/show`. This installation has
architecture `qwen3vl` but template `{{ .Prompt }}`. Normal `/api/chat` calls spent the entire output
budget in the `thinking` field even with `think=false`, leaving no JSON result.

For precisely this positively identified case, the adapter supplies Qwen role delimiters and an
already closed thinking prefix through `/api/generate` with `raw=true`. The output is constrained
by the same JSON schema. Other Ollama templates use `/api/chat`. No model files, weights, global
Ollama templates, or credentials are modified. This is inference-time formatting and validation.

## Physical skills

The controller snapshots the actual Genesis joint configuration by official joint name, including
the lifter and fingers. `simulation/openarm_v2/kinematics.py` evaluates IK candidates on a private
data copy of the already compiled official model, using only forward kinematics and Jacobians.
It never changes the live scene. This distinction matters: Genesis immediately redraws a direct
joint-position write and clears the contact solver's state, even without stepping physics. Trial
poses followed by restoring the start pose would visibly replay the motion before execution.

Joint-limited damped least-squares IK drives the same position actuators and joint paths. If a
redundant arm posture left by a handover is a poor IK starting point, it tries the official spawn
posture as a calculation seed. Only motor commands execute candidates in Genesis; endpoint
verification reads the actual physics state. Direct joint-position writes are reserved for scene
setup and model-fidelity checks.

A motion result measures its settled Cartesian position and orientation. A valid IK solution
that physical collision prevents the motors from reaching returns `MOTION_BLOCKED`. Both arms
receive gravity compensation while either arm moves, so the stationary donor continues holding.
The CPU and GPU paths both read one actual Genesis qpos snapshot and compute the
gravity lever arms with the official mass/COM geometry. Independent tests compare
these torques against Genesis's live link Jacobians. One batched actuator-bias
write supports both arms per five physics steps; redundant controller calls reuse
that update. Joint limits are immutable model data read once at construction.

Native viewer pacing targets 1x wall-clock real time by default, configurable with
`OPENARM_SIM_SPEED`. It is a throughput target, bounded by computation, not a
physics dt multiplier. It does not rescale gravity, force limits or the physical
clock. Actual arm travel uses a separate `OPENARM_MOTION_SPEED` (default 6).
Free transit and carry waypoints form a shape-preserving joint spline with
quintic time scaling; position and velocity references are sent through Genesis's
native combined PD control at up to 4ms intervals. Arm stiffness and inertia-based
damping improve tracking without changing force caps or finger motors. Endpoints
are measured from physics, and contact checks continue during loaded transport.
The controller clears velocity references on completion and interruption.
Large loaded reorientations interpolate the object center and hand attitude
together using the observed pinch offset, rather than swinging a held object
through a single short joint stroke. Receiver insertion and table reorientation
also use continuous waypoints while checking the donor/receiver pinch.

Gripper waits observe actual joint settling, while preserving the two-finger and
lift-follow grasp checks. Calibrated pour-angle/mouth adjustments and the ballistic
launch restore the original actuator response and simulation durations. Upright
bottle transport and throw wind-up use fast transit. Every intervening physics
step still executes; stop checks remain between steps. `scripts/verify_arm_motion.py`
compares measured simulation seconds for approach, grasp and loaded A-to-B travel,
independently of playback. Wine mode retains the production fluid and samples
spillage and airborne volume during the move. `scripts/verify_execution_speed.py`
measures actual native task wall time independently from initial compilation and
language inference, while requiring the physical grasp/place postconditions.
A commanded orientation is retained across carry segments rather than accumulating each new
measurement's tracking error as a new desired orientation.

The fingertip midpoint and gripper limits come from the official hand meshes and keyframe.
`scripts/measure_grasp_geometry.py --check` re-derives the midpoint and checks the constants against
the installed model. Table grasps open, stage above the workspace, enter along the approach axis,
close the official fingers, require both finger links, and perform a 30mm follow test.

For a planned handover, the table pick uses a 15mm rearward tangential bias on the cube face,
leaving room for the other hand. This changes which portion of a face is pinched; it does not
change the mesh or write the cube pose. The grasp still requires contacts and physical follow.

## Direct handover

`simulation/openarm_v2/handover.py` reads the attached MJCF and solves both seven-joint arms and
the cube orientation together. It uses the *measured* donor-to-cube transform, official joint
limits with a margin, and collision meshes. The receiving pads approach the opposite portion
of the same side faces. Candidate poses reject arm-arm and hand-table penetration.
MuJoCo is a model/geometry reader here: no `mj_step` and no second dynamic simulation exist.
Genesis validates every candidate by actually moving, touching, carrying, and releasing.

The runtime protocol is:

1. Keep the donor's real pinch while moving it to the coupled pose and settling.
2. Open the receiving hand. Move to a 120mm standoff outside the held object.
3. Insert along a Cartesian line in 10mm increments, checking the donor's hold throughout.
4. Reject an approach that displaces the cube by over 6mm. Close the receiver and require both fingers.
5. Open the donor and withdraw 80mm along its approach axis. Require no donor-object contacts.
6. Require the receiving pinch, lift the object another 30mm with that arm alone, and measure follow.

A joint interpolation straight from home to the receiving pose was observed to sweep an open
finger through the cube, producing spurious intermediate contacts and losing the grasp. The
standoff and straight insertion were necessary; endpoint IK and a two-finger contact list alone
did not establish a successful handover.

For placement, the received grasp returns to the table attitude by interpolating rotation while
keeping the cube centre high. Carrying uses Cartesian segments of at most 25mm, followed by
measured object-position correction. This avoids a joint-space arc dipping through the other
cube. Placement refuses an occupied target rather than accidentally stacking objects.

## Execution, recovery, and verification

`execution/task_executor.py` observes before and after each skill and records the evidence with
its action index. A failed table approach or empty pinch can be reacquired once from the current
measured state. A lost transport or a failed handover is reported, not replayed from a scene reset.
The initial `randomize` option and an explicit reset are setup operations; retries do not reset.

The discourse context stores only the last object, hand, and released target. It advances from
successful execution steps and is cleared with a scene reset. It is a reference aid, not evidence
of ownership; hand occupancy is always observed from Genesis contacts.

`verification/goal_verification.py` checks every final object predicate, including objects placed
earlier in the task. Holding requires both fingers and an elevated object. Placement requires
horizontal error <=40mm, vertical error <=50mm, no finger contact, linear speed <20mm/s, and
less than 2mm displacement over a 100ms observation window. Unmentioned objects must stay within
6mm of their initial positions. The displayed result includes measured errors and tolerances.
A truncated plan or an unfinished skill cannot become a success just because some final position
happens to match.

## Broker and UI

`run_genesis_gui.py` registers an owner before constructing Genesis, preventing duplicate native
viewers. The broker accepts snapshots and results only from its registered owner; the destructive
queue-read endpoint also checks the owner ID. Planning runs in a worker on an immutable scene
snapshot, keeping native macOS rendering on the main thread. Only that main thread drives Genesis.
A stop during planning can be reported without allowing a late language response to move motors.

The operator page at `/` and `/simple` shows the entire interpreted program, assumptions, current
step, actual provider, measured hand occupancy, verification, and failures. Queued resets and stop
requests remain pending until acknowledged by execution. `/api/scene` serves the viewer's latest
snapshot and returns 503 while none exists; the backend never invents a scene.

The native viewer also supports `--headless` and `--once` for testing the same broker path without
a window. `scripts/verify_language_tasks.py` exercises the actual planner and executor and writes
inspectable JSON. Existing official-model fidelity and position-grid tests remain applicable.

## Dynamic manipulation and liquid scene

The task schema also contains `throw` and `pour`. A cube throw acquires the cube,
commands an accelerating upward joint stroke, and opens the actual fingers. It does
not set the cube velocity. Verification independently searches the recorded Genesis
contacts/positions/velocities for an upward contact-free interval, a measured apex,
and gravitational acceleration. A dropped cube or a contact-supported lift fails.
The default requested rise is 30mm measured after release.

Wine work uses a separate scene profile with the same attached official robot/cell,
a free bottle, a fixed receiving glass, and red SPH particles. The bottle and glass
are Innerscene CC0 assets; unchanged GLBs, source hashes, and reproducible open-vessel
mesh derivatives are included in `assets/wine`. Hollow material shells couple to
SPH through SDFs; wall friction is 0.2 on the bottle and 0.8 on the glass to
approximate a dissipative wet boundary at this coarse resolution. Convex body/neck proxies on the same physical bottle provide rigid
gripper/support contacts without filling the liquid cavity. No attachment or pose
tracking constraint holds the bottle. Bottle acquisition retracts behind the body
before moving sideways, since a cube-style overflight hits the tall neck.

`scene_selection.py` selects the workpiece profile before language grounding. The
native owner closes its previous scene and creates the new one on its main thread;
it never creates a second runtime. Switching requires empty hands and clears the
reference context. `PREPARING_SCENE` and the owner's startup lease cover compilation.
The wine profile uses an available GPU by default (Metal on this Mac), unless
`GENESIS_USE_GPU` explicitly selects a device. Robot gravity geometry is evaluated
from the actual observed joint snapshot to avoid per-link GPU transfers; this is
forward geometry only, with Genesis retaining every dynamic state and payload force.

Wine measurement counts active particles inside the transformed vessel cavities and
reports reference volume in mL from the actual Genesis SPH reference particle volume (0.8 times spacing cubed,
3,600 particles, 45mL initially, with 2.5mm spacing). It separately
reports particles in the bottle, glass, flight, and on the table. These are coarse
SPH volume estimates, not a calibrated real-wine material model. Liquid moves only
through gravity, pressure, and rigid/SPH coupling. Particle/vessel setters are
restricted to scene setup/reset. `scripts/verify_dynamic_tasks.py` runs language,
compilation, physical skills, and independent event/volume verification together.
Containment uses distance to the rotational cavity profile with a particle-radius
margin, including the sloped floor. Padding only the interpolated radius wrongly
classified near-floor particles as spills. Bottle/glass/flight/spill sets remain
disjoint and all active particles are counted. Spill checks include acquisition
and transport and allow at most 0.1mL in the entire task.

Vessel MJCF collision geoms use group 3, outside Genesis's visual groups 0–2.
Alpha zero alone does not keep a collision geom out of the imported visual set.
The transparent visual meshes contain a single outward-facing skin with back-face
culling: the native rasterizer sorts objects rather than the faces of a hollow
transparent shell, so overlapping inner/outer walls otherwise produce triangular
blending artifacts. The watertight collision/SDF shells retain their complete
cavities and are unchanged by this display preparation. This is rasterized
transparency, without optical refraction. Base, stem, and bowl are separate visual
geoms so that the rasterizer can order their transparent draw calls. The wine
camera includes the entire glass.

SPH uses `vis_mode="recon"` with a smooth splashsurf surface of the observed render
particles, not a separate fluid simulation. `LiquidSurfaceRenderer` adapts this
scene's Genesis 1.4.4 SPH mesh upload. A clean CPU forkserver preloads imports;
each reconstruction job exits after returning the mesh, bounding its native heap.
Only one job may be outstanding; frame sequence checks and explicit reset discard
stale results. Native updates are asynchronous; offscreen validation waits for
the current surface. No installed Genesis code or global renderer functions are
modified. Reconstructed geometry never affects particle positions or measured mL.

Wine stepping uses 2ms simulation steps with 32 substeps and WCSPH. The rigid
solver uses Newton with convex contact resolution and `impratio=10`; the default
elliptic signorini combination produced invalid forces on Metal when the initial
glass interfered with the open fingers. The glass is now outside the spawn sweep.
The default cube profile retains its existing 1ms solver configuration.
The viscosity coefficient is 0.003 and surface-tension coefficient is 1.5e-4 for
resolved water-like flow at this scale; these are not calibrated wine properties.
The scene validates Genesis 1.4.4's explicit-viscosity bound
`substep_dt <= 0.01 * support_radius**2 / mu` before opening a viewer. Changing
spacing alone from 4.5mm to 2.5mm with the former mu=0.003 and 0.5ms substep
violated that bound and produced the rejected exploding configuration. Hollow
SDF walls use 0.4mm cells (128–512 resolution) to resolve the thin vessel material.
The SPH neighbor hash uses 32 cubed buckets (`hash_grid_res=(0.16, 0.16, 0.16)`
with 5mm cells), instead of the default 150 cubed. The simulation bounds remain
unchanged. Genesis filters bucket candidates by actual distance, including across
hash wraps, so this reduces empty-bucket work without clipping escaped liquid or
changing the support radius. CPU/Metal timings are diagnostics, not a promise of
real-time fluid simulation.

Pour control observes the bottle-to-wrist transform and corrects the measured
mouth position, holds the angle once flow begins, and meters liquid at 20ms
intervals before increasing tilt. Explicit motion durations include interpolation
time. Closing keeps the mouth over the bowl until the opening points upward; only
then does the bottle move away. Final checks require measured glass volume,
bounded total spill, zero airborne liquid, particle conservation, release, and a
stable upright bottle at its starting position. Existing spills remain in the
reported total; incremental spill is recorded separately. Nonzero volumes are
shown to four decimal places, with a nonzero lower-bound label below that scale.
Measured vessel clearance must be nonnegative as well as having safe planned
clearance. These are motion-endpoint measurements; the rigid solver handles
contact during interpolation.
Vessel motion uses quintic time interpolation with zero endpoint velocity and
acceleration. Near flow onset the tilt increments are 2 degrees; the mouth is
aimed over the receiving opening, 18mm above the rim. The neck and bowl must keep
at least 1mm of planned clearance, checked against samples of the actual glass
shell and the bottle's physical contact proxies. A previous fixed-vessel fixture
put the neck through the bowl wall; its low-spill result is invalid and is not
robot-task evidence. Actual clearance is also recorded after each bottle motion.
Closing anticipates the measured outflow during the motor's stopping interval.
Flow holds last up to two seconds before increasing the tilt again. Closure first
raises the opening to 85 degrees, then returns upright in stages, keeping the last
liquid above the receiving opening.

Implementation references: [Genesis SPH setup](https://genesis-world.readthedocs.io/en/latest/user_guide/physics/beyond_rigid_bodies.html),
[SPH materials](https://genesis-world.readthedocs.io/en/latest/api_reference/engine/material/sph.html),
[rigid/particle coupling](https://genesis-world.readthedocs.io/en/latest/user_guide/theory/coupling/index.html),
and [the exact v1.4.4 SPH solver](https://github.com/Genesis-Embodied-AI/genesis-world/blob/v1.4.4/genesis/engine/solvers/sph_solver.py).

Informal Japanese upward-release phrases and bounded noun-only wine requests also
have labelled deterministic parses. Unknown trailing tasks still fail. For model
programs, an unspecified wine hand is selected from observed ownership/geometry,
not a model's arbitrary default. A redundant unspecified bottle placement directly
after a pour is merged with that skill's measured return. Model assumptions that
claim success are discarded; only the physical verifier can establish completion.

The native owner installs a step observer after scene construction. Between every
completed physics step it checks the cached stop request and publishes the usual
heartbeat. Long SPH gripper/settling waits therefore do not starve the broker or
delay cancellation until the entire simulated wait finishes. An interrupted
transit clears its velocity reference and holds the observed joint position.
The measured object/particle state is retained; cancellation never resets it.
