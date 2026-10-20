# Language tasks on the official OpenArm v2 / Genesis scene

The native viewer owns the only live simulation. FastAPI is a command, progress, and result broker;
it does not instantiate an environment. The same `TaskExecutor` is used by the native owner,
headless verification scripts, and the asynchronous execution wrapper.

## Language and grounding

`planning/agent_planner.py` asks a configured provider for a `TaskProgram`, not motor commands.
The bounded Pydantic schema in `planning/task_program.py` has eight operations: pick, handover,
place, approach, move, open, close, home. Each action can name an object, acting arm, receiving
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
