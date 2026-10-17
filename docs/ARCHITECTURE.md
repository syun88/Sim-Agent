# Architecture

The backend is the authority for state. The planner sees only serialized simulation state and can choose from a fixed skill library. It cannot generate executable code. Each skill returns measured timing, error categories, and endpoint poses.

## Simulation engine

One engine runs the task: Genesis World, on the official OpenArm v2 Cell model.

```mermaid
flowchart TD
  A[Agent / planner / skills / verification] -->|SimulationBackend| B[Genesis World backend]
  D[(Official openarm-mujoco v2 Cell MJCF)] --> B
  D -.->|sites, keyframe, actuator mapping| C[MuJoCo compiler as the model reader]
  C --> B
```

The agent layers never import an engine. `backend/app/simulation/base.py` declares the contract
(`SimulationBackend`) that the engine implements, and `backend/app/simulation/__init__.py` is the one
place an engine is constructed. `backend/app/control/motion.py`, `backend/app/skills/robot_skills.py` and
`backend/app/verification/goal_verification.py` read poses, Jacobians, contacts and grasp state only
through that contract. The seam exists so an engine can be replaced or added without touching the agent,
and so every engine difference is declared where the agent can see it (`grasp_mode`) rather than hidden in
a controller branch.

### Model description

`assets/scenes/openarm_v2_demo.xml` is authored with two path tokens and no machine-specific paths. They
are resolved at load time from the installed `openarm-mujoco` 2.3.0 package
(`backend/app/simulation/openarm_v2/scene.py`), so the same XML string is compiled on any machine. The
official model is attached through MuJoCo's `<model>`/`<attach>` mechanism and is never copied or
substituted. `backend/app/simulation/openarm_v2/definition.py` holds the joint, link, site, object, zone
and grasp-geometry names the agent uses.

The description has one reader. Genesis does not implement the MJCF site abstraction, so the gripper
control point and the zone origins are read from the *same XML string* by MuJoCo's compiler
(`_OfficialModelMetadata`) as a parent link plus a local offset. Genesis supplies the physics; the
naming, the keyframe, the actuator mapping and the grasp geometry all come from the official asset.

### The physical grasp

The grasp is the official hand's own mechanism, measured from the official model:

- **Fingertip midpoint.** Each finger body's `*_collision_00` geom is the fingertip pad (the hand's
  foremost, pad-facing segment, `condim=4` so torsional friction keeps a pinched object from
  spinning). The midpoint of the two pads' centroids at the closed configuration, in the gripper
  control point frame, is `(-0.0066, 0, -0.1634)` for both hands (`FINGERTIP_MIDPOINT`);
  `scripts/measure_grasp_geometry.py` re-measures it from the installed model and the fidelity
  test fails if the constant drifts.
- **Gripper travel.** The official keyframe holds both hands fully open (`key_ctrl` `+/-0.7854`, the
  open range end); 0 closes the pads onto the symmetry plane. The project commands open `0.7854` /
  `-0.7854` and closes to `0.02` / `-0.02` (the pads stop short of touching each other; on an object the
  position actuators press into it from there, and the official finger mimic equality keeps the two
  fingers symmetric).
- **Approach orientation.** Fingers forward (+x world) so the pads close across the object's `+-y`
  side faces, tilted 10 deg nose-down: at the pinch alignment the palm's lowest point sits 14.5 mm
  above the cell table at 10 deg versus 1.7 mm at 0 deg (measured), so the tilt is what keeps the
  official palm geometry off the table while the fingertips stay on the object's centre.
- **Approach path.** The arm first rises to a 1.24 m staging altitude above its current pose (the
  highest the official arm reaches with the horizontal orientation; measured 1.0 mm IK residual at
  1.24, 12.7 mm at 1.26), because a joint-interpolated swing toward a lateral target otherwise yaws
  the extended hand through the workspace and can sweep an object off its place. It then swings
  above the object, drops onto the approach axis 6 cm behind the pinch point, and slides straight in.
  The skill measures how much the approach displaced the object and fails (`APPROACH_COLLISION`)
  beyond 5 mm.
- **Grasp verification.** After closing, both the inner and the outer finger must contact the
  object (`GRASP_FAILED` otherwise), and a 3 cm verification lift must carry the object with the
  gripper, with contact maintained (`GRASP_LOST` otherwise). Nothing writes the object's pose:
  the only pose writes are scene reset and setup.

### What the engine does differently

| | Genesis World |
|---|---|
| Scene | the resolved official MJCF string, Genesis physics and rasterizer |
| Grasp | **physical friction pinch**: the position-controlled fingertip pads press the object's side faces; the official finger mimic equality keeps the two fingers symmetric; friction holds the object: `physical-friction-pinch` |
| Grasp alignment | the hand's `FINGERTIP_MIDPOINT`, measured from the official fingertip pad geoms, is IK-aligned to the object centre |
| Bias torque during a joint path | `apply_gravity_compensation()` writes the gravity torque into the constant term of the actuator model |
| Camera | one fixed dashboard camera aimed at the middle of the table |

`env.grasp_mode` is reported with every grasp result, in `/api/health`, and in the dashboard badge.
The skills verify the pinch from the finger-object contacts and from the object following a short
verification lift, and the goal verification requires the object to be untouched by any finger before
declaring it released.

### Spawn pose

The request "spawn the joints at 90 degrees" has two readings.  The elbow (joint 4) already spawns at
90 degrees in the official model keyframe.  Setting *all seven* arm joints to 90 degrees makes the two
arms interpenetrate, so the project keeps the official keyframe as the single documented spawn pose.
`SPAWN_ARM_QPOS` in `backend/app/simulation/openarm_v2/definition.py` names that decision, and
`GenesisOpenArmV2Environment.reset()` builds the full qpos from it.

Measured on the compiled MJCF with `mj_kinematics` + `mj_collision` (both arms at the same pose):

| Pose | Joint values (deg) | left ee (m) | right ee (m) | height above table (m) | collision-free | worst penetration |
|---|---|---|---|---|---|---|
| official keyframe | [0, 0, 0, 90, 0, 0, 0] | (0.401, 0.153, 1.120) | (0.401, -0.153, 1.120) | 0.120 | **yes** | — |
| only joint4 = 90° | [0, 0, 0, 90, 0, 0, 0] | (0.401, 0.153, 1.120) | (0.401, -0.153, 1.120) | 0.120 | **yes** | — |
| all seven joints = 90° | [90, 90, 90, 90, 90, 90, 90] | (0.401, -0.067, 1.340) | (-0.031, -0.373, 1.340) | 0.340 | **no** | arm-arm -0.072 m |
| joints 2 and 4 = 90° | [0, 90, 0, 90, 0, 0, 0] | (0.401, -0.067, 1.340) | (0.401, -0.374, 1.340) | 0.340 | **no** | arm-arm -0.072 m |
| joint2 = 45°, joint4 = 90° | [0, 45, 0, 90, 0, 0, 0] | (0.401, -0.002, 1.184) | (0.401, -0.309, 1.184) | 0.184 | **no** | arm-arm -0.018 m |
| joint2 = -45°, joint4 = 90° | [0, -45, 0, 90, 0, 0, 0] | (0.401, 0.309, 1.184) | (0.401, 0.002, 1.184) | 0.184 | **no** | arm-arm -0.018 m |

Only the official keyframe (which is also "only joint4 = 90°") is collision-free.  The all-90° pose
produces 19 arm-arm contacts, the deepest being `openarm_lifter_link | openarm_left_link3` at -0.072 m
and `openarm_left_link3 | openarm_right_base_link` at -0.064 m.  No arm-cube or arm-table contacts
appear in any candidate.  `test_spawn_pose_is_the_official_model_keyframe` asserts the project constant
matches the model, and `test_starts_in_the_documented_scene` asserts a freshly constructed engine is in
that pose.

### Genesis World engine: status

Verified by `backend/tests/test_official_model_fidelity.py`, `backend/tests/test_openarm.py` and
`backend/tests/test_transform.py`, `tests/test_memory_serialization.py` (34 tests, `pytest -q`):

- the ground truth for structure is the model description, compiled by MuJoCo and evaluated with its own
  kinematics routines (`mj_kinematics`, `mj_comPos`, `mj_jac`) - pure kinematics, no solver, no stepping;
- the initial scene after construction *and* after `reset()` is the model's own keyframe plus the shared
  object placement, to 1e-5 (the server builds one backend at startup and never resets it, so a task that
  arrives first must not begin from the model's zero configuration);
- forward kinematics, the gripper control point pose and the 3x7 linear and angular Jacobians the
  controller consumes match the official model's own routines to 1e-5 and 1e-6, at the home pose and at an
  arbitrary configuration;
- an inverse kinematics solution is checked by the official model's forward kinematics, so a wrong
  convention cannot pass by agreeing with itself;
- the complete approach/grasp/lift/place/release/retreat sequence executes for both arms, reports its
  labelled grasp mode, asserts the both-finger pinch and the verification lift on every grasp, and
  passes independent verification.

Three engine differences had to be handled explicitly, and each is covered by a test:

1. **Equalities.** Genesis builds every equality it parses and reads no `active` flag, and it has no
   runtime enable/disable. The scene therefore declares no inactive equalities at all: a grasp is the
   fingers' physical pinch, so no constraint that could move an object is expressed.
   `genesis_scene_xml()` remains as a guard, dropping any inactive equality a future scene edit
   introduces before the XML reaches Genesis.
   `test_genesis_builds_the_official_model` asserts Genesis built exactly the official two finger
   mimics.
2. **Bias feed-forward.** A Genesis position controller holds a commanded pose `gravity_torque / kp` short
   of its target. Measured on this machine with the same commanded pose and the same control path: 10.16 mm
   of sag without compensation, 0.41 mm with it. `apply_gravity_compensation()` writes the arm's gravity
   torque into the constant term of Genesis's actuator model (`act_bias[0]`, the analogue of the model's
   `biasprm[0]`), which keeps the damping implicit in the solver and leaves the cubes their own gravity. An
   earlier attempt to feed the same torque through `control_dofs_force` diverged on long joint paths,
   because FORCE mode skips the implicit damping the solver folds into the mass matrix for POSITION mode.
3. **Sites.** Genesis has no MJCF site, so the gripper control point and the zone origins come from the
   model description as a parent link plus a local offset (above).

Measured placement error of the red-cube task through the task API, with the physical friction pinch:
7.6 mm horizontally and 23.0 mm vertically against a 40 mm / 50 mm tolerance (right-arm blue-cube task:
5.4 mm). The residual is where the released object comes to rest after its 4 cm release drop onto the
zone, not a grasp defect: the transport holds the object between the fingertip pads the whole way.

`scripts/verify_pick_place_genesis.py` runs the same physical pick & place from nine initial object
positions (six left-arm, three right-arm); all nine passed on this machine with 0.0 mm approach
displacement, a both-finger pinch, the object following the verification lift, and 5.3-18.5 mm final
horizontal error (`outputs/pick_place_positions.json`).

## OpenArm v2

- Model: `openarm-mujoco` 2.3.0, v2 Cell, attached bimanual robot.
- Actuator mapping: 1 lifter, 7 left arm, 1 left gripper, 7 right arm, 1 right gripper.
- Active task arm: left.
- Controller: damped least-squares inverse kinematics on the gripper control point, executed as an
  interpolated joint path. The rotation algebra is NumPy (`backend/app/simulation/transform.py`) and is
  pinned against MuJoCo's `mju_*` functions by `backend/tests/test_transform.py`.

## Execution

- Pydantic plan validation rejects unknown skills, objects, targets, unsupported arms, empty or oversized plans.
- The executor serializes simulation control through one engine/task.
- Failure verification is independent of the AI plan.
- `reset(randomize=True)` uses one shared placement function for both engines, whose seed is documented
  in `backend/app/simulation/openarm_v2/scene.py`.

## Memory

- AuraDB uses parameterized Cypher and constraints.
- Experience is keyed by task, execution, object, skill, outcome, and failure.
- If AuraDB is unavailable, records go to `outputs/memory-fallback.json` and the UI reports `Fallback`.
- Retrieved experience is stored back into plans only as a compact summary (instruction, outcome, retry
  count, skills). Embedding whole records nested the entire history inside one record and grew the
  fallback file to 4.2 GB (one record: 345 MB); the summaries keep it flat.
