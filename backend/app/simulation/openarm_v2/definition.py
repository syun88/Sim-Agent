"""Names and grasp geometry of the official OpenArm v2 bimanual Cell model.

Every name here exists in the official `openarm-mujoco` model, so any engine addresses the same robot.
The engine backend reuses them unchanged, so the agent, the skills and the verification step name the
same robot whatever engine runs it. The grasp geometry - the fingertip midpoint the IK aligns to the
object centre, and the gripper travel - is measured from that same model description (see
`measured_fingertip_midpoint` below and `scripts/measure_grasp_geometry.py`), and
`backend/tests/test_official_model_fidelity.py` fails if the constants here drift from the installed
model.
"""

from __future__ import annotations

import numpy as np

ARM_JOINTS = {
    "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}
ARM_ACTUATORS = {
    "left": [f"left_joint{i}_ctrl" for i in range(1, 8)],
    "right": [f"right_joint{i}_ctrl" for i in range(1, 8)],
}
ARM_LINKS = {
    "left": [f"openarm_left_link{i}" for i in range(1, 7)] + ["openarm_left_ee_base_link"],
    "right": [f"openarm_right_link{i}" for i in range(1, 7)] + ["openarm_right_ee_base_link"],
}
GRIPPER_ACTUATORS = {"left": "left_finger1_ctrl", "right": "right_finger1_ctrl"}
GRIPPER_JOINTS = {"left": "openarm_left_finger_joint1", "right": "openarm_right_finger_joint1"}
#: Fully open finger joint command. The official keyframe holds both hands at the open end of the
#: finger range (`key_ctrl` 0.785398 / -0.785398): rotating the fingers to that end swings the pads
#: apart, and 0 closes them onto the gripper's symmetry plane. The left hand's range is
#: `[0, 0.7854]`, so its open end is 0.7854 - not 0.0, which is the closed end.
GRIPPER_OPEN = {"left": 0.7854, "right": -0.7854}
#: Nearly closed: the pads stop short of touching each other, and on an object the position
#: actuators press into it from there.
GRIPPER_CLOSED = {"left": 0.02, "right": -0.02}
EE_SITES = {"left": "left_ee_control_point", "right": "right_ee_control_point"}
EE_LINKS = {"left": "openarm_left_ee_base_link", "right": "openarm_right_ee_base_link"}

#: Graspable objects of the demo scene.
OBJECTS = ("red_cube", "blue_cube")
#: Target zones of the demo scene, each carrying a `<name>_site` marker in the MJCF.
ZONES = ("blue_zone", "green_zone")

#: Spawn pose for each arm, expressed in degrees.
#:
#: This is the official OpenArm v2 model keyframe: the elbow (joint 4) is at 90 degrees and all
#: other arm joints are at 0 degrees.  The request "spawn the joints at 90 degrees" has two
#: readings: the elbow already spawns at 90 degrees, while setting *all seven* arm joints to
#: 90 degrees makes the two arms interpenetrate (measured worst arm-arm penetration -0.072 m
#: on the compiled MJCF).  The project therefore keeps the official keyframe as the documented
#: spawn pose.
SPAWN_ARM_QPOS_DEG = np.array([0.0, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0])
SPAWN_ARM_QPOS = np.deg2rad(SPAWN_ARM_QPOS_DEG)

#: Fingertip midpoint of one hand, in the gripper control point frame: the midpoint between the
#: two fingertip pad centroids at the closed configuration (the `openarm_<side>_ee_{inner,outer}_
#: finger` bodies' `*_collision_00` geoms of the official model), on the gripper's symmetry plane.
#:
#: This is the point the IK aligns to the object centre: the fingertip pads then pinch the
#: object's side faces symmetrically about its centre, with the contact height centred on the
#: object. The constants are the measured values; `measured_fingertip_midpoint` re-derives them
#: from the installed model and the fidelity test compares the two.
FINGERTIP_MIDPOINT = {
    "left": np.array([-0.0066, 0.0, -0.1634]),
    "right": np.array([-0.0066, 0.0, -0.1634]),
}

#: Height at which the transport releases the object above the target zone (world z, metres).
#: The carried object's bottom face stays about 4 cm above the cell table during transport, and
#: the object then falls the rest of the way to rest under gravity.
RELEASE_HEIGHT = 1.07

#: Gripper approach orientation used by the verified demonstration: the fingers point forward
#: (+x world) so the pads close across the object's +-y side faces, tilted `GRASP_TILT_DEG`
#: nose-down. The tilt is what keeps the official palm geometry off the cell table at the pinch
#: alignment: with the fingertip midpoint at a resting cube's centre, the palm's lowest point sits
#: 2.2 cm above the table top at 10 degrees, versus 0.7 mm at 0 degrees (measured on the compiled
#: MJCF; see docs/ARCHITECTURE.md).
GRASP_TILT_DEG = 10.0
_TILT = np.deg2rad(GRASP_TILT_DEG)
GRASP_ROTATION = (
    np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    @ np.array(
        [
            [np.cos(_TILT), 0.0, np.sin(_TILT)],
            [0.0, 1.0, 0.0],
            [-np.sin(_TILT), 0.0, np.cos(_TILT)],
        ]
    )
)


def measured_fingertip_midpoint(xml: str) -> dict[str, np.ndarray]:
    """Measure each hand's fingertip midpoint from the official model description.

    The fingertip pad of one finger is that finger body's `*_collision_00` collision geom (the
    official hand's foremost, pad-facing segment); the fingertip midpoint is the midpoint of the
    two pads' vertex centroids at the closed configuration, expressed in the gripper control
    point frame. The model description is the authority for this geometry, so the constants in
    `FINGERTIP_MIDPOINT` must be this measurement, not an independent guess.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    for side in ("left", "right"):
        for joint in (GRIPPER_JOINTS[side], f"openarm_{side}_finger_joint2"):
            identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            data.qpos[model.jnt_qposadr[identifier]] = 0.0
    mujoco.mj_forward(model, data)

    midpoints = {}
    for side in ("left", "right"):
        ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_LINKS[side])
        ee_pos = data.xpos[ee].astype(float)
        ee_mat = data.xmat[ee].reshape(3, 3).astype(float)
        centroids = []
        for finger in ("inner", "outer"):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_ee_{finger}_finger")
            pad = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"finger_{finger}_{side}_collision_00"
            )
            assert model.geom_bodyid[pad] == body, "the official model's fingertip pad moved"
            mesh = model.geom_dataid[pad]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start : start + count].astype(float)
            world = vertices @ data.geom_xmat[pad].reshape(3, 3).T + data.geom_xpos[pad]
            centroids.append(((world - ee_pos) @ ee_mat).mean(axis=0))
        midpoints[side] = (centroids[0] + centroids[1]) / 2.0
    return midpoints
