"""Names and grasp geometry of the official OpenArm v2 bimanual Cell model.

Every name here exists in the official `openarm-mujoco` model, so any engine addresses the same robot.
The engine backend reuses them unchanged, so the agent, the skills and the verification step name the
same robot whatever engine runs it.
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
GRIPPER_OPEN = {"left": 0.0, "right": -0.7854}
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

#: Offset from the grasped object centre to the gripper control point, in world axes.
GRASP_OFFSET = np.array([0.083, 0.0, -0.019])
#: Gripper approach orientation used by the verified demonstration.
GRASP_ROTATION = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
