"""Measure the grasp geometry the skills use, from the official OpenArm v2 model.

The official `openarm-mujoco` model description is the authority for the grasp geometry, so the
constants in `backend/app/simulation/openarm_v2/definition.py` must be this script's measurements:

- `FINGERTIP_MIDPOINT` - the midpoint between the two fingertip pad centroids of one hand, in the
  gripper control point frame, at the closed configuration. This is the point the IK aligns to
  the object centre.
- `GRIPPER_OPEN` / `GRIPPER_CLOSED` - checked against the official finger joint ranges and the
  keyframe, which holds both hands fully open.

Run with `--check` to assert that the constants in `definition.py` equal the measurements
(the same assertion runs in the automated test suite on every machine).

The script also prints the clearance diagnostics the approach orientation was chosen from: the
palm's lowest point above the cell table when the fingertip midpoint sits on a resting cube's
centre, at several nose-down tilts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.simulation.openarm_v2.definition import (  # noqa: E402
    FINGERTIP_MIDPOINT,
    GRASP_ROTATION,
    GRASP_TILT_DEG,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    EE_LINKS,
    GRIPPER_JOINTS,
    measured_fingertip_midpoint,
)
from backend.app.simulation.openarm_v2.scene import (  # noqa: E402
    INITIAL_OBJECT_POSES,
    resolved_scene_xml,
)

#: Table top of the official cell (`cell_table_col`: z = 1.00, half-thickness 0.005).
TABLE_TOP = 1.005


def check_gripper_travel(model: mujoco.MjModel) -> list[str]:
    """The open command must be the keyframe end of the finger range; closed is near zero."""
    problems = []
    for side in ("left", "right"):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINTS[side])
        low, high = model.jnt_range[joint]
        key = model.key_qpos[0][model.jnt_qposadr[joint]]
        if abs(GRIPPER_OPEN[side] - key) > 1e-3:
            problems.append(f"{side} GRIPPER_OPEN {GRIPPER_OPEN[side]} is not the keyframe {key:.4f}")
        if abs(GRIPPER_OPEN[side] - (low if key > 0 else high)) > 1e-3 and abs(GRIPPER_OPEN[side] - (high if key > 0 else low)) > 1e-3:
            problems.append(f"{side} GRIPPER_OPEN {GRIPPER_OPEN[side]} is not a range end of [{low}, {high}]")
        if not (low - 1e-6 <= GRIPPER_CLOSED[side] <= high + 1e-6):
            problems.append(f"{side} GRIPPER_CLOSED {GRIPPER_CLOSED[side]} is outside the range [{low}, {high}]")
    return problems


def palm_clearance(model: mujoco.MjModel, data: mujoco.MjData, tilt_deg: float) -> float:
    """Palm clearance above the cell table when the fingertip midpoint sits on a resting cube.

    The resting cube centre is `INITIAL_OBJECT_POSES['red_cube']`; at tilt `tilt_deg` the control
    point sits at `cube - R_tilt @ FINGERTIP_MIDPOINT`, and the clearance is the palm mesh's
    lowest world point minus the table top.
    """
    tilt = np.deg2rad(tilt_deg)
    rot_y = np.array(
        [[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0], [-np.sin(tilt), 0, np.cos(tilt)]]
    )
    rotation = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]) @ rot_y
    cube = np.asarray(INITIAL_OBJECT_POSES["red_cube"], float)
    control_point = cube - rotation @ FINGERTIP_MIDPOINT["left"]

    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_LINKS["left"])
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_kinematics(model, data)
    ee_mat = data.xmat[ee].reshape(3, 3).astype(float)
    # rotate the hand to `rotation`: R_world = rotation @ R_ee^{-1}

    # Palm vertices in the ee frame (from the measurement configuration, q closed) - recompute here
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ee_base_link_left_collision_00")
    mesh = model.geom_dataid[palm]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[start : start + count].astype(float)
    local = (vertices @ data.geom_xmat[palm].reshape(3, 3).T + data.geom_xpos[palm]) - data.xpos[ee]
    local = local @ ee_mat  # palm vertices in the ee frame
    world = (rotation @ local.T).T + control_point
    return float(world[:, 2].min() - TABLE_TOP)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="assert definition.py matches the model")
    args = parser.parse_args()

    xml = resolved_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    measured = measured_fingertip_midpoint(xml)
    print("Fingertip midpoint (control-point frame, per side):")
    for side, value in measured.items():
        print(f"  {side}: ({value[0]:+.5f}, {value[1]:+.5f}, {value[2]:+.5f})  definition.py: "
              f"{FINGERTIP_MIDPOINT[side][0]:+.4f}, {FINGERTIP_MIDPOINT[side][1]:+.4f}, {FINGERTIP_MIDPOINT[side][2]:+.4f}")

    print("\nWorld grasp offset at the documented orientation "
          f"(GRASP_ROTATION, tilt {GRASP_TILT_DEG} deg), object centre relative to control point:")
    for side in ("left", "right"):
        offset = GRASP_ROTATION @ FINGERTIP_MIDPOINT[side]
        print(f"  {side}: ({offset[0]:+.4f}, {offset[1]:+.4f}, {offset[2]:+.4f})")

    print("\nPalm clearance above the cell table at the pinch alignment, by nose-down tilt:")
    for tilt in (0.0, 5.0, 10.0, 15.0):
        print(f"  tilt {tilt:4.1f} deg: {palm_clearance(model, data, tilt) * 1000:+7.1f} mm")

    problems = check_gripper_travel(model)
    for side in ("left", "right"):
        if np.linalg.norm(measured[side] - FINGERTIP_MIDPOINT[side]) > 1e-4:
            problems.append(
                f"{side} FINGERTIP_MIDPOINT {FINGERTIP_MIDPOINT[side].tolist()} differs from the "
                f"measured {measured[side].tolist()}"
            )
    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nAll grasp-geometry constants match the official model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
