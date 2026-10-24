"""Measure a mobile base's open-loop drive model in Genesis and store it.

System identification for an imported wheeled robot: command each body
twist axis in turn, measure the achieved body-frame twist from the
backend's own measured base pose, and assemble the 3x3 model A
(achieved = A @ commanded). The generic environment reads the stored
`drive_calibration.json` from the robot's profile directory and commands
through A^-1, cancelling the leak between axes (CAD omni wheels scrub;
on LeKiwi a forward command otherwise yaws hard).

Usage: uv run python scripts/calibrate_mobile_base.py <robot_id>
"""

import math
import sys

import numpy as np

from backend.app.simulation import create_backend


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def measure(env, cmd, spinup=0.5, window=1.5, sample=0.1):
    env.reset()
    for _ in range(100):
        env.step(1)
    env.command_base_velocity(*cmd)
    for _ in range(int(spinup / env.timestep)):
        env.step(1)
    twists = []
    prev = env.base_pose()
    steps = int(sample / env.timestep)
    for _ in range(int(window / sample)):
        for _ in range(steps):
            env.step(1)
        now = env.base_pose()
        dx, dy = now[0] - prev[0], now[1] - prev[1]
        c, s = math.cos(prev[2]), math.sin(prev[2])
        twists.append([(c * dx + s * dy) / sample,
                       (-s * dx + c * dy) / sample,
                       _wrap(now[2] - prev[2]) / sample])
        prev = now
    env.command_base_velocity(0., 0., 0.)
    return np.mean(twists, axis=0)


def main():
    robot_id = sys.argv[1]
    env = create_backend(robot_id=robot_id)
    basis = [(0.1, 0., 0.), (0., 0.1, 0.), (0., 0., 0.5)]
    model = np.zeros((3, 3))
    for i, cmd in enumerate(basis):
        achieved = measure(env, cmd)
        model[:, i] = achieved / np.asarray(cmd)[i]
        print(f"cmd {cmd}: achieved {np.round(achieved, 3)}")
    print("drive model A =\n", np.round(model, 3))
    from pathlib import Path
    import json
    target = Path(env.profile["directory"]) / "drive_calibration.json"
    target.write_text(json.dumps({"model": model.tolist()}, indent=2) + "\n")
    print("wrote", target)
    # Held-out check: a combined twist the model never saw.
    cmd = np.array([0.08, 0.05, 0.3])
    achieved = measure(env, cmd)
    print(f"held-out cmd {cmd}: raw achieved {np.round(achieved, 3)} "
          f"(model predicts {np.round(model @ cmd, 3)}; re-run this script "
          "after the environment picks the calibration up to see it applied)")


if __name__ == "__main__":
    main()
