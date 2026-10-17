"""Record the verified red-cube task to a low-resolution MP4, as a backup for live demos.

The video is produced from a real run of the same engine and the same skills the app uses, and it is
only written when every skill and the independent verification passed: a failed run deletes the file
instead of shipping a clip that looks like a success.

Frames are sampled on *simulated time*, not on the number of frames already written, so the clip
shows the motion instead of the first sampling instant repeated.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run as a script: put the repository root on the path so `backend` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio

from backend.app.control.motion import OpenArmV2Controller
from backend.app.simulation import create_backend
from backend.app.skills.robot_skills import SkillRunner
from backend.app.verification.goal_verification import verify_place_goal

WIDTH, HEIGHT, FPS = 320, 240, 20
#: One recorded frame every this many simulated steps (dt = 1 ms, so 50 steps is 20 fps of sim time).
RECORD_EVERY_STEPS = 50
#: Static tail after the plan ends, long enough to show the cube settled in the zone.
TAIL_STEPS = 1000

PLAN = [
    {"skill": "approach_object", "arm": "left", "object": "red_cube"},
    {"skill": "grasp_object", "arm": "left", "object": "red_cube"},
    {"skill": "lift_object", "arm": "left", "object": "red_cube"},
    {"skill": "move_to_target", "arm": "left", "target": "blue_zone"},
    {"skill": "release_object", "arm": "left", "object": "red_cube"},
    {"skill": "retreat", "arm": "left"},
]


class Recorder:
    """Wraps `env.step` so the simulated motion is sampled at a fixed simulated-time interval."""

    def __init__(self, env, writer):
        self._step = env.step
        self._write = writer.append_data
        self._env = env
        self.steps = 0
        self._recorded = 0
        self.frames = 0

    def restore(self):
        """The original `env.step`, for putting back in a `finally` block."""
        return self._step

    def __call__(self, steps: int = 1) -> None:
        self._step(steps)
        self.steps += steps
        if self.steps // RECORD_EVERY_STEPS > self._recorded:
            self._recorded = self.steps // RECORD_EVERY_STEPS
            rendered = self._env.render(WIDTH, HEIGHT)
            if rendered is not None:
                self._write(rendered)
                self.frames += 1


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / "openarm-v2-demo.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    env = create_backend()
    skills = SkillRunner(env, OpenArmV2Controller(env))
    writer = imageio.get_writer(output, fps=FPS, codec="libx264", quality=8, macro_block_size=None)
    recorder = Recorder(env, writer)
    try:
        env.step = recorder
        results = skills.execute_plan(PLAN)
        verification = verify_place_goal(env, "red_cube", "blue_zone")
        for _ in range(TAIL_STEPS // RECORD_EVERY_STEPS):
            recorder(RECORD_EVERY_STEPS)
    finally:
        env.step = recorder.restore()
        writer.close()

    failed = [result for result in results if not result["success"]]
    if failed or not verification["success"]:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Refusing to save a failed demonstration as backup: skills={failed}, "
            f"verification={verification}"
        )
    print(output)
    print(f"frames={recorder.frames}, sim_time={recorder.steps / 1000:.1f}s, "
          f"skills={[result['success'] for result in results]}")
    print(f"grasp_mode={env.grasp_mode}, horizontal_error="
          f"{verification['horizontal_position_error'] * 1000:.1f} mm")


if __name__ == "__main__":
    main()
