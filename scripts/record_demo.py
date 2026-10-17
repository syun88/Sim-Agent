from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from backend.app.control.motion import OpenArmV2Controller
from backend.app.simulation.openarm_v2.environment import OpenArmV2Environment
from backend.app.skills.robot_skills import SkillRunner


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / "openarm-v2-demo.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    env = OpenArmV2Environment()
    controller = OpenArmV2Controller(env)
    skills = SkillRunner(env, controller)

    width, height, fps = 320, 240, 20
    writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8, macro_block_size=None)

    original_step = env.step
    frame_count = 0

    def recording_step(steps: int = 1) -> None:
        nonlocal frame_count
        original_step(steps)
        if frame_count % 50 == 0:
            rendered = env.render(width, height)
            if rendered is not None:
                writer.append_data(rendered)
                frame_count += 1

    env.step = recording_step
    plan = [
        {"skill": "approach_object", "arm": "left", "object": "red_cube"},
        {"skill": "grasp_object", "arm": "left", "object": "red_cube"},
        {"skill": "lift_object", "arm": "left", "object": "red_cube"},
        {"skill": "move_to_target", "arm": "left", "target": "blue_zone"},
        {"skill": "release_object", "arm": "left", "object": "red_cube"},
        {"skill": "retreat", "arm": "left"},
    ]
    results = skills.execute_plan(plan)
    for _ in range(200):
        original_step(50)
        rendered = env.render(width, height)
        if rendered is not None:
            writer.append_data(rendered)
            frame_count += 1
    writer.close()
    if not all(result["success"] for result in results):
        output.unlink(missing_ok=True)
        raise RuntimeError("Refusing to save failed demonstration as backup")
    print(output)
    print(f"frames={frame_count}, skills={[r['success'] for r in results]}")


if __name__ == "__main__":
    main()
