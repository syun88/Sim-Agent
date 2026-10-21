"""Language -> production executor -> measured Genesis throw and SPH pouring.

No object/particle pose writes occur inside a task. Scene reset is explicit between
cases. A zero exit code requires the measured postconditions, not planner success.
"""
from pathlib import Path
import argparse
import json
import logging
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.execution.task_executor import TaskExecutor
from backend.app.planning.agent_planner import AgentPlanner
from backend.app.simulation import create_backend

CASES = {
    "throw_right": ("cubes", "右手で青いキューブを上に投げて"),
    "throw_left": ("cubes", "左手で赤いキューブを上に投げて"),
    "pour": ("wine", "ワインをグラスに20ml注いで"),
    "pour_10": ("wine", "Pour 10 ml of wine into the glass"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--provider", choices=["fallback", "local"], default="fallback")
    parser.add_argument("--json", type=Path, default=ROOT / "outputs/dynamic_tasks.json")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--record", action="store_true", help="Record actual pouring motion and its final state")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    os.environ["OPENARM_LOCAL_PLANNER"] = "1" if args.provider == "local" else "0"
    planner = AgentPlanner()
    planner.api_key = planner.base_url = planner.model = ""
    env, executor, profile = None, None, None
    records = []
    for name in CASES if args.case == "all" else [args.case]:
        requested, instruction = CASES[name]
        if requested != profile:
            if env is not None:
                env.close()
            env = create_backend(requested)
            executor = TaskExecutor(env)
            profile = requested
        env.reset()
        executor.reset_context()
        env.step(200)
        print(f"{name}: {instruction}; device={env.compute_backend}", flush=True)
        recording = False
        writer = None
        original_step = env.step
        if args.record and requested == "wine":
            import imageio.v2 as imageio
            from PIL import Image
            args.json.parent.mkdir(parents=True, exist_ok=True)
            env._camera.set_pose(pos=(.85, -.24, 1.38), lookat=(.60, .07, 1.13))
            image = env.render()
            if image is None:
                raise RuntimeError("Could not render the actual wine scene")
            Image.fromarray(image).save(args.json.with_name(args.json.stem + "-initial.png"))
            writer = imageio.get_writer(args.json.with_suffix(".mp4"), fps=25, codec="libx264")
            interval = max(1, round(.04 / env.timestep))
            recorded_steps = 0
            largest_flow = -1.
            def record_step(steps=1):
                nonlocal recorded_steps, largest_flow
                if not recording:
                    return original_step(steps)
                for _ in range(steps):
                    original_step()
                    recorded_steps += 1
                    if recorded_steps % interval == 0:
                        image = env.render()
                        if image is None:
                            raise RuntimeError("Wine rendering failed during the recorded task")
                        writer.append_data(image)
                        liquid = env.fluid_state()
                        if liquid["in_flight_ml"] > largest_flow:
                            largest_flow = liquid["in_flight_ml"]
                            Image.fromarray(image).save(args.json.with_name(args.json.stem + "-flow.png"))
            env.step = record_step
        def emit(kind, **payload):
            nonlocal recording
            if kind == "skill_started":
                print("start", payload["step"]["skill"], flush=True)
                recording = payload["step"]["skill"] == "pour_liquid"
            elif kind == "skill_finished":
                r = payload["result"]
                print("finish", r["skill"], r["success"], r.get("detail", ""), flush=True)
        try:
            plan = planner.plan(instruction, executor.world_state())
            result = executor.run(plan, emit=emit)
        except Exception as exc:
            result = {"success": False, "error": type(exc).__name__ + ": " + str(exc)}
        finally:
            env.step = original_step
            if writer is not None:
                writer.close()
                image = env.render()
                if image is not None:
                    Image.fromarray(image).save(args.json.with_name(args.json.stem + "-final.png"))
        physics = None
        if profile == "wine":
            from backend.app.simulation.genesis_backend import wine
            physics = {"particle_size": wine.PARTICLE_SIZE, "mu": wine.LIQUID_MU,
                       "gamma": wine.LIQUID_GAMMA, "stiffness": wine.LIQUID_STIFFNESS,
                       "dt": wine.WINE_DT, "substeps": wine.WINE_SUBSTEPS,
                       "pressure_solver": wine.PRESSURE_SOLVER, "hash_grid_size": wine.HASH_GRID_SIZE,
                       "maximum_spill_ml": wine.MAX_SPILL_ML}
        records.append({"case": name, "scene": profile, "instruction": instruction,
                        "physics": physics, "result": result})
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(name, result["success"], result.get("verification", result.get("error")), flush=True)
        if not result["success"]:
            return 1
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
