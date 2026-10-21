"""Measure physical arm travel in simulation seconds, independently of viewer pacing.

Wine mode uses the production filled bottle, physical pinch and upright transport
helper. It does not run a pour. A prior native pour record can provide the measured
approach/grasp/clear_table baseline; its origin is retained in the output.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.control.motion import OpenArmV2Controller
from backend.app.simulation import create_backend
from backend.app.skills.robot_skills import SkillRunner
from backend.app.skills.pouring import move_held_vessel


class MeasuredController(OpenArmV2Controller):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.moves = []

    def move_to_site(self, side, position, *args, **kwargs):
        initial = self.env._sim_step
        wall = time.perf_counter()
        result = super().move_to_site(side, position, *args, **kwargs)
        self.moves.append({"target": np.asarray(position).tolist(),
                           "via_count": len(kwargs.get("via") or []),
                           "simulation_duration": (self.env._sim_step - initial) * self.timestep,
                           "wall_duration": time.perf_counter() - wall, **asdict(result)})
        return result


def measured_baseline(path):
    record = json.loads(path.read_text())
    if "phases" in record:
        if not record["success"]:
            raise ValueError("The comparison must be a successful physical measurement")
        return {phase["phase"]: phase["simulation_duration"] for phase in record["phases"]}
    last = record["initial_world_state"]["sim_time"]
    durations = {}
    for skill in record["skills"][:2]:
        end = skill["world_state"]["sim_time"]
        durations[skill["skill"]] = end - last
        last = end
    trace = record["skills"][2]["fluid_trace"]
    durations["transport"] = next(t["sim_time"] for t in trace if t["phase"] == "clear_table") - trace[0]["sim_time"]
    return durations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["cubes", "wine"], default="wine")
    parser.add_argument("--speed", type=float, default=6.)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--realtime-factor", type=float, default=1.)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--json", type=Path, default=ROOT / "outputs/arm_motion.json")
    args = parser.parse_args()
    os.environ["GENESIS_SHOW_VIEWER"] = "1" if args.native else "0"
    os.environ["OPENARM_SIM_SPEED"] = str(args.realtime_factor)
    env = create_backend(args.scene)
    controller = MeasuredController(env, motion_speed=args.speed)
    runner = SkillRunner(env, controller)
    obj = "wine_bottle" if args.scene == "wine" else "red_cube"
    side = "left"
    result = {"scene": args.scene, "motion_speed": args.speed,
              "viewer_realtime_factor": env.simulation_speed, "physics_dt": env.timestep,
              "compute_backend": env.compute_backend, "phases": [], "success": False,
              "scope": "physical approach, grasp/lift and upright A-to-B transfer; excludes pouring"}
    baseline = measured_baseline(args.baseline) if args.baseline else {}
    if args.baseline:
        result["baseline_source"] = str(args.baseline.resolve())
        result["baseline_scope"] = "previous measured physical run; compare simulation seconds, not wall-clock playback"
    samples = []
    liquid_samples = []

    def save():
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    def observe():
        if env._sim_step % max(1, round(.01 / env.timestep)) == 0:
            samples.append((env._sim_step * env.timestep, env.ee_pose(side)[0].tolist()))
        if args.scene == "wine" and env._sim_step % max(1, round(.05 / env.timestep)) == 0:
            liquid_samples.append({"sim_time": env._sim_step * env.timestep, **env.fluid_state()})

    def phase(name, action):
        samples.clear()
        move_index = len(controller.moves)
        before = env._sim_step
        wall = time.perf_counter()
        evidence = action()
        elapsed = (env._sim_step - before) * env.timestep
        speeds = [np.linalg.norm(np.asarray(b[1]) - a[1]) / (b[0] - a[0]) for a, b in zip(samples, samples[1:])]
        measured = {"phase": name, "simulation_duration": elapsed,
                    "wall_duration": time.perf_counter() - wall,
                    "peak_measured_ee_speed_m_s": float(max(speeds, default=0.)),
                    "moves": controller.moves[move_index:], "evidence": evidence}
        if name in baseline:
            measured.update(before_simulation_duration=baseline[name], speedup=baseline[name] / elapsed)
        result["phases"].append(measured)
        save()
        print(name, "sim_s", elapsed, "speedup", measured.get("speedup"),
              "success", evidence.get("success", True), flush=True)
        if evidence.get("success") is False:
            raise RuntimeError(str(evidence))

    try:
        env.step(200)
        env.step_observer = observe
        phase("approach_object", lambda: runner.approach_object(side, obj))
        phase("grasp_object", lambda: runner.grasp_object(side, obj))
        position, rotation = env.object_pose(obj)
        if args.scene == "wine":
            target = np.array([env.zone_position("wine_glass")[0], position[1], max(position[2], 1.12)])
            phase("transport", lambda: move_held_vessel(runner, side, obj, "wine_glass", target,
                                                        rotation, "clear_table", 1.2, transport=True))
        else:
            target = position + [.12, 0., 0.]
            phase("transport", lambda: runner._carry_to(side, obj, target))
        env.step_observer = None
        env.step(max(1, round(.2 / env.timestep)))
        result["final_contacts"] = env.object_contacts(obj, side)
        actual = env.object_pose(obj)[0]
        result["target"] = target.tolist()
        result["actual_position"] = actual.tolist()
        result["position_error"] = float(np.linalg.norm(actual - target))
        result["final_liquid"] = env.fluid_state() if args.scene == "wine" else None
        result["liquid_samples"] = liquid_samples
        result["maximum_sampled_spill_ml"] = max((sample["spilled_ml"] for sample in liquid_samples), default=0.)
        result["maximum_sampled_airborne_ml"] = max((sample["in_flight_ml"] for sample in liquid_samples), default=0.)
        liquid = result["final_liquid"]
        result["success"] = (len(set(result["final_contacts"])) >= 2 and result["position_error"] <= .012
                             and (not liquid or (liquid["finite"] and liquid["spilled_ml"] == 0.
                                                 and liquid["in_flight_ml"] == 0.
                                                 and liquid["bottle_ml"] == liquid["initial_ml"]
                                                 and result["maximum_sampled_spill_ml"] == 0.
                                                 and result["maximum_sampled_airborne_ml"] == 0.)))
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
        print(result["error"], flush=True)
    finally:
        env.step_observer = None
        save()
        env.close()
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
