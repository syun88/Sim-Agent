"""Fixed-vessel Genesis diagnostic: containment, gravity-driven flow, and surface.

This does not establish a robot grasp. Vessel/particle setters are used once for
each explicit diagnostic setup; all subsequent liquid motion is Genesis SPH.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import backend.app.simulation.genesis_backend.wine as wine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", type=float, default=wine.PARTICLE_SIZE)
    parser.add_argument("--mu", type=float, default=wine.LIQUID_MU)
    parser.add_argument("--gamma", type=float, default=wine.LIQUID_GAMMA)
    parser.add_argument("--stiffness", type=float, default=wine.LIQUID_STIFFNESS)
    parser.add_argument("--substeps", type=int, default=wine.WINE_SUBSTEPS)
    parser.add_argument("--tilt", type=float, default=110.)
    parser.add_argument("--pressure-solver", choices=["WCSPH", "DFSPH"], default=wine.PRESSURE_SOLVER)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Compare the same Genesis physics on the CPU")
    parser.add_argument("--hash-grid-size", type=float, default=wine.HASH_GRID_SIZE[0])
    parser.add_argument("--free-hold-only", action="store_true", help="Test the dynamic bottle resting on the table")
    parser.add_argument("--hold-seconds", type=float, default=2.)
    parser.add_argument("--json", type=Path, default=ROOT / "outputs/wine_fluid_validation.json")
    args = parser.parse_args()
    wine.PARTICLE_SIZE, wine.LIQUID_MU, wine.LIQUID_GAMMA = args.spacing, args.mu, args.gamma
    wine.LIQUID_STIFFNESS = args.stiffness
    bound = wine.validate_fluid_timestep(args.spacing, args.mu, wine.WINE_DT / args.substeps)
    import genesis as gs
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False,
                     sim_options=gs.options.SimOptions(dt=wine.WINE_DT, substeps=args.substeps),
                     rigid_options=gs.options.RigidOptions(constraint_solver=gs.constraint_solver.Newton,
                                                            contact_resolution=gs.contact_resolution.convex, impratio=10.),
                     sph_options=gs.options.SPHOptions(particle_size=args.spacing, pressure_solver=args.pressure_solver,
                                                       hash_grid_res=(args.hash_grid_size,) * 3,
                                                       lower_bound=(.05, -.55, .80), upper_bound=(.9, .55, 1.75)))
    scene.add_entity(gs.morphs.Plane(pos=(0., 0., 1.006)), surface=gs.surfaces.Default(color=(.65, .65, .62, 1.)))
    world = wine.WineWorld(scene, gs, fixed_bottle=not args.free_hold_only, async_surface=False)
    camera = scene.add_camera(res=(768, 576), pos=(.85, -.24, 1.38), lookat=(.60, .07, 1.13), fov=50, GUI=False)
    scene.build()
    world.capture_setup()
    start = time.perf_counter()
    hold_trace = []
    for index in range(int(args.hold_seconds / wine.WINE_DT)):
        scene.step()
        if index % 250 == 249:
            hold_trace.append({"time": (index + 1) * wine.WINE_DT,
                               "max_speed": float(world.liquid.get_particles_vel().norm(dim=-1).max().cpu()),
                               **world.measure()})
            print("hold_progress", {key: hold_trace[-1][key] for key in
                                   ("time", "max_speed", "bottle_ml", "spilled_ml", "in_flight_ml")}, flush=True)
    hold = world.measure()
    hold_speed = float(world.liquid.get_particles_vel().norm(dim=-1).max().cpu())
    velocity_quantiles = np.percentile(world.liquid.get_particles_vel().norm(dim=-1).cpu().numpy(), [50, 95, 99, 100]).tolist()
    pressure_quantiles = np.percentile(scene.sph_solver.particles.p.to_numpy(), [50, 95, 99, 100]).tolist()
    contained = hold["finite"] and hold["bottle_ml"] >= hold["initial_ml"] * .999 and hold_speed < .1
    print("hold", contained, hold, "max_speed", hold_speed, flush=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    if args.record:
        from PIL import Image
        initial = camera.render(rgb=True)[0]
        Image.fromarray(initial).save(args.json.with_name(args.json.stem + "-initial.png"))
    trace = []
    if contained and not args.free_hold_only:
        rotation = Rotation.from_euler("y", args.tilt, degrees=True)
        mouth = wine.GLASS_START + [-.005, 0., .161]
        position = mouth - rotation.apply(wine.BOTTLE_MOUTH)
        clearance = wine.vessel_clearance(position, rotation.as_matrix(), wine.GLASS_START, np.eye(3))
        if clearance < .001:
            raise RuntimeError(f"Diagnostic vessels intersect or lack clearance: {clearance:g}m")
        world.reset(tilt_degrees=args.tilt, bottle_position=position)
        for index in range(1500):
            scene.step()
            if index % 20 == 0:
                trace.append({"time": index * wine.WINE_DT, **world.measure()})
                if args.record:
                    frames.append(camera.render(rgb=True)[0])
        final = world.measure()
    else:
        final = hold
    success = bool(contained and final["finite"] and final["particle_count"] == hold["particle_count"]
                   and final["glass_ml"] > 5. and final["in_flight_ml"] < 2.
                   and final["spilled_ml"] <= wine.MAX_SPILL_ML)
    if args.free_hold_only:
        success = bool(contained and hold["spilled_ml"] <= wine.MAX_SPILL_ML)
    record = {"success": success, "diagnostic": "free_bottle_hold" if args.free_hold_only else "fixed_vessels_not_robot_grasp",
              "config": {"particle_size": args.spacing, "mu": args.mu, "gamma": args.gamma,
                         "compute_backend": gs.backend.name,
                         "hash_grid_size": args.hash_grid_size,
                         "stiffness": args.stiffness,
                         "dt": wine.WINE_DT, "substeps": args.substeps, "viscosity_dt_bound": bound,
                         "tilt_degrees": args.tilt, "maximum_spill_ml": wine.MAX_SPILL_ML,
                         "pressure_solver": args.pressure_solver},
              "hold": hold, "hold_max_speed": hold_speed, "final": final, "trace": trace,
              "hold_trace": hold_trace,
              "velocity_quantiles": velocity_quantiles, "pressure_quantiles": pressure_quantiles,
              "elapsed_seconds": time.perf_counter() - start}
    args.json.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    if frames:
        import imageio.v2 as imageio
        from PIL import Image
        Image.fromarray(frames[len(frames)//3]).save(args.json.with_name(args.json.stem + "-flow.png"))
        imageio.mimwrite(args.json.with_name(args.json.stem + "-flow.mp4"), frames, fps=25, codec="libx264")
    print("flow", success, final, flush=True)
    world.surface_renderer.close()
    scene.destroy()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
