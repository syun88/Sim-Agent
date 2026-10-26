"""CC0 hollow vessels and real SPH liquid in the same Genesis scene as OpenArm.

Only reset/setup writes vessel or particle poses. During a task the bottle moves
through finger contacts and liquid moves through Genesis's rigid/SPH coupling.
"""
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).resolve().parents[4] / "assets" / "wine"
BOTTLE_START = np.array([.44, .29, 1.091])
GLASS_START = np.array([.62, .06, 1.006])
BOTTLE_MOUTH = np.array([0., 0., .125])
PARTICLE_SIZE = .0025
# Small-scale resolved liquid coefficients; these are not calibrated wine data.
# Genesis's explicit viscosity acts in its acceleration equation. At this
# resolution mu=.003 requires a substep below 83.3 microseconds; the former
# 0.5ms substep was unstable. The current 62.5us substep also damps the dynamic
# vessel's pressure ringing, which a fixed-vessel-only diagnostic did not expose.
# These coefficients are not a calibrated material model of a particular wine.
LIQUID_MU = .003
LIQUID_GAMMA = .00015
LIQUID_STIFFNESS = 500.
WINE_DT = .002
WINE_SUBSTEPS = 32
MAX_SPILL_ML = .1
PRESSURE_SOLVER = "WCSPH"
# This is the periodic neighbor-hash size, not the simulation boundary. Genesis
# still rejects candidates outside the support radius using their real positions.
# A 150^3 default grid cleared millions of empty slots for only 3,600 particles.
HASH_GRID_SIZE = (.16, .16, .16)


@lru_cache(maxsize=1)
def _glass_shell_samples():
    import trimesh
    mesh = trimesh.load(ASSETS / "glass_collision.obj", force="mesh", process=False)
    return np.vstack((mesh.vertices, mesh.triangles_center))


def vessel_clearance(bottle_position, bottle_rotation, glass_position, glass_rotation):
    """Distance of glass shell samples from the bottle's physical contact proxies.

    Fixed/fixed objects do not produce rigid contacts in Genesis. Check this
    geometry explicitly so a fixed-vessel fluid diagnostic cannot silently pass
    with the neck inserted through the receiving wall.
    """
    glass_world = _glass_shell_samples() @ np.asarray(glass_rotation).T + glass_position
    local = (glass_world - bottle_position) @ np.asarray(bottle_rotation)
    clearances = []
    for center_z, radius, half_height in ((-.0122, .02625, .0728), (.10995, .00875, .01505)):
        distance = np.column_stack((np.linalg.norm(local[:, :2], axis=1) - radius,
                                    np.abs(local[:, 2] - center_z) - half_height))
        signed = np.linalg.norm(np.maximum(distance, 0.), axis=1) + np.minimum(distance.max(axis=1), 0.)
        clearances.append(float(signed.min()))
    return min(clearances)


def validate_fluid_timestep(particle_size, viscosity, substep_dt):
    # Genesis 1.4.4 SPHSolver.build derives this bound from its explicit
    # non-pressure viscosity force. Check it before constructing the viewer,
    # rather than losing the engine warning at logging_level='error'.
    stable_dt = .01 * (2. * particle_size) ** 2 / viscosity if viscosity else float("inf")
    if substep_dt > stable_dt * (1. + 1e-6):
        raise ValueError(f"SPH substep {substep_dt:g}s exceeds viscosity stability bound {stable_dt:g}s")
    return stable_dt


def inside_cavity(points, inner_z, inner_radius, margin):
    """Particle footprints overlapping a rotational cavity, with metric margin.

    Padding radius alone understates the margin on a sloping floor. Use distance
    to the actual r/z profile so contained particles near that floor are not
    reported as table spills when their centres lie within the SPH footprint.
    """
    rz = np.column_stack((np.linalg.norm(points[:, :2], axis=1), points[:, 2]))
    inside = ((rz[:, 0] <= np.interp(rz[:, 1], inner_z, inner_radius))
              & (rz[:, 1] >= inner_z[0]) & (rz[:, 1] <= inner_z[-1]))
    boundary = np.column_stack((inner_radius, inner_z))
    boundary = np.vstack((boundary, [0., inner_z[-1]], boundary[0]))
    start, segment = boundary[:-1], np.diff(boundary, axis=0)
    delta = rz[:, None, :] - start
    fraction = np.clip((delta * segment).sum(axis=2) / np.maximum((segment * segment).sum(axis=1), 1e-20), 0., 1.)
    distance_squared = ((delta - fraction[:, :, None] * segment) ** 2).sum(axis=2).min(axis=1)
    return inside | (distance_squared <= margin ** 2)


def vessel_xml(name: str, fixed: bool) -> str:
    bottle = name == "wine_bottle"
    kind = "bottle" if bottle else "glass"
    rgba = ".24 .43 .28 .28" if bottle else ".65 .84 .91 .30"
    visuals = ["bottle_open"] if bottle else ["glass_base", "glass_stem", "glass_bowl"]
    visual_assets = "".join(f'<mesh name="{part}" file="{ASSETS / (part + ".obj")}"/>' for part in visuals)
    visual_geoms = "".join(f'<geom group="1" type="mesh" mesh="{part}" contype="0" conaffinity="0" mass="0" rgba="{rgba}"/>'
                           for part in visuals)
    # Genesis renders MJCF groups 0-2, including collision geoms with alpha=0.
    # Keep physical proxies/SDF shells in group 3 to avoid overlapping surfaces.
    contact = '''<geom name="body_contact" group="3" type="cylinder" pos="0 0 -.0122" size=".02625 .0728"
        mass=".112" friction="1.2 .02 .002" rgba="0 0 0 0"/>
        <geom name="neck_contact" group="3" type="cylinder" pos="0 0 .10995" size=".00875 .01505"
        mass=".008" friction="1.2 .02 .002" rgba="0 0 0 0"/>''' if bottle else ""
    return f'''<mujoco model="{name}"><compiler angle="radian"/>
      <asset>{visual_assets}
      <mesh name="shell" file="{ASSETS / (kind + '_collision.obj')}"/></asset>
      <worldbody><body name="{name}" pos="0 0 0">
      {'' if fixed else '<freejoint/>'}
      {visual_geoms}
      {contact}
      <geom group="3" type="mesh" mesh="shell" mass="{0. if bottle else .08}"
        contype="{1073741824 if bottle else 1}" conaffinity="{0 if bottle else 1}"
        friction="1.2 .02 .002" rgba="0 0 0 0"/>
      </body></worldbody></mujoco>'''


class WineWorld:
    def __init__(self, scene, gs, *, fixed_bottle=False, async_surface=True,
                 bottle_position=None, glass_position=None):
        # The vessel positions are the official cell's constants by default. An imported arm has its own table height
        # and its own reach, so the caller may place them: measured 2026-10-27, the official bottle sits 0.69 m from a
        # 0.3 m arm and every approach posture is rejected for want of reach, not for want of a fluid model.
        self.bottle_start = np.asarray(BOTTLE_START if bottle_position is None else bottle_position, float)
        self.glass_start = np.asarray(GLASS_START if glass_position is None else glass_position, float)
        self.scene = scene
        self.solver = "SPH"
        material = gs.materials.Rigid(needs_coup=True, coup_friction=.2, coup_softness=.0003,
                                     sdf_cell_size=.0004, sdf_min_res=128, sdf_max_res=512)
        self.bottle = scene.add_entity(gs.morphs.MJCF(file=vessel_xml("wine_bottle", fixed_bottle), pos=tuple(self.bottle_start),
                                                    convexify=False, watertighten=None), material=material,
                                      surface=gs.surfaces.Default(vis_mode="visual", roughness=.15, double_sided=False))
        # One physical bottle, with convex grip/support contacts and a hollow SDF
        # for fluid. The contact proxies must not turn the liquid cavity solid.
        for geom in self.bottle.geoms:
            if geom.type != gs.GEOM_TYPE.MESH:
                geom._needs_coup = False
        self.glass = scene.add_entity(gs.morphs.MJCF(file=vessel_xml("wine_glass", True), pos=tuple(self.glass_start),
                                                   convexify=False, watertighten=None),
                                      material=material.model_copy(update={"coup_friction": .8}),
                                      surface=gs.surfaces.Default(vis_mode="visual", roughness=.12, double_sided=False))
        self.liquid = scene.add_entity(
            gs.morphs.Cylinder(pos=tuple(self.bottle_start + [0, 0, -.085 + .012 + .020]), radius=.021, height=.040),
            material=gs.materials.SPH.Liquid(rho=990., mu=LIQUID_MU, gamma=LIQUID_GAMMA,
                                              stiffness=LIQUID_STIFFNESS, sampler="regular"),
            surface=gs.surfaces.Default(color=(.42, .008, .025, 1.), roughness=.12,
                                        vis_mode="recon"), name="red_wine")
        from .fluid_surface import LiquidSurfaceRenderer
        self.surface_renderer = LiquidSurfaceRenderer(scene, self.liquid, asynchronous=async_surface)
        self.initial_particles = None
        self.ml_per_particle = 0.

    def capture_setup(self):
        self.initial_particles = self.liquid.get_particles_pos().cpu().numpy().copy() - self.bottle_start
        # Genesis assigns each SPH particle 0.8 * spacing^3 reference volume,
        # not a full spacing cube. Use the engine's actual mass/volume model.
        self.ml_per_particle = float(self.scene.sph_solver.particle_volume) * 1e6

    def reset(self, *, tilt_degrees=0., bottle_position=None):
        self.surface_renderer.reset()
        position = self.bottle_start if bottle_position is None else np.asarray(bottle_position, float)
        rotation = Rotation.from_euler("y", tilt_degrees, degrees=True)
        q = rotation.as_quat()
        # Explicit scene setup only. No skill calls these setters.
        self.bottle.set_pos(position.astype(np.float32), zero_velocity=True)
        self.bottle.set_quat(q[[3, 0, 1, 2]].astype(np.float32), zero_velocity=True)
        points = self.initial_particles @ rotation.as_matrix().T + position
        self.liquid.set_particles_pos(points.astype(np.float32))
        self.liquid.set_particles_vel(np.zeros_like(points, dtype=np.float32))
        self.liquid.set_particles_active(np.ones(len(points), dtype=np.int32))

    def pose(self, entity):
        p = entity.get_pos().cpu().numpy().astype(float)
        q = entity.get_quat().cpu().numpy().astype(float)
        return p, Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()

    def measure(self):
        points = self.liquid.get_particles_pos().cpu().numpy().astype(float)
        active = self.liquid.get_particles_active().cpu().numpy().astype(bool)
        bp, br = self.pose(self.bottle)
        gp, gr = self.pose(self.glass)
        local = (points - bp) @ br
        # The shoulder/neck are narrower than the body. Use the hollow mesh's
        # actual inner profile so particles beside the neck aren't counted inside.
        inner_z = np.array([.007, .012, .208, .233, .257, .300]) * .70 - .085
        inner_r = np.array([0., .035, .035, .0285, .010, .010]) * .70
        in_bottle = inside_cavity(local, inner_z, inner_r, PARTICLE_SIZE / 2)
        glass_local = (points - gp) @ gr
        # Radius of the source glass's inner bowl, interpolated from its mesh profile.
        profile = np.array([[.094, 0.], [.096, .016], [.102, .0232], [.116, .0338], [.130, .0392],
                            [.163, .0402], [.179, .03744], [.195, .0332], [.220, .0292]])
        in_glass = ~in_bottle & inside_cavity(glass_local, profile[:, 0] * .65, profile[:, 1] * .65, PARTICLE_SIZE / 2)
        spilled = active & ~in_glass & ~in_bottle & (points[:, 2] < gp[2] + .03)
        return {"solver": self.solver, "pressure_solver": self.scene.sph_solver._pressure_solver,
                "particle_count": int(active.sum()), "initial_ml": len(points) * self.ml_per_particle,
                "ml_per_particle": self.ml_per_particle, "volume_basis": "genesis_sph_rest_volume",
                "bottle_ml": int((active & in_bottle).sum()) * self.ml_per_particle,
                "glass_ml": int((active & in_glass).sum()) * self.ml_per_particle,
                "spilled_ml": int(spilled.sum()) * self.ml_per_particle,
                "in_flight_ml": int((active & ~in_bottle & ~in_glass & ~spilled).sum()) * self.ml_per_particle,
                "bottle_mouth": (bp + br @ BOTTLE_MOUTH).tolist(),
                "finite": bool(np.isfinite(points).all())}
