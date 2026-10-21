from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from backend.app.planning.agent_planner import AgentPlanner, PlannerProviderError
from backend.app.planning.scene_selection import scene_for_instruction
from backend.app.planning.task_program import TaskAction, TaskProgram, compile_program
from backend.app.skills.dynamic_skills import evaluate_throw
from backend.app.simulation.genesis_backend.wine import (
    WineWorld, vessel_xml, vessel_clearance, BOTTLE_MOUTH, GLASS_START, inside_cavity,
    validate_fluid_timestep, PARTICLE_SIZE, LIQUID_MU, WINE_DT, WINE_SUBSTEPS,
)
from backend.app.simulation.genesis_backend.fluid_surface import reconstruct_surface
from backend.app.verification.goal_verification import verify_task_goal


def test_throw_measures_free_flight_and_rejects_a_drop_or_contact_supported_rise():
    times = np.arange(.24, .37, .002)
    trace = [{"t": float(t), "position": [0., 0., float(1.2 + (t - .24) - 4.905 * (t - .24) ** 2)],
              "velocity": [0., 0., float(1. - 9.81 * (t - .24))], "contacts": []} for t in times]
    assert evaluate_throw(trace, .03)["success"]
    assert not evaluate_throw([{**row, "velocity": [0., 0., -.1]} for row in trace], .03)["success"]
    assert not evaluate_throw([{**row, "contacts": ["finger"]} for row in trace], .03)["success"]


def test_new_language_operations_keep_identity_amount_and_physical_preconditions(monkeypatch):
    for key in ("AI_AND_API_KEY", "AI_AND_BASE_URL", "AI_AND_MODEL", "OPENARM_LOCAL_PLANNER"):
        monkeypatch.delenv(key, raising=False)
    state = {"objects": [{"id": "blue_cube", "type": "cube", "position": [.5, -.2, 1.03]},
                         {"id": "wine_bottle", "type": "bottle", "position": [.44, .29, 1.09]},
                         {"id": "wine_glass", "type": "glass", "position": [.55, .05, 1.006], "graspable": False}],
             "targets": [{"id": "wine_glass"}], "fluid": {"bottle_ml": 50.}}
    planner = AgentPlanner()
    throw = planner.plan("右手で青いキューブを上に投げて", state)
    assert throw.steps[-1].skill == "throw_object" and throw.steps[-1].arm == "right"
    pour = planner.plan("ワインをグラスに20ml注いで", state)
    assert [step.skill for step in pour.steps] == ["approach_object", "grasp_object", "pour_liquid"]
    assert pour.steps[-1].volume_ml == 20 and pour.steps[-1].target == "wine_glass"
    with pytest.raises(PlannerProviderError):
        planner.plan("ワインをグラスに80ml注いで", state)
    with pytest.raises(PlannerProviderError):
        planner.plan("ワインをグラスに1ml注いで", state)
    with pytest.raises(PlannerProviderError):
        planner.plan("グラスを掴んで", state)
    with pytest.raises(PlannerProviderError):
        planner.plan("ワインをグラスに20ml注いで、20ml注いで、20ml注いで", state)
    with pytest.raises(PlannerProviderError):
        planner.plan("ボトルをグラスに置いて", state)
    returned = compile_program(TaskProgram(actions=[
        TaskAction(op="pour", object="wine_bottle", volume_ml=10., receiver="right"),
        TaskAction(op="place", object="it"),
    ]), state)
    assert [s.skill for s in returned.steps] == ["approach_object", "grasp_object", "pour_liquid"]
    assert returned.program["actions"][0]["receiver"] is None
    assert returned.assumptions and returned.goal["checks"][0]["type"] == "pour"
    informal_throw = planner.plan("青いやつを右の手でひょいっと上に放って", state)
    assert informal_throw.steps[-1].skill == "throw_object" and informal_throw.steps[-1].arm == "right"
    informal_pour = planner.plan("赤ワインを10ml、そこのグラスにお願い。終わったらボトルを戻しておいて", state)
    assert informal_pour.steps[-1].skill == "pour_liquid" and informal_pour.steps[-1].volume_ml == 10.
    with pytest.raises(PlannerProviderError):
        planner.plan("赤ワインを10ml、そこのグラスにお願い。そして未知の作業を実行して", state)


def test_model_default_bottle_hand_is_grounded_and_success_claim_is_discarded(monkeypatch):
    planner = AgentPlanner()
    planner.enabled = planner.local_enabled = True
    planner.api_key = planner.base_url = planner.model = ""
    monkeypatch.setattr(planner, "_ai_plan", lambda *_: {"payload": {
        "actions": [{"op": "pour", "object": "wine_bottle", "arm": "right", "volume_ml": 20.}],
        "assumptions": ["The pour is successful"]}, "raw": "test provider"})
    state = {"objects": [{"id": "wine_bottle", "type": "bottle", "position": [.44, .29, 1.091]},
                         {"id": "wine_glass", "type": "glass", "graspable": False}],
             "targets": [{"id": "wine_glass"}], "fluid": {"bottle_ml": 40.}}
    plan = planner.plan("Serve 10 ml from the bottle into the glass, please", state)
    assert plan.steps[-1].arm == "left" and plan.steps[-1].volume_ml == 10. and not plan.assumptions


def test_liquid_volume_uses_the_engine_reference_volume_instead_of_spacing_cubed():
    world = object.__new__(WineWorld)
    reference_volume = .8 * .0045 ** 3
    world.scene = SimpleNamespace(sph_solver=SimpleNamespace(particle_volume=reference_volume))
    world.liquid = SimpleNamespace(get_particles_pos=lambda: torch.zeros((3, 3)))
    world.capture_setup()
    assert world.ml_per_particle == pytest.approx(reference_volume * 1e6)
    assert world.ml_per_particle != pytest.approx(.0045 ** 3 * 1e6)


def test_liquid_time_step_rejects_the_resolution_change_that_caused_explosion():
    assert WINE_DT / WINE_SUBSTEPS <= validate_fluid_timestep(PARTICLE_SIZE, LIQUID_MU, WINE_DT / WINE_SUBSTEPS)
    with pytest.raises(ValueError, match="stability bound"):
        validate_fluid_timestep(.0025, .003, .002 / 4)
    assert validate_fluid_timestep(.0025, .003, .002 / 32) == pytest.approx(1 / 12000)


def test_cavity_margin_uses_distance_to_sloping_floor_and_rejects_real_spills():
    z = np.array([.007, .012, .208, .233, .257, .300]) * .7 - .085
    radius = np.array([0., .035, .035, .0285, .010, .010]) * .7
    points = np.array([[.024, 0., -.0769], [.03, 0., -.0769], [0., 0., -.087], [0., 0., -.06]])
    assert inside_cavity(points, z, radius, .00125).tolist() == [True, False, False, True]


@pytest.mark.parametrize("pending,expected", [(0., True), (.0125, False)])
@pytest.mark.parametrize("initial_spill", [0., .025])
def test_finished_pour_must_account_for_airborne_liquid_and_existing_spills(pending, expected, initial_spill):
    from backend.app.simulation.genesis_backend.wine import BOTTLE_START
    liquid = {"glass_ml": 10., "bottle_ml": 35. - pending - initial_spill, "spilled_ml": initial_spill,
              "in_flight_ml": pending, "particle_count": 3600, "finite": True}
    env = SimpleNamespace(fluid_state=lambda: liquid, object_contacts=lambda _: [],
                          object_pose=lambda _: (BOTTLE_START, np.eye(3)), object_linear_speed=lambda _: 0.,
                          name="genesis", grasp_mode="physical-friction-pinch")
    plan = SimpleNamespace(steps=[None], goal={"type": "direct_command", "checks": [
        {"type": "pour", "object": "wine_bottle", "volume_ml": 10., "action_index": 0}]})
    results = [{"skill": "pour_liquid", "action_index": 0, "success": True,
                "fluid_trace": [{"vessel_clearance_mm": 2.}],
                "initial_liquid": {"glass_ml": 0., "spilled_ml": initial_spill, "particle_count": 3600}}]
    verification = verify_task_goal(env, plan, results)
    assert verification["success"] is expected
    assert verification["checks"][0]["spilled_ml"] == initial_spill
    assert verification["checks"][0]["additional_spilled_ml"] == 0.
    results[0]["fluid_trace"] = [{"vessel_clearance_mm": -1.}]
    assert not verify_task_goal(env, plan, results)["success"]


def test_fluid_diagnostic_rejects_neck_through_glass_wall_even_when_both_are_fixed():
    from scipy.spatial.transform import Rotation
    for angle in (85., 103., 112.):
        rotation = Rotation.from_euler("y", angle, degrees=True).as_matrix()
        intersecting = GLASS_START + [-.005, 0., .133] - rotation @ BOTTLE_MOUTH
        safe = GLASS_START + [-.005, 0., .161] - rotation @ BOTTLE_MOUTH
        assert vessel_clearance(intersecting, rotation, GLASS_START, np.eye(3)) < 0.
        assert vessel_clearance(safe, rotation, GLASS_START, np.eye(3)) > .001


def test_liquid_surface_is_connected_and_follows_observed_particles_without_modifying_them():
    import trimesh
    spacing = .0025
    points = np.array(np.meshgrid(np.arange(6), np.arange(6), np.arange(9), indexing="ij"),
                      dtype=np.float32).reshape(3, -1).T * spacing
    original = points.copy()
    vertices, faces, normals = reconstruct_surface(points, spacing / 2)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    assert np.array_equal(points, original)
    assert len(mesh.split(only_watertight=False)) == 1
    assert len(faces) > 100 and normals.shape == vertices.shape
    assert np.isfinite(vertices).all() and np.isfinite(normals).all()
    assert np.all(mesh.bounds >= points.min(axis=0) - spacing * 2)
    assert np.all(mesh.bounds <= points.max(axis=0) + spacing * 2)
    offset = np.array([.3, -.2, 1.1], dtype=np.float32)
    moved, moved_faces, _ = reconstruct_surface(points + offset, spacing / 2)
    assert len(moved_faces) == len(faces)
    assert moved.min(axis=0) - vertices.min(axis=0) == pytest.approx(offset, abs=1e-5)
    assert moved.max(axis=0) - vertices.max(axis=0) == pytest.approx(offset, abs=1e-5)


def test_surface_worker_exits_cleanly_when_the_viewer_disconnected(monkeypatch):
    from backend.app.simulation.genesis_backend import fluid_surface
    class ClosedConsumer:
        closed = False
        def send(self, _):
            raise BrokenPipeError("viewer closed")
        def close(self):
            self.closed = True
    consumer = ClosedConsumer()
    monkeypatch.setattr(fluid_surface, "reconstruct_surface", lambda *_: ([], [], []))
    fluid_surface._reconstruct_job(consumer, 1, np.empty((0, 3)), .00125)
    assert consumer.closed


@pytest.mark.parametrize("name,fixed,count,mass", [("wine_bottle", False, 3, .12),
                                                 ("wine_glass", True, 1, .08)])
def test_vessel_collision_meshes_are_not_rendered_over_the_transparent_visual(name, fixed, count, mass):
    import mujoco
    model = mujoco.MjModel.from_xml_string(vessel_xml(name, fixed))
    collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    assert int(collision.sum()) == count
    assert np.all(model.geom_group[collision] == 3)
    assert int((model.geom_group < 3).sum()) == (1 if name == "wine_bottle" else 3)
    assert float(model.body_mass.sum()) == pytest.approx(mass)


@pytest.mark.parametrize("instruction,expected", [("青いキューブを投げて", "cubes"), ("ワインを注いで", "wine"),
                                                   ("Pour wine into the glass", "wine")])
def test_scene_is_selected_before_language_grounding(instruction, expected):
    assert scene_for_instruction(instruction) == expected


def test_cpu_gravity_geometry_matches_actual_genesis_jacobians(env):
    env.set_arm_qpos("left", np.array([.2, -.2, .3, 1.2, -.1, .2, .1]))
    env.set_arm_qpos("right", np.array([-.3, .1, -.2, 1., .2, -.1, -.2]))
    env._refresh_gravity_torque()
    # Independent reference: Genesis link masses and live COM Jacobians. Calling
    # the same private geometry helper twice would no longer test equivalence.
    for side, dofs in env._dofs.items():
        torque = np.zeros(7)
        for link in env._arm_links[side]:
            index = np.array([link.idx_local])
            mass = float(env._entity.get_links_mass(index).cpu().numpy().item())
            com = env._entity.get_links_COM(index).cpu().numpy()[0]
            jacobian = env._entity.get_jacobian(link, local_point=com).cpu().numpy()
            torque -= jacobian[:3, dofs].T @ (mass * env._gravity)
        assert env._gravity_torque_values[side] == pytest.approx(torque, abs=1e-4)


def test_a_long_physics_wait_can_be_interrupted_between_real_steps(env):
    previous = env.step_observer
    start = env.frame().sim_time
    observed = 0
    def interrupt():
        nonlocal observed
        observed += 1
        if observed == 7:
            raise InterruptedError("stop requested")
    try:
        env.step_observer = interrupt
        with pytest.raises(InterruptedError):
            env.step(100)
        assert observed == 7
        assert env.frame().sim_time == pytest.approx(start + 7 * env.timestep)
    finally:
        env.step_observer = previous
