"""Physical relations the planner and the skills share, as data rather than as scattered constants.

The planner decides *what* to do; it must not do arithmetic, and a skill must not carry a private
constant that nobody can check. Every relation here is named, has units and a one-line statement of what
it assumes, and is evaluated by one entry point - so a caller asks for "the release speed that gives a
3 cm rise" the way it asks IK for a joint solution, and gets numbers plus the relation that produced
them. The engine stays the verifier: a relation proposes, a measurement decides.

Measured basis for the ones that encode a robot property (2026-10-25, OpenManipulator-X + pinch grip):
a 20 cm stroke over 0.35 s released the cube at 0.805 m/s; the same stroke commanded faster released
slower, because friction needs contact time to transfer momentum.
"""
from __future__ import annotations

GRAVITY = 9.81


def release_speed_for_rise(rise_m: float) -> float:
    """Speed a projectile needs at release to rise ``rise_m`` metres. v = sqrt(2*g*h)."""
    return (2. * GRAVITY * max(float(rise_m), 0.)) ** .5


def rise_for_release_speed(speed_m_s: float) -> float:
    """Free rise of a projectile released upward at ``speed_m_s``. h = v^2/(2*g)."""
    return float(speed_m_s) ** 2 / (2. * GRAVITY)


def ramp_for_release_speed(stroke_m: float, speed_m_s: float, release_phase: float = .9) -> float:
    """Seconds for a quadratic ramp of ``stroke_m`` to reach ``speed_m_s`` at ``release_phase``.

    A quadratic ramp covers ``stroke*p^2`` by phase ``p``, so its speed there is ``2*p*stroke/T``.
    """
    speed = max(float(speed_m_s), 1e-6)
    return 2. * float(release_phase) * max(float(stroke_m), 0.) / speed


def friction_speed_bound(friction: float, contact_time_s: float) -> float:
    """Largest speed change friction can give in ``contact_time_s``. dv <= mu*g*T.

    This is the relation that explained why a short, violent stroke cannot throw: at mu*g = 9.8 m/s^2
    the 0.77 m/s a 3 cm rise needs 0.079 s of contact, so the stroke has to last at least that long.
    """
    return max(float(friction), 0.) * GRAVITY * max(float(contact_time_s), 0.)


def contact_time_for_speed(speed_m_s: float, friction: float) -> float:
    """Contact time friction needs to deliver ``speed_m_s``. T = v/(mu*g)."""
    return float(speed_m_s) / max(float(friction) * GRAVITY, 1e-9)


def pour_time_for_volume(volume_ml: float, flow_ml_s: float) -> float:
    """Seconds a stream of ``flow_ml_s`` needs to deliver ``volume_ml``."""
    return max(float(volume_ml), 0.) / max(float(flow_ml_s), 1e-9)


def closing_reserve(flow_ml_s: float, closing_time_s: float, cap_ml: float) -> float:
    """Liquid still arriving while the pour is closed off, bounded by ``cap_ml``.

    A pour cannot stop instantly: the stream keeps running while the bottle comes upright, so the
    requested volume is undershot unless the stop is commanded early. The bound is the caller's, because
    how much overshoot is acceptable is a task decision rather than a physical constant.
    """
    return min(max(float(cap_ml), 0.), max(float(flow_ml_s), 0.) * max(float(closing_time_s), 0.))


def launch_for_target(delta_m, flight_time_s, gravity: float = GRAVITY) -> dict:
    """Release velocity that carries a projectile through ``delta_m`` after ``flight_time_s``.

    Gravity acts on z only, so the horizontal part is constant and the vertical part carries the whole
    of ``0.5*g*T^2``. The flight time is the caller's choice: a short time needs a fast, flat throw and a
    long one needs a high, slow arc, and which is reachable is the robot's question, not physics'.
    """
    dx, dy, dz = (float(value) for value in delta_m)
    time = max(float(flight_time_s), 1e-6)
    velocity = (dx / time, dy / time, (dz + .5 * gravity * time ** 2) / time)
    speed = sum(component ** 2 for component in velocity) ** .5
    direction = tuple(component / speed for component in velocity) if speed > 1e-9 else (0., 0., 1.)
    return {"velocity": velocity, "speed": speed, "direction": direction, "flight_time_s": time}


#: name -> (callable, unit of the result, arguments, what the relation assumes)
RELATIONS = {
    "release_speed_for_rise": (
        release_speed_for_rise, "m/s", {"rise_m": "m"},
        "Projectile launched upward; no drag. v = sqrt(2*g*h)."),
    "rise_for_release_speed": (
        rise_for_release_speed, "m", {"speed_m_s": "m/s"},
        "Upward launch, free flight only (contact ended at release). h = v^2/(2*g)."),
    "ramp_for_release_speed": (
        ramp_for_release_speed, "s", {"stroke_m": "m", "speed_m_s": "m/s", "release_phase": "-"},
        "Quadratic ramp from rest; release at the given phase while still accelerating."),
    "friction_speed_bound": (
        friction_speed_bound, "m/s", {"friction": "-", "contact_time_s": "s"},
        "Signed friction accelerates the payload uniformly during contact: dv <= mu*g*T."),
    "contact_time_for_speed": (
        contact_time_for_speed, "s", {"speed_m_s": "m/s", "friction": "-"},
        "Inverse of friction_speed_bound: the contact time a speed change needs."),
    "pour_time_for_volume": (
        pour_time_for_volume, "s", {"volume_ml": "ml", "flow_ml_s": "ml/s"},
        "Constant stream; the measured flow rate of the running pour is the right input."),
    "launch_for_target": (
        launch_for_target, "m/s and -, -, m", {"delta_m": "m", "flight_time_s": "s"},
        "Projectile through a point after a chosen flight time; horizontal part constant, z carries "
        "0.5*g*T^2. Returns velocity, speed and unit direction."),
    "closing_reserve": (
        closing_reserve, "ml", {"flow_ml_s": "ml/s", "closing_time_s": "s", "cap_ml": "ml"},
        "The stream keeps flowing while the vessel is returned upright; capped by the caller."),
}


def names() -> list[str]:
    return sorted(RELATIONS)


def describe() -> list[dict]:
    return [{"name": name, "unit": unit, "args": args, "assumes": doc}
            for name, (_, unit, args, doc) in sorted(RELATIONS.items())]


def evaluate(name: str, **arguments) -> dict:
    """Evaluate one relation by name. Unknown names raise: a silent zero would look like physics."""
    if name not in RELATIONS:
        raise KeyError(f"unknown physical relation {name!r}; known: {', '.join(names())}")
    function, unit, expected, doc = RELATIONS[name]
    missing = [arg for arg in expected if arg not in arguments]
    if missing:
        raise KeyError(f"{name} needs {missing}")
    value = function(**{arg: arguments[arg] for arg in expected})
    result = {"relation": name, "unit": unit,
              "inputs": {arg: arguments[arg] for arg in expected}, "assumes": doc}
    # Some relations answer with a vector (a launch velocity has a direction as well as a size); those
    # keep their structure, everything else is a scalar a caller can use directly.
    if isinstance(value, dict):
        return {**result, "value": value, "vector": True}
    return {**result, "value": float(value), "vector": False}
