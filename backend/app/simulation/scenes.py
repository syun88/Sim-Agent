"""Which scenes exist, as data, so one list answers "what can I select?" and "what is in it?".

Adding a scene used to mean editing the environment's validation, the API, and the UI. This module is the
single list: an id, what the scene puts on the table, and what a person should read. The environment still
owns how each one is built - this only says which ones exist.

The `mixed` scene is the cube cell plus the wine vessels, from the asset bundle that ships with the repo
(`assets/scenes/openarm_v2_demo.xml` and `assets/wine/`): cubes to stack and a bottle/glass to pour, in one
world, which neither existing profile offers.
"""

from __future__ import annotations

#: scene id -> what it contains. `objects` is the vocabulary the planner will see; `needs` names the pieces the
#: environment has to build, so a scene cannot claim objects its builder does not create.
SCENES: dict[str, dict] = {
    'cubes': {
        'description': 'Two cubes and two zones: stacking, handover, placement.',
        'objects': ['red_cube', 'blue_cube'],
        'zones': ['blue_zone', 'green_zone'],
        'needs': ['cubes'],
        'fluid': False,
        'verified': True,
    },
    'wine': {
        'description': 'Bottle and glass with a fluid surface: pouring.',
        'objects': ['wine_bottle', 'wine_glass'],
        'zones': [],
        'needs': ['wine'],
        'fluid': True,
        # Measured 2026-11-10: building this scene stops in Genesis' mesh-to-SDF conversion
        # (`rigid_geom._compute_sd` -> `T_mesh_to_sdf`) and had not finished after ten minutes at ~350% CPU.
        # The wine world was never built in this session, so the earlier "verified" was an assumption.
        'verified': False,
        'note': 'build stalls in Genesis mesh-to-SDF conversion; not yet built on this host',
    },
    'mixed': {
        'description': 'Cubes, a bottle and a glass together: pour and stack in one scene.',
        'objects': ['red_cube', 'blue_cube', 'wine_bottle', 'wine_glass'],
        'zones': ['blue_zone', 'green_zone'],
        'needs': ['cubes', 'wine'],
        'fluid': True,
        # The fluid-coupling SDF is what decided whether this scene exists at all: at the official 0.4 mm /
        # 512^3 it had not finished after fifteen minutes (696 MB of cache per shell); at 1.0 mm / 160 it
        # builds in 222 s with all four objects present (measured 2026-11-10). Coarser coupling is the price.
        'sdf': {'cell_mm': 1.0, 'min_res': 64, 'max_res': 160},
        'verified': True,
        'note': 'builds in ~222 s at 1.0 mm fluid-coupling SDF (official 0.4 mm does not finish in 15 min)',
    },
}

DEFAULT = 'cubes'


def ids() -> list[str]:
    return list(SCENES)


def is_valid(profile: str) -> bool:
    return profile in SCENES


def needs(profile: str) -> tuple[str, ...]:
    return tuple((SCENES.get(profile) or {}).get('needs') or ())


def sdf(profile: str) -> dict | None:
    """The fluid-coupling SDF resolution this scene should be built with, if it differs from the default."""
    return (SCENES.get(profile) or {}).get('sdf')


def has_fluid(profile: str) -> bool:
    return bool((SCENES.get(profile) or {}).get('fluid'))


def describe(active: str | None = None) -> list[dict]:
    """The list a client renders: what each scene holds, which is running, and which can actually be built."""
    return [{'id': profile, 'description': entry['description'], 'objects': list(entry['objects']),
             'zones': list(entry['zones']), 'fluid': bool(entry['fluid']),
             'verified': bool(entry.get('verified')), 'note': entry.get('note'),
             'selectable': bool(entry.get('verified')),
             'active': profile == active}
            for profile, entry in SCENES.items()]


def is_selectable(profile: str, *, force: bool = False) -> bool:
    """A scene can be selected when it has been built at least once (or when the caller insists)."""
    if not is_valid(profile):
        return False
    return force or bool((SCENES.get(profile) or {}).get('verified'))


def check(profile: str) -> str:
    if not is_valid(profile):
        raise ValueError(f'Unknown scene profile {profile!r}; known: {ids()}')
    return profile
