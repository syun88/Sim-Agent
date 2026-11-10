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
        # The same coupling-SDF decision as `mixed`: at the official 0.4 mm / 512^3 the build had not finished
        # in fifteen minutes; at 1.0 mm / 160 it builds in 22.8 s with 3600 particles and 45 mL in the bottle
        # (measured 2026-11-10). Coarser coupling is the price, and whether it costs pour accuracy is the next
        # measurement - this note says what has been established, not what is hoped.
        'sdf': {'cell_mm': 1.0, 'min_res': 64, 'max_res': 160},
        # Measured 2026-11-10: at 2.5 mm particles, dt 2 ms and 32 substeps the scene advanced 7.9 ms of
        # simulated time per wall second (about 1/127 of real time), which is why a wine task felt broken. At
        # 3.5 mm / dt 3 ms / 20 substeps it advances 49.3 ms per wall second (6.2x faster) and stays inside
        # the viscosity stability bound (150 us substep vs 163 us limit).
        'fluid': {'particle_mm': 3.5, 'dt_ms': 3, 'substeps': 20},
        'verified': True,
        # Measured 2026-11-10 on this scene: the approach and grasp succeed, then pouring fails with
        # POUR_SPILL - bottle 45 -> 40.4 mL, glass 3.1 mL, about 1.5 mL spilled. The official 0.4 mm coupling
        # cannot be built here (it stalls), so this is not yet comparable: the coarse SDF is usable for the
        # scene, and not yet shown to be usable for the pour.
        'note': 'builds in ~23 s at 1.0 mm coupling SDF (official 0.4 mm stalls)',
        'warnings': ['pour measured POUR_SPILL at 1.0 mm coupling: glass 3.1 mL, ~1.5 mL spilled of 45 mL'],
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
        # Buildable, but the fluid timestep makes ordinary manipulation far slower: measured 2026-11-10, a
        # single cube placement had not finished after ten minutes here and takes about twelve seconds in
        # `cubes`. The scene is offered with that warning rather than hidden, because pour-while-stacking is
        # exactly what it is for - and a person should decide whether that cost is worth it.
        'note': 'builds in ~222 s at 1.0 mm coupling SDF',
        'warnings': ['manipulation is ~50x slower than cubes (WINE_DT): a cube placement took over ten minutes'],
    },
}

DEFAULT = 'cubes'


def ids() -> list[str]:
    return list(SCENES)


def hosts(profile: str, instruction_scene: str) -> bool:
    """Whether `profile` already contains everything `instruction_scene` needs.

    This is what decides whether an instruction may rebuild the world. Measured 2026-11-10: a person selected
    `mixed` and asked to place a cube, and the instruction-driven switch rebuilt `cubes`, silently discarding
    the scene they had chosen. A scene that already hosts the objects must not be replaced.
    """
    wanted = set((SCENES.get(instruction_scene) or {}).get('objects') or ())
    have = set((SCENES.get(profile) or {}).get('objects') or ())
    return bool(wanted) and wanted <= have


def is_valid(profile: str) -> bool:
    return profile in SCENES


def needs(profile: str) -> tuple[str, ...]:
    return tuple((SCENES.get(profile) or {}).get('needs') or ())


def fluid(profile: str) -> dict | None:
    """The particle size and timestep this scene should run with, if it differs from the engine's defaults."""
    return (SCENES.get(profile) or {}).get('fluid')


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
             'warnings': list(entry.get('warnings') or []),
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
