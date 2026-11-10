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
        'verified': True,
    },
    'mixed': {
        'description': 'Cubes, a bottle and a glass together: pour and stack in one scene.',
        'objects': ['red_cube', 'blue_cube', 'wine_bottle', 'wine_glass'],
        'zones': ['blue_zone', 'green_zone'],
        'needs': ['cubes', 'wine'],
        'fluid': True,
        # Measured 2026-11-10: building this scene ran for over ten minutes at ~350% CPU without finishing, so
        # it is not offered. A scene a person can select and then wait forever for is worse than an honest no.
        'verified': False,
        'note': 'not buildable yet: the build did not finish in ten minutes on this host',
    },
}

DEFAULT = 'cubes'


def ids() -> list[str]:
    return list(SCENES)


def is_valid(profile: str) -> bool:
    return profile in SCENES


def needs(profile: str) -> tuple[str, ...]:
    return tuple((SCENES.get(profile) or {}).get('needs') or ())


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
