"""What exists in the scene, as *data*: objects, zones, what each is called, which hand spawns it.

The scene vocabulary used to be spread over five places - the object-reference regexes, the Japanese
fallback lexicon, the planner's prompt text, the spawn assignment, and the web UI's labels. Adding a cup or
a third block meant editing all of them and hoping none was missed, so "another task" was really "another
code change".

This module is the one list. It ships with the cell's own objects, and a project can add its own by writing
`assets/catalog/scene_catalog.json` (the default file documents the shape). Everything that needs to know
what an object is called reads it from here.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[3] / 'assets' / 'catalog' / 'scene_catalog.json'

#: Used when the file is absent or unreadable: the objects this cell has always had.
FALLBACK = {
    'objects': [
        {'id': 'red_cube', 'kind': 'cube', 'size_m': .04, 'spawn_arm': 'left',
         'aliases': {'ja': ['赤', 'あか', '赤いキューブ', '赤いブロック'], 'en': ['red', 'red cube', 'red cube']},
         'label': {'ja': '赤いキューブ', 'en': 'red cube'}},
        {'id': 'blue_cube', 'kind': 'cube', 'size_m': .04, 'spawn_arm': 'right',
         'aliases': {'ja': ['青', 'あお', '青いキューブ', '青いブロック'], 'en': ['blue', 'blue cube']},
         'label': {'ja': '青いキューブ', 'en': 'blue cube'}},
        {'id': 'wine_bottle', 'kind': 'vessel', 'graspable': True,
         'aliases': {'ja': ['ワインボトル', 'ボトル', '瓶'], 'en': ['wine', 'wine bottle', 'bottle']},
         'label': {'ja': 'ワインボトル', 'en': 'wine bottle'}},
        {'id': 'wine_glass', 'kind': 'vessel', 'graspable': False,
         'aliases': {'ja': ['グラス', 'ワイングラス'], 'en': ['glass', 'wine glass']},
         'label': {'ja': 'グラス', 'en': 'wine glass'}},
    ],
    'zones': [
        {'id': 'blue_zone', 'label': {'ja': '青いゾーン', 'en': 'blue zone'},
         'aliases': {'ja': ['青いゾーン', '青のエリア'], 'en': ['blue zone', 'blue target']}},
        {'id': 'green_zone', 'label': {'ja': '緑のゾーン', 'en': 'green zone'},
         'aliases': {'ja': ['緑のゾーン', '緑のエリア'], 'en': ['green zone', 'green target']}},
    ],
}


@dataclass(frozen=True)
class Entry:
    id: str
    kind: str
    aliases: dict = field(default_factory=dict)
    label: dict = field(default_factory=dict)
    graspable: bool = True
    spawn_arm: str | None = None
    size_m: float | None = None

    def names(self) -> list[str]:
        return [name for language in self.aliases.values() for name in language]


@dataclass(frozen=True)
class Catalog:
    objects: tuple[Entry, ...]
    zones: tuple[Entry, ...]
    source: str

    def object(self, object_id: str) -> Entry | None:
        return next((entry for entry in self.objects if entry.id == object_id), None)

    def spawn_arms(self) -> list[tuple[str, str]]:
        """Which hand each object starts nearest, for scenes that spawn one per hand."""
        return [(entry.id, entry.spawn_arm) for entry in self.objects if entry.spawn_arm]

    def graspable(self) -> tuple[str, ...]:
        return tuple(entry.id for entry in self.objects if entry.graspable)

    def prompt_objects(self) -> str:
        return ', '.join(entry.id for entry in self.objects)

    def prompt_zones(self) -> str:
        return ', '.join(entry.id for entry in self.zones)


def _entries(raw: list[dict]) -> tuple[Entry, ...]:
    return tuple(Entry(id=item['id'], kind=item.get('kind', 'object'), aliases=item.get('aliases', {}),
                       label=item.get('label', {}), graspable=bool(item.get('graspable', True)),
                       spawn_arm=item.get('spawn_arm'), size_m=item.get('size_m')) for item in raw)


def load(path: str | Path | None = None) -> Catalog:
    """Read the catalog file, falling back to the built-in list if it is missing or unusable."""
    resolved = Path(path) if path is not None else Path(os.environ.get('OPENARM_CATALOG', DEFAULT_CATALOG_PATH))
    try:
        payload = json.loads(Path(resolved).read_text())
        return Catalog(_entries(payload['objects']), _entries(payload.get('zones', [])), str(resolved))
    except (OSError, ValueError, KeyError):
        return Catalog(_entries(FALLBACK['objects']), _entries(FALLBACK['zones']), 'builtin')


def alias_patterns(entries) -> dict[str, str]:
    """One regex per id, from the names the catalog says it is called."""
    patterns = {}
    for entry in entries:
        names = sorted(entry.names(), key=len, reverse=True)
        if not names:
            continue
        patterns[entry.id] = '|'.join(re.escape(name) for name in names)
    return patterns
