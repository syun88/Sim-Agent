"""Which scenes exist is one list, and a new one is a data edit plus a builder that follows it."""
from types import SimpleNamespace

import pytest

from backend.app.planning.scene_selection import scene_for_instruction
from backend.app.simulation import scenes


def test_the_registry_lists_every_scene_with_what_it_contains():
    listed = {entry['id']: entry for entry in scenes.describe()}
    assert {'cubes', 'wine', 'mixed'} <= set(listed)
    assert 'red_cube' in listed['cubes']['objects'] and listed['wine']['objects'] == ['wine_bottle', 'wine_glass']
    assert set(listed['mixed']['objects']) == {'red_cube', 'blue_cube', 'wine_bottle', 'wine_glass'}


def test_the_mixed_scene_declares_both_builders_and_fluid():
    assert scenes.needs('mixed') == ('cubes', 'wine')
    assert scenes.has_fluid('mixed') is True
    assert scenes.needs('cubes') == ('cubes',) and scenes.has_fluid('cubes') is False


def test_the_active_scene_is_marked_in_the_listing():
    listed = scenes.describe(active='mixed')
    assert [entry['id'] for entry in listed if entry['active']] == ['mixed']


def test_only_scenes_that_have_actually_been_built_are_offered():
    """Measured 2026-11-10: mixed builds in 222 s at 1.0 mm coupling SDF; wine keeps the official 0.4 mm
    and did not finish in fifteen minutes, so it stays unselectable until it is built once."""
    listed = {entry['id']: entry for entry in scenes.describe()}
    assert listed['cubes']['selectable'] is True
    assert listed['mixed']['selectable'] is True and scenes.sdf('mixed') == {'cell_mm': 1.0, 'min_res': 64, 'max_res': 160}
    assert listed['wine']['selectable'] is False
    assert 'SDF' in listed['wine']['note']
    assert scenes.is_selectable('wine') is False
    assert scenes.is_selectable('wine', force=True) is True


def test_an_unknown_scene_is_refused_by_name():
    with pytest.raises(ValueError, match='Unknown scene profile'):
        scenes.check('kitchen')
    assert scenes.is_valid('mixed') and not scenes.is_valid('kitchen')


def test_an_instruction_that_needs_both_kinds_of_object_asks_for_the_mixed_scene():
    assert scene_for_instruction('赤いキューブをワイングラスの横に置いて') == 'mixed'
    assert scene_for_instruction('ワインを注いで') == 'wine'
    assert scene_for_instruction('赤いキューブを左において') == 'cubes'
    assert scene_for_instruction('こんにちは', current='wine') == 'wine'
