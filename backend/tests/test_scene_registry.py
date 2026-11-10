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


def test_a_scene_that_has_not_been_built_is_not_offered():
    """Measured 2026-11-10: the mixed scene ran over ten minutes without finishing, so it is not a button."""
    listed = {entry['id']: entry for entry in scenes.describe()}
    assert listed['mixed']['verified'] is False and listed['mixed']['selectable'] is False
    assert listed['mixed']['note']
    assert listed['cubes']['selectable'] is True
    assert listed['wine']['selectable'] is False, 'the wine build stalls in mesh-to-SDF conversion'
    assert 'SDF' in listed['wine']['note'] or 'SDF' in listed['mixed']['note']
    assert scenes.is_selectable('mixed') is False
    assert scenes.is_selectable('mixed', force=True) is True


def test_an_unknown_scene_is_refused_by_name():
    with pytest.raises(ValueError, match='Unknown scene profile'):
        scenes.check('kitchen')
    assert scenes.is_valid('mixed') and not scenes.is_valid('kitchen')


def test_an_instruction_that_needs_both_kinds_of_object_asks_for_the_mixed_scene():
    assert scene_for_instruction('赤いキューブをワイングラスの横に置いて') == 'mixed'
    assert scene_for_instruction('ワインを注いで') == 'wine'
    assert scene_for_instruction('赤いキューブを左において') == 'cubes'
    assert scene_for_instruction('こんにちは', current='wine') == 'wine'
