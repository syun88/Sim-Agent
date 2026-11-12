"""Adding an object or a zone is a data edit, not a code edit."""
import json

import pytest

from backend.app.planning import catalog, language_fallback, references


@pytest.fixture()
def custom_catalog(tmp_path, monkeypatch):
    payload = {
        'objects': [
            {'id': 'red_cube', 'kind': 'cube', 'spawn_arm': 'left',
             'aliases': {'ja': ['赤いキューブ'], 'en': ['red cube']}, 'label': {'ja': '赤いキューブ'}},
            {'id': 'green_block', 'kind': 'cube', 'size_m': .05, 'spawn_arm': 'right',
             'aliases': {'ja': ['緑のブロック', 'みどり'], 'en': ['green block']},
             'label': {'ja': '緑のブロック', 'en': 'green block'}},
        ],
        'zones': [{'id': 'shelf_zone', 'aliases': {'ja': ['棚'], 'en': ['shelf']},
                   'label': {'ja': '棚', 'en': 'shelf'}}],
    }
    path = tmp_path / 'scene_catalog.json'
    path.write_text(json.dumps(payload, ensure_ascii=False))
    monkeypatch.setenv('OPENARM_CATALOG', str(path))
    monkeypatch.setattr(references, 'OBJECT_PATTERNS', references._patterns())
    return path


def test_a_new_object_can_be_referred_to_without_touching_the_parser(custom_catalog):
    state = {'objects': [{'id': 'red_cube'}, {'id': 'green_block'}]}
    assert references.named_objects('緑のブロックを右手で持って', state) == ['green_block']
    assert references.named_objects('green block on red cube', state) == ['green_block', 'red_cube']


def test_a_new_object_is_not_found_when_the_scene_does_not_contain_it(custom_catalog):
    state = {'objects': [{'id': 'red_cube'}]}
    assert references.named_objects('緑のブロックを持って', state) == []


def test_the_catalog_says_which_hand_the_object_starts_near(custom_catalog):
    assert catalog.load().spawn_arms() == [('red_cube', 'left'), ('green_block', 'right')]


def test_the_prompt_vocabulary_comes_from_the_catalog(custom_catalog):
    vocabulary = catalog.load()
    assert 'green_block' in vocabulary.prompt_objects()
    assert vocabulary.prompt_zones() == 'shelf_zone'


def test_a_zone_alias_reaches_the_language_fallback(custom_catalog):
    """The fallback asks the catalog which zone names exist, then keeps its own spatial vocabulary."""
    assert language_fallback.target_in('棚に置いて') == 'shelf_zone'
    assert language_fallback.target_in('青いゾーンに置いて') == 'blue_zone'


def test_a_missing_catalog_falls_back_to_the_cells_own_objects(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENARM_CATALOG', str(tmp_path / 'absent.json'))
    loaded = catalog.load()
    assert loaded.source == 'builtin'
    assert {'red_cube', 'blue_cube'} <= {entry.id for entry in loaded.objects}
