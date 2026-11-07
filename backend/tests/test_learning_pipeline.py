"""The mechanism predictor must be judged by its held-out score, not by having been trained."""
import json
from pathlib import Path

import pytest

from backend.app.learning import mechanism_dataset, mechanism_model
from backend.app.learning.mechanism_dataset import FEATURES, example_of, label_of, load_examples
from backend.app.learning.mechanism_model import MechanismModel, leave_one_out

RECORD = {
    'task_id': 't1', 'instruction': '右手でキューブを掴んでから、掴んでいる状態のキューブの上に赤いキューブを重ねて',
    'success': False,
    'initial_world_state': {'robot': {'left_arm': {'held_object': None}, 'right_arm': {'held_object': None}},
                            'objects': [{'id': 'red_cube'}, {'id': 'blue_cube'}]},
    'agent_attempts': [{'attempt': 1,
                        'plan': {'model_called': True,
                                 'program': {'actions': [{'op': 'pick', 'object': 'blue_cube', 'arm': 'right'},
                                                         {'op': 'place', 'object': 'red_cube', 'arm': 'left',
                                                          'target': 'blue_cube', 'relation': 'on'}]}},
                        'result': {'skills': [{'skill': 'move_to_object', 'success': False,
                                               'error': 'CARRY_COLLISION',
                                               'evidence': {'object': 'red_cube', 'carrying': 'blue_cube',
                                                            'displacement_m': 0.0074, 'tolerance_m': 0.005,
                                                            'phase': 'carry'}}]}}],
    'evidence_ledger': [],
}


def test_features_come_from_the_plan_and_the_scene_not_from_the_failure():
    features = mechanism_dataset.features_of(RECORD)
    assert set(features) == set(FEATURES)
    assert features['action_count'] == 2
    assert features['relative_placements'] == 1
    assert features['mentions_maintain_hold'] == 1
    assert features['both_arms'] == 1


def test_the_label_is_the_attributed_mechanism():
    assert label_of(RECORD) == 'CARRY_CONTACT'
    assert example_of(RECORD)['label'] == 'CARRY_CONTACT'


def test_a_successful_run_is_its_own_label():
    assert label_of({**RECORD, 'success': True}) == mechanism_dataset.SUCCESS


def test_a_record_without_evidence_has_no_label_rather_than_a_guess():
    bare = {**RECORD, 'agent_attempts': [{'attempt': 1, 'plan': RECORD['agent_attempts'][0]['plan'],
                                          'result': {'skills': [{'skill': 'x', 'success': False}]}}]}
    assert label_of(bare) == 'UNKNOWN'


def test_loading_reports_every_record_it_could_not_use(tmp_path):
    good = tmp_path / 'good.json'
    good.write_text(json.dumps(RECORD), encoding='utf-8')
    bad = tmp_path / 'bad.json'
    bad.write_text('{"task_id": "x"}', encoding='utf-8')
    broken = tmp_path / 'broken.json'
    broken.write_text('not json', encoding='utf-8')
    examples, skipped = load_examples([good, bad, broken])
    assert len(examples) == 1
    assert {Path(item['file']).name for item in skipped} == {'bad.json', 'broken.json'}


def _separable_examples():
    examples = []
    for index in range(12):
        label = 'HOLD' if index % 2 else 'PATH'
        features = {name: 0 for name in FEATURES}
        features['handover_count'] = 1 if label == 'HOLD' else 0
        examples.append({'task_id': str(index), 'instruction': 'i', 'features': features, 'label': label})
    return examples


def test_the_optimiser_learns_a_separable_toy_set():
    model = MechanismModel.fit(_separable_examples())
    label, probability = model.predict(_separable_examples()[0]['features'])
    assert label == 'PATH' and probability > 0.9


def test_saved_models_reload_and_refuse_a_different_feature_set(tmp_path):
    model = MechanismModel.fit(_separable_examples())
    path = tmp_path / 'model.json'
    model.save(path)
    reloaded = MechanismModel.load(path)
    assert reloaded.predict(_separable_examples()[0]['features']) == model.predict(
        _separable_examples()[0]['features'])
    data = json.loads(path.read_text(encoding='utf-8'))
    data['features'] = list(reversed(data['features']))
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='different feature set'):
        MechanismModel.load(path)


def test_leave_one_out_reports_the_baseline_it_must_beat():
    report = leave_one_out(_separable_examples())
    assert report['examples'] == 12
    assert report['accuracy'] == pytest.approx(1.0)
    assert report['majority_class'] in {'HOLD', 'PATH'}
    assert report['accuracy'] > report['baseline_accuracy'], 'a separable set must beat the majority class'


def test_adoption_needs_a_large_enough_sample(monkeypatch):
    from scripts.train_mechanism_predictor import main
    monkeypatch.setattr(mechanism_model, 'MIN_EXAMPLES_FOR_ADOPTION', 40)
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'record.json'
        path.write_text(json.dumps(RECORD), encoding='utf-8')
        report = Path(directory) / 'report.json'
        assert main(['--input', str(path), '--report', str(report)]) == 0
        loaded = json.loads(report.read_text(encoding='utf-8'))
    assert loaded['adopted'] is False
    assert 'NOT_ENOUGH_CLASSES' in loaded['reason'] or 'below the 40' in loaded['reason']


def _blind_spot_examples():
    """A separable majority plus one class the features cannot separate at all."""
    examples = []
    for index in range(20):
        label = 'HOLD' if index % 2 else 'PATH'
        features = {name: 0 for name in FEATURES}
        features['handover_count'] = 1 if label == 'HOLD' else 0
        examples.append({'task_id': str(index), 'instruction': 'i', 'features': features, 'label': label})
    for index in range(6):
        features = {name: 0 for name in FEATURES}
        features['relative_placements'] = index % 2       # identical on average to both other classes
        examples.append({'task_id': f'b{index}', 'instruction': 'i', 'features': features,
                         'label': 'APPROACH_CONTACT'})
    return examples


def test_a_class_the_model_never_predicts_blocks_adoption(monkeypatch, tmp_path):
    """Measured 2026-11-08: 0.54 accuracy over a 0.28 baseline while APPROACH_CONTACT was 0/7."""
    from scripts.train_mechanism_predictor import main
    monkeypatch.setattr(mechanism_model, 'MIN_EXAMPLES_FOR_ADOPTION', 5)
    # Force a blind spot regardless of what the optimiser happens to do.
    real_predict = MechanismModel.predict

    def predict(self, features):
        label, probability = real_predict(self, features)
        return ('PATH' if label == 'APPROACH_CONTACT' else label), probability
    monkeypatch.setattr(MechanismModel, 'predict', predict)
    import json
    records = []
    for example in _blind_spot_examples():
        record = {**RECORD, 'task_id': example['task_id'], 'success': example['label'] == mechanism_dataset.SUCCESS,
                  'agent_attempts': [{'attempt': 1, 'plan': {'model_called': False, 'program': {'actions': [
                      {'op': 'pick', 'object': 'red_cube', 'arm': 'left'}]}},
                      'result': {'skills': []}}], 'evidence_ledger': []}
        path = tmp_path / f"{example['task_id']}.json"
        path.write_text(json.dumps(record), encoding='utf-8')
        records.append(str(path))
    monkeypatch.setattr('scripts.train_mechanism_predictor.load_examples',
                        lambda paths: (_blind_spot_examples(), []))
    report = tmp_path / 'verdict.json'
    assert main(['--input', str(tmp_path), '--report', str(report)]) == 0
    verdict = json.loads(report.read_text(encoding='utf-8'))
    assert verdict['adopted'] is False
    assert 'APPROACH_CONTACT' in verdict['reason']


def test_the_report_carries_per_class_recall_and_the_confidence_split():
    scores = leave_one_out(_separable_examples())
    assert set(scores['per_class']) == {'HOLD', 'PATH'}
    assert all(stats['recall'] == pytest.approx(1.0) for stats in scores['per_class'].values())
    # A perfect model has no wrong predictions, so the wrong-side mean is honestly absent, not zero.
    assert scores['confidence_when_right'] is not None
    assert scores['confidence_when_wrong'] is None


def test_a_genuinely_separable_set_is_adopted(monkeypatch):
    """The guard must not refuse everything: a detectable, confident model passes."""
    from scripts import train_mechanism_predictor as trainer
    monkeypatch.setattr(trainer, 'load_examples', lambda paths: (_separable_examples(), []))
    monkeypatch.setattr(trainer, 'MIN_EXAMPLES_FOR_ADOPTION', 5)
    import json
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / 'verdict.json'
        assert trainer.main(['--input', directory, '--report', str(report)]) == 0
        verdict = json.loads(report.read_text(encoding='utf-8'))
    assert verdict['adopted'] is True, verdict['reason']
