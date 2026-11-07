#!/usr/bin/env python3
"""Train and *judge* the mechanism predictor on saved execution records.

The output is a decision, not a model file: it prints whether the evidence warrants adopting the predictor,
and refuses to adopt when the sample is too small or the leave-one-out score does not beat a majority
baseline. A model that loses to "always answer the most common mechanism" is not worth a wiring change, and
saying so is the point of the script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.app.learning.mechanism_dataset import load_examples
from backend.app.learning.mechanism_model import (MIN_EXAMPLES_FOR_ADOPTION, MechanismModel,
                                                 leave_one_out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input', nargs='+', required=True,
                        help='execution JSON files or directories of them')
    parser.add_argument('--out', default=None, help='write the trained model here (JSON)')
    parser.add_argument('--report', default=None, help='write the evaluation report here (JSON)')
    args = parser.parse_args(argv)

    paths = []
    for entry in args.input:
        path = Path(entry)
        paths.extend(sorted(path.glob('*.json')) if path.is_dir() else [path])
    examples, skipped = load_examples(paths)

    report = {'records': len(paths), 'examples': len(examples), 'skipped': skipped}
    if len(examples) < 2 or len({example['label'] for example in examples}) < 2:
        report['adopted'] = False
        report['reason'] = ('NOT_ENOUGH_CLASSES: a predictor needs at least two mechanisms to separate; '
                            f'{len(examples)} example(s) with {len({e["label"] for e in examples})} class(es)')
        if args.report:
            Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
        print(json.dumps({'records': report['records'], 'examples': report['examples'],
                          'adopted': report['adopted'], 'reason': report['reason']},
                         ensure_ascii=False, indent=1))
        return 0

    scores = leave_one_out(examples)
    report.update(scores)
    gain = None
    if scores['accuracy'] is not None and scores['baseline_accuracy'] is not None:
        gain = scores['accuracy'] - scores['baseline_accuracy']
    report['gain_over_baseline'] = gain

    reasons = []
    if len(examples) < MIN_EXAMPLES_FOR_ADOPTION:
        reasons.append(f'{len(examples)} examples is below the {MIN_EXAMPLES_FOR_ADOPTION} needed for a '
                       f'held-out score to mean anything')
    if gain is None or gain <= 0.:
        reasons.append('the held-out score does not beat the majority baseline')
    report['adopted'] = not reasons
    report['reason'] = '; '.join(reasons) if reasons else 'beats the baseline on a large enough sample'

    if args.out:
        model = MechanismModel.fit(examples, metadata={'examples': len(examples),
                                                       'adopted': report['adopted'],
                                                       'accuracy': scores['accuracy']})
        model.save(args.out)
        report['model_path'] = args.out
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({key: report[key] for key in
                      ('records', 'examples', 'accuracy', 'baseline_accuracy', 'gain_over_baseline',
                       'class_counts', 'adopted', 'reason')}, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
