#!/usr/bin/env python3
"""Read-only S2 ledger audit from a saved TaskAgent result; never connect to Genesis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.execution.failure_evidence import _canonical


def replay(result):
    ledger = result.get('evidence_ledger', [])
    cited = {receipt['id'] for receipt in ledger}
    findings = []
    if len(cited) != len(ledger):
        findings.append('DUPLICATE_EVIDENCE_ID')
    for attempt in result.get('agent_attempts', []):
        reference = attempt.get('failure_evidence_id')
        if reference and reference not in cited:
            findings.append('UNKNOWN_ATTEMPT_EVIDENCE_ID')
    for decision in result.get('agent_trace', []):
        reference = decision.get('evidence_id')
        if reference and reference not in cited:
            findings.append('UNKNOWN_TRACE_EVIDENCE_ID')
    for receipt in ledger:
        if receipt.get('classification') == 'REACHABILITY' and 'unreachable_by_m' not in receipt.get('measurement', {}):
            findings.append('UNMEASURED_REACHABILITY')
        state = receipt.get('known_inputs')
        # A fingerprint is a digest of exactly the input snapshot, not of prose or elapsed time.
        if state is not None:
            digest = hashlib.sha256(_canonical(state).encode()).hexdigest()
            if digest != receipt.get('before_fingerprint'):
                findings.append('INPUT_FINGERPRINT_MISMATCH')
    return {'ok':not findings, 'findings':findings, 'evidence_ids':[r['id'] for r in ledger],
            'failures':[{'id':r['id'], 'classification':r['classification'],
                         'unchanged_visible_inputs':r['before_fingerprint']==r['after_fingerprint'],
                         'unknown_inputs':r.get('unknown_inputs',[])} for r in ledger],
            'note':'saved-record audit only; no independent sensor verification or retry decision'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record',type=Path,help='saved TaskAgent result JSON')
    args=parser.parse_args()
    report=replay(json.loads(args.record.read_text()))
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0 if report['ok'] else 1


if __name__=='__main__':
    raise SystemExit(main())
