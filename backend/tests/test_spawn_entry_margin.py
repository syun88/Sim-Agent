"""The entry margin is a separate budget from reach - pin the rule that names it."""
from backend.app.robots.registry import entry_margin_warning


def test_a_millimetre_per_side_is_reported_with_both_numbers():
    note = entry_margin_warning(.084, .086)

    assert note is not None
    assert '84.0 mm' in note and '1.0 mm' in note and '86.0 mm' in note


def test_a_comfortable_margin_says_nothing():
    # (0.086 - 0.074) / 2 evaluates to 0.0059999999999999985, so the exact 6 mm boundary is not a stable case to
    # pin; 8 mm is.
    assert entry_margin_warning(.070, .086) is None


def test_no_recorded_aperture_cannot_be_judged():
    assert entry_margin_warning(.084, 0.) is None
