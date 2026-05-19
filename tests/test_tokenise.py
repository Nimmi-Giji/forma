import pytest
from src.tokenise import tokenise_event

# ── Pass tests ────────────────────────────────────────────────────────────────

def test_progressive_pass():
    row = {
        'type': 'Pass',
        'location': [30, 40],
        'pass': {'end_location': [45, 40], 'cross': False, 'switch': False}
    }
    assert tokenise_event(row) == 'PASS_PROG'

def test_short_pass_is_skipped():
    """A 5m pass that doesn't progress should return None."""
    row = {
        'type': 'Pass',
        'location': [50, 40],
        'pass': {'end_location': [54, 40], 'cross': False, 'switch': False}
    }
    assert tokenise_event(row) is None

def test_cross():
    row = {
        'type': 'Pass',
        'location': [100, 10],
        'pass': {'end_location': [110, 40], 'cross': True, 'switch': False}
    }
    assert tokenise_event(row) == 'PASS_CROSS'

def test_switch_pass():
    row = {
        'type': 'Pass',
        'location': [60, 20],
        'pass': {'end_location': [65, 60], 'cross': False, 'switch': True}
    }
    assert tokenise_event(row) == 'PASS_BACK'

# ── Dribble tests ─────────────────────────────────────────────────────────────

def test_successful_dribble():
    row = {'type': 'Dribble', 'dribble': {'outcome': 'Complete'}}
    assert tokenise_event(row) == 'DRIBBLE_W'

def test_failed_dribble():
    row = {'type': 'Dribble', 'dribble': {'outcome': 'Incomplete'}}
    assert tokenise_event(row) == 'DRIBBLE_L'

def test_dribble_with_dict_outcome():
    """StatsBomb sometimes returns outcome as a dict, not a string."""
    row = {'type': 'Dribble', 'dribble': {'outcome': {'id': 8, 'name': 'Complete'}}}
    assert tokenise_event(row) == 'DRIBBLE_W'

# ── Shot tests ────────────────────────────────────────────────────────────────

def test_goal():
    row = {'type': 'Shot', 'shot': {'outcome': 'Goal'}}
    assert tokenise_event(row) == 'SHOT_ON'

def test_shot_off_target():
    row = {'type': 'Shot', 'shot': {'outcome': 'Off T'}}
    assert tokenise_event(row) == 'SHOT_OFF'

# ── Edge cases ────────────────────────────────────────────────────────────────

def test_none_nested_dict_doesnt_crash():
    """StatsBomb can return None for nested dicts — should not crash."""
    row = {'type': 'Pass', 'location': [50, 40], 'pass': None}
    result = tokenise_event(row)   # should return None, not raise an error
    assert result is None

def test_unknown_event_type_is_skipped():
    row = {'type': 'Ball Receipt', 'location': [50, 50]}
    assert tokenise_event(row) is None