import math

def _is_true(val):
    """Safe boolean — treats NaN, None, and False all as False.
    NaN == True is False in Python, so this is safe."""
    return val == True

def _str(val):
    """Extract a plain string from a value that might be:
    - already a string ('Off T')
    - NaN (return empty string)
    - None (return empty string)
    - a dict {'id':1, 'name':'Saved'} — fallback just in case
    """
    if isinstance(val, dict):
        return val.get('name', '')
    if val is None:
        return ''
    try:
        if math.isnan(float(val)):
            return ''
    except (TypeError, ValueError):
        pass
    return str(val)


def tokenise_event(row):
    t = row.get('type', '')

    # ── Pass ─────────────────────────────────────────────────────────────────
    if t == 'Pass':
        if _is_true(row.get('pass_cross')):   return 'PASS_CROSS'
        if _is_true(row.get('pass_switch')):  return 'PASS_BACK'

        start = row.get('location')          or [0, 0]
        end   = row.get('pass_end_location') or [0, 0]
        if end[0] - start[0] >= 10:           return 'PASS_PROG'
        return None  # non-progressive, skip

    # ── Carry ─────────────────────────────────────────────────────────────────
    if t == 'Carry':
        loc = row.get('location') or [60, 40]
        end = row.get('carry_end_location') or loc
        
        # Wide = within 18m of either touchline (y < 18 or y > 62 on 80-wide pitch)
        is_wide    = loc[1] < 18 or loc[1] > 62
        # Progressive = moves ball 5m+ toward goal
        is_prog    = end[0] - loc[0] >= 5
        
        if is_wide and is_prog:   return 'CARRY_WIDE_PROG'   # winger advancing
        if is_wide:               return 'CARRY_WIDE'         # wide possession
        if is_prog:               return 'CARRY_PROG'         # central progression

        return 'CARRY'                                         # recycling possession


    # ── Dribble ───────────────────────────────────────────────────────────────
    if t == 'Dribble':
        outcome = _str(row.get('dribble_outcome'))
        return 'DRIBBLE_W' if outcome == 'Complete' else 'DRIBBLE_L'

    # ── Shot ──────────────────────────────────────────────────────────────────
    if t == 'Shot':
        outcome  = _str(row.get('shot_outcome'))
        loc      = row.get('location') or [100, 40]
        in_box   = loc[0] > 102 and 18 < loc[1] < 62   # inside penalty area
        
        if outcome in ('Saved', 'Goal'):
            return 'SHOT_ON_BOX' if in_box else 'SHOT_ON'
        if outcome in ('Off T', 'Blocked'):
            return 'SHOT_OFF_BOX' if in_box else 'SHOT_OFF'
        return None

    # ── Pressure ──────────────────────────────────────────────────────────────
    if t == 'Pressure':
        return 'PRESS'

    # ── Interception ──────────────────────────────────────────────────────────
    if t == 'Interception':
        outcome = _str(row.get('interception_outcome'))
        return 'INT' if outcome == 'Won' else None

    # ── Clearance ─────────────────────────────────────────────────────────────
    if t == 'Clearance':
        return 'CLEARANCE'

    # ── Foul Won ──────────────────────────────────────────────────────────────
    if t == 'Foul Won':
        return 'FOUL_W'

    return None