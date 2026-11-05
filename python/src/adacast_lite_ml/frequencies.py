"""Pandas calendar frequencies and row-based forecast grids."""
import re
from pandas.tseries.frequencies import to_offset


def normalize_frequency(value):
    match = re.fullmatch(r"(\d+)?([A-Za-z]+)(-.*)?", str(value).strip())
    if not match:
        raise ValueError(f"Invalid frequency {value!r}")
    count, unit, suffix = match.groups()
    # Reference model 'm' is monthly; minutes use 't' or 'min'.
    aliases = {"m":"ME", "M":"ME", "a":"YE", "A":"YE", "y":"YE", "Y":"YE", "q":"QE", "Q":"QE",
               "t":"min", "T":"min", "H":"h", "S":"s", "sec":"s", "D":"D", "d":"D", "B":"B", "b":"B", "w":"W"}
    text = (count or "") + aliases.get(unit, unit) + (suffix or "")
    try:
        offset = to_offset(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported frequency {value!r}: {error}") from error
    if offset.n <= 0:
        raise ValueError("Frequency must be positive")
    return offset.freqstr
