"""Parsing og bearbeiding av tidsseksjoner (start-slutt-intervaller) i en video."""
from __future__ import annotations

from typing import List, Tuple

from .events import format_timestamp

TimeRange = Tuple[float, float]


def parse_time_ranges(spec: str) -> List[TimeRange]:
    """Parse en kommaseparert liste av intervaller til (start, slutt) i sekunder.

    Godtar både mm:ss og rene sekunder, f.eks.:
    "12:30-13:00,45:10-45:40" eller "750-780,2710-2740"
    """
    ranges: List[TimeRange] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Ugyldig tidsintervall: '{part}' (forventet 'start-slutt')")
        start_str, end_str = part.split("-", 1)
        start = _parse_time(start_str.strip())
        end = _parse_time(end_str.strip())
        if end <= start:
            raise ValueError(f"Ugyldig tidsintervall: '{part}' (slutt må være etter start)")
        ranges.append((start, end))
    return ranges


def _parse_time(value: str) -> float:
    if ":" in value:
        parts = value.split(":")
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + float(part)
        return seconds
    return float(value)


def merge_ranges(ranges: List[TimeRange], min_gap_seconds: float = 0.0) -> List[TimeRange]:
    """Slå sammen overlappende eller nærliggende intervaller (sortert på starttid)."""
    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    merged = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + min_gap_seconds:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def pad_ranges(
    ranges: List[TimeRange], pad_seconds: float, max_time: float
) -> List[TimeRange]:
    """Utvid hvert intervall med pad_seconds på hver side, klippet til [0, max_time]."""
    padded = []
    for start, end in ranges:
        padded_start = max(0.0, start - pad_seconds)
        padded_end = min(max_time, end + pad_seconds) if max_time > 0 else end + pad_seconds
        padded.append((padded_start, padded_end))
    return padded


def format_ranges(ranges: List[TimeRange]) -> str:
    return ", ".join(f"{format_timestamp(s)}-{format_timestamp(e)}" for s, e in ranges)


def total_duration(ranges: List[TimeRange]) -> float:
    return sum(end - start for start, end in ranges)
