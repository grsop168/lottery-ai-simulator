from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from lottery_sim.models import Draw5D


FEATURE_WINDOWS: Tuple[int, ...] = (5, 10, 30, 60, 100, 200)


@dataclass(frozen=True)
class Pl5TrainingRows:
    feature_names: Tuple[str, ...]
    features: Tuple[Tuple[float, ...], ...]
    targets: Tuple[int, ...]
    target_issues: Tuple[str, ...]


def pl5_feature_names(windows: Sequence[int] = FEATURE_WINDOWS) -> Tuple[str, ...]:
    names: List[str] = []
    for window in windows:
        names.extend(f"freq_{window}_digit_{digit}" for digit in range(10))
    names.extend(f"freq_all_digit_{digit}" for digit in range(10))
    names.extend(f"omission_digit_{digit}" for digit in range(10))
    names.extend(f"avg_gap_digit_{digit}" for digit in range(10))
    names.extend(f"max_gap_digit_{digit}" for digit in range(10))
    names.extend(("lag_1", "lag_2", "lag_3", "lag_5"))
    names.extend(("previous_sum", "previous_span", "previous_odd_count", "previous_big_count", "previous_duplicate_count"))
    return tuple(names)


def build_pl5_prefix_features(
    history: Sequence[Draw5D],
    position: int,
    windows: Sequence[int] = FEATURE_WINDOWS,
) -> Tuple[float, ...]:
    if position < 0 or position >= 5:
        raise ValueError("position must be between 0 and 4")
    ordered = tuple(sorted(history, key=lambda draw: int(draw.issue)))
    digits = tuple(draw.numbers[position] for draw in ordered)
    values: List[float] = []
    for window in windows:
        scoped = digits[-window:]
        denominator = len(scoped)
        counts = Counter(scoped)
        values.extend(counts[digit] / denominator if denominator else 0.0 for digit in range(10))
    all_counts = Counter(digits)
    values.extend(all_counts[digit] / len(digits) if digits else 0.0 for digit in range(10))

    for digit in range(10):
        last = _last_index(digits, digit)
        values.append(float(len(digits) if last < 0 else len(digits) - 1 - last))
    for digit in range(10):
        gaps = _appearance_gaps(digits, digit)
        values.append(sum(gaps) / len(gaps) if gaps else float(max(len(digits), 1)))
    for digit in range(10):
        gaps = _appearance_gaps(digits, digit)
        values.append(float(max(gaps)) if gaps else float(max(len(digits), 1)))

    values.extend(float(_lag(digits, lag)) for lag in (1, 2, 3, 5))
    if ordered:
        previous = ordered[-1].numbers
        counts = Counter(previous)
        values.extend((
            float(sum(previous)),
            float(max(previous) - min(previous)),
            float(sum(value % 2 for value in previous)),
            float(sum(value >= 5 for value in previous)),
            float(sum(count - 1 for count in counts.values() if count > 1)),
        ))
    else:
        values.extend((0.0,) * 5)
    return tuple(values)


def build_pl5_training_rows(
    draws: Sequence[Draw5D],
    position: int,
    min_history: int,
    windows: Sequence[int] = FEATURE_WINDOWS,
) -> Pl5TrainingRows:
    ordered = tuple(sorted(draws, key=lambda draw: int(draw.issue)))
    features: List[Tuple[float, ...]] = []
    targets: List[int] = []
    issues: List[str] = []
    digits: List[int] = []
    totals = [0] * 10
    last_seen = [-1] * 10
    gap_sums = [0] * 10
    gap_counts = [0] * 10
    max_gaps = [0] * 10
    for target_index, draw in enumerate(ordered):
        if target_index >= min_history:
            values: List[float] = []
            for window in windows:
                scoped = digits[-window:]
                counts = Counter(scoped)
                values.extend(counts[digit] / len(scoped) if scoped else 0.0 for digit in range(10))
            values.extend(totals[digit] / len(digits) if digits else 0.0 for digit in range(10))
            values.extend(float(len(digits) if last_seen[digit] < 0 else len(digits) - 1 - last_seen[digit]) for digit in range(10))
            values.extend(gap_sums[digit] / gap_counts[digit] if gap_counts[digit] else float(max(len(digits), 1)) for digit in range(10))
            values.extend(float(max_gaps[digit]) if gap_counts[digit] else float(max(len(digits), 1)) for digit in range(10))
            values.extend(float(_lag(digits, lag)) for lag in (1, 2, 3, 5))
            previous = ordered[target_index - 1].numbers
            previous_counts = Counter(previous)
            values.extend((
                float(sum(previous)), float(max(previous) - min(previous)),
                float(sum(value % 2 for value in previous)), float(sum(value >= 5 for value in previous)),
                float(sum(count - 1 for count in previous_counts.values() if count > 1)),
            ))
            features.append(tuple(values))
            targets.append(draw.numbers[position])
            issues.append(draw.issue)
        digit = draw.numbers[position]
        if last_seen[digit] >= 0:
            gap = target_index - last_seen[digit]
            gap_sums[digit] += gap
            gap_counts[digit] += 1
            max_gaps[digit] = max(max_gaps[digit], gap)
        last_seen[digit] = target_index
        totals[digit] += 1
        digits.append(digit)
    return Pl5TrainingRows(pl5_feature_names(windows), tuple(features), tuple(targets), tuple(issues))


def _lag(values: Sequence[int], lag: int) -> int:
    return values[-lag] if len(values) >= lag else -1


def _last_index(values: Sequence[int], digit: int) -> int:
    for index in range(len(values) - 1, -1, -1):
        if values[index] == digit:
            return index
    return -1


def _appearance_gaps(values: Sequence[int], digit: int) -> Tuple[int, ...]:
    indexes = [index for index, value in enumerate(values) if value == digit]
    return tuple(right - left for left, right in zip(indexes, indexes[1:]))
