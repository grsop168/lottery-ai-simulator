from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence


@dataclass(frozen=True)
class Candidate:
    rank: int
    strategy_name: str
    numbers: Any
    number_text: str
    reason: str


def generate_candidates(
    history: Sequence[Any],
    game: Any,
    strategies: Sequence[Any],
    filler_strategy: Any,
    count: int = 10,
) -> List[Candidate]:
    if count <= 0:
        raise ValueError("候选数量必须大于0")

    candidates: List[Candidate] = []
    seen = set()

    for strategy in strategies:
        _try_add_candidate(candidates, seen, history, game, strategy)
        if len(candidates) >= count:
            return _rank_candidates(candidates[:count])

    max_attempts = count * 100
    attempts = 0
    while len(candidates) < count and attempts < max_attempts:
        attempts += 1
        _try_add_candidate(candidates, seen, history, game, filler_strategy)

    if len(candidates) < count:
        raise ValueError("无法生成足够的不重复候选号码")
    return _rank_candidates(candidates[:count])


def render_recommendation_report(
    game_name: str,
    candidates: Iterable[Candidate],
    history_count: int,
    latest_issue: str = "",
) -> str:
    lines = [
        f"{game_name} 候选号码报告",
        "",
        f"历史期数：{history_count}",
    ]
    if latest_issue:
        lines.append(f"最新期号：{latest_issue}")

    lines.extend([
        "",
        "排名 | 策略 | 候选号码 | 说明",
        "---: | --- | --- | ---",
    ])
    for candidate in candidates:
        lines.append(
            f"{candidate.rank} | "
            f"{candidate.strategy_name} | "
            f"{candidate.number_text} | "
            f"{candidate.reason}"
        )

    lines.extend([
        "",
        "风险提示：候选号码只用于模拟研究和后续验证，彩票具有强随机性，不能保证命中。",
    ])
    return "\n".join(lines)


def diversify_ranked_candidates(
    candidates: Sequence[Candidate],
    count: int = 10,
    minimum_different_positions: int = 2,
) -> List[Candidate]:
    """Greedily select a stable diverse subset from an already probability-ranked pool."""
    if count <= 0:
        raise ValueError("candidate count must be positive")
    selected: List[Candidate] = []
    selected_numbers = set()
    for distance in range(minimum_different_positions, -1, -1):
        for candidate in candidates:
            if candidate.number_text in selected_numbers:
                continue
            if all(_different_positions(candidate.number_text, item.number_text) >= distance for item in selected):
                selected.append(candidate)
                selected_numbers.add(candidate.number_text)
                if len(selected) == count:
                    return _rerank_candidates(selected, "多样化Top10")
    return _rerank_candidates(selected, "多样化Top10")


def _different_positions(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))


def _rerank_candidates(candidates: Sequence[Candidate], strategy_name: str) -> List[Candidate]:
    return [
        Candidate(index, strategy_name, candidate.numbers, candidate.number_text, candidate.reason)
        for index, candidate in enumerate(candidates, start=1)
    ]


def _try_add_candidate(candidates, seen, history, game, strategy) -> None:
    pick = game.validate_pick(strategy.generate(history))
    number_text = _number_text(pick)
    if number_text in seen:
        return
    seen.add(number_text)
    candidates.append(Candidate(
        rank=0,
        strategy_name=strategy.name,
        numbers=pick,
        number_text=number_text,
        reason="基于历史窗口生成",
    ))


def _rank_candidates(candidates: Sequence[Candidate]) -> List[Candidate]:
    return [
        Candidate(
            rank=index,
            strategy_name=candidate.strategy_name,
            numbers=candidate.numbers,
            number_text=candidate.number_text,
            reason=candidate.reason,
        )
        for index, candidate in enumerate(candidates, start=1)
    ]


def _number_text(value: Any) -> str:
    if hasattr(value, "number_text"):
        return value.number_text
    if isinstance(value, tuple):
        if value and all(isinstance(item, int) and 0 <= item <= 9 for item in value):
            return "".join(str(item) for item in value)
        return " ".join(str(item) for item in value)
    return str(value)
