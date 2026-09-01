from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

from lottery_sim.betting import BehavioralBettor, BetDistributionEstimator, UniformBettor
from lottery_sim.ml.house_min_payout import _inverse_softmax
from lottery_sim.models import Draw5D


@dataclass(frozen=True)
class MechanismMetrics:
    mechanism: str
    log_likelihood: float
    average_log_probability: float
    brier_score: float
    average_bet_popularity_percentile: float
    average_expected_payout_percentile: float


@dataclass(frozen=True)
class PercentileStatistics:
    coldest_10_rate: float
    coldest_20_rate: float
    hottest_10_rate: float
    chi_square_statistic: float
    chi_square_p_value: float
    ks_statistic: float
    ks_p_value: float
    coldest_10_bootstrap_ci: Tuple[float, float]
    coldest_20_bootstrap_ci: Tuple[float, float]
    hottest_10_bootstrap_ci: Tuple[float, float]


@dataclass(frozen=True)
class HouseDrawDiagnostic:
    issue: str
    number_text: str
    bet_popularity_percentile: float
    expected_payout_percentile: float
    bet_probability: float
    expected_ticket_count: float
    expected_payout: float
    house_probability: float


@dataclass(frozen=True)
class HouseAnalysisResult:
    game_code: str
    draw_count: int
    bettor_model: str
    house_temperature: float
    simulated_ticket_count: int
    mechanisms: Tuple[MechanismMetrics, ...]
    percentile_statistics: PercentileStatistics
    latest_low_payout_numbers: Tuple[Dict[str, float | str], ...]
    draw_diagnostics: Tuple[HouseDrawDiagnostic, ...]
    disclaimer: str = "House hypothesis 是实验建模假设，不代表真实开奖机制或真实全国投注分布。"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def analyze_house_history(
    draws: Sequence[Draw5D],
    bettor_model: str = "behavioral",
    house_temperature: float = 1_000_000.0,
    simulated_ticket_count: int = 1_000_000,
    random_seed: int = 20260505,
    manual_pick_ratio: float = 0.60,
    machine_pick_ratio: float = 0.40,
) -> HouseAnalysisResult:
    ordered = tuple(sorted(draws, key=lambda draw: int(draw.issue)))
    bettor = UniformBettor() if bettor_model == "uniform" else BehavioralBettor()
    if bettor_model not in {"uniform", "behavioral"}:
        raise ValueError("bettor_model must be uniform or behavioral")
    estimator = BetDistributionEstimator(
        bettor,
        simulated_ticket_count=simulated_ticket_count,
        random_seed=random_seed,
        manual_pick_ratio=manual_pick_ratio,
        machine_pick_ratio=machine_pick_ratio,
    )
    payout_per_ticket = 100_000.0
    uniform_full_probability = 1.0 / 100_000
    random_log_likelihood = 0.0
    house_log_likelihood = 0.0
    random_brier_total = 0.0
    house_brier_total = 0.0
    percentiles: List[float] = []
    diagnostics: List[HouseDrawDiagnostic] = []
    history: List[Draw5D] = []

    for draw in ordered:
        market = estimator.estimate(history)
        expected_payout = market.expected_ticket_counts * payout_per_ticket
        house_probabilities = _inverse_softmax(expected_payout, house_temperature)
        actual_index = int(draw.number_text)
        popularity = _midrank_percentile(market.probabilities, market.probabilities[actual_index])
        payout_percentile = _midrank_percentile(expected_payout, expected_payout[actual_index])
        percentiles.append(popularity)
        random_log_likelihood += math.log(uniform_full_probability)
        house_log_likelihood += math.log(max(float(house_probabilities[actual_index]), 1e-300))
        random_brier_total += 0.9
        house_brier_total += _full_to_position_brier(house_probabilities, draw.numbers)
        diagnostics.append(HouseDrawDiagnostic(
            issue=draw.issue,
            number_text=draw.number_text,
            bet_popularity_percentile=popularity,
            expected_payout_percentile=payout_percentile,
            bet_probability=float(market.probabilities[actual_index]),
            expected_ticket_count=float(market.expected_ticket_counts[actual_index]),
            expected_payout=float(expected_payout[actual_index]),
            house_probability=float(house_probabilities[actual_index]),
        ))
        history.append(draw)

    count = len(ordered)
    denominator = count or 1
    percentile_array = np.asarray(percentiles, dtype=np.float64)
    percentile_stats = _percentile_statistics(percentile_array, random_seed)
    average_percentile = float(percentile_array.mean()) if count else 0.0
    mechanisms = (
        MechanismMetrics("uniform_betting_random_draw", random_log_likelihood, random_log_likelihood / denominator, random_brier_total / denominator, 0.5, 0.5),
        MechanismMetrics(f"{bettor_model}_betting_random_draw", random_log_likelihood, random_log_likelihood / denominator, random_brier_total / denominator, average_percentile, average_percentile),
        MechanismMetrics(f"{bettor_model}_betting_house_min_payout_draw", house_log_likelihood, house_log_likelihood / denominator, house_brier_total / denominator, average_percentile, average_percentile),
    )
    latest_market = estimator.estimate(history)
    latest_expected_payout = latest_market.expected_ticket_counts * payout_per_ticket
    latest_house_probability = _inverse_softmax(latest_expected_payout, house_temperature)
    coldest = np.argsort(latest_expected_payout, kind="stable")[:20]
    latest_rows = tuple({
        "number_text": f"{int(index):05d}",
        "bet_probability": float(latest_market.probabilities[index]),
        "expected_ticket_count": float(latest_market.expected_ticket_counts[index]),
        "expected_payout": float(latest_expected_payout[index]),
        "house_probability": float(latest_house_probability[index]),
    } for index in coldest)
    return HouseAnalysisResult(
        game_code="pl5",
        draw_count=count,
        bettor_model=bettor_model,
        house_temperature=house_temperature,
        simulated_ticket_count=simulated_ticket_count,
        mechanisms=mechanisms,
        percentile_statistics=percentile_stats,
        latest_low_payout_numbers=latest_rows,
        draw_diagnostics=tuple(diagnostics),
    )


def save_house_analysis_json(result: HouseAnalysisResult, path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def render_house_analysis_text(result: HouseAnalysisResult) -> str:
    lines = [
        "排列五 House hypothesis 历史异常分析",
        f"历史期数：{result.draw_count}",
        f"投注行为模型：{result.bettor_model}",
        f"House temperature：{result.house_temperature}",
        "",
        "机制 | Log likelihood | 平均Log概率 | Brier | 平均投注热度percentile | 平均返奖percentile",
        "--- | ---: | ---: | ---: | ---: | ---:",
    ]
    for item in result.mechanisms:
        lines.append(f"{item.mechanism} | {item.log_likelihood:.4f} | {item.average_log_probability:.6f} | {item.brier_score:.6f} | {item.average_bet_popularity_percentile:.2%} | {item.average_expected_payout_percentile:.2%}")
    stats_result = result.percentile_statistics
    lines.extend((
        "", "历史开奖号码投注热度 percentile：",
        f"- 最冷10%：{stats_result.coldest_10_rate:.2%}，95% bootstrap CI {stats_result.coldest_10_bootstrap_ci[0]:.2%}–{stats_result.coldest_10_bootstrap_ci[1]:.2%}",
        f"- 最冷20%：{stats_result.coldest_20_rate:.2%}，95% bootstrap CI {stats_result.coldest_20_bootstrap_ci[0]:.2%}–{stats_result.coldest_20_bootstrap_ci[1]:.2%}",
        f"- 最热10%：{stats_result.hottest_10_rate:.2%}，95% bootstrap CI {stats_result.hottest_10_bootstrap_ci[0]:.2%}–{stats_result.hottest_10_bootstrap_ci[1]:.2%}",
        f"- Chi-square：{stats_result.chi_square_statistic:.4f}，p={stats_result.chi_square_p_value:.6g}",
        f"- KS：{stats_result.ks_statistic:.4f}，p={stats_result.ks_p_value:.6g}",
        "", "最新估计低返奖号码：", "号码 | 投注概率 | 预计注数 | 预计返奖 | House概率", "--- | ---: | ---: | ---: | ---:",
    ))
    for row in result.latest_low_payout_numbers:
        lines.append(f"{row['number_text']} | {row['bet_probability']:.8g} | {row['expected_ticket_count']:.4f} | {row['expected_payout']:.2f} | {row['house_probability']:.8g}")
    lines.extend(("", result.disclaimer))
    return "\n".join(lines)


def _midrank_percentile(values: np.ndarray, actual: float) -> float:
    less = int(np.count_nonzero(values < actual))
    equal = int(np.count_nonzero(values == actual))
    return (less + 0.5 * equal) / values.size


def _full_to_position_brier(probabilities: np.ndarray, actual: Tuple[int, int, int, int, int]) -> float:
    cube = probabilities.reshape((10, 10, 10, 10, 10))
    total = 0.0
    for position, actual_digit in enumerate(actual):
        axes = tuple(axis for axis in range(5) if axis != position)
        row = cube.sum(axis=axes)
        target = np.zeros(10, dtype=np.float64)
        target[actual_digit] = 1.0
        total += float(np.square(row - target).sum())
    return total / 5


def _percentile_statistics(values: np.ndarray, seed: int) -> PercentileStatistics:
    if values.size == 0:
        return PercentileStatistics(0, 0, 0, 0, 1, 0, 1, (0, 0), (0, 0), (0, 0))
    cold10 = values <= 0.1
    cold20 = values <= 0.2
    hot10 = values >= 0.9
    counts, _ = np.histogram(values, bins=np.linspace(0, 1, 11))
    chi = stats.chisquare(counts, np.full(10, values.size / 10))
    ks = stats.kstest(values, "uniform")
    return PercentileStatistics(
        coldest_10_rate=float(cold10.mean()),
        coldest_20_rate=float(cold20.mean()),
        hottest_10_rate=float(hot10.mean()),
        chi_square_statistic=float(chi.statistic),
        chi_square_p_value=float(chi.pvalue),
        ks_statistic=float(ks.statistic),
        ks_p_value=float(ks.pvalue),
        coldest_10_bootstrap_ci=_bootstrap_rate_ci(cold10, seed),
        coldest_20_bootstrap_ci=_bootstrap_rate_ci(cold20, seed + 1),
        hottest_10_bootstrap_ci=_bootstrap_rate_ci(hot10, seed + 2),
    )


def _bootstrap_rate_ci(indicators: np.ndarray, seed: int, samples: int = 2000) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    rates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        rates[index] = rng.choice(indicators, size=indicators.size, replace=True).mean()
    lower, upper = np.quantile(rates, (0.025, 0.975))
    return float(lower), float(upper)
