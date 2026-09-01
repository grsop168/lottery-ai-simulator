from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats

from lottery_sim.betting import BehavioralBettor, BehavioralBettorConfig, BetDistributionEstimator
from lottery_sim.ml.house_min_payout import _inverse_softmax
from lottery_sim.models import Draw5D


@dataclass(frozen=True)
class SensitivityConfig:
    config_id: str
    manual_pick_ratio: float
    digit_preference: str
    four_avoidance: str
    repeat_preference: str
    sequence_preference: str
    recent_follow: str
    hot_preference: str
    cold_preference: str
    keeper_tendency: str
    temperature_level: str
    house_temperature: float
    bettor_parameters: Dict[str, object]


@dataclass(frozen=True)
class SegmentEvaluation:
    config_id: str
    draw_count: int
    log_likelihood: float
    average_log_probability: float
    uniform_log_likelihood: float
    log_likelihood_delta: float
    brier_score: float
    uniform_brier: float
    brier_improvement: float
    cold10_ratio: float
    cold20_ratio: float
    hot10_ratio: float
    chi_square: float
    chi_square_p_value: float
    chi_square_adjusted_p: float = 1.0
    ks_statistic: float = 0.0
    ks_p_value: float = 1.0
    ks_adjusted_p: float = 1.0
    cold10_bootstrap_ci: Tuple[float, float] = (0.0, 0.0)
    cold20_bootstrap_ci: Tuple[float, float] = (0.0, 0.0)
    hot10_bootstrap_ci: Tuple[float, float] = (0.0, 0.0)

    @property
    def better_than_uniform(self) -> bool:
        return self.log_likelihood_delta > 0 and self.brier_improvement > 0


@dataclass(frozen=True)
class SensitivityPair:
    rank: int
    config: SensitivityConfig
    train: SegmentEvaluation
    holdout: SegmentEvaluation
    status: str


@dataclass(frozen=True)
class HouseSensitivityResult:
    game_code: str
    train_draws: int
    holdout_draws: int
    split_issue: str
    parameter_space: Dict[str, object]
    scanned_combination_count: int
    selected_top_n: int
    train_better_count: int
    train_better_ratio: float
    holdout_better_count: int
    holdout_better_ratio: float
    fdr_significant_train_count: int
    fdr_significant_holdout_count: int
    robust_cluster_count: int
    robust_parameter_regions: Tuple[Tuple[str, ...], ...]
    failed_parameter_regions: Dict[str, int]
    train_results: Tuple[SegmentEvaluation, ...]
    selected_results: Tuple[SensitivityPair, ...]
    conclusion_level: str
    conclusion: str
    disclaimer: str = "House hypothesis sensitivity analysis 是实验性稳健性检验，不代表真实开奖或投注分布。"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


PARAMETER_SPACE: Dict[str, object] = {
    "manual_pick_ratio": (0.2, 0.4, 0.6, 0.8),
    "machine_pick_ratio": "1 - manual_pick_ratio",
    "digit_preference": ("weak", "medium", "strong"),
    "four_avoidance": ("none", "weak", "strong"),
    "repeat_preference": ("none", "weak", "strong"),
    "sequence_preference": ("none", "weak", "strong"),
    "recent_follow": ("0", "low", "medium", "high"),
    "hot_preference": ("0", "low", "medium", "high"),
    "cold_preference": ("0", "low", "medium", "high"),
    "keeper_tendency": ("0", "low", "medium", "high"),
    "house_temperature": ("very_high", "high", "medium", "low", "very_low"),
    "grid_reduction": "32 deterministic level-balanced combinations",
}


def deterministic_sensitivity_configs(count: int = 32) -> Tuple[SensitivityConfig, ...]:
    manuals = (0.2, 0.4, 0.6, 0.8)
    digit_levels = ("weak", "medium", "strong")
    tri_levels = ("none", "weak", "strong")
    quad_levels = ("0", "low", "medium", "high")
    temp_levels = ("very_high", "high", "medium", "low", "very_low")
    temperatures = {"very_high": 5_000_000.0, "high": 2_000_000.0, "medium": 1_000_000.0, "low": 500_000.0, "very_low": 200_000.0}
    result = []
    for index in range(count):
        manual = manuals[index % 4]
        digit = digit_levels[(index // 4) % 3]
        four = tri_levels[(index * 2 + index // 3) % 3]
        repeat = tri_levels[(index * 5 + 1) % 3]
        sequence = tri_levels[(index * 7 + 2) % 3]
        recent = quad_levels[(index * 3 + index // 4) % 4]
        hot = quad_levels[(index * 5 + 1) % 4]
        cold = quad_levels[(index * 7 + 2) % 4]
        keeper = quad_levels[(index * 11 + 3) % 4]
        temp = temp_levels[(index * 3 + index // 5) % 5]
        bettor = _bettor_config(digit, four, repeat, sequence, recent, hot, cold, keeper)
        result.append(SensitivityConfig(
            config_id=f"house-{index + 1:03d}", manual_pick_ratio=manual,
            digit_preference=digit, four_avoidance=four, repeat_preference=repeat,
            sequence_preference=sequence, recent_follow=recent, hot_preference=hot,
            cold_preference=cold, keeper_tendency=keeper, temperature_level=temp,
            house_temperature=temperatures[temp], bettor_parameters=asdict(bettor),
        ))
    return tuple(result)


def run_house_sensitivity(
    draws: Sequence[Draw5D],
    train_count: int = 5000,
    top_n: int = 20,
    configs: Sequence[SensitivityConfig] | None = None,
    simulated_ticket_count: int = 1_000_000,
    random_seed: int = 20260505,
    bootstrap_samples: int = 1000,
) -> HouseSensitivityResult:
    ordered = tuple(sorted(draws, key=lambda draw: int(draw.issue)))
    if len(ordered) <= train_count:
        raise ValueError("draw history must contain both train and holdout segments")
    actual_configs = tuple(configs or deterministic_sensitivity_configs())
    train_draws = ordered[:train_count]
    holdout_draws = ordered[train_count:]
    train_raw = [
        evaluate_sensitivity_segment(config, train_draws, (), simulated_ticket_count, random_seed, 0)
        for config in actual_configs
    ]
    train_adjusted = _apply_fdr(train_raw)
    ranked = sorted(zip(actual_configs, train_adjusted), key=lambda item: _train_rank_key(item[1]))
    selected = ranked[:min(top_n, len(ranked))]
    holdout_raw = [
        evaluate_sensitivity_segment(config, holdout_draws, train_draws, simulated_ticket_count, random_seed, bootstrap_samples)
        for config, _ in selected
    ]
    holdout_adjusted = _apply_fdr(holdout_raw)
    pairs = []
    for rank, ((config, train_result), holdout_result) in enumerate(zip(selected, holdout_adjusted), start=1):
        consistent = train_result.better_than_uniform and holdout_result.better_than_uniform
        cold_consistent = (train_result.cold10_ratio >= 0.1) == (holdout_result.cold10_ratio >= 0.1)
        status = "potentially robust" if consistent and cold_consistent else ("likely overfit" if train_result.better_than_uniform and not holdout_result.better_than_uniform else "no support")
        pairs.append(SensitivityPair(rank, config, train_result, holdout_result, status))
    train_better = sum(result.better_than_uniform for result in train_adjusted)
    holdout_better = sum(pair.holdout.better_than_uniform for pair in pairs)
    robust_regions = _robust_regions(pairs)
    robust_clusters = len(robust_regions)
    failure_summary = {
        "likely_overfit": sum(pair.status == "likely overfit" for pair in pairs),
        "no_support": sum(pair.status == "no support" for pair in pairs),
        "train_not_better": sum(not pair.train.better_than_uniform for pair in pairs),
        "holdout_not_better": sum(not pair.holdout.better_than_uniform for pair in pairs),
    }
    significant_train = sum(min(result.chi_square_adjusted_p, result.ks_adjusted_p) < 0.05 for result in train_adjusted)
    significant_holdout = sum(min(pair.holdout.chi_square_adjusted_p, pair.holdout.ks_adjusted_p) < 0.05 for pair in pairs)
    level, conclusion = _support_level(pairs, robust_clusters)
    return HouseSensitivityResult(
        game_code="pl5", train_draws=len(train_draws), holdout_draws=len(holdout_draws),
        split_issue=holdout_draws[0].issue, parameter_space=PARAMETER_SPACE,
        scanned_combination_count=len(actual_configs), selected_top_n=len(pairs),
        train_better_count=train_better, train_better_ratio=train_better / len(actual_configs),
        holdout_better_count=holdout_better, holdout_better_ratio=holdout_better / len(pairs) if pairs else 0.0,
        fdr_significant_train_count=significant_train, fdr_significant_holdout_count=significant_holdout,
        robust_cluster_count=robust_clusters, robust_parameter_regions=robust_regions,
        failed_parameter_regions=failure_summary, train_results=tuple(train_adjusted),
        selected_results=tuple(pairs), conclusion_level=level, conclusion=conclusion,
    )


def evaluate_sensitivity_segment(
    config: SensitivityConfig,
    evaluation_draws: Sequence[Draw5D],
    initial_history: Sequence[Draw5D],
    simulated_ticket_count: int,
    random_seed: int,
    bootstrap_samples: int,
) -> SegmentEvaluation:
    bettor_config = BehavioralBettorConfig(**config.bettor_parameters)
    estimator = BetDistributionEstimator(
        BehavioralBettor(bettor_config), simulated_ticket_count=simulated_ticket_count,
        random_seed=random_seed, manual_pick_ratio=config.manual_pick_ratio,
        machine_pick_ratio=1.0 - config.manual_pick_ratio,
    )
    history = list(initial_history)
    log_likelihood = 0.0
    brier_total = 0.0
    percentiles = []
    for draw in evaluation_draws:
        market = estimator.estimate(history)
        payouts = market.expected_ticket_counts * 100_000.0
        house = _inverse_softmax(payouts, config.house_temperature)
        actual_index = int(draw.number_text)
        log_likelihood += math.log(max(float(house[actual_index]), 1e-300))
        brier_total += _position_brier(house, draw.numbers)
        percentiles.append(_midrank(market.probabilities, market.probabilities[actual_index]))
        history.append(draw)
    count = len(evaluation_draws)
    denominator = count or 1
    uniform_ll = count * math.log(1.0 / 100_000)
    values = np.asarray(percentiles, dtype=np.float64)
    cold10 = values <= 0.1
    cold20 = values <= 0.2
    hot10 = values >= 0.9
    bins, _ = np.histogram(values, bins=np.linspace(0, 1, 11))
    chi = stats.chisquare(bins, np.full(10, denominator / 10))
    ks = stats.kstest(values, "uniform")
    return SegmentEvaluation(
        config_id=config.config_id, draw_count=count, log_likelihood=log_likelihood,
        average_log_probability=log_likelihood / denominator, uniform_log_likelihood=uniform_ll,
        log_likelihood_delta=log_likelihood - uniform_ll, brier_score=brier_total / denominator,
        uniform_brier=0.9, brier_improvement=0.9 - brier_total / denominator,
        cold10_ratio=float(cold10.mean()), cold20_ratio=float(cold20.mean()), hot10_ratio=float(hot10.mean()),
        chi_square=float(chi.statistic), chi_square_p_value=float(chi.pvalue),
        ks_statistic=float(ks.statistic), ks_p_value=float(ks.pvalue),
        cold10_bootstrap_ci=_bootstrap_ci(cold10, random_seed, bootstrap_samples),
        cold20_bootstrap_ci=_bootstrap_ci(cold20, random_seed + 1, bootstrap_samples),
        hot10_bootstrap_ci=_bootstrap_ci(hot10, random_seed + 2, bootstrap_samples),
    )


def benjamini_hochberg(p_values: Sequence[float]) -> Tuple[float, ...]:
    count = len(p_values)
    if not count:
        return ()
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, float(p_values[index]) * count / rank)
        adjusted[index] = min(1.0, running)
    return tuple(adjusted)


def save_house_sensitivity_json(result: HouseSensitivityResult, path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def render_house_sensitivity_text(result: HouseSensitivityResult) -> str:
    lines = [
        "排列五 House hypothesis 参数稳健性分析", "",
        f"数据分割：TRAIN 前{result.train_draws}期；HOLDOUT 后{result.holdout_draws}期；首个HOLDOUT期号 {result.split_issue}",
        f"扫描组合：{result.scanned_combination_count}；冻结验证Top N：{result.selected_top_n}",
        f"TRAIN同时优于Uniform：{result.train_better_count}/{result.scanned_combination_count} ({result.train_better_ratio:.2%})",
        f"HOLDOUT仍同时优于Uniform：{result.holdout_better_count}/{result.selected_top_n} ({result.holdout_better_ratio:.2%})",
        f"BH-FDR后显著：TRAIN {result.fdr_significant_train_count}；HOLDOUT {result.fdr_significant_holdout_count}",
        f"稳健相邻区域数量：{result.robust_cluster_count}", "",
        "参数空间：", json.dumps(result.parameter_space, ensure_ascii=False, sort_keys=True), "",
        "失败参数区域摘要：", json.dumps(result.failed_parameter_regions, ensure_ascii=False, sort_keys=True),
        f"稳健参数区域：{json.dumps(result.robust_parameter_regions, ensure_ascii=False)}", "",
        "TRAIN Top20 与冻结 HOLDOUT：",
        "排名 | 配置 | TRAIN ΔLL | TRAIN Brier改善 | HOLDOUT ΔLL | HOLDOUT Brier改善 | HOLDOUT cold10 | HOLDOUT cold20 | HOLDOUT hot10 | Chi FDR | KS FDR | 判断",
        "---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---",
    ]
    for pair in result.selected_results:
        lines.append(
            f"{pair.rank} | {pair.config.config_id} | {pair.train.log_likelihood_delta:.4f} | {pair.train.brier_improvement:.6f} | "
            f"{pair.holdout.log_likelihood_delta:.4f} | {pair.holdout.brier_improvement:.6f} | {pair.holdout.cold10_ratio:.2%} | "
            f"{pair.holdout.cold20_ratio:.2%} | {pair.holdout.hot10_ratio:.2%} | {pair.holdout.chi_square_adjusted_p:.6g} | "
            f"{pair.holdout.ks_adjusted_p:.6g} | {pair.status}"
        )
    lines.extend(("", f"最终支持等级：{result.conclusion_level}", result.conclusion, "", result.disclaimer))
    return "\n".join(lines)


def _bettor_config(digit: str, four: str, repeat: str, sequence: str, recent: str, hot: str, cold: str, keeper: str) -> BehavioralBettorConfig:
    digit_map = {"weak": 0.08, "medium": 0.18, "strong": 0.32}
    four_map = {"none": 0.0, "weak": -0.15, "strong": -0.35}
    repeat_map = {"none": 0.0, "weak": 0.12, "strong": 0.25}
    sequence_map = {"none": (0.0, 0.0), "weak": (0.06, 0.20), "strong": (0.18, 0.60)}
    recent_map = {"0": 0.0, "low": 0.06, "medium": 0.16, "high": 0.30}
    hot_map = {"0": 0.0, "low": 0.10, "medium": 0.22, "high": 0.40}
    cold_map = {"0": 0.0, "low": -0.04, "medium": -0.08, "high": -0.16}
    keeper_map = {"0": 0.0, "low": 0.05, "medium": 0.12, "high": 0.22}
    adjacent, straight = sequence_map[sequence]
    return BehavioralBettorConfig(
        preferred_digit_weight=digit_map[digit], avoided_four_weight=four_map[four],
        repeated_digit_weight=repeat_map[repeat], adjacent_sequence_weight=adjacent,
        full_straight_weight=straight, recent_draw_match_weight=recent_map[recent],
        hot_digit_weight=hot_map[hot], cold_digit_weight=cold_map[cold],
        fixed_number_weight=keeper_map[keeper],
    )


def _position_brier(probabilities: np.ndarray, actual: Tuple[int, int, int, int, int]) -> float:
    cube = probabilities.reshape((10, 10, 10, 10, 10))
    total = 0.0
    for position, digit in enumerate(actual):
        axes = tuple(axis for axis in range(5) if axis != position)
        row = cube.sum(axis=axes)
        target = np.zeros(10); target[digit] = 1.0
        total += float(np.square(row - target).sum())
    return total / 5


def _midrank(values: np.ndarray, actual: float) -> float:
    return (np.count_nonzero(values < actual) + 0.5 * np.count_nonzero(values == actual)) / values.size


def _bootstrap_ci(indicators: np.ndarray, seed: int, samples: int) -> Tuple[float, float]:
    if not samples:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    rates = np.asarray([rng.choice(indicators, indicators.size, replace=True).mean() for _ in range(samples)])
    return tuple(float(value) for value in np.quantile(rates, (0.025, 0.975)))


def _apply_fdr(results: Sequence[SegmentEvaluation]) -> List[SegmentEvaluation]:
    chi = benjamini_hochberg([item.chi_square_p_value for item in results])
    ks = benjamini_hochberg([item.ks_p_value for item in results])
    return [replace(item, chi_square_adjusted_p=chi[index], ks_adjusted_p=ks[index]) for index, item in enumerate(results)]


def _train_rank_key(result: SegmentEvaluation) -> Tuple[float, float, float, float, str]:
    direction = min(result.cold10_ratio - 0.1, result.cold20_ratio - 0.2)
    consistency = -abs((result.cold10_ratio - 0.1) - (result.cold20_ratio - 0.2))
    return (-result.log_likelihood_delta, -result.brier_improvement, -direction, -consistency, result.config_id)


def _config_distance(left: SensitivityConfig, right: SensitivityConfig) -> int:
    fields = ("manual_pick_ratio", "digit_preference", "four_avoidance", "repeat_preference", "sequence_preference", "recent_follow", "hot_preference", "cold_preference", "keeper_tendency", "temperature_level")
    return sum(getattr(left, field) != getattr(right, field) for field in fields)


def _robust_regions(pairs: Sequence[SensitivityPair]) -> Tuple[Tuple[str, ...], ...]:
    robust = [pair for pair in pairs if pair.status == "potentially robust" and min(pair.holdout.chi_square_adjusted_p, pair.holdout.ks_adjusted_p) < 0.05]
    neighbors = {index: set() for index in range(len(robust))}
    for left in range(len(robust)):
        for right in range(left + 1, len(robust)):
            if _config_distance(robust[left].config, robust[right].config) <= 2:
                neighbors[left].add(right); neighbors[right].add(left)
    regions = []
    remaining = {index for index, linked in neighbors.items() if linked}
    while remaining:
        stack = [remaining.pop()]
        component = set(stack)
        while stack:
            current = stack.pop()
            for neighbor in neighbors[current]:
                if neighbor not in component:
                    component.add(neighbor); remaining.discard(neighbor); stack.append(neighbor)
        regions.append(tuple(sorted(robust[index].config.config_id for index in component)))
    return tuple(sorted(regions))


def _support_level(pairs: Sequence[SensitivityPair], robust_clusters: int) -> Tuple[str, str]:
    qualified = [pair for pair in pairs if pair.status == "potentially robust" and pair.holdout.chi_square_adjusted_p < 0.05 and pair.holdout.ks_adjusted_p < 0.05]
    if not qualified:
        return "NO SUPPORT", "没有配置同时满足训练/验证概率评分改善、方向一致和多重比较校正后的双重显著性。"
    if robust_clusters == 0:
        return "WEAK / UNSTABLE SUPPORT", "仅有孤立配置满足部分条件，没有形成相邻稳定参数区域。"
    ratio = len(qualified) / max(len(pairs), 1)
    if ratio >= 0.5 and robust_clusters >= 3:
        return "STRONG ROBUST SUPPORT", "多个相邻参数区域在训练和验证段呈现一致、多重校正后显著的支持。"
    return "MODERATE ROBUST SUPPORT", "至少一个相邻参数区域在训练和验证段呈现一致、多重校正后显著的支持。"
