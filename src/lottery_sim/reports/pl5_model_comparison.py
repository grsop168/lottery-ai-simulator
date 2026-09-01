from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

from lottery_sim.backtest.pl5_metrics import Pl5ModelMetrics


@dataclass(frozen=True)
class ModelComparisonResult:
    game_code: str
    models: Tuple[Pl5ModelMetrics, ...]
    disclaimer: str = "历史模拟结果不代表未来盈利能力。"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def save_model_comparison_json(result: ModelComparisonResult, path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def render_model_comparison_text(result: ModelComparisonResult) -> str:
    lines = [
        "排列五模型对比",
        "模型 | 回测期数 | Top1 | Top3 | Top5 | 平均命中位数 | 完整命中 | LogLoss | Brier | Top10覆盖 | Top50覆盖 | Top100覆盖 | Top10返奖率 | Top10净收益率",
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for metric in result.models:
        top10 = metric.returns.get("top10")
        lines.append(" | ".join((
            metric.model_name,
            str(metric.backtest_draws),
            _percent(metric.mean_position_accuracy),
            _percent(metric.mean_top3_coverage),
            _percent(metric.mean_top5_coverage),
            f"{metric.average_correct_positions:.3f}",
            str(metric.exact_hits),
            f"{metric.log_loss:.4f}",
            f"{metric.brier_score:.4f}",
            _percent(metric.candidate_exact_coverage.get("top10", 0.0)),
            _percent(metric.candidate_exact_coverage.get("top50", 0.0)),
            _percent(metric.candidate_exact_coverage.get("top100", 0.0)),
            _percent(top10.return_rate if top10 else 0.0),
            _percent(top10.profit_roi if top10 else 0.0),
        )))
    lines.extend(("", result.disclaimer))
    return "\n".join(lines)


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"
