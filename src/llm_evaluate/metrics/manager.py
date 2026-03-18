from __future__ import annotations

from collections import defaultdict

from llm_evaluate.metrics.text_metrics import char_f1, exact_match, rouge_l, token_f1


class MetricsManager:
    def __init__(self, metric_names: list[str]):
        self.metric_names = metric_names
        self._impls = {
            "exact_match": exact_match,
            "token_f1": token_f1,
            "rouge_l": rouge_l,
            "char_f1": char_f1,
        }

    def score_sample(self, prediction: str, reference: str | None) -> dict[str, float]:
        if reference is None:
            return {name: float("nan") for name in self.metric_names}
        scores: dict[str, float] = {}
        for name in self.metric_names:
            if name not in self._impls:
                raise ValueError(f"Unsupported metric: {name}")
            scores[name] = self._impls[name](prediction, reference)
        return scores

    def aggregate(self, sample_scores: list[dict[str, float]]) -> dict[str, float]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for item in sample_scores:
            for k, v in item.items():
                if v == v:  # skip nan
                    buckets[k].append(v)
        return {k: (sum(vs) / len(vs) if vs else float("nan")) for k, vs in buckets.items()}
