#!/usr/bin/env python3
"""Compare a BitsAndBytes candidate against a recorded FP16 Reranker baseline."""

import argparse
import json
import math
import urllib.request
from pathlib import Path
from typing import Any


TOP1_AGREEMENT_MIN = 0.95
MAX_ABSOLUTE_METRIC_DROP = 0.0121


def post_rerank(base_url: str, case: dict[str, Any]) -> list[int]:
    payload = json.dumps(
        {"query": case["query"], "documents": case["documents"]}, ensure_ascii=False
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/rerank",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=180) as response:
        result = json.load(response)
    return [item["index"] for item in result["results"]]


def dcg(ranking: list[int], relevance: dict[int, int], k: int = 10) -> float:
    return sum(
        (2 ** relevance.get(index, 0) - 1) / math.log2(rank + 2)
        for rank, index in enumerate(ranking[:k])
    )


def ndcg(ranking: list[int], relevance: dict[int, int], k: int = 10) -> float:
    ideal = sorted(relevance, key=lambda index: relevance[index], reverse=True)
    denominator = dcg(ideal, relevance, k)
    return dcg(ranking, relevance, k) / denominator if denominator else 0.0


def mrr(ranking: list[int], relevance: dict[int, int], k: int = 10) -> float:
    for rank, index in enumerate(ranking[:k], start=1):
        if relevance.get(index, 0) > 0:
            return 1 / rank
    return 0.0


def evaluate(cases: list[dict[str, Any]], rankings: list[list[int]]) -> dict[str, float]:
    ndcgs: list[float] = []
    mrrs: list[float] = []
    for case, ranking in zip(cases, rankings, strict=True):
        relevance = {int(index): int(score) for index, score in case["relevance"].items()}
        ndcgs.append(ndcg(ranking, relevance))
        mrrs.append(mrr(ranking, relevance))
    return {"ndcg_at_10": sum(ndcgs) / len(ndcgs), "mrr_at_10": sum(mrrs) / len(mrrs)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tests/fixtures/reranker_eval.json")
    parser.add_argument("--candidate-url")
    parser.add_argument("--record-fp16-url")
    parser.add_argument("--baseline-file", default="reranker_fp16_baseline.json")
    args = parser.parse_args()
    cases = json.loads(Path(args.dataset).read_text(encoding="utf-8"))

    if args.record_fp16_url:
        rankings = [post_rerank(args.record_fp16_url, case) for case in cases]
        record = {"model_mode": "fp16", "rankings": rankings, "metrics": evaluate(cases, rankings)}
        Path(args.baseline_file).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0

    if not args.candidate_url:
        parser.error("--candidate-url is required unless --record-fp16-url is used")
    baseline = json.loads(Path(args.baseline_file).read_text(encoding="utf-8"))
    candidate_rankings = [post_rerank(args.candidate_url, case) for case in cases]
    baseline_rankings = baseline["rankings"]
    top1_agreement = sum(
        baseline_ranking[0] == candidate_ranking[0]
        for baseline_ranking, candidate_ranking in zip(baseline_rankings, candidate_rankings, strict=True)
    ) / len(cases)
    baseline_metrics = evaluate(cases, baseline_rankings)
    candidate_metrics = evaluate(cases, candidate_rankings)
    ndcg_drop = baseline_metrics["ndcg_at_10"] - candidate_metrics["ndcg_at_10"]
    mrr_drop = baseline_metrics["mrr_at_10"] - candidate_metrics["mrr_at_10"]
    passed = (
        top1_agreement >= TOP1_AGREEMENT_MIN
        and ndcg_drop <= MAX_ABSOLUTE_METRIC_DROP
        and mrr_drop <= MAX_ABSOLUTE_METRIC_DROP
    )
    report = {
        "top1_agreement": top1_agreement,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "ndcg_at_10_absolute_drop": ndcg_drop,
        "mrr_at_10_absolute_drop": mrr_drop,
        "thresholds": {
            "top1_agreement_min": TOP1_AGREEMENT_MIN,
            "max_absolute_metric_drop": MAX_ABSOLUTE_METRIC_DROP,
        },
        "passed": passed,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
