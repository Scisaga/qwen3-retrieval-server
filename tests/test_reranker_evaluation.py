import importlib.util
from pathlib import Path


def _load_evaluation_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_reranker_quantization.py"
    spec = importlib.util.spec_from_file_location("reranker_evaluation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quantized_quality_limit_accepts_recorded_drop():
    module = _load_evaluation_module()

    assert module.TOP1_AGREEMENT_MIN == 0.95
    assert module.MAX_ABSOLUTE_METRIC_DROP == 0.0121
    assert 0.012070209993973569 <= module.MAX_ABSOLUTE_METRIC_DROP
