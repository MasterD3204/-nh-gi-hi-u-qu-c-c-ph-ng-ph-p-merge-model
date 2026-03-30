from .registry import build_default_registry
from .specs import EvaluationRequest, GenerationSettings


def evaluate_experiment(*args, **kwargs):
    from .evaluator import evaluate_experiment as _evaluate_experiment

    return _evaluate_experiment(*args, **kwargs)


__all__ = [
    "EvaluationRequest",
    "GenerationSettings",
    "build_default_registry",
    "evaluate_experiment",
]
