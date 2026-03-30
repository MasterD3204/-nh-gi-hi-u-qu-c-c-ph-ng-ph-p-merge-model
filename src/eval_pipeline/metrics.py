from __future__ import annotations

import ast
import operator
import re
from typing import Any


BOXED_PATTERN = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"


def extract_last_boxed(text: str) -> str | None:
    matches = re.findall(BOXED_PATTERN, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_final_answer_reasoning(text: str) -> str:
    boxed_answer = extract_last_boxed(text)
    if boxed_answer:
        return boxed_answer

    patterns = [
        r"\*?\*?[Ff]inal\s+[Aa]nswer\*?\*?\s*[:\-]?\s*(.+?)(?:\n|$)",
        r"[Tt]he\s+answer\s+is\s*[:\-]?\s*(.+?)(?:\.|$|\n)",
        r"[Tt]herefore[,]?\s*(.+?)(?:\.|$|\n)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    think_end = text.rfind("</think>")
    if think_end != -1:
        after_think = text[think_end:]
        equals_match = re.search(r"=\s*(.+?)(?:\n|$)", after_think)
        if equals_match:
            return equals_match.group(1).strip()

    stripped = text.strip()
    if not stripped:
        return "[KHONG_EXTRACT_DUOC]"

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return lines[-1] if lines else "[KHONG_EXTRACT_DUOC]"


def normalize_reasoning_answer(answer: str) -> str:
    normalized = answer.strip().lower()
    normalized = re.sub(r"[\s,]+", " ", normalized)
    normalized = re.sub(r"\\text\{([^}]*)\}", r"\1", normalized)
    normalized = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", normalized)
    normalized = re.sub(r"[$\\{}]", "", normalized)
    return normalized.strip()


def _safe_eval_numeric(expression: str) -> float:
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return float(operators[type(node.op)](evaluate(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return float(operators[type(node.op)](evaluate(node.left), evaluate(node.right)))
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    parsed = ast.parse(expression.replace("^", "**"), mode="eval")
    return evaluate(parsed)


def compare_reasoning_answers(predicted: str, ground_truth: str) -> tuple[bool, str]:
    predicted_normalized = normalize_reasoning_answer(predicted)
    ground_truth_normalized = normalize_reasoning_answer(ground_truth)

    if predicted_normalized == ground_truth_normalized:
        return True, "EXACT_MATCH"

    try:
        predicted_value = _safe_eval_numeric(predicted_normalized)
        ground_truth_value = _safe_eval_numeric(ground_truth_normalized)
        if abs(predicted_value - ground_truth_value) < 1e-6:
            return True, "NUMERIC_MATCH"
    except Exception:
        pass

    fraction_pattern = r"(\d+)\s*/\s*(\d+)"
    predicted_fraction = re.search(fraction_pattern, predicted_normalized)
    ground_truth_fraction = re.search(fraction_pattern, ground_truth_normalized)
    if predicted_fraction and ground_truth_fraction:
        predicted_value = int(predicted_fraction.group(1)) / int(predicted_fraction.group(2))
        ground_truth_value = int(ground_truth_fraction.group(1)) / int(ground_truth_fraction.group(2))
        if abs(predicted_value - ground_truth_value) < 1e-6:
            return True, "FRACTION_MATCH"

    return False, "MISMATCH"


def normalize_choice_answer(answer: str) -> str:
    match = re.search(r"[a-e]", answer.lower())
    return match.group(0) if match else answer.strip().lower()


def extract_mathqa_choice(text: str) -> str:
    lowered = text.lower()
    marker = "the correct answer is:"
    if marker in lowered:
        marker_index = lowered.index(marker) + len(marker)
        answer = lowered[marker_index:].strip()
        if answer:
            return normalize_choice_answer(answer[0])

    patterns = [
        r"the correct answer is\s*[:\-]?\s*([a-e])",
        r"answer is\s*[:\-]?\s*([a-e])",
        r"answer\s*[:\-]?\s*([a-e])",
        r"\boption\s*([a-e])\b",
        r"\(([a-e])\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return normalize_choice_answer(match.group(1))

    return "unknown"


def compare_choice_answers(predicted: str, ground_truth: str) -> tuple[bool, str]:
    predicted_normalized = normalize_choice_answer(predicted)
    ground_truth_normalized = normalize_choice_answer(ground_truth)
    is_correct = predicted_normalized == ground_truth_normalized
    return is_correct, "EXACT_MATCH" if is_correct else "MISMATCH"


def normalize_numeric_text(text: str) -> str:
    normalized = text.strip().lower()
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("=", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def extract_gsm8k_ground_truth(answer_text: str) -> str:
    if "####" in answer_text:
        return normalize_numeric_text(answer_text.split("####")[-1])
    return normalize_numeric_text(extract_final_answer_reasoning(answer_text))


def extract_gsm8k_prediction(text: str) -> str:
    boxed_answer = extract_last_boxed(text)
    if boxed_answer:
        return normalize_numeric_text(boxed_answer)

    patterns = [
        r"\*?\*?[Ff]inal\s+[Aa]nswer\*?\*?\s*[:\-]?\s*(.+?)(?:\n|$)",
        r"[Tt]he\s+answer\s+is\s*[:\-]?\s*(.+?)(?:\.|$|\n)",
        r"[Tt]herefore[,]?\s*(.+?)(?:\.|$|\n)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return normalize_numeric_text(match.group(1))

    numeric_candidates = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if numeric_candidates:
        return normalize_numeric_text(numeric_candidates[-1])

    return normalize_numeric_text(text)


def compare_numeric_or_text(predicted: str, ground_truth: str) -> tuple[bool, str]:
    predicted_normalized = normalize_numeric_text(predicted)
    ground_truth_normalized = normalize_numeric_text(ground_truth)
    if predicted_normalized == ground_truth_normalized:
        return True, "EXACT_MATCH"

    try:
        predicted_value = float(predicted_normalized)
        ground_truth_value = float(ground_truth_normalized)
        if abs(predicted_value - ground_truth_value) < 1e-6:
            return True, "NUMERIC_MATCH"
    except Exception:
        pass

    return False, "MISMATCH"


def summarize_accuracy(sample_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(sample_results)
    correct = sum(1 for result in sample_results if result["metric"]["is_correct"])
    accuracy = correct / total if total else 0.0
    return {
        "metric_name": "accuracy",
        "num_samples": total,
        "num_correct": correct,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
    }

