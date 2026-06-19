"""TriviaQA official-style answer normalization and EM/F1 scoring."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


def normalize_answer(text: str) -> str:
    """Official TriviaQA normalization (lowercase, articles, punctuation)."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def handle_punc(s: str) -> str:
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in s)

    def lower(s: str) -> str:
        return s.lower()

    def replace_underscore(s: str) -> str:
        return s.replace("_", " ")

    return white_space_fix(
        remove_articles(handle_punc(lower(replace_underscore(text))))
    ).strip()


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def metric_max_over_ground_truths(
    metric_fn, prediction: str, ground_truths: list[str]
) -> float:
    if not ground_truths:
        return 0.0
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def get_ground_truths(answer_obj: dict[str, Any]) -> list[str]:
    """Collect aliases from a HuggingFace TriviaQA answer dict."""
    truths: list[str] = []
    for key in (
        "normalized_aliases",
        "aliases",
        "normalized_value",
        "value",
        "matched_wiki_entity_name",
        "normalized_matched_wiki_entity_name",
    ):
        val = answer_obj.get(key)
        if isinstance(val, list):
            truths.extend(v for v in val if isinstance(v, str) and v.strip())
        elif isinstance(val, str) and val.strip():
            truths.append(val)

    seen: set[str] = set()
    unique: list[str] = []
    for truth in truths:
        norm = normalize_answer(truth)
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(truth)
    return unique


def _clean_answer_span(text: str) -> str:
    text = text.strip().strip("\"'`“”‘’")
    for stop in ("\n", "?", "Question:", "Note:", "Explanation:"):
        if stop in text:
            text = text.split(stop)[0]
    text = text.strip().rstrip(".,;:!-")
    if len(text) > 120:
        match = re.match(r"^[^.!?]+[.!?]?", text)
        if match:
            text = match.group(0)
    return text.strip()


def extract_final_answer(raw: str) -> str:
    """Extract a single short answer span from free-form model output."""
    text = (raw or "").strip()
    if not text:
        return ""

    for pat in (
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:|:)\s*(.+?)(?:\n|$)",
        r"\\boxed\s*\{(.+?)\}",
    ):
        matches = list(re.finditer(pat, text, re.IGNORECASE | re.DOTALL))
        if matches:
            return _clean_answer_span(matches[-1].group(1))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    if len(lines) >= 2 and len(lines[-1]) <= 80 and len(lines[0]) > len(lines[-1]):
        return _clean_answer_span(lines[-1])

    return _clean_answer_span(lines[0])


def substring_match_label(prediction: str, ground_truths: list[str]) -> bool:
    """Legacy substring heuristic kept only for audit comparison."""
    pred = prediction.strip().lower()
    for truth in ground_truths:
        alias = truth.lower()
        if alias in pred or pred in alias:
            return True
    return False


def evaluate_prediction(
    raw_answer: str,
    answer_obj: dict[str, Any],
) -> dict[str, Any]:
    ground_truths = get_ground_truths(answer_obj)
    extracted = extract_final_answer(raw_answer)
    em = metric_max_over_ground_truths(exact_match_score, extracted, ground_truths)
    f1 = metric_max_over_ground_truths(f1_score, extracted, ground_truths)

    best_alias = ""
    best_f1 = -1.0
    for truth in ground_truths:
        score = f1_score(extracted, truth)
        if score > best_f1:
            best_f1 = score
            best_alias = truth

    return {
        "raw_answer": raw_answer,
        "extracted_answer": extracted,
        "ground_truths": ground_truths,
        "best_matching_alias": best_alias,
        "exact_match": bool(em),
        "f1": float(f1),
        "legacy_substring_match": substring_match_label(raw_answer, ground_truths),
    }
