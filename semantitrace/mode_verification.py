from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np

from semantitrace.metrics import contains_positive_signature, normalize_text
from semantitrace.records import infer_record_mode


COLOR_ALIASES: dict[str, list[str]] = {
    "black": ["black", "dark"],
    "blue": ["blue", "navy"],
    "green": ["green"],
    "orange": ["orange"],
    "pink": ["pink", "magenta", "fuchsia"],
    "red": ["red"],
    "teal": ["teal", "turquoise", "cyan"],
    "white": ["white", "pale"],
    "yellow": ["yellow", "gold", "golden"],
}

POSITION_ALIASES: dict[str, list[str]] = {
    "upper left": ["upper left", "top left", "upper-left", "top-left"],
    "upper right": ["upper right", "top right", "upper-right", "top-right"],
    "lower left": ["lower left", "bottom left", "lower-left", "bottom-left"],
    "lower right": ["lower right", "bottom right", "lower-right", "bottom-right"],
    "center": ["center", "centre", "middle", "central"],
}

OBJECT_ALIASES: dict[str, list[str]] = {
    "bottle": ["bottle", "water bottle", "plastic bottle"],
    "bottle cap": ["bottle cap", "cap", "lid"],
    "bowl": ["bowl", "small bowl", "dish"],
    "coffee cup": ["coffee cup", "cup", "mug"],
    "cup": ["cup", "mug"],
    "orange": ["orange", "ripe orange", "fruit"],
    "potted succulent": ["potted succulent", "succulent", "potted plant", "plant"],
    "spice jar": ["spice jar", "jar", "container"],
    "water bottle": ["water bottle", "bottle"],
}

OBJECT_STOPWORDS = {
    "a",
    "an",
    "and",
    "ball",
    "clean",
    "dark",
    "fresh",
    "large",
    "little",
    "new",
    "object",
    "ripe",
    "single",
    "small",
    "the",
    "tiny",
}


def response_is_negated(response: str) -> bool:
    text = normalize_text(response)
    return bool(
        re.search(r"\bnone\b", text)
        or "not visible" in text
        or "no visible" in text
        or "no object" in text
        or "no such object" in text
        or "cannot see" in text
        or "cant see" in text
        or "do not see" in text
        or "not present" in text
        or "not shown" in text
    )


def _aliases_for_color(color: str) -> list[str]:
    key = normalize_text(color)
    return COLOR_ALIASES.get(key, [key])


def _aliases_for_position(position: str) -> list[str]:
    key = normalize_text(position)
    return POSITION_ALIASES.get(key, [key])


def _object_aliases(record: dict[str, Any]) -> list[str]:
    plan = record.get("nontext_plan") if isinstance(record.get("nontext_plan"), dict) else {}
    verification = record.get("nontext_verification") if isinstance(record.get("nontext_verification"), dict) else {}
    parsed = verification.get("parsed") if isinstance(verification.get("parsed"), dict) else {}
    candidates = [
        str(plan.get("object_class", "")),
        str(parsed.get("object", "")),
        str(record.get("trap_signature", "")),
    ]
    aliases: list[str] = []
    for candidate in candidates:
        norm = normalize_text(candidate)
        if not norm:
            continue
        aliases.append(norm)
        aliases.extend(OBJECT_ALIASES.get(norm, []))
        for token in norm.split():
            if len(token) >= 3 and token not in OBJECT_STOPWORDS:
                aliases.append(token)
            aliases.extend(OBJECT_ALIASES.get(token, []))
    deduped: list[str] = []
    for alias in aliases:
        norm_alias = normalize_text(alias)
        if norm_alias and norm_alias not in deduped:
            deduped.append(norm_alias)
    return deduped


def _modeb_target(record: dict[str, Any]) -> dict[str, str]:
    plan = record.get("nontext_plan") if isinstance(record.get("nontext_plan"), dict) else {}
    return {
        "position": str(plan.get("position_region") or (record.get("selected_canvas") or {}).get("position") or ""),
        "color": str(plan.get("color") or ""),
        "object": str(plan.get("object_class") or ""),
    }


def score_modeb_response(response: str, record: dict[str, Any]) -> dict[str, Any]:
    text = normalize_text(response)
    target = _modeb_target(record)
    negated = response_is_negated(response)
    color_aliases = _aliases_for_color(target["color"])
    position_aliases = _aliases_for_position(target["position"])
    object_aliases = _object_aliases(record)
    color_hit = any(normalize_text(alias) in text for alias in color_aliases if alias)
    position_hit = any(
        normalize_text(alias) in text or normalize_text(alias).replace(" ", "") in text.replace(" ", "")
        for alias in position_aliases
        if alias
    )
    object_hit = any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in object_aliases if alias)
    if negated:
        color_hit = object_hit = position_hit = False
    color_object_hit = bool(color_hit and object_hit)
    strict_hit = bool(color_object_hit and position_hit)
    return {
        "mode": "mode_b",
        "target": target,
        "negated": negated,
        "color_hit": bool(color_hit),
        "object_hit": bool(object_hit),
        "position_hit": bool(position_hit),
        "color_object_hit": color_object_hit,
        "strict_hit": strict_hit,
        "hit": color_object_hit,
    }


def score_response(response: str, record: dict[str, Any]) -> dict[str, Any]:
    mode = infer_record_mode(record)
    if mode == "mode_b":
        return score_modeb_response(response, record)
    hit = contains_positive_signature(response, str(record.get("trap_signature", "")))
    return {
        "mode": mode,
        "hit": bool(hit),
        "text_signature_hit": bool(hit),
        "strict_hit": bool(hit),
    }


def per_canary_rates(details: list[dict[str, Any]], key: str, total_records: int) -> np.ndarray:
    grouped: list[list[bool]] = [[] for _ in range(total_records)]
    for detail in details:
        record_index = int(detail["record_index"])
        if 0 <= record_index < total_records:
            grouped[record_index].append(bool(detail.get(key, False)))
    return np.asarray([sum(values) / len(values) for values in grouped if values], dtype=np.float64)


def per_canary_rates_from_predicate(
    details: list[dict[str, Any]],
    predicate: Any,
    total_records: int,
) -> np.ndarray:
    grouped: list[list[bool]] = [[] for _ in range(total_records)]
    for detail in details:
        record_index = int(detail["record_index"])
        if 0 <= record_index < total_records:
            grouped[record_index].append(bool(predicate(detail)))
    return np.asarray([sum(values) / len(values) for values in grouped if values], dtype=np.float64)


def target_rank_in_topk(rank: Any, top_k: int) -> bool:
    try:
        return int(rank) <= int(top_k)
    except (TypeError, ValueError):
        return False


def detail_response_hit(detail: dict[str, Any], prefix: str, *, strict: bool = False) -> bool:
    response_key = f"{prefix}_response_{'strict_' if strict else ''}hit"
    if response_key in detail:
        return bool(detail[response_key])
    legacy_key = f"{prefix}_{'strict_' if strict else ''}hit"
    return bool(detail.get(legacy_key, False))


def detail_target_gated_hit(
    detail: dict[str, Any],
    prefix: str,
    top_k: int,
    *,
    strict: bool = False,
) -> bool:
    rank_key = "target_rank" if prefix == "watermarked" else f"{prefix}_target_rank"
    return detail_response_hit(detail, prefix, strict=strict) and target_rank_in_topk(
        detail.get(rank_key),
        top_k,
    )


def summarize_by_mode(details: list[dict[str, Any]], records: list[dict[str, Any]], *, hit_key: str = "watermarked_hit") -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        groups["all"].append(idx)
        groups[infer_record_mode(record)].append(idx)
    rows: list[dict[str, Any]] = []
    for subset, record_indices in groups.items():
        index_set = set(record_indices)
        subset_details = [d for d in details if int(d.get("record_index", -1)) in index_set]
        ranks = np.asarray([int(d["target_rank"]) for d in subset_details if "target_rank" in d], dtype=float)
        hits = per_canary_rates(subset_details, hit_key, len(records))
        rows.append(
            {
                "subset": subset,
                "num_canaries": len(record_indices),
                "num_queries": len(subset_details),
                "hit_rate": float(hits.mean()) if hits.size else 0.0,
                "recall_at_1": float(np.mean(ranks <= 1)) if ranks.size else 0.0,
                "recall_at_3": float(np.mean(ranks <= 3)) if ranks.size else 0.0,
                "recall_at_10": float(np.mean(ranks <= 10)) if ranks.size else 0.0,
                "mean_target_rank": float(ranks.mean()) if ranks.size else 0.0,
            }
        )
    return rows
