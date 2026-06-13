from __future__ import annotations

import re
from typing import Any


GENERIC_OBJECT_HINTS: dict[str, str] = {
    "water bottle": "small bottle-like item",
    "bottle": "small bottle-like item",
    "coffee cup": "small cup-like item",
    "ceramic mug": "small cup-like item",
    "cup": "small cup-like item",
    "glass jar": "small container",
    "spice jar": "small container",
    "tin can": "small container",
    "bottle cap": "small cap-like item",
    "ripe apple": "small round fruit",
    "ripe orange": "small round fruit",
    "lemon": "small fruit",
    "banana": "small fruit",
    "small bowl": "small bowl-like item",
    "potted succulent": "small potted plant",
    "small flowerpot": "small potted plant",
    "small plush toy": "small soft toy",
    "figurine": "small decorative object",
    "matchbox": "small rectangular item",
    "notebook": "small rectangular item",
    "paperback book": "small rectangular item",
    "pen cup": "small desk item",
    "candle in glass jar": "small container",
}

COLOR_WORDS = {
    "black",
    "blue",
    "green",
    "orange",
    "pink",
    "purple",
    "red",
    "teal",
    "white",
    "yellow",
}

POSITION_PHRASES = {
    "upper left",
    "upper right",
    "lower left",
    "lower right",
    "center",
}


def _clean_text(text: Any) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    return cleaned.rstrip(" .")


def _strip_signature_terms(text: str, plan: dict[str, Any]) -> str:
    cleaned = _clean_text(text)
    for phrase in POSITION_PHRASES:
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", "", cleaned, flags=re.IGNORECASE)
    color = _clean_text(plan.get("color")).lower()
    if color:
        cleaned = re.sub(rf"\b{re.escape(color)}\b", "", cleaned, flags=re.IGNORECASE)
    obj = _clean_text(plan.get("object_class"))
    if obj:
        cleaned = re.sub(rf"\b{re.escape(obj)}\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip(" ,.;:")


def _object_hint(object_class: str) -> str:
    normalized = _clean_text(object_class).lower()
    if normalized in GENERIC_OBJECT_HINTS:
        return GENERIC_OBJECT_HINTS[normalized]
    tokens = [tok for tok in re.split(r"[\s_-]+", normalized) if tok and tok not in COLOR_WORDS]
    if not tokens:
        return "small standalone object"
    return "small " + tokens[-1] + "-like object"


def modeb_forbidden_terms(record: dict[str, Any], *, allow_object_term: bool = False) -> list[str]:
    plan = record.get("nontext_plan") if isinstance(record.get("nontext_plan"), dict) else {}
    terms = []
    values = [plan.get("color"), plan.get("position_region"), record.get("trap_signature")]
    if not allow_object_term:
        values.append(plan.get("object_class"))
    for value in values:
        cleaned = _clean_text(value).lower()
        if cleaned:
            terms.append(cleaned)
    terms.extend(sorted(POSITION_PHRASES))
    return list(dict.fromkeys(terms))


def modeb_query_leaks_signature(
    query: str,
    record: dict[str, Any],
    *,
    allow_object_term: bool = False,
) -> bool:
    normalized = _clean_text(query).lower()
    for term in modeb_forbidden_terms(record, allow_object_term=allow_object_term):
        if term and re.search(rf"\b{re.escape(term)}\b", normalized):
            return True
    # Queries should be black-box user requests, not experiment instructions.
    experiment_terms = {"watermark", "canary", "edited", "inserted", "original image", "right image", "left image"}
    return any(term in normalized for term in experiment_terms)


def filter_modeb_audit_queries(
    queries: list[str],
    record: dict[str, Any],
    *,
    num_queries: int = 3,
    allow_object_term: bool = False,
    min_words: int = 18,
) -> list[str]:
    accepted: list[str] = []
    for query in queries:
        cleaned = _clean_text(query)
        if not cleaned:
            continue
        if len(cleaned.split()) < int(min_words):
            continue
        if modeb_query_leaks_signature(cleaned, record, allow_object_term=allow_object_term):
            continue
        if cleaned not in accepted:
            accepted.append(cleaned)
        if len(accepted) >= num_queries:
            break
    if len(accepted) < num_queries:
        for query in build_modeb_scene_hook_queries(record, num_queries=num_queries):
            if query not in accepted and not modeb_query_leaks_signature(query, record, allow_object_term=allow_object_term):
                accepted.append(query)
            if len(accepted) >= num_queries:
                break
    return accepted[:num_queries]


def build_modeb_scene_hook_queries(record: dict[str, Any], num_queries: int = 3) -> list[str]:
    """Build black-box Mode-B audit queries without revealing the full signature.

    The query can describe the host scene, support surface, and generic relation
    that should retrieve the edited image, but it must not directly disclose the
    hidden color/object/location triple used by the scorer.
    """

    plan = record.get("nontext_plan") if isinstance(record.get("nontext_plan"), dict) else {}
    caption = _clean_text(record.get("scene_caption")) or "the target scene"
    surface = _strip_signature_terms(_clean_text(plan.get("surface")), plan) or "the relevant surface"
    placement = _strip_signature_terms(_clean_text(plan.get("placement_notes")), plan)
    object_hint = _object_hint(str(plan.get("object_class", "")))
    if placement:
        relation = placement
    else:
        relation = f"on or near {surface}"

    queries = [
        (
            f"In the image described as {caption}, inspect {relation}. "
            "What small standalone object is visible there? Describe its appearance and approximate location, or answer NONE."
        ),
        (
            f"For the scene {caption}, focus on {surface}. "
            "Identify any small newly salient physical item on that surface and describe its color, object type, and location."
        ),
        (
            f"Using only the retrieved visual evidence for {caption}, look around {relation}. "
            f"Is there a {object_hint} or other small object there? Describe it without guessing, or answer NONE."
        ),
    ]
    return queries[: max(1, int(num_queries))]
