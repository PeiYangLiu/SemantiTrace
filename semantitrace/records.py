from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def resolve_record_path(path: str | Path, roots: list[str | Path]) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    resolved_roots = [resolve_repo_path(root) for root in roots]
    deduped_roots: list[Path] = []
    for root in resolved_roots:
        if root not in deduped_roots:
            deduped_roots.append(root)
    for root in deduped_roots:
        candidate = root / p
        if candidate.exists():
            return candidate
    return deduped_roots[0] / p if deduped_roots else resolve_repo_path(p)


def default_record_roots(records_path: str | Path, record_root: str | Path | None = None) -> list[Path]:
    resolved_records_path = resolve_repo_path(records_path)
    roots: list[Path] = []
    if record_root is not None:
        roots.append(resolve_repo_path(record_root))
    roots.extend([resolved_records_path.parent, resolved_records_path.parent.parent, ROOT])
    deduped_roots: list[Path] = []
    for root in roots:
        if root not in deduped_roots:
            deduped_roots.append(root)
    return deduped_roots


def infer_record_mode(record: dict[str, Any]) -> str:
    record_id = str(record.get("id", "")).lower()
    parasitism_mode = str(record.get("parasitism_mode", "")).lower()
    selected_mode = str((record.get("selected_canvas") or {}).get("mode", "")).lower()
    if record_id.startswith("modea-") or "text mutation" in parasitism_mode:
        return "mode_a"
    if (
        record_id.startswith("modeb-")
        or record_id.startswith("nontextmodeb-")
        or "non-text" in parasitism_mode
        or "natural object" in parasitism_mode
        or "object insertion" in parasitism_mode
        or "non_text" in selected_mode
    ):
        return "mode_b"
    return "unknown"


def load_records_with_resolved_paths(
    records_path: str | Path,
    max_records: int | None = None,
    *,
    record_root: str | Path | None = None,
    require_images: bool = True,
) -> list[dict[str, Any]]:
    resolved_records_path = resolve_repo_path(records_path)
    rows = json.loads(resolved_records_path.read_text(encoding="utf-8"))
    roots = default_record_roots(resolved_records_path, record_root)
    usable: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        for key in ("watermarked_image_path", "anchor_image_path", "comparison_image_path"):
            if key in record and record[key]:
                resolved = resolve_record_path(record[key], roots)
                record[f"_resolved_{key}"] = str(resolved)
        if require_images:
            wm = record.get("_resolved_watermarked_image_path")
            if not wm or not Path(wm).is_file():
                continue
        record["_record_mode"] = infer_record_mode(record)
        usable.append(record)
        if max_records is not None and len(usable) >= max_records:
            break
    if require_images and not usable:
        root_msg = ", ".join(str(root) for root in roots)
        raise FileNotFoundError(f"No usable records in {records_path}; tried roots: {root_msg}")
    return usable
