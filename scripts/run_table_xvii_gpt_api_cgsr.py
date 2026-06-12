#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantitrace.mode_verification import score_response
from semantitrace.records import infer_record_mode, resolve_repo_path


DEFAULT_VARIANTS = "without_ret,without_sig,without_blending,full_semantitrace"
DEFAULT_BLOB_ROOT = (
    "outputs/table_xvii_ablation_n250_blob/"
    "semantitrace_table_xvii_ablation_n250_20260609"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run GPT API CGSR for Table XVII ablation outputs")
    parser.add_argument("--records", default="n_scaling_subsets_20260609/combined_a125_b125_n250.json")
    parser.add_argument("--output_dir", default="outputs/table_xvii_ablation_n250")
    parser.add_argument("--blob_root", default=DEFAULT_BLOB_ROOT)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--model", default="gpt-5.1_2025-11-13")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max_output_tokens", type=int, default=96)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--top_k", type=int, default=3)
    return parser.parse_args()


def bbox_for_record(record: dict[str, Any]) -> list[int]:
    for container in (
        record.get("injection_metrics"),
        record.get("selected_canvas"),
        record.get("nontext_plan"),
        record.get("anchor"),
    ):
        if isinstance(container, dict):
            bbox = container.get("effective_mask_bbox") or container.get("bbox")
            if bbox and len(bbox) == 4:
                return [int(v) for v in bbox]
    raise ValueError(f"Record {record.get('id')} lacks a valid bbox")


def expanded_crop(image: Image.Image, bbox: list[int], pad_ratio: float = 0.35) -> Image.Image:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad = int(max(width, height) * pad_ratio)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.width, x2 + pad)
    y2 = min(image.height, y2 + pad)
    return image.crop((x1, y1, x2, y2))


def direct_prompt(record: dict[str, Any]) -> str:
    mode = infer_record_mode(record)
    if mode == "mode_b":
        return (
            "Inspect this local crop from an image. If a small inserted or salient object is visible, "
            "describe its color, object type, and approximate location in a concise phrase. "
            "If no such object is visible, answer NONE."
        )
    return (
        "Inspect this local crop from an image. Transcribe the exact short text, label, or acronym "
        "that is most prominent in the crop. Return only the visible text; if no text is readable, answer NONE."
    )


def image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                parts.append(str(value))
            elif isinstance(content, dict) and content.get("text"):
                parts.append(str(content["text"]))
    return "\n".join(parts)


_thread_local = threading.local()


def get_client():
    client = getattr(_thread_local, "client", None)
    if client is not None:
        return client
    from openai import AzureOpenAI
    from azure.identity import (
        AzureCliCredential,
        ChainedTokenCredential,
        ManagedIdentityCredential,
        get_bearer_token_provider,
    )

    scope = os.environ.get("TRAPI_SCOPE", "api://trapi/.default")
    apipath = os.environ.get("TRAPI_APIPATH", "gcr/shared")
    api_version = os.environ.get("TRAPI_API_VERSION", "2025-04-01-preview")
    endpoint = os.environ.get("TRAPI_ENDPOINT", f"https://trapi.research.microsoft.com/{apipath}")
    credential = get_bearer_token_provider(
        ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential()),
        scope,
    )
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=credential,
        api_version=api_version,
    )
    _thread_local.client = client
    return client


def resolve_wm_path(record: dict[str, Any], variant: str, output_dir: Path, blob_root: Path) -> Path:
    value = Path(str(record["watermarked_image_path"]))
    candidates = [
        output_dir / variant / "watermarked" / value.name,
        blob_root / variant / "watermarked" / value.name,
        value,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(str(value))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def call_gpt(record_index: int, source: dict[str, Any], wm_path: Path, model: str, max_output_tokens: int) -> dict[str, Any]:
    image = Image.open(wm_path).convert("RGB")
    crop = expanded_crop(image, bbox_for_record(source))
    response = get_client().responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": direct_prompt(source)},
                    {"type": "input_image", "image_url": image_to_data_url(crop)},
                ],
            }
        ],
        max_output_tokens=max_output_tokens,
    )
    text = response_text(response)
    scored = score_response(text, source)
    return {
        "record_index": record_index,
        "source_semantitrace_id": source.get("id"),
        "model": model,
        "response": text,
        "hit": bool(scored.get("hit", False)),
        "strict_hit": bool(scored.get("strict_hit", scored.get("hit", False))),
        "score": scored,
    }


def call_with_retries(
    record_index: int,
    source: dict[str, Any],
    wm_path: Path,
    model: str,
    max_output_tokens: int,
    max_retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    last_error = ""
    for attempt in range(max_retries):
        try:
            return call_gpt(record_index, source, wm_path, model, max_output_tokens)
        except Exception as exc:  # noqa: BLE001 - details are persisted for audit/debugging.
            last_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            time.sleep(retry_sleep * (2**attempt))
    return {
        "record_index": record_index,
        "source_semantitrace_id": source.get("id"),
        "model": model,
        "response": "",
        "hit": False,
        "strict_hit": False,
        "score": {"error": last_error},
        "error": last_error,
    }


def combine_direct_with_retrieval(retrieval_details: list[dict[str, Any]], direct_details: list[dict[str, Any]], top_k: int) -> float:
    direct_hits = {int(detail["record_index"]): bool(detail.get("hit", False)) for detail in direct_details}
    effective = [
        bool(direct_hits.get(int(detail["record_index"]), False)) and int(detail["target_rank"]) <= top_k
        for detail in retrieval_details
    ]
    return float(sum(effective) / len(effective)) if effective else 0.0


def evaluate_variant(
    variant: str,
    source_records: list[dict[str, Any]],
    output_dir: Path,
    blob_root: Path,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    variant_dir = output_dir / variant
    records = json.loads((variant_dir / "canary_records.json").read_text(encoding="utf-8"))
    details_path = variant_dir / "gpt51_direct_details.jsonl"
    existing = load_jsonl(details_path)
    by_index = {int(row["record_index"]): row for row in existing if "error" not in row}
    lock = threading.Lock()
    total = len(records)
    missing = [idx for idx in range(total) if idx not in by_index]
    print(f"[{variant}] existing={len(by_index)} missing={len(missing)} workers={args.workers}", flush=True)

    def task(idx: int) -> dict[str, Any]:
        return call_with_retries(
            idx,
            source_records[idx],
            resolve_wm_path(records[idx], variant, output_dir, blob_root),
            args.model,
            args.max_output_tokens,
            args.max_retries,
            args.retry_sleep,
        )

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task, idx): idx for idx in missing}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                by_index[int(result["record_index"])] = result
                completed += 1
                if completed % 10 == 0 or completed == len(missing):
                    write_jsonl(details_path, [by_index[idx] for idx in sorted(by_index)])
                    print(
                        f"[{variant}] completed {completed}/{len(missing)} "
                        f"hit={bool(result.get('hit'))} id={result.get('source_semantitrace_id')}",
                        flush=True,
                    )
    write_jsonl(details_path, [by_index[idx] for idx in sorted(by_index)])
    details = [by_index[idx] for idx in sorted(by_index)]
    retrieval = load_jsonl(variant_dir / "retrieval_rank_details.jsonl")
    return {
        "api_direct_cgsr": float(sum(bool(row.get("hit")) for row in details) / len(details)) if details else 0.0,
        "api_effective_cgsr_at_top3": combine_direct_with_retrieval(retrieval, details, args.top_k),
        "api_num_records": len(details),
        "api_model": args.model,
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    blob_root = resolve_repo_path(args.blob_root)
    source_records = json.loads(resolve_repo_path(args.records).read_text(encoding="utf-8"))
    variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]
    summary_path = output_dir / "table_xvii_ablation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    by_method = {row["method"]: row for row in summary}

    for variant in variants:
        metrics = evaluate_variant(variant, source_records, output_dir, blob_root, args)
        by_method[variant].update(metrics)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{variant}] {json.dumps(metrics, ensure_ascii=False)}", flush=True)

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
