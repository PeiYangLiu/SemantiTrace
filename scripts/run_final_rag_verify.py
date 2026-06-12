#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantitrace.backends.real import OpenCLIPEncoder, QwenVLMClient
from semantitrace.config import load_config
from semantitrace.metrics import contains_positive_signature
from semantitrace.rag import ImageRAGIndex
from semantitrace.verification import Verifier


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Combine final canaries and run clean-control RAG verification")
    parser.add_argument("--text_records", default="outputs/flux2_klein_opus130_v7/canary_records.json")
    parser.add_argument("--oi_records", default="outputs/flux2_klein_opus_oi7_freeedit_v2/canary_records.json")
    parser.add_argument("--extra_records", nargs="*", default=[], help="Additional canary_records.json files to append")
    parser.add_argument("--extra_limit_per_file", type=int, default=None, help="Limit records loaded from each extra file")
    parser.add_argument(
        "--extra_select_lowest_quality_delta",
        action="store_true",
        help="Sort each extra file by injection_metrics.quality_local_delta before applying --extra_limit_per_file",
    )
    parser.add_argument("--target_records", type=int, default=None, help="Optional final cap after combining")
    parser.add_argument("--output_dir", default="outputs/flux2_klein_opus130_v7_plus_oi_freeedit_v2_rag_verify")
    parser.add_argument("--config", default="configs/semantitrace_flux2_klein_opus_struct.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--combine_only", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_records(path: str | Path, source_run: str | None) -> list[dict[str, Any]]:
    records = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    for record in records:
        if source_run is not None:
            record.setdefault("source_run", source_run)
        record["source_record_id"] = record.get("id")
    return records


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def combine_records(args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    records = (
        load_records(args.text_records, "outputs/flux2_klein_opus130_v7")
        + load_records(args.oi_records, "outputs/flux2_klein_opus_oi7_freeedit_v2")
    )
    for extra_path in args.extra_records:
        extra = load_records(extra_path, str(resolve_path(extra_path).parent.relative_to(ROOT)))
        if args.extra_select_lowest_quality_delta:
            extra.sort(key=lambda r: float((r.get("injection_metrics") or {}).get("quality_local_delta", 999.0)))
        if args.extra_limit_per_file is not None:
            extra = extra[: args.extra_limit_per_file]
        records.extend(extra)
    if args.target_records is not None:
        records = records[: args.target_records]

    seen: set[str] = set()
    duplicates: list[str] = []
    combined: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        signature = str(record["trap_signature"])
        if signature in seen:
            duplicates.append(signature)
        seen.add(signature)

        new_id = f"canary-{idx:04d}"
        wm_src = resolve_path(record["watermarked_image_path"])
        clean_src = resolve_path(record["anchor_image_path"])
        wm_dst = out_dir / "watermarked" / f"{new_id}_{wm_src.name}"
        clean_dst = out_dir / "clean" / f"{new_id}_{clean_src.name}"
        copy_image(wm_src, wm_dst)
        copy_image(clean_src, clean_dst)

        rec = json.loads(json.dumps(record))
        rec["id"] = new_id
        rec["watermarked_image_path"] = str(wm_dst.relative_to(ROOT))
        rec["anchor_image_path"] = str(clean_dst.relative_to(ROOT))
        if isinstance(rec.get("anchor"), dict):
            rec["anchor"]["image_path"] = str(clean_dst.relative_to(ROOT))
        combined.append(rec)

    if duplicates:
        raise ValueError(f"Duplicate trap signatures would bias CER: {sorted(set(duplicates))}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "canary_records.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {
        "num_records": len(combined),
        "num_unique_signatures": len({r["trap_signature"] for r in combined}),
        "source_records": [args.text_records, args.oi_records],
        "extra_records": args.extra_records,
        "extra_limit_per_file": args.extra_limit_per_file,
        "target_records": args.target_records,
    }
    (out_dir / "combine_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return combined


def rag_prompt(query: str) -> str:
    return (
        "Answer the user using the retrieved visual evidence. "
        "If text is visible, transcribe it exactly.\n\n"
        f"User query: {query}"
    )


def build_clients(config_path: str, device: str) -> tuple[OpenCLIPEncoder, QwenVLMClient, Verifier]:
    cfg = load_config(config_path)
    models = cfg.get("models", {})
    vlm_cfg = cfg.get("vlm", {})
    encoder = OpenCLIPEncoder(
        model_name=models.get("clip_model", "ViT-L-14"),
        pretrained=models.get("clip_pretrained", "openai"),
        device=device,
    )
    vlm = QwenVLMClient(
        model_name=models.get("surrogate_vlm", "Qwen/Qwen3-VL-8B-Instruct"),
        device=device,
        torch_dtype=vlm_cfg.get("torch_dtype", "bfloat16"),
    )
    return encoder, vlm, Verifier(cfg.get("verification", {}))


def load_existing_details(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    details: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                details.append(json.loads(line))
    return details


def run_verify(args: argparse.Namespace, out_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    encoder, vlm, verifier = build_clients(args.config, args.device)
    image_paths = [resolve_path(r["watermarked_image_path"]) for r in records]
    image_ids = [str(r["id"]) for r in records]
    index = ImageRAGIndex(encoder).build(image_paths, image_ids)

    details_path = out_dir / "rag_verify_details.jsonl"
    details = load_existing_details(details_path)
    skip = len(details)
    done = 0

    with details_path.open("a", encoding="utf-8") as out:
        for record_index, record in enumerate(records):
            signature = str(record["trap_signature"])
            original_text = str((record.get("selected_canvas") or {}).get("text") or "")
            queries = list(record.get("probe_queries", []))[: verifier.num_probes_per_canary]
            for probe_index, query in enumerate(queries):
                flat_index = record_index * verifier.num_probes_per_canary + probe_index
                if flat_index < skip:
                    continue

                hits = index.search(str(query), args.top_k)
                wm_image = Image.open(hits[0].image_path).convert("RGB") if hits else None
                clean_image = Image.open(resolve_path(record["anchor_image_path"])).convert("RGB")
                prompt = rag_prompt(str(query))
                watermarked_response = vlm.generate(
                    wm_image,
                    prompt,
                    temperature=0.0,
                    max_new_tokens=args.max_new_tokens,
                )
                clean_response = vlm.generate(
                    clean_image,
                    prompt,
                    temperature=0.0,
                    max_new_tokens=args.max_new_tokens,
                )
                detail = {
                    "record_index": record_index,
                    "probe_index": probe_index,
                    "id": record["id"],
                    "signature": signature,
                    "original_text": original_text,
                    "source_run": record.get("source_run"),
                    "query": str(query),
                    "watermarked_response": watermarked_response,
                    "clean_response": clean_response,
                    "watermarked_hit": contains_positive_signature(watermarked_response, signature),
                    "clean_hit": contains_positive_signature(clean_response, signature),
                    "watermarked_hits_retrieval": [hit.__dict__ for hit in hits],
                    "clean_hits_retrieval": [
                        {
                            "image_id": f"{record['id']}::clean",
                            "image_path": record["anchor_image_path"],
                            "score": 1.0,
                            "rank": 1,
                        }
                    ],
                }
                out.write(json.dumps(detail, ensure_ascii=False) + "\n")
                out.flush()
                details.append(detail)
                done += 1
                print(
                    f"[{len(details):03d}/{len(records) * verifier.num_probes_per_canary:03d}] "
                    f"{record['id']} probe={probe_index} sig={signature} "
                    f"wm_hit={detail['watermarked_hit']} clean_hit={detail['clean_hit']}",
                    flush=True,
                )

    signatures = [str(r["trap_signature"]) for r in records]
    suspect_responses = [str(d["watermarked_response"]) for d in details]
    clean_responses = [str(d["clean_response"]) for d in details]
    suspect_samples = verifier.compute_per_canary_cer(suspect_responses, signatures)
    clean_samples = verifier.compute_per_canary_cer(clean_responses, signatures)
    test = verifier.welch_t_test(suspect_samples, clean_samples)
    report = {
        "num_canaries": len(records),
        "num_probes_per_canary": verifier.num_probes_per_canary,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "suspect_cer": float(suspect_samples.mean()) if suspect_samples.size else 0.0,
        "suspect_per_canary_cer": suspect_samples.tolist(),
        "clean_cer": float(clean_samples.mean()) if clean_samples.size else 0.0,
        "clean_per_canary_cer": clean_samples.tolist(),
        "test_result": test,
        "details": details,
        "new_details_this_run": done,
    }
    (out_dir / "rag_verify_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.output_dir)
    records = combine_records(args, out_dir)
    print(f"Combined {len(records)} records -> {out_dir / 'canary_records.json'}", flush=True)
    if args.combine_only:
        return
    report = run_verify(args, out_dir, records)
    print(json.dumps(report["test_result"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
