from __future__ import annotations

import json
import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from semantitrace import metrics
from semantitrace.backends.deterministic import DeterministicEncoder
from semantitrace.backends.real import HFSigLIPEncoder, OpenCLIPEncoder
from semantitrace.baselines import AQUABaseline, PGDBaseline
from semantitrace.defenses import MahalanobisOODDetector
from semantitrace.experiments.datasets import ImageCorpus, create_synthetic_corpus, load_corpus
from semantitrace.experiments.generators import AnswerGenerator, build_answer_generator
from semantitrace.experiments.tables import write_csv, write_json
from semantitrace.experiments.transforms import apply_transform
from semantitrace.rag import ImageRAGIndex
from semantitrace.utils.image import list_images
from semantitrace.verification import Verifier
from semantitrace.pipeline import SemantiTracePipeline

logger = logging.getLogger(__name__)


@dataclass
class MethodArtifacts:
    name: str
    records: list[dict[str, Any]]
    corpus_paths: list[str]
    corpus_ids: list[str]


class MainExperimentRunner:
    def __init__(
        self,
        config_path: str | os.PathLike[str] = "configs/main_experiment.yaml",
        output_dir: str | os.PathLike[str] | None = None,
        device: str | None = None,
        dry_run_sample: bool = False,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if output_dir:
            self.config.setdefault("run", {})["output_dir"] = str(output_dir)
        if device:
            self.config.setdefault("run", {})["device"] = device
        self.output_dir = Path(self.config.get("run", {}).get("output_dir", "outputs/main_experiment"))
        self.device = self.config.get("run", {}).get("device", "cuda")
        self.seed = int(self.config.get("run", {}).get("seed", 42))
        self.dry_run_sample = dry_run_sample
        self.rng = random.Random(self.seed)

    def run(self, stages: list[str] | None = None) -> dict[str, Any]:
        stages = stages or ["efficacy", "stealth", "ood", "robustness"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        corpora = self._load_corpora()
        all_rows: dict[str, list[dict[str, Any]]] = {
            "efficacy": [],
            "stealth": [],
            "ood": [],
            "robustness": [],
            "skipped": [],
        }

        for corpus in corpora:
            logger.info("Running dataset %s with %d images", corpus.name, len(corpus.records))
            method_artifacts = self._materialize_methods(corpus)

            if "stealth" in stages:
                for artifacts in method_artifacts:
                    all_rows["stealth"].append(self._stealth_row(corpus, artifacts))

            if "ood" in stages:
                for retriever_spec in self.config.get("retrievers", []):
                    encoder = self._safe_build_encoder(retriever_spec, all_rows["skipped"])
                    if encoder is None:
                        continue
                    for artifacts in method_artifacts:
                        all_rows["ood"].append(self._ood_row(corpus, artifacts, retriever_spec["name"], encoder))

            if "efficacy" in stages:
                for retriever_spec in self.config.get("retrievers", []):
                    encoder = self._safe_build_encoder(retriever_spec, all_rows["skipped"])
                    if encoder is None:
                        continue
                    for generator_spec in self.config.get("generators", []):
                        generator = self._safe_build_generator(generator_spec, all_rows["skipped"])
                        if generator is None:
                            continue
                        for artifacts in method_artifacts:
                            all_rows["efficacy"].append(
                                self._evaluate_efficacy(corpus, artifacts, retriever_spec["name"], encoder, generator)
                            )

            if "robustness" in stages and self.config.get("robustness", {}).get("enabled", True):
                base_retriever = self.config.get("retrievers", [{}])[0]
                encoder = self._safe_build_encoder(base_retriever, all_rows["skipped"])
                base_generator = self._first_available_generator(all_rows["skipped"])
                if encoder is not None and base_generator is not None:
                    for artifacts in method_artifacts:
                        if artifacts.name != "semantitrace":
                            continue
                        all_rows["robustness"].extend(
                            self._evaluate_robustness(corpus, artifacts, base_retriever["name"], encoder, base_generator)
                        )

        self._write_outputs(all_rows)
        return all_rows

    def _load_corpora(self) -> list[ImageCorpus]:
        if self.dry_run_sample:
            corpus = create_synthetic_corpus(self.output_dir / "_dryrun_images", count=12, seed=self.seed)
            return [corpus]
        corpora = []
        for name, spec in self.config.get("datasets", {}).items():
            corpora.append(
                load_corpus(
                    name=name,
                    image_dir=spec.get("image_dir"),
                    manifest=spec.get("manifest"),
                    max_images=spec.get("max_images"),
                )
            )
        return corpora

    def _materialize_methods(self, corpus: ImageCorpus) -> list[MethodArtifacts]:
        artifacts: list[MethodArtifacts] = []
        for method_spec in self.config.get("methods", []):
            method_name = method_spec["name"]
            method_type = method_spec["type"]
            method_dir = self.output_dir / corpus.name / method_name
            method_dir.mkdir(parents=True, exist_ok=True)
            if method_type == "semantitrace":
                artifacts.append(self._materialize_semantitrace(corpus, method_spec, method_dir))
            else:
                artifacts.append(self._materialize_baseline(corpus, method_spec, method_dir))
        return artifacts

    def _materialize_semantitrace(self, corpus: ImageCorpus, method_spec: dict[str, Any], method_dir: Path) -> MethodArtifacts:
        protocol = self.config["protocol"]
        subset_dir = self._stage_corpus(corpus, method_dir / "_corpus")
        pipeline = SemantiTracePipeline(method_spec.get("config", "configs/default.yaml"), device=self.device)
        records = pipeline.inject_canaries(
            subset_dir,
            method_dir,
            num_canaries=min(int(protocol["num_canaries"]), len(corpus.records)),
        )
        for record in records:
            record["method"] = "semantitrace"
            record["insertion_policy"] = "replace"
            record["index_image_path"] = record["watermarked_image_path"]
        paths, ids = self._build_index_corpus(corpus, records)
        write_json(method_dir / "records.json", records)
        return MethodArtifacts("semantitrace", records, paths, ids)

    def _materialize_baseline(self, corpus: ImageCorpus, method_spec: dict[str, Any], method_dir: Path) -> MethodArtifacts:
        protocol = self.config["protocol"]
        n = min(int(protocol["num_canaries"]), len(corpus.records))
        selected = self.rng.sample(corpus.records, n) if n < len(corpus.records) else list(corpus.records)
        records: list[dict[str, Any]] = []
        pgd = PGDBaseline(epsilon=float(method_spec.get("epsilon", 8 / 255)), seed=self.seed)
        aqua = AQUABaseline(seed=self.seed)

        for idx, item in enumerate(selected):
            image = Image.open(item.image_path).convert("RGB")
            signature = self._signature(idx)
            if method_spec["type"] == "naive":
                out_image = image
                prompt = f"unmodified anchor assigned signature {signature}"
                policy = "replace"
            elif method_spec["type"] == "pgd":
                baseline = pgd.apply(image, signature)
                out_image, prompt = baseline.image, baseline.trigger_prompt
                policy = "replace"
            elif method_spec["type"] == "aqua_acronym":
                baseline = aqua.acronym(signature)
                out_image, prompt = baseline.image, baseline.trigger_prompt
                policy = "add"
            elif method_spec["type"] == "aqua_spatial":
                baseline = aqua.spatial(signature)
                out_image, prompt = baseline.image, baseline.trigger_prompt
                policy = "add"
            else:
                raise ValueError(f"Unknown method type: {method_spec['type']}")

            out_path = method_dir / "watermarked" / f"{method_spec['name']}_{idx:04d}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_image.save(out_path)
            record = {
                "id": f"{method_spec['name']}-{idx:04d}",
                "method": method_spec["name"],
                "anchor_image_path": item.image_path,
                "watermarked_image_path": str(out_path),
                "index_image_path": str(out_path),
                "insertion_policy": policy,
                "trap_signature": signature,
                "trigger_prompt": prompt,
                "parasitism_mode": "Synthetic" if policy == "add" else "Perturbation",
                "probe_queries": self._probe_queries(signature),
            }
            records.append(record)
        paths, ids = self._build_index_corpus(corpus, records)
        write_json(method_dir / "records.json", records)
        return MethodArtifacts(method_spec["name"], records, paths, ids)

    def _build_index_corpus(self, corpus: ImageCorpus, records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        replacements = {
            str(Path(record["anchor_image_path"]).resolve()): record["index_image_path"]
            for record in records
            if record.get("insertion_policy") == "replace"
        }
        paths: list[str] = []
        ids: list[str] = []
        for item in corpus.records:
            key = str(Path(item.image_path).resolve())
            paths.append(replacements.get(key, item.image_path))
            ids.append(item.image_id)
        for record in records:
            if record.get("insertion_policy") == "add":
                paths.append(record["index_image_path"])
                ids.append(record["id"])
        return paths, ids

    def _evaluate_efficacy(
        self,
        corpus: ImageCorpus,
        artifacts: MethodArtifacts,
        retriever_name: str,
        encoder,
        generator: AnswerGenerator,
    ) -> dict[str, Any]:
        top_k = int(self.config["protocol"]["retrieval_top_k"])
        index = ImageRAGIndex(encoder).build(artifacts.corpus_paths, artifacts.corpus_ids)
        records_by_path = {record["index_image_path"]: record for record in artifacts.records}
        responses: list[str] = []
        clean_responses: list[str] = []
        ranks: list[int] = []
        raw_hits = 0
        clean_hits = 0
        adjusted_hits = 0
        total_queries = 0

        for record in artifacts.records:
            record_ranks: list[int] = []
            for query in record["probe_queries"][: int(self.config["protocol"]["num_probe_queries"])]:
                hits = index.search(query, top_k=top_k)
                hit_dicts = [hit.__dict__ for hit in hits]
                response = generator.answer(query, hit_dicts, records_by_path)
                responses.append(response)
                rank = next(
                    (hit.rank for hit in hits if Path(hit.image_path).resolve() == Path(record["index_image_path"]).resolve()),
                    top_k + 1,
                )
                record_ranks.append(rank)
                clean_response = generator.answer(query, self._clean_control_hit(record), {})
                clean_responses.append(clean_response)
                raw_hit = self._contains_signature(response, record["trap_signature"])
                clean_hit = self._contains_signature(clean_response, record["trap_signature"])
                raw_hits += int(raw_hit)
                clean_hits += int(clean_hit)
                adjusted_hits += int(raw_hit and not clean_hit)
                total_queries += 1
            ranks.append(int(min(record_ranks) if record_ranks else top_k + 1))

        verifier = Verifier(
            {
                "num_probes_per_canary": int(self.config["protocol"]["num_probe_queries"]),
                "significance_level": float(self.config["protocol"]["significance_level"]),
            }
        )
        signatures = [record["trap_signature"] for record in artifacts.records]
        samples = verifier.compute_per_canary_cer(responses, signatures)
        clean_samples = verifier.compute_per_canary_cer(clean_responses, signatures)
        default_clean_test = verifier.welch_t_test(samples)
        clean_control_test = verifier.welch_t_test(samples, clean_samples)
        adjusted_cgsr = adjusted_hits / total_queries if total_queries else 0.0
        return {
            "dataset": corpus.name,
            "method": artifacts.name,
            "retriever": retriever_name,
            "generator": generator.name,
            "num_canaries": len(artifacts.records),
            "rank": float(np.mean(ranks)) if ranks else None,
            "cgsr": adjusted_cgsr,
            "adjusted_cgsr": adjusted_cgsr,
            "raw_cgsr": raw_hits / total_queries if total_queries else 0.0,
            "clean_cgsr": clean_hits / total_queries if total_queries else 0.0,
            "raw_positive_queries": raw_hits,
            "clean_positive_queries": clean_hits,
            "adjusted_positive_queries": adjusted_hits,
            "total_queries": total_queries,
            "p_value": clean_control_test["p_value"],
            "reject_h0": clean_control_test["reject_h0"],
            "p_value_default_clean": default_clean_test["p_value"],
            "reject_h0_default_clean": default_clean_test["reject_h0"],
            "clean_control": "anchor_image",
        }

    @staticmethod
    def _contains_signature(response: str, signature: str) -> bool:
        return metrics.contains_positive_signature(response, signature)

    @staticmethod
    def _clean_control_hit(record: dict[str, Any]) -> list[dict[str, Any]]:
        hit = {
            "image_id": f"{record['id']}::clean",
            "image_path": record["anchor_image_path"],
            "score": 1.0,
            "rank": 1,
        }
        for key in ("selected_canvas", "injection_metrics"):
            if key in record:
                hit[key] = record[key]
        return [hit]

    def _stealth_row(self, corpus: ImageCorpus, artifacts: MethodArtifacts) -> dict[str, Any]:
        psnrs: list[float] = []
        original_images = []
        edited_images = []
        for record in artifacts.records:
            anchor = Image.open(record["anchor_image_path"]).convert("RGB")
            edited = Image.open(record["watermarked_image_path"]).convert("RGB").resize(anchor.size)
            original_images.append(anchor)
            edited_images.append(edited)
            if self.config.get("stealth", {}).get("compute_psnr", True):
                psnrs.append(metrics.compute_psnr(np.asarray(anchor), np.asarray(edited)))
        fid = None
        if self.config.get("stealth", {}).get("compute_fid", True) and original_images and edited_images:
            encoder = DeterministicEncoder()
            fid = metrics.compute_fid(encoder.encode_images(original_images), encoder.encode_images(edited_images))
        return {
            "dataset": corpus.name,
            "method": artifacts.name,
            "num_images": len(artifacts.records),
            "psnr": float(np.mean([p for p in psnrs if np.isfinite(p)])) if psnrs and any(np.isfinite(psnrs)) else "inf",
            "fid": fid,
        }

    def _ood_row(self, corpus: ImageCorpus, artifacts: MethodArtifacts, retriever_name: str, encoder) -> dict[str, Any]:
        clean_images = [Image.open(path).convert("RGB") for path in corpus.image_paths]
        suspect_images = [Image.open(record["watermarked_image_path"]).convert("RGB") for record in artifacts.records]
        ood_cfg = self.config.get("ood", {})
        detector = MahalanobisOODDetector(
            percentile=float(ood_cfg.get("mahalanobis_percentile", 99.0)),
            regularization=float(ood_cfg.get("mahalanobis_regularization", 1e-4)),
            max_components=int(ood_cfg.get("mahalanobis_max_components", 64)),
            variance_keep=float(ood_cfg.get("mahalanobis_variance_keep", 0.95)),
        ).fit(encoder.encode_images(clean_images))
        rejected = detector.reject_embeddings(encoder.encode_images(suspect_images)) if suspect_images else np.array([])
        return {
            "dataset": corpus.name,
            "method": artifacts.name,
            "retriever": retriever_name,
            "num_images": len(suspect_images),
            "ood_reject_rate": float(np.mean(rejected)) if rejected.size else 0.0,
        }

    def _evaluate_robustness(
        self,
        corpus: ImageCorpus,
        artifacts: MethodArtifacts,
        retriever_name: str,
        encoder,
        generator: AnswerGenerator,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        robust_dir = self.output_dir / corpus.name / artifacts.name / "robustness"
        for op in self.config.get("robustness", {}).get("operations", []):
            transformed_records = []
            for record in artifacts.records:
                image = Image.open(record["watermarked_image_path"]).convert("RGB")
                out = apply_transform(image, op)
                out_path = robust_dir / op["name"] / Path(record["watermarked_image_path"]).name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out.save(out_path)
                nr = dict(record)
                nr["index_image_path"] = str(out_path)
                nr["watermarked_image_path"] = str(out_path)
                transformed_records.append(nr)
            paths, ids = self._build_index_corpus(corpus, transformed_records)
            transformed = MethodArtifacts(artifacts.name, transformed_records, paths, ids)
            row = self._evaluate_efficacy(corpus, transformed, retriever_name, encoder, generator)
            row["operation"] = op["name"]
            rows.append(row)
        return rows

    def _safe_build_encoder(self, spec: dict[str, Any], skipped: list[dict[str, Any]]):
        try:
            return self._build_encoder(spec)
        except Exception as exc:
            skipped.append({"component": "retriever", "name": spec.get("name"), "reason": str(exc)})
            logger.warning("Skipping retriever %s: %s", spec.get("name"), exc)
            return None

    def _build_encoder(self, spec: dict[str, Any]):
        backend = spec.get("backend", "deterministic")
        if backend == "deterministic":
            return DeterministicEncoder()
        if backend == "open_clip":
            return OpenCLIPEncoder(
                model_name=spec.get("model_name", "ViT-L-14"),
                pretrained=spec.get("pretrained", "openai"),
                device=self.device,
            )
        if backend == "siglip":
            return HFSigLIPEncoder(model_name=spec["model_name"], device=self.device)
        raise ValueError(f"Unknown retriever backend: {backend}")

    def _safe_build_generator(self, spec: dict[str, Any], skipped: list[dict[str, Any]]):
        try:
            return build_answer_generator(spec, device=self.device)
        except Exception as exc:
            skipped.append({"component": "generator", "name": spec.get("name"), "reason": str(exc)})
            logger.warning("Skipping generator %s: %s", spec.get("name"), exc)
            return None

    def _first_available_generator(self, skipped: list[dict[str, Any]]):
        for spec in self.config.get("generators", []):
            gen = self._safe_build_generator(spec, skipped)
            if gen is not None:
                return gen
        return None

    def _write_outputs(self, rows: dict[str, list[dict[str, Any]]]) -> None:
        write_json(self.output_dir / "main_experiment_report.json", rows)
        for name, table_rows in rows.items():
            write_csv(self.output_dir / f"{name}.csv", table_rows)

    def _stage_corpus(self, corpus: ImageCorpus, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in corpus.records:
            src = Path(item.image_path).resolve()
            dst = target_dir / src.name
            if dst.exists():
                continue
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        return target_dir

    def _signature(self, idx: int) -> str:
        consonants = "BCDFGHJKLMNPQRSTVWXYZ"
        vowels = "AEIOU"
        rng = random.Random(self.seed + idx)
        return "".join(rng.choice(consonants if i % 2 == 0 else vowels) for i in range(3))

    def _probe_queries(self, signature: str) -> list[str]:
        return [
            f"I am looking for the image containing the rare verification mark {signature}. What exact text is visible?",
            f"Search for the visual canary labeled {signature}. Please quote the label exactly.",
            f"Can you confirm the exact acronym {signature} in the retrieved image?",
        ][: int(self.config["protocol"]["num_probe_queries"])]
