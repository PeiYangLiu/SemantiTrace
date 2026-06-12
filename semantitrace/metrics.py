from __future__ import annotations

import math
import re
import unicodedata
import warnings

import numpy as np


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_positive_signature(response: str, signature: str) -> bool:
    """Return true only when a response affirmatively contains the signature.

    VLMs often echo the queried canary in negative answers such as
    "FALEK is not visible". Plain substring matching would count those as
    successes, so signature mentions inside local negation contexts are ignored.
    """

    normalized_signature = normalize_text(signature)
    if not normalized_signature:
        return False
    normalized_response = normalize_text(response)
    signature_pattern = rf"\b{re.escape(normalized_signature)}\b"
    if re.search(signature_pattern, normalized_response) is None:
        return False

    sentences = re.split(r"[\n.!?;:]+", str(response))
    for sentence in sentences:
        normalized_sentence = normalize_text(sentence)
        if re.search(signature_pattern, normalized_sentence) is None:
            continue
        if _signature_sentence_is_negated(normalized_sentence, normalized_signature):
            continue
        if not _signature_sentence_is_affirmative(normalized_sentence, normalized_signature):
            continue
        return True
    return False


def _signature_sentence_is_affirmative(sentence: str, signature: str) -> bool:
    signature_pattern = re.escape(signature)
    if re.fullmatch(rf"(?:the\s+)?{signature_pattern}", sentence):
        return True
    if re.match(rf"^{signature_pattern}\b", sentence) and not re.search(r"\b(?:not|none|cannot|cant|closest)\b", sentence):
        return True
    positive_patterns = [
        rf"\b(?:exact\s+)?(?:text|label|mark|signature|code|acronym|transcription)\b.*\b(?:is|says|reads|appears|visible|shown|written|printed)\b.*\b{signature_pattern}\b",
        rf"\b(?:can|could)\s+(?:clearly\s+)?(?:read|see|identify|transcribe)\s+\b{signature_pattern}\b",
        rf"\b(?:read|transcribe|identify)\s+\b{signature_pattern}\b",
        rf"\b(?:is|says|reads|appears|visible|shown|written|printed)\b\s+(?:as\s+)?[\"']?\b{signature_pattern}\b",
        rf"\b{signature_pattern}\b\s+(?:is\s+)?(?:visible|shown|written|printed|readable|present)\b",
        rf"\bvisible\s+(?:near|in|on).*?\bis\s+[\"']?\b{signature_pattern}\b",
    ]
    return any(re.search(pattern, sentence) for pattern in positive_patterns)


def _signature_sentence_is_negated(sentence: str, signature: str) -> bool:
    negation_patterns = [
        r"\bnot\s+(?:visible|present|legible|readable|found|shown|seen|there)\b",
        r"\bno\s+(?:clearly\s+)?(?:visible\s+)?(?:text|mark|label|signature|code|acronym)\b",
        r"\bthere\s+is\s+no\b",
        r"\bnone\s+(?:of\s+the\s+)?(?:provided\s+)?(?:images?|panels?)\s+(?:show|contain|include|has|have)\b",
        r"\bnone\s+(?:contain|contains|show|shows|include|includes)\b",
        r"\bnor\s+(?:is|are|does|do)\s+(?:there\s+)?(?:any\s+)?(?:visible\s+)?(?:text|mark|label|signature|code|acronym)?\b",
        r"\bdoes\s+not\s+(?:contain|show|include|appear)\b",
        r"\bdo\s+not\s+(?:see|read|find|detect)\b",
        r"\bcannot\s+(?:see|read|find|detect|confirm|identify|make\s+out)\b",
        r"\bcannot\s+be\s+(?:determined|confirmed|identified|read|seen|found)\b",
        r"\bcant\s+(?:see|read|find|detect|confirm|identify|make\s+out)\b",
        r"\bcan\s+not\s+(?:see|read|find|detect|confirm|identify|make\s+out)\b",
        r"\bnot\s+able\s+to\s+(?:see|read|find|detect|confirm|identify|make\s+out)\b",
        r"\bnot\s+match\s+any\b",
        r"\bdoes\s+not\s+match\s+any\b",
        r"\bnot\s+(?:an\s+)?exact\s+match\b",
    ]
    if any(re.search(pattern, sentence) for pattern in negation_patterns):
        return True
    signature_pattern = re.escape(signature)
    local_negations = [
        rf"\b{signature_pattern}\b\s+(?:is|was|appears|seems)?\s*not\b",
        rf"\b{signature_pattern}\b\s+(?:cannot|cant|can\s+not)\s+(?:be\s+)?(?:seen|read|found|detected|confirmed|identified|made\s+out)\b",
        rf"\b{signature_pattern}\b\s+(?:is|was|appears|seems)?\s*(?:not\s+)?(?:visible|present|legible|readable)\s+in\s+the\s+(?:provided\s+)?image\b",
        rf"\b{signature_pattern}\b\s+(?:is|was|appears|seems)?\s*(?:not\s+)?(?:visible|present|legible|readable)\s+in\s+(?:any|none)\b",
        rf"\b(?:no|none)\b.*\b(?:exact\s+)?(?:text|mark|label|signature|code|acronym)?\s*\b{signature_pattern}\b",
        rf"\b(?:no|none)\b.*\b(?:contain|contains|show|shows|include|includes)\b.*\b{signature_pattern}\b",
        rf"\bnot\s+(?:an\s+)?exact\s+match\s+(?:to|for)\s+\b{signature_pattern}\b",
        rf"\bnot\s+(?:the\s+)?\b{signature_pattern}\b",
    ]
    return any(re.search(pattern, sentence) for pattern in local_negations)


def compute_psnr(original: np.ndarray, edited: np.ndarray) -> float:
    original = original.astype(np.float64)
    edited = edited.astype(np.float64)
    mse = float(np.mean((original - edited) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((255.0**2) / mse)


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)
    if real_features.ndim != 2 or fake_features.ndim != 2:
        raise ValueError("FID expects two 2D feature matrices")

    mu_r, mu_f = real_features.mean(axis=0), fake_features.mean(axis=0)
    sigma_r = np.atleast_2d(np.cov(real_features, rowvar=False))
    sigma_f = np.atleast_2d(np.cov(fake_features, rowvar=False))
    eps = 1e-6
    sigma_r = sigma_r + np.eye(sigma_r.shape[0]) * eps
    sigma_f = sigma_f + np.eye(sigma_f.shape[0]) * eps

    try:
        from scipy.linalg import sqrtm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            covmean = sqrtm(sigma_r @ sigma_f)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    except Exception:
        vals, vecs = np.linalg.eigh(sigma_r @ sigma_f)
        vals = np.clip(vals, 0.0, None)
        covmean = (vecs * np.sqrt(vals)) @ vecs.T

    diff = mu_r - mu_f
    return float(diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * covmean))


def compute_lpips(original: np.ndarray, edited: np.ndarray, net: str = "alex") -> float:
    try:
        import lpips
        import torch
    except ImportError as exc:
        raise ImportError("LPIPS requires optional dependency: pip install lpips torch") from exc

    loss_fn = lpips.LPIPS(net=net)

    def to_tensor(img: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0

    with torch.no_grad():
        return float(loss_fn(to_tensor(original), to_tensor(edited)).item())


def conditional_generation_success_rate(responses: list[str], signatures: list[str]) -> float:
    if len(responses) != len(signatures):
        raise ValueError("responses and signatures must have the same length")
    if not responses:
        return 0.0
    hits = 0
    for response, signature in zip(responses, signatures):
        hits += int(contains_positive_signature(response, signature))
    return hits / len(responses)


def retrieval_rank(ranked_ids: list[str], target_id: str) -> int | None:
    try:
        return ranked_ids.index(target_id) + 1
    except ValueError:
        return None
