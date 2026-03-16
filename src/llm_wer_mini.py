"""
LLM-WER Mini — Lightweight LLM-based Word Error Rate evaluation for ASR.

Usage:
    python llm_wer_mini.py --input data.csv [--provider gemini] [--output results.csv]

Input CSV must have columns: audio_filepath, reference, prediction, language
Optional column: provider (per-row override)

Environment variables:
    GEMINI_API_KEY    — required if provider is 'gemini' (default)
    OPENAI_API_KEY    — required if provider is 'openai'
    ANTHROPIC_API_KEY — required if provider is 'anthropic'
    OPENAI_MODEL      — override OpenAI model (default: gpt-4o-mini)
    ANTHROPIC_MODEL   — override Anthropic model (default: claude-sonnet-4-20250514)
"""

import argparse
import json
import re
import string
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import jiwer
import pandas as pd
from tqdm import tqdm

from llm_providers import call_llm

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = (Path(__file__).parent / "prompt_template.txt").read_text()

# ---------------------------------------------------------------------------
# Text normalization (simplified from reference — no indic-nlp dependency)
# ---------------------------------------------------------------------------
INDIC_PUNCTUATION = "।॥॰''\"‛‟′″´˝^°¤।॥॰¯'—–‑°¬´ۭ۪\u200b\u200c\u200d\u200e\u200f"
ALL_PUNCTUATION = string.punctuation + INDIC_PUNCTUATION


def normalize_text(text: str) -> str:
    """Basic normalization: lowercase, strip punctuation, collapse whitespace."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"([,\-\.\(\)\[\]\{\}/\\])\B", r" ", text)
    text = text.translate(str.maketrans("", "", ALL_PUNCTUATION)).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# WER / CER (from reference)
# ---------------------------------------------------------------------------
def compute_wer(ref: str, hyp: str) -> float:
    ref, hyp = ref.strip(), hyp.strip()
    n, m = len(ref.split()), len(hyp.split())
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0
    out = jiwer.process_words(ref, hyp)
    denom = max(m, n)
    return (out.substitutions + out.deletions + out.insertions) / denom


def compute_cer(ref: str, hyp: str) -> float:
    ref, hyp = ref.strip(), hyp.strip()
    n, m = len(ref), len(hyp)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0
    out = jiwer.process_characters(ref, hyp)
    denom = max(m, n)
    return (out.substitutions + out.deletions + out.insertions) / denom


# ---------------------------------------------------------------------------
# Segment diffing (from reference)
# ---------------------------------------------------------------------------
def get_diff_segments(reference: str, prediction: str) -> list[dict[str, Any]]:
    """Find mismatched segments between reference and prediction."""
    ref_words = reference.strip().split()
    pred_words = prediction.strip().split()
    if not ref_words and not pred_words:
        return []
    matcher = SequenceMatcher(None, ref_words, pred_words)
    return [
        {
            "reference": " ".join(ref_words[i1:i2]),
            "prediction": " ".join(pred_words[j1:j2]),
            "tag": tag,
            "seg_idx": idx,
        }
        for idx, (tag, i1, i2, j1, j2) in enumerate(matcher.get_opcodes())
    ]


# ---------------------------------------------------------------------------
# LLM equivalence check — now with full sentence context
# ---------------------------------------------------------------------------
def check_equivalence(
        ref_segment: str,
        pred_segment: str,
        full_reference: str,
        full_prediction: str,
        provider: str,
        cache: dict,
) -> tuple[bool, str]:
    """
    Ask the LLM whether two mismatched segments are equivalent,
    providing full sentence context for disambiguation.

    Returns:
        (is_equivalent, reasoning)
    """
    # Cache on segment pair + provider (context aids LLM but same segments
    # in different sentences are rare enough that segment-level cache is fine)
    cache_key = (ref_segment, pred_segment, provider)
    if cache_key in cache:
        return cache[cache_key]

    input_obj = json.dumps(
        {
            "index": 0,
            "full_reference": full_reference,
            "full_prediction": full_prediction,
            "reference_segment": ref_segment,
            "prediction_segment": pred_segment,
        },
        ensure_ascii=False,
    )
    prompt = f"{PROMPT_TEMPLATE}\n\n**INPUT:**\n{input_obj}"

    result = call_llm(
        prompt=prompt,
        system_prompt="You are an expert linguistic analyst. Respond with ONLY valid JSON.",
        provider=provider,
    )

    is_equivalent = False
    reasoning = ""
    if result and isinstance(result, dict):
        # Handle both 'equivalent' and 'equivalence' keys (compatibility)
        is_equivalent = bool(
            result.get("equivalent", result.get("equivalence", False))
        )
        reasoning = result.get("reasoning", "")

    cache[cache_key] = (is_equivalent, reasoning)
    return is_equivalent, reasoning


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_llm_wer(input_path: str, provider: str = "gemini", output_path: str | None = None):
    """
    Run the LLM-WER pipeline.

    Args:
        input_path:  Path to CSV with columns: audio_filepath, reference, prediction, language
        provider:    Default LLM provider ('gemini', 'openai', 'anthropic')
        output_path: Path to save results CSV (default: <input_stem>_llm_wer.csv)
    """
    # --- Load ---
    df = pd.read_csv(input_path)
    required = {"audio_filepath", "reference", "prediction", "language"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if output_path is None:
        output_path = str(Path(input_path).stem) + "_llm_wer.csv"

    # --- Normalize ---
    df["norm_reference"] = df["reference"].astype(str).apply(normalize_text)
    df["norm_prediction"] = df["prediction"].astype(str).apply(normalize_text)

    # --- Original WER/CER ---
    df["original_wer"] = [
        compute_wer(r, p)
        for r, p in zip(df["norm_reference"], df["norm_prediction"])
    ]
    df["original_cer"] = [
        compute_cer(r, p)
        for r, p in zip(df["norm_reference"], df["norm_prediction"])
    ]

    # --- Segment diffing + LLM adjudication ---
    llm_cache: dict = {}
    llm_logs: list[dict] = []
    corrected_preds = []
    corrected_refs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        row_provider = str(row.get("provider", provider)).strip() or provider
        full_ref = row["norm_reference"]
        full_pred = row["norm_prediction"]
        segments = get_diff_segments(full_ref, full_pred)
        pred_parts = []
        ref_parts = []

        for seg in segments:
            if seg["tag"] == "equal":
                pred_parts.append(seg["reference"])
                ref_parts.append(seg["reference"])
            elif seg["reference"].strip() and seg["prediction"].strip():
                is_eq, reasoning = check_equivalence(
                    ref_segment=seg["reference"],
                    pred_segment=seg["prediction"],
                    full_reference=full_ref,
                    full_prediction=full_pred,
                    provider=row_provider,
                    cache=llm_cache,
                )
                llm_logs.append({
                    "reference_segment": seg["reference"],
                    "prediction_segment": seg["prediction"],
                    "full_reference": full_ref,
                    "full_prediction": full_pred,
                    "equivalent": is_eq,
                    "reasoning": reasoning,
                    "provider": row_provider,
                })
                if is_eq:
                    pred_parts.append(seg["reference"])
                    ref_parts.append(seg["reference"])
                else:
                    pred_parts.append(seg["prediction"])
                    ref_parts.append(seg["reference"])
            else:
                # insertion or deletion — keep as-is
                pred_parts.append(seg["prediction"])
                ref_parts.append(seg["reference"])

        corrected_preds.append(" ".join(pred_parts).strip())
        corrected_refs.append(" ".join(ref_parts).strip())

    # --- Corrected WER/CER ---
    df["corrected_prediction"] = corrected_preds
    df["corrected_reference"] = corrected_refs
    df["corrected_wer"] = [
        compute_wer(r, p)
        for r, p in zip(df["corrected_reference"], df["corrected_prediction"])
    ]
    df["corrected_cer"] = [
        compute_cer(r, p)
        for r, p in zip(df["corrected_reference"], df["corrected_prediction"])
    ]

    # --- Summary ---
    print("\n" + "=" * 50)
    print(f"  Original  WER: {df['original_wer'].mean():.4f}")
    print(f"  Original  CER: {df['original_cer'].mean():.4f}")
    print(f"  Corrected WER: {df['corrected_wer'].mean():.4f}")
    print(f"  Corrected CER: {df['corrected_cer'].mean():.4f}")
    print(f"  LLM calls made: {len(llm_logs)} ({len(llm_cache)} unique)")
    print("=" * 50)

    # --- Save ---
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    if llm_logs:
        logs_path = str(Path(output_path).stem) + "_llm_logs.csv"
        pd.DataFrame(llm_logs).to_csv(logs_path, index=False)
        print(f"LLM logs saved to: {logs_path}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM-WER Mini: Lightweight LLM-based WER/CER evaluation"
    )
    parser.add_argument("--input", "-i", required=True, help="Input CSV path")
    parser.add_argument(
        "--provider", "-p", default="gemini",
        choices=["gemini", "openai", "anthropic"],
        help="Default LLM provider (default: gemini)",
    )
    parser.add_argument("--output", "-o", default=None, help="Output CSV path")
    args = parser.parse_args()

    run_llm_wer(args.input, args.provider, args.output)