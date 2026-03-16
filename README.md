# LLM-WER Mini v0.1

Lightweight LLM-based Word Error Rate (WER) and Character Error Rate (CER) evaluation for ASR systems, with special focus on Indic languages.

Simplified from [sarvamai/llm_wer](https://github.com/sarvamai/llm_wer), with key improvements to the equivalence logic.

## How It Works

1. **Normalize** reference and prediction text (lowercase, strip punctuation, collapse whitespace)
2. **Compute original WER/CER** using `jiwer`
3. **Diff** reference vs prediction into segments using `SequenceMatcher`
4. **Ask an LLM** (with full sentence context) whether each mismatched segment is equivalent — covers:
    - Cross-script transliteration (Amazon → अमेज़न)
    - Spelling variations (कुत्ता → कुत्त)
    - Phonetic contractions (यह → ये)
    - Number/symbol equivalence (सौ रुपये → ₹100)
    - **Semantic translation equivalence** (राजा → king, कुत्ता → dog)
5. **Recompute corrected WER/CER** — only genuine errors remain

## Changes from v1

| Change | Detail |
|--------|--------|
| **Semantic translation rule** | New Section 5 in prompt — handles cross-language translation (राजा=king, पानी=water) that v1 missed |
| **Full sentence context** | LLM now receives the complete reference + prediction sentences alongside the segment pair, enabling better disambiguation |
| **Reasoning in logs** | LLM logs now include the reasoning and full sentence context for each judgment |
| **Key compatibility** | Handles both `equivalent` and `equivalence` keys from LLM responses |

## Supported LLM Providers

| Provider | Model (default) | Env Variable |
|----------|----------------|--------------|
| **Gemini** (default) | `gemini-2.5-flash` | `GEMINI_API_KEY` |
| OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |

Override OpenAI/Anthropic models via `OPENAI_MODEL` / `ANTHROPIC_MODEL` env vars.

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"
```

## Input Format

CSV with columns:

| Column | Description |
|--------|-------------|
| `audio_filepath` | Path to the audio file |
| `reference` | Ground truth transcription |
| `prediction` | ASR prediction |
| `language` | Language code/name (e.g., `hindi`, `tamil`) |
| `provider` | *(optional)* Per-row LLM provider override |

## Usage

```bash
# Default (Gemini)
python llm_wer_mini.py --input data.csv

# With OpenAI
python llm_wer_mini.py --input data.csv --provider openai

# Custom output path
python llm_wer_mini.py --input data.csv --output results.csv
```

## Output

The output CSV contains all original columns plus:

| Column | Description |
|--------|-------------|
| `norm_reference` | Normalized reference |
| `norm_prediction` | Normalized prediction |
| `original_wer` | Standard WER |
| `original_cer` | Standard CER |
| `corrected_prediction` | Prediction after LLM equivalence corrections |
| `corrected_reference` | Reference after LLM equivalence corrections |
| `corrected_wer` | LLM-adjusted WER |
| `corrected_cer` | LLM-adjusted CER |

A separate `*_llm_logs.csv` file records each LLM judgment with:

| Column | Description |
|--------|-------------|
| `reference_segment` | The mismatched reference segment |
| `prediction_segment` | The mismatched prediction segment |
| `full_reference` | Complete normalized reference sentence (context) |
| `full_prediction` | Complete normalized prediction sentence (context) |
| `equivalent` | LLM verdict (True/False) |
| `reasoning` | LLM's explanation for the verdict |
| `provider` | Which LLM provider was used |

## Files

```
llm-wer-mini/
├── llm_wer_mini.py       # Main pipeline + CLI
├── llm_providers.py       # LLM provider abstraction (Gemini/OpenAI/Anthropic)
├── prompt_template.txt    # Indic language equivalence prompt (with semantic translation)
├── requirements.txt       # Dependencies
├── sample_input.csv       # Example input
└── README.md
```

## What we took from Sarvam's repo (concept/approach):

- The overall methodology — diff segments → ask LLM → recompute WER/CER
- The SequenceMatcher-based diffing approach
- The WER/CER computation logic using jiwer with max(m, n) clamping
- The prompt template's structure and equivalence rules 1–4 (formatting, spoken/written, cross-script, phonetic contractions) — these were trimmed but the rules and examples are largely from the original

## What we wrote fresh / changed significantly:

- llm_providers.py — entirely new, Sarvam uses Vertex AI + OpenAI SDK with threading/workers; ours uses direct Gemini GenAI, OpenAI, and Anthropic SDKs
- Section 5 (Semantic Translation Equivalence) — this is our addition, doesn't exist in Sarvam's prompt at all
- Full sentence context in the LLM call — Sarvam only sends the segment pair, we send full_reference + full_prediction alongside
- The normalization is simplified (no indic-nlp-library dependency)
- No threading, no JSONL caching, no Google Sheets integration, no Pydantic validation
- The CLI, the per-row provider override, the reasoning in logs — all new