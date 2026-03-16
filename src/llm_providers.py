"""
LLM provider abstraction for Gemini, OpenAI, and Anthropic.
"""

import os
import json
import re


def _extract_json(text: str) -> dict | None:
    """Extract JSON object from LLM response text."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    for pattern in [
        r"```(?:json)?\s*(\{.*?\})\s*```",
        r"(\{.*\})",
    ]:
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if match.groups() else match.group(0))
            except json.JSONDecodeError:
                continue
    return None


def call_llm(prompt: str, system_prompt: str, provider: str = "gemini") -> dict | None:
    """
    Call an LLM and return parsed JSON response.

    Args:
        prompt: User prompt
        system_prompt: System instruction
        provider: One of 'gemini', 'openai', 'anthropic'

    Returns:
        Parsed dict from JSON response, or None on failure.
    """
    provider = provider.lower().strip()

    try:
        if provider == "gemini":
            return _call_gemini(prompt, system_prompt)
        elif provider == "openai":
            return _call_openai(prompt, system_prompt)
        elif provider == "anthropic":
            return _call_anthropic(prompt, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: '{provider}'. Use 'gemini', 'openai', or 'anthropic'.")
    except Exception as e:
        print(f"[{provider}] API error: {e}")
        return None


def _call_gemini(prompt: str, system_prompt: str) -> dict | None:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
        ),
        contents=prompt,
    )
    return _extract_json(response.text) if response.text else None


def _call_openai(prompt: str, system_prompt: str) -> dict | None:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    text = response.choices[0].message.content
    return _extract_json(text) if text else None


def _call_anthropic(prompt: str, system_prompt: str) -> dict | None:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=1024,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    return _extract_json(text) if text else None