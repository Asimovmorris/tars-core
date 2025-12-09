# test_chat_audio.py
"""
Test client for the /chat_audio endpoint of TARS.

Usage (from project root, e.g. 'tars core/'):
    python test_chat_audio.py --file samples\hello.wav
    python test_chat_audio.py --file samples\hello.wav --lang en --mode analyst

This script:
  - Verifies the audio file exists and is not too large.
  - Guesses a MIME type from the file extension.
  - Checks that the TARS API is up via /health.
  - Sends the audio to /chat_audio.
  - Prints a clean, human-readable summary of the response.
  - If audio is returned, saves it as '<input_basename>_reply.mp3'.

Design practices:
  1) Clear CLI interface (--file, --lang, --mode, --api-base).
  2) Input validation & normalization (file size, lang, mode).
  3) No binary spam in logs (audio_base64 not printed).
  4) Explicit errors with context (HTTP status, JSON body).
  5) Separation of concerns (I/O, HTTP, printing, saving audio).
  6) Backwards compatible defaults (no lang/mode → current behavior).
  7) Minimal but informative logging (sizes, MIME, base64 length).
  8) Safe handling of optional fields (language/mode optional).
  9) Defensive JSON parsing (validate before using).
 10) Human-oriented summary instead of raw JSON blob.
"""

import argparse
import base64
import json
from pathlib import Path
from typing import Optional

import requests

# Must match the FastAPI server address & port
DEFAULT_API_BASE = "http://127.0.0.1:8000"

# Keep in sync with MAX_AUDIO_BYTES in tars/audio/service.py
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MiB

EXT_TO_MIME = {
    ".wav": "audio/wav",
    ".wave": "audio/x-wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
}

ALLOWED_MODES = {
    "analyst",
    "critic",
    "synthetic",
}


def guess_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    return EXT_TO_MIME.get(ext, "application/octet-stream")


def load_audio_file(path: Path) -> bytes:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if size > MAX_AUDIO_BYTES:
        raise ValueError(
            f"Audio file too large ({size} bytes); max allowed is {MAX_AUDIO_BYTES} bytes."
        )
    with path.open("rb") as f:
        return f.read()


def normalize_lang(lang: Optional[str]) -> Optional[str]:
    """
    Normalize language code.

    - Trim whitespace.
    - Lowercase.
    - Accept e.g. "en", "es", "EN", "Es-ES".
    - Return None if empty.
    """
    if not lang:
        return None
    cleaned = lang.strip()
    if not cleaned:
        return None
    # Very simple normalization; you can extend this later if needed.
    return cleaned.lower()


def normalize_mode(mode: Optional[str]) -> Optional[str]:
    """
    Normalize and validate reasoning mode.

    - Trim, lowercase.
    - Map a few common variants.
    - Return None if unrecognized (server will ignore).
    """
    if not mode:
        return None
    cleaned = mode.strip().lower()
    if not cleaned:
        return None

    # Map some friendly aliases
    if cleaned in {"analyst", "analysis"}:
        cleaned = "analyst"
    elif cleaned in {"critic", "devil", "devils_advocate", "devil's advocate"}:
        cleaned = "critic"
    elif cleaned in {"synthetic", "synth", "synthesizer"}:
        cleaned = "synthetic"

    if cleaned not in ALLOWED_MODES:
        print(
            f"[warn] Unrecognized mode '{mode}'. "
            f"Allowed modes: {sorted(ALLOWED_MODES)}. Ignoring mode hint."
        )
        return None

    return cleaned


def health_check(api_base: str) -> None:
    url = f"{api_base.rstrip('/')}/health"
    resp = requests.get(url, timeout=5)
    if resp.status_code != 200:
        raise RuntimeError(
            f"/health returned status {resp.status_code}: {resp.text!r}"
        )
    try:
        data = resp.json()
    except Exception:
        data = None
    print(f"[health] OK. Response: {data}")


def call_chat_audio(
    api_base: str,
    audio_path: Path,
    lang: Optional[str],
    mode: Optional[str],
) -> dict:
    audio_bytes = load_audio_file(audio_path)
    mime_type = guess_mime_type(audio_path)

    print(f"[info] Using file: {audio_path}")
    print(f"[info] Size: {len(audio_bytes)} bytes")
    print(f"[info] MIME type guess: {mime_type}")
    if lang:
        print(f"[info] Using language hint: {lang}")
    if mode:
        print(f"[info] Using mode hint: {mode}")

    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    url = f"{api_base.rstrip('/')}/chat_audio"

    payload = {
        "audio_base64": audio_b64,
        "mime_type": mime_type,
        "sample_rate": None,  # optional; server doesn't rely on it
    }
    # Only include language/mode if they are not None
    if lang is not None:
        payload["language"] = lang
    if mode is not None:
        payload["mode"] = mode

    resp = requests.post(url, json=payload, timeout=60)

    print(f"[http] Status: {resp.status_code}")
    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(
            f"Response from {url} is not valid JSON: {e}, raw text={resp.text!r}"
        ) from e

    if resp.status_code != 200:
        print("[error] Non-200 status. Response JSON:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        raise RuntimeError("Server returned error status.")

    return data


def save_reply_audio(input_path: Path, response_data: dict) -> None:
    audio_b64 = response_data.get("audio_base64") or ""
    if not audio_b64.strip():
        print("[info] No audio_base64 in response (TTS may have failed or been disabled).")
        return

    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except Exception as e:
        print(f"[warn] Failed to decode audio_base64 from response: {e}")
        return

    out_path = input_path.with_name(input_path.stem + "_reply.mp3")
    with out_path.open("wb") as f:
        f.write(audio_bytes)

    print(f"[info] Saved reply audio to: {out_path}")


def print_clean_summary(response_data: dict) -> None:
    """
    Print a human-friendly summary of the /chat_audio response
    without dumping the full audio_base64 field.
    """
    reply = response_data.get("reply", "")
    mode = response_data.get("mode", "")
    transcript = response_data.get("transcript", "")
    error = response_data.get("error")
    latency_ms = response_data.get("latency_ms")

    audio_b64 = response_data.get("audio_base64") or ""
    audio_len = len(audio_b64)

    print("\n[summary]")
    print(f"  Mode:        {mode!r}")
    print(f"  Transcript:  {transcript!r}")
    print(f"  Reply:       {reply!r}")
    print(f"  Latency:     {latency_ms} ms")
    print(f"  Error:       {error}")
    print(f"  Audio bytes: {'present' if audio_len > 0 else 'none'} "
          f"(base64 length={audio_len})")

    # For deep debugging, you can still inspect full JSON with audio omitted:
    # data_copy = dict(response_data)
    # if "audio_base64" in data_copy:
    #     data_copy["audio_base64"] = f"<omitted, length={audio_len}>"
    # print("\n[response JSON (audio_base64 omitted)]")
    # print(json.dumps(data_copy, indent=2, ensure_ascii=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Test client for TARS /chat_audio endpoint."
    )
    parser.add_argument(
        "--file",
        "-f",
        required=True,
        help="Path to input audio file (WAV/MP3/OGG).",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"Base URL for TARS API (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Optional STT language hint, e.g. 'en' or 'es'. If omitted, Whisper auto-detects.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Optional reasoning mode hint: 'analyst', 'critic', or 'synthetic'.",
    )

    args = parser.parse_args(argv)
    audio_path = Path(args.file)

    # Normalize optional fields
    lang = normalize_lang(args.lang)
    mode = normalize_mode(args.mode)

    try:
        # 1. Check server health first
        health_check(args.api_base)

        # 2. Call /chat_audio with optional lang/mode
        data = call_chat_audio(args.api_base, audio_path, lang, mode)

        # 3. Print a clean summary (without dumping the whole audio_base64)
        print_clean_summary(data)

        # 4. Save reply audio, if present
        save_reply_audio(audio_path, data)

    except Exception as e:
        print(f"[fatal] {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


