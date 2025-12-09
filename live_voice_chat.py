# live_voice_chat.py
"""
Live voice chat with TARS over /chat_audio, with dynamic recording:

    microphone -> WAV bytes -> /chat_audio -> reply text + MP3 -> speakers

Recording behavior:
    - Maximum duration (e.g. 15–45 seconds) as a safety ceiling.
    - Early stop when your speech ends: if there's at least some speech
      and then ~1.8 seconds of sustained silence, recording stops and TARS
      starts thinking immediately.

Usage (from project root, e.g. 'tars core/'):
    python live_voice_chat.py
    python live_voice_chat.py --lang en --mode analyst
    python live_voice_chat.py --lang es --mode critic --duration 30
    python live_voice_chat.py --list-devices
    python live_voice_chat.py --input-device-index 2
    python live_voice_chat.py --voice-style brief
    python live_voice_chat.py --debug

Design / enhancements in this version:
  1) Dynamic recording with chunked audio and RMS-based silence detection.
  2) 1.8 seconds of silence after speech ends triggers early stop.
  3) 'duration' is treated as max duration (ceiling), not fixed length.
  4) Minimum speech requirement before we consider silence = valid utterance.
  5) Adjustable max duration (up to 45s) with clear safeguards.
  6) Explicit device listing and device selection.
  7) No binary spam in logs (audio_base64 never printed).
  8) Optional debug mode to inspect JSON (audio omitted).
  9) Clean, human-friendly summaries and latency reporting.
 10) Voice style control wired cleanly to server ('brief', 'story', 'technical', 'default').
"""

import argparse
import base64
import io
import json
import os
import tempfile
import time
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
import wave
from playsound import playsound

# API defaults
DEFAULT_API_BASE = "http://127.0.0.1:8000"

# Audio recording defaults
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_MAX_DURATION_SECONDS = 15.0  # max recording time per utterance

# Silence / VAD tuning
CHUNK_DURATION_SEC = 0.2           # length of each recorded chunk
SILENCE_DURATION_SEC = 1.8         # how long silence must last after speech to stop
SILENCE_THRESHOLD = 200.0          # RMS threshold for silence vs speech (int16 scale)
MIN_SPEECH_DURATION_SEC = 0.5      # minimum non-silent speech required to treat as valid utterance

# Keep in sync with server expectation if you change format
REQUEST_MIME_TYPE = "audio/wav"

ALLOWED_MODES = {
    "analyst",
    "critic",
    "synthetic",
}

ALLOWED_VOICE_STYLES = {"brief", "default", "story", "technical"}


# ---------------------------------------------------------------------------
# Normalization helpers (lang, mode, voice_style)
# ---------------------------------------------------------------------------

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
    return cleaned.lower()


def normalize_mode(mode: Optional[str]) -> Optional[str]:
    """
    Normalize and validate reasoning mode.

    Returns one of: 'analyst', 'critic', 'synthetic', or None if unrecognized.
    """
    if not mode:
        return None

    cleaned = mode.strip().lower()
    if not cleaned:
        return None

    # Map aliases
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


def normalize_voice_style(style: Optional[str]) -> Optional[str]:
    """
    Normalize and validate voice style.

    Returns one of: 'brief', 'default', 'story', 'technical', or None.
    This is passed through to the server as 'voice_style'.
    """
    if not style:
        return None

    cleaned = style.strip().lower()
    if not cleaned:
        return None

    if cleaned in {"brief", "short", "concise"}:
        return "brief"
    if cleaned in {"story", "narrative"}:
        return "story"
    if cleaned in {"technical", "tech", "dense"}:
        return "technical"
    if cleaned in {"default", "normal"}:
        return "default"

    print(
        f"[warn] Unrecognized voice style '{style}'. "
        f"Allowed: {sorted(ALLOWED_VOICE_STYLES)}. Ignoring voice-style hint."
    )
    return None


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def list_devices_and_exit() -> int:
    """
    Print available audio devices and exit.
    """
    print("=== Available audio devices ===")
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"[fatal] Could not query audio devices: {e}")
        return 1

    for idx, dev in enumerate(devices):
        print(
            f"[{idx}] {dev['name']}  "
            f"(max_input_channels={dev['max_input_channels']}, "
            f"max_output_channels={dev['max_output_channels']})"
        )
    print("================================")
    return 0


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def health_check(api_base: str) -> None:
    """
    Call /health on the TARS API and print a short status.
    """
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


# ---------------------------------------------------------------------------
# Audio recording (dynamic: mic -> WAV bytes with silence detection)
# ---------------------------------------------------------------------------

def record_audio_to_wav_bytes_dynamic(
    max_duration_seconds: float,
    samplerate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    device: Optional[int] = None,
) -> Optional[bytes]:
    """
    Dynamically record from the microphone until:

      - User has spoken for at least MIN_SPEECH_DURATION_SEC
      - And then there is at least SILENCE_DURATION_SEC of continuous silence

    Or until max_duration_seconds is reached, whichever comes first.

    Returns:
        WAV-encoded bytes if a valid utterance was captured,
        or None if the recording was mostly silence.
    """
    if max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be > 0")

    if device is not None:
        print(f"[record] Using input device index: {device}")

    print(
        f"[record] Recording (up to {max_duration_seconds:.1f} s). "
        f"Silence stop: {SILENCE_DURATION_SEC:.1f} s, "
        f"silence threshold: {SILENCE_THRESHOLD:.1f} (RMS). "
        "Speak normally; recording will stop when you fall silent."
    )

    # Ensure at least 1 frame per chunk
    block_size = max(1, int(CHUNK_DURATION_SEC * samplerate))
    frames = []
    total_frames = 0
    speech_frames = 0
    last_non_silent_frame_idx: Optional[int] = None

    try:
        with sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            dtype="int16",
            device=device,
        ) as stream:
            start_time = time.monotonic()
            while True:
                chunk, overflowed = stream.read(block_size)
                if overflowed:
                    print("[warn] Audio buffer overflowed; some data may be lost.")

                # chunk shape: (block_size, channels)
                frames.append(chunk.copy())
                total_frames += chunk.shape[0]

                # Compute RMS for this chunk
                audio_np = chunk.astype(np.float32)
                rms = float(np.sqrt(np.mean(audio_np ** 2))) if audio_np.size > 0 else 0.0

                current_time = time.monotonic()
                elapsed_sec = current_time - start_time

                # Detect speech vs silence
                if rms >= SILENCE_THRESHOLD:
                    # Non-silent chunk
                    speech_frames += chunk.shape[0]
                    last_non_silent_frame_idx = total_frames

                # Check if we've reached max duration ceiling
                if elapsed_sec >= max_duration_seconds:
                    print(
                        f"[record] Reached max duration ({max_duration_seconds:.1f}s). "
                        "Stopping recording."
                    )
                    break

                # If we have some speech and then enough silence, stop early
                if last_non_silent_frame_idx is not None:
                    last_speech_time_sec = last_non_silent_frame_idx / float(samplerate)
                    silence_since_sec = elapsed_sec - last_speech_time_sec

                    if (
                        (speech_frames / float(samplerate)) >= MIN_SPEECH_DURATION_SEC
                        and silence_since_sec >= SILENCE_DURATION_SEC
                    ):
                        print(
                            f"[record] Detected end of speech at t≈{elapsed_sec:.1f}s "
                            f"(silence {silence_since_sec:.1f}s). Stopping recording."
                        )
                        break
    except Exception as e:
        raise RuntimeError(f"Failed to record audio from microphone: {e}") from e

    if total_frames == 0:
        print("[warn] No audio frames captured. Please check your microphone.")
        return None

    total_duration = total_frames / float(samplerate)
    speech_duration = speech_frames / float(samplerate)
    print(
        f"[record] Total recorded: {total_duration:.2f}s, "
        f"estimated speech: {speech_duration:.2f}s"
    )

    # If we never got enough speech, treat as silence / noise
    if speech_duration < MIN_SPEECH_DURATION_SEC:
        print(
            "[warn] Detected too little speech in recording. "
            "Mic may not be configured correctly, or you spoke too briefly. "
            "Skipping this utterance.\n"
        )
        return None

    # Concatenate all chunks
    audio_all = np.concatenate(frames, axis=0)

    # Encode as WAV bytes
    buf = io.BytesIO()
    try:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(samplerate)
            wf.writeframes(audio_all.tobytes())
    except Exception as e:
        raise RuntimeError(f"Failed to encode recorded audio as WAV: {e}") from e

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Call /chat_audio
# ---------------------------------------------------------------------------

def call_chat_audio(
    api_base: str,
    wav_bytes: bytes,
    lang: Optional[str],
    mode: Optional[str],
    voice_style: Optional[str],
    debug: bool = False,
) -> dict:
    """
    Send WAV audio bytes to /chat_audio and return the JSON response.
    """
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    url = f"{api_base.rstrip('/')}/chat_audio"
    payload = {
        "audio_base64": audio_b64,
        "mime_type": REQUEST_MIME_TYPE,
        "sample_rate": DEFAULT_SAMPLE_RATE,
    }
    if lang is not None:
        payload["language"] = lang
    if mode is not None:
        payload["mode"] = mode
    if voice_style is not None:
        payload["voice_style"] = voice_style

    try:
        resp = requests.post(url, json=payload, timeout=90)
    except Exception as e:
        raise RuntimeError(f"Error calling {url}: {e}") from e

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

    if debug:
        # Show JSON minus audio for debugging
        dbg = dict(data)
        if "audio_base64" in dbg:
            dbg["audio_base64"] = f"<omitted, length={len(dbg['audio_base64'])}>"
        print("\n[debug] Full JSON response (audio omitted):")
        print(json.dumps(dbg, indent=2, ensure_ascii=False))

    return data


# ---------------------------------------------------------------------------
# Playback (MP3 bytes -> speakers)
# ---------------------------------------------------------------------------

def play_reply_audio_from_response(response_data: dict) -> None:
    """
    Decode audio_base64 from response, save to a temporary MP3 file,
    and play it via the default audio device.
    """
    audio_b64 = response_data.get("audio_base64") or ""
    if not audio_b64.strip():
        print("[info] No audio in response (TTS may have failed or been disabled).")
        return

    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except Exception as e:
        print(f"[warn] Failed to decode audio_base64 from response: {e}")
        return

    # Unique temp file per reply to avoid permission conflicts on Windows
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="tars_reply_", suffix=".mp3")
        os.close(fd)
    except Exception as e:
        print(f"[warn] Failed to create temporary reply audio file: {e}")
        return

    try:
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)
    except Exception as e:
        print(f"[warn] Failed to write temporary reply audio file: {e}")
        return

    print(f"[play] Playing reply audio...")
    try:
        playsound(tmp_path)
    except Exception as e:
        print(f"[warn] Failed to play audio: {e}")
    finally:
        # Attempt to clean up; ignore errors
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Pretty summary
# ---------------------------------------------------------------------------

def print_clean_summary(response_data: dict) -> None:
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
    print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Live voice chat client for TARS /chat_audio endpoint (dynamic recording with silence detection)."
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
    parser.add_argument(
        "--voice-style",
        default=None,
        help="Optional voice style: 'brief', 'story', 'technical', or 'default' (server decides defaults).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_MAX_DURATION_SECONDS,
        help=(
            f"Maximum recording duration in seconds (default: {DEFAULT_MAX_DURATION_SECONDS}). "
            "Recording will stop earlier if you speak and then fall silent for ~1.8s."
        ),
    )
    parser.add_argument(
        "--input-device-index",
        type=int,
        default=None,
        help="Optional index of the input (microphone) device. Use --list-devices to inspect.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full JSON response (audio omitted) for debugging.",
    )

    args = parser.parse_args(argv)

    if args.list_devices:
        return list_devices_and_exit()

    # Normalize optional fields
    lang = normalize_lang(args.lang)
    mode = normalize_mode(args.mode)
    voice_style = normalize_voice_style(args.voice_style)

    # Safeguard on max duration (here we allow up to 45s)
    if args.duration <= 0 or args.duration > 45:
        print("[warn] Duration out of recommended range (0 < d <= 45). Using default 15.0s.")
        max_duration = DEFAULT_MAX_DURATION_SECONDS
    else:
        max_duration = args.duration

    # 1. Check server health first
    try:
        health_check(args.api_base)
    except Exception as e:
        print(f"[fatal] Health check failed: {e}")
        return 1

    print("\n=== Live voice chat with TARS (dynamic recording) ===")
    print(f"Language hint: {lang or 'auto-detect'}")
    print(f"Mode hint:     {mode or 'current TARS mode'}")
    print(f"Voice style:   {voice_style or 'server-default'}")
    if args.input_device_index is not None:
        print(f"Input device:  index {args.input_device_index}")
    else:
        print("Input device:  default (system-selected)")
    print(
        f"Max duration:  {max_duration:.1f} s | Silence stop: {SILENCE_DURATION_SEC:.1f} s "
        f"| Silence threshold: {SILENCE_THRESHOLD:.1f} RMS\n"
    )
    print("Press ENTER to start recording a message.")
    print("Type 'q' and press ENTER to quit.\n")

    try:
        while True:
            user_input = input("[prompt] Press ENTER to speak, or 'q' + ENTER to quit: ").strip()
            if user_input.lower() == "q":
                print("[info] Quitting.")
                break

            # 2. Record audio (dynamic, with silence detection)
            try:
                wav_bytes = record_audio_to_wav_bytes_dynamic(
                    max_duration_seconds=max_duration,
                    samplerate=DEFAULT_SAMPLE_RATE,
                    channels=DEFAULT_CHANNELS,
                    device=args.input_device_index,
                )
            except Exception as e:
                print(f"[error] Recording failed: {e}")
                continue

            # If recording was near-silence or invalid, skip
            if wav_bytes is None:
                continue

            # 3. Send to /chat_audio
            try:
                t0 = time.monotonic()
                response_data = call_chat_audio(
                    args.api_base,
                    wav_bytes,
                    lang,
                    mode,
                    voice_style,
                    debug=args.debug,
                )
                roundtrip_ms = int((time.monotonic() - t0) * 1000)
            except Exception as e:
                print(f"[error] /chat_audio call failed: {e}")
                continue

            print(f"[info] End-to-end interaction time: {roundtrip_ms} ms")

            # 4. Print summary
            print_clean_summary(response_data)

            # 5. Play reply audio (if any)
            play_reply_audio_from_response(response_data)

    except KeyboardInterrupt:
        print("\n[info] KeyboardInterrupt received. Exiting...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

