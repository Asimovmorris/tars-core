# tars/main.py
"""
TARS CLI entrypoint.

Default behavior (unchanged):
- Text in -> TARSCore -> text out

Optional upgrades (disabled by default):
- --voice-input : record mic audio -> STT -> feed into TARSCore
- --speak       : synthesize TARS replies -> playback
- --lang        : STT language hint (e.g., en, es), default: auto-detect
- --device      : input device index (sounddevice)
- --format      : reply audio format (wav recommended if you enabled server-side FX)
- --voice       : OpenAI TTS voice name (server/client may ignore depending on pipeline)

Notes:
- This file is NOT your /chat_audio server. It is a standalone CLI.
- Voice behavior is implemented here only for local testing / development.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from typing import Optional, Tuple

from tars.core.chat import TARSCore

# Optional imports (only needed if voice flags are used)
try:
    import numpy as np
    import sounddevice as sd
    import wave
except Exception:
    np = None  # type: ignore
    sd = None  # type: ignore
    wave = None  # type: ignore

# Use your service layer (you pasted it earlier)
try:
    from tars.audio.service import stt_with_safeguards, tts_with_safeguards
except Exception:
    stt_with_safeguards = None  # type: ignore
    tts_with_safeguards = None  # type: ignore


# -----------------------------
# Recording (minimal, robust)
# -----------------------------

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
CHUNK_SEC = 0.20

# Simple “good enough” VAD (matches your live client spirit, but shorter)
SILENCE_HOLD_SEC = 1.8
MIN_SPEECH_SEC = 0.6
BASE_RMS_THRESHOLD = 180.0
THRESHOLD_MARGIN = 140.0
NOISE_EST_CHUNKS = 10


def _require_voice_stack() -> None:
    """
    Fail fast with a clear message if user requests voice mode without deps.
    """
    missing = []
    if np is None:
        missing.append("numpy")
    if sd is None:
        missing.append("sounddevice")
    if wave is None:
        missing.append("wave(stdlib?)")
    if stt_with_safeguards is None:
        missing.append("tars.audio.service (stt_with_safeguards)")
    if missing:
        raise RuntimeError(
            "Voice features requested but dependencies are missing: "
            + ", ".join(missing)
            + ". Install requirements and ensure tars.audio.service imports correctly."
        )


def _rms_int16(x) -> float:
    if x.size == 0:
        return 0.0
    xf = x.astype("float32")
    return float((xf * xf).mean() ** 0.5)


def record_mic_to_wav_bytes(
    *,
    device_index: Optional[int],
    max_seconds: float = 20.0,
    samplerate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    debug: bool = False,
) -> Optional[bytes]:
    """
    Records from mic until:
      - >= MIN_SPEECH_SEC speech observed
      - followed by SILENCE_HOLD_SEC silence
    Returns WAV bytes or None if no usable speech.
    """
    _require_voice_stack()

    block = max(1, int(CHUNK_SEC * samplerate))

    frames = []
    total_frames = 0
    speech_frames = 0
    last_speech_t = None

    noise_sum = 0.0
    noise_seen = 0
    thr = BASE_RMS_THRESHOLD + THRESHOLD_MARGIN

    start = time.monotonic()
    if device_index is not None:
        print(f"[voice] Using input device index: {device_index}")

    try:
        with sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            dtype="int16",
            device=device_index,
        ) as stream:
            while True:
                chunk, overflowed = stream.read(block)
                _ = overflowed

                frames.append(chunk.copy())
                total_frames += chunk.shape[0]

                mono = chunk.reshape(-1)
                r = _rms_int16(mono)

                # noise estimation (first N chunks)
                if noise_seen < NOISE_EST_CHUNKS:
                    noise_sum += r
                    noise_seen += 1
                    avg = noise_sum / max(1, noise_seen)
                    thr = max(BASE_RMS_THRESHOLD, avg + THRESHOLD_MARGIN)
                else:
                    # slow adaptation when likely silence
                    if r < thr:
                        noise_sum = 0.95 * noise_sum + 0.05 * r
                        avg = noise_sum / max(1, noise_seen)
                        thr = max(BASE_RMS_THRESHOLD, avg + THRESHOLD_MARGIN)

                is_speech = r >= thr
                now = time.monotonic()
                elapsed = now - start

                if debug:
                    print(f"[vad] t={elapsed:5.2f}s rms={r:7.1f} thr={thr:7.1f} speech={is_speech}")

                if is_speech:
                    speech_frames += chunk.shape[0]
                    last_speech_t = now

                if elapsed >= max_seconds:
                    print(f"[voice] Reached max recording ({max_seconds:.1f}s).")
                    break

                if last_speech_t is not None:
                    speech_sec = speech_frames / float(samplerate)
                    silence_since = now - last_speech_t
                    if speech_sec >= MIN_SPEECH_SEC and silence_since >= SILENCE_HOLD_SEC:
                        if debug:
                            print(f"[voice] End-of-speech detected (silence {silence_since:.2f}s).")
                        break

    except Exception as e:
        raise RuntimeError(f"Microphone recording failed: {e}") from e

    speech_sec = speech_frames / float(samplerate) if samplerate > 0 else 0.0
    if speech_sec < MIN_SPEECH_SEC:
        print("[voice] Too little speech detected; try again.")
        return None

    audio_all = np.concatenate(frames, axis=0)

    buf = io.BytesIO()
    try:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # int16
            wf.setframerate(samplerate)
            wf.writeframes(audio_all.tobytes())
    except Exception as e:
        raise RuntimeError(f"Failed to encode WAV: {e}") from e

    return buf.getvalue()


def play_wav_bytes(wav_bytes: bytes) -> None:
    """
    Minimal WAV player (PCM16/mono/stereo OK; uses sounddevice).
    """
    _require_voice_stack()

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        ch = wf.getnchannels()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sw != 2:
        raise RuntimeError(f"Unsupported WAV sampwidth={sw}; expected PCM16.")

    x = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch)

    sd.play(x, samplerate=sr)
    sd.wait()


# -----------------------------
# CLI main
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TARS CLI (text prototype) with optional voice input/output.")
    p.add_argument("--voice-input", action="store_true", help="Use microphone + STT for user input.")
    p.add_argument("--speak", action="store_true", help="Use TTS to speak TARS replies.")
    p.add_argument("--lang", default=None, help="STT language hint (e.g., en, es). Default: auto-detect.")
    p.add_argument("--device", type=int, default=None, help="Microphone device index (sounddevice).")
    p.add_argument("--max-record", type=float, default=20.0, help="Max voice recording seconds (voice-input).")
    p.add_argument("--format", default="wav", choices=["wav", "mp3"], help="TTS output format (wav recommended).")
    p.add_argument("--voice", default=None, help="TTS voice name/preset (OpenAI).")
    p.add_argument("--debug-vad", action="store_true", help="Print VAD diagnostics during voice recording.")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    # Default behavior remains: text prompt
    tars = TARSCore()
    print("TARS (text-only prototype). Type 'exit' to quit.\n")
    print(f"[Session mode: {tars.state.mode.value}]\n")

    if args.voice_input or args.speak:
        # Fail early if dependencies missing
        _require_voice_stack()

    try:
        while True:
            # -------- read user input (text or voice) --------
            try:
                if args.voice_input:
                    print("\n[voice] Press ENTER to record. Type 'q' + ENTER to quit.")
                    cmd = input("[prompt] ").strip().lower()
                    if cmd == "q":
                        print("[Session ended]")
                        break

                    wav_bytes = record_mic_to_wav_bytes(
                        device_index=args.device,
                        max_seconds=args.max_record,
                        debug=args.debug_vad,
                    )
                    if wav_bytes is None:
                        continue

                    # STT
                    user = stt_with_safeguards(
                        audio_bytes=wav_bytes,
                        mime_type="audio/wav",
                        language=args.lang,
                    ).strip()

                    print(f"You (STT): {user}")

                else:
                    user = input("You: ").strip()

            except (EOFError, KeyboardInterrupt):
                print("\n[Session ended]")
                break

            if user.lower() in {"exit", "quit"}:
                print("TARS: Session terminated. Goodbye.")
                break

            # -------- process --------
            try:
                reply = tars.process_user_text(user)
            except RuntimeError as e:
                print(f"TARS (error): {e}")
                break

            print(f"TARS: {reply}\n")

            # -------- optional speak --------
            if args.speak:
                try:
                    audio_bytes = tts_with_safeguards(
                        text=reply,
                        voice=args.voice,
                        audio_format=args.format,
                    )
                    if args.format == "wav":
                        play_wav_bytes(audio_bytes)
                    else:
                        # MP3 playback is intentionally not implemented here to keep it lean.
                        # Use live_voice_chat.py for advanced playback control.
                        print("[warn] MP3 playback not supported in this CLI. Use --format wav.")
                except Exception as e:
                    # Your preference: vocal-first, but if it fails, fall back to default voice route.
                    # Here "default voice" means retry with voice=None (engine default).
                    try:
                        if args.voice is not None:
                            print(f"[audio] Voice failed ({e}). Retrying with default voice...")
                            audio_bytes = tts_with_safeguards(
                                text=reply,
                                voice=None,
                                audio_format="wav",
                            )
                            play_wav_bytes(audio_bytes)
                        else:
                            print(f"[audio] TTS failed: {e}")
                    except Exception as e2:
                        print(f"[audio] TTS fallback failed: {e2}")

    finally:
        tars.close(summary=None)


if __name__ == "__main__":
    main()

