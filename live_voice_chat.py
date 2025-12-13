# live_voice_chat.py
"""
Live voice chat with TARS over /chat_audio, with dynamic recording and
conversation-aware interaction policy.

    microphone -> WAV bytes -> /chat_audio -> reply text + audio -> speakers

This version upgrades playback so barge-in can work reliably:
- Duck volume on detected user speech during playback
- Stop playback if user continues speaking for > duck window
- Fallback to legacy playsound when decoding is not available

It preserves your overall architecture:
  1) record
  2) call /chat_audio
  3) print summary
  4) play audio
  5) grace feedback

Quit: type 'q' then ENTER at the prompt.
"""

import argparse
import base64
import io
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import sounddevice as sd
import wave

# playsound fallback remains (simple, but can't duck/stop reliably)
from playsound import playsound


# =============================================================================
# Defaults + Constants
# =============================================================================

DEFAULT_API_BASE = os.getenv("TARS_BASE_URL", "http://127.0.0.1:8000")

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1

# Ceiling: absolute never exceed
HARD_MAX_DURATION_SECONDS = 45.0

# Conversational max (still a ceiling, but user configurable)
DEFAULT_MAX_DURATION_SECONDS = 15.0

# Chunk duration for streaming read
CHUNK_DURATION_SEC = 0.2  # 200 ms

# Align with your Android tuning goals
SILENCE_HOLD_SEC = 2.0      # end-of-speech requires >= 2.0s silence
MIN_SPEECH_SEC = 0.65       # require >= 0.65s speech before ending turn
POST_TARS_GRACE_SEC = 10.0  # after TARS speaks, you get ~10s to respond

# VAD / thresholding (recording)
BASE_RMS_THRESHOLD = 180.0
THRESHOLD_MARGIN = 140.0
NOISE_EST_CHUNKS = 10
HYSTERESIS_MARGIN = 30.0

# Payload guards
MAX_WAV_BYTES = 2_000_000  # safety ceiling
REQUEST_MIME_TYPE = "audio/wav"

ALLOWED_MODES = {"analyst", "critic", "synthetic"}
ALLOWED_VOICE_STYLES = {"brief", "default", "story", "technical"}

# =============================================================================
# NEW: Playback + barge-in tuning
# =============================================================================

# Barge-in policy (your preference):
# - Duck for 3 seconds on detected speech
# - If speech continues through duck window -> stop
BARGE_DUCK_SECONDS = 3.0
BARGE_DUCK_GAIN = 0.20  # 20% volume during duck
BARGE_MIN_SPEECH_HOLD_SEC = 0.35  # require sustained speech this long to count
BARGE_MONITOR_BLOCK_SEC = 0.10    # mic monitoring resolution while playing

# Playback output sample rate target (we resample if needed)
PLAYBACK_TARGET_SR = 24000  # common for TTS wav; if mp3 decoded, may differ

# If audio missing from server response, try local offline fallback (optional)
ENABLE_OFFLINE_TTS_FALLBACK = os.getenv("TARS_OFFLINE_TTS_FALLBACK", "0").strip() == "1"

# Hint to server (it may ignore safely). Prefer wav for best FX + best duck/stop.
PREFERRED_REPLY_AUDIO_FORMAT = os.getenv("TARS_REPLY_AUDIO_FORMAT", "wav").strip().lower()  # wav|mp3

# =============================================================================
# Session persistence / logs
# =============================================================================

STATE_DIR = os.path.join(os.path.expanduser("~"), ".tars_voice")
os.makedirs(STATE_DIR, exist_ok=True)

SESSION_ID_FILE = os.path.join(STATE_DIR, "session_id.txt")
TRACE_DIR = os.path.join(STATE_DIR, "traces")
os.makedirs(TRACE_DIR, exist_ok=True)


# =============================================================================
# Helpers: normalization
# =============================================================================

def normalize_lang(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    cleaned = lang.strip()
    if not cleaned:
        return None
    return cleaned.lower()


def normalize_mode(mode: Optional[str]) -> Optional[str]:
    if not mode:
        return None
    cleaned = mode.strip().lower()
    if not cleaned:
        return None

    if cleaned in {"analyst", "analysis"}:
        cleaned = "analyst"
    elif cleaned in {"critic", "devil", "devils_advocate", "devil's advocate"}:
        cleaned = "critic"
    elif cleaned in {"synthetic", "synth", "synthesizer"}:
        cleaned = "synthetic"

    if cleaned not in ALLOWED_MODES:
        print(f"[warn] Unrecognized mode '{mode}'. Allowed: {sorted(ALLOWED_MODES)}. Ignoring.")
        return None
    return cleaned


def normalize_voice_style(style: Optional[str]) -> Optional[str]:
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

    print(f"[warn] Unrecognized voice style '{style}'. Allowed: {sorted(ALLOWED_VOICE_STYLES)}. Ignoring.")
    return None


def safe_trim_slash(url: str) -> str:
    return url.rstrip("/")


# =============================================================================
# Session persistence
# =============================================================================

def get_or_create_session_id() -> str:
    if os.path.exists(SESSION_ID_FILE):
        try:
            sid = open(SESSION_ID_FILE, "r", encoding="utf-8").read().strip()
            if sid:
                return sid
        except Exception:
            pass

    sid = str(uuid.uuid4())
    try:
        with open(SESSION_ID_FILE, "w", encoding="utf-8") as f:
            f.write(sid)
    except Exception:
        pass
    return sid


def sanitize_session_id(session_id: str) -> str:
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_"))
    return safe or "default"


def trace_path_for_session(session_id: str) -> str:
    safe = sanitize_session_id(session_id)
    return os.path.join(TRACE_DIR, f"{safe}.jsonl")


def append_trace(session_id: str, record: Dict[str, Any]) -> None:
    record = dict(record)
    record.pop("audio_base64", None)
    record.pop("request_audio_base64", None)
    record["ts"] = time.time()

    path = trace_path_for_session(session_id)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# =============================================================================
# Device helper
# =============================================================================

def list_devices_and_exit() -> int:
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


# =============================================================================
# Health check
# =============================================================================

def health_check(api_base: str) -> Dict[str, Any]:
    url = f"{safe_trim_slash(api_base)}/health"
    resp = requests.get(url, timeout=5)
    if resp.status_code != 200:
        raise RuntimeError(f"/health returned status {resp.status_code}: {resp.text!r}")
    try:
        return resp.json()
    except Exception:
        return {"status": "ok", "raw": resp.text}


# =============================================================================
# VAD / recording utilities
# =============================================================================

def compute_rms_int16(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    xf = x.astype(np.float32)
    return float(np.sqrt(np.mean(xf * xf)))


@dataclass
class VadState:
    noise_floor_rms: float = 0.0
    threshold_rms: float = 0.0
    noise_chunks_seen: int = 0
    last_class: str = "silence"  # or "speech"


def update_vad_threshold(vs: VadState, chunk_rms: float) -> None:
    if vs.noise_chunks_seen < NOISE_EST_CHUNKS:
        vs.noise_floor_rms += chunk_rms
        vs.noise_chunks_seen += 1
        avg = vs.noise_floor_rms / max(1, vs.noise_chunks_seen)
        vs.threshold_rms = max(BASE_RMS_THRESHOLD, avg + THRESHOLD_MARGIN)
        return

    if chunk_rms < vs.threshold_rms:
        vs.noise_floor_rms = (0.95 * vs.noise_floor_rms) + (0.05 * chunk_rms)
        vs.threshold_rms = max(BASE_RMS_THRESHOLD, vs.noise_floor_rms + THRESHOLD_MARGIN)


def vad_classify(vs: VadState, chunk_rms: float) -> str:
    thr = vs.threshold_rms if vs.threshold_rms > 0 else BASE_RMS_THRESHOLD + THRESHOLD_MARGIN

    if vs.last_class == "speech":
        effective_thr = max(0.0, thr - HYSTERESIS_MARGIN)
    else:
        effective_thr = thr + HYSTERESIS_MARGIN

    cls = "speech" if chunk_rms >= effective_thr else "silence"
    vs.last_class = cls
    return cls


def record_audio_to_wav_bytes_dynamic(
    max_duration_seconds: float,
    samplerate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    device: Optional[int] = None,
    debug_vad: bool = False,
) -> Tuple[Optional[bytes], Dict[str, Any]]:
    if max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be > 0")

    if max_duration_seconds > HARD_MAX_DURATION_SECONDS:
        max_duration_seconds = HARD_MAX_DURATION_SECONDS

    if device is not None:
        print(f"[record] Using input device index: {device}")

    print(
        f"[record] Recording (up to {max_duration_seconds:.1f}s). "
        f"Stop-on-silence: {SILENCE_HOLD_SEC:.1f}s, "
        f"min speech: {MIN_SPEECH_SEC:.2f}s, "
        f"chunk: {CHUNK_DURATION_SEC:.2f}s"
    )

    block_size = max(1, int(CHUNK_DURATION_SEC * samplerate))

    frames: List[np.ndarray] = []
    total_frames = 0
    speech_frames = 0
    overflow_count = 0
    last_speech_time: Optional[float] = None

    vs = VadState(noise_floor_rms=0.0, threshold_rms=max(BASE_RMS_THRESHOLD, BASE_RMS_THRESHOLD + THRESHOLD_MARGIN))
    start_time = time.monotonic()

    try:
        with sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            dtype="int16",
            device=device,
        ) as stream:
            while True:
                chunk, overflowed = stream.read(block_size)
                if overflowed:
                    overflow_count += 1

                frames.append(chunk.copy())
                total_frames += chunk.shape[0]

                audio_np = chunk.reshape(-1)
                r = compute_rms_int16(audio_np)

                update_vad_threshold(vs, r)
                cls = vad_classify(vs, r)

                now = time.monotonic()
                elapsed = now - start_time

                if debug_vad:
                    print(f"[vad] t={elapsed:5.2f}s rms={r:7.1f} thr={vs.threshold_rms:7.1f} cls={cls}")

                if cls == "speech":
                    speech_frames += chunk.shape[0]
                    last_speech_time = now

                if elapsed >= max_duration_seconds:
                    print(f"[record] Reached max duration ({max_duration_seconds:.1f}s). Stopping.")
                    break

                if last_speech_time is not None:
                    speech_sec = speech_frames / float(samplerate)
                    silence_since = now - last_speech_time

                    if speech_sec >= MIN_SPEECH_SEC and silence_since >= SILENCE_HOLD_SEC:
                        print(
                            f"[record] End of speech detected at t≈{elapsed:.1f}s "
                            f"(silence {silence_since:.1f}s)."
                        )
                        break

    except Exception as e:
        raise RuntimeError(f"Failed to record audio from microphone: {e}") from e

    if total_frames == 0:
        return None, {
            "error": "no_frames",
            "total_frames": 0,
            "speech_frames": 0,
            "overflow_count": overflow_count,
        }

    total_duration = total_frames / float(samplerate)
    speech_duration = speech_frames / float(samplerate)

    silence_after = 0.0
    if last_speech_time is not None:
        silence_after = (time.monotonic() - last_speech_time)

    stats = {
        "total_duration_sec": total_duration,
        "speech_duration_sec": speech_duration,
        "silence_after_speech_sec": silence_after,
        "noise_floor_rms_est": float(vs.noise_floor_rms / max(1, min(vs.noise_chunks_seen, NOISE_EST_CHUNKS))) if vs.noise_chunks_seen else 0.0,
        "threshold_rms_final": float(vs.threshold_rms),
        "overflow_count": overflow_count,
        "samplerate": samplerate,
        "channels": channels,
        "block_size": block_size,
    }

    print(
        f"[record] Total recorded: {total_duration:.2f}s | "
        f"estimated speech: {speech_duration:.2f}s | "
        f"overflow: {overflow_count}"
    )

    if speech_duration < MIN_SPEECH_SEC:
        print("[warn] Too little speech detected; skipping utterance.")
        return None, stats

    audio_all = np.concatenate(frames, axis=0)

    buf = io.BytesIO()
    try:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_all.tobytes())
    except Exception as e:
        raise RuntimeError(f"Failed to encode WAV: {e}") from e

    wav_bytes = buf.getvalue()

    if len(wav_bytes) > MAX_WAV_BYTES:
        print(f"[warn] WAV too large ({len(wav_bytes)} bytes). Consider shorter duration.")
        stats["wav_bytes_warning"] = True

    return wav_bytes, stats


# =============================================================================
# Networking: /chat_audio with retries
# =============================================================================

def post_json_with_retry(
    url: str,
    payload: Dict[str, Any],
    timeout_sec: int,
    attempts: int = 2,
    backoff_base: float = 0.35,
) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout_sec)
            return resp
        except Exception as e:
            last_exc = e
            if attempt < attempts:
                sleep_s = backoff_base * attempt
                print(f"[warn] HTTP attempt {attempt}/{attempts} failed: {e}. Retrying in {sleep_s:.2f}s...")
                time.sleep(sleep_s)
            else:
                break
    raise RuntimeError(f"HTTP failed after {attempts} attempts: {last_exc}")


def call_chat_audio(
    api_base: str,
    wav_bytes: bytes,
    lang: Optional[str],
    mode: Optional[str],
    voice_style: Optional[str],
    session_id: str,
    debug: bool = False,
    timeout_sec: int = 90,
    prefer_reply_format: Optional[str] = None,
    prefer_voice: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send WAV audio bytes to /chat_audio and return JSON response.

    Includes:
      - session_id (Android parity)
      - channel="VOICE"
      - client_request_id (helps correlate logs)

    Also includes *optional* server hints:
      - reply_audio_format (wav/mp3)  [WAV-first by default]
      - reply_voice (OpenAI voice preset)
    """
    client_request_id = str(uuid.uuid4())
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    url = f"{safe_trim_slash(api_base)}/chat_audio"

    # Resolve preferred reply format (wav-first default)
    fmt = (prefer_reply_format or PREFERRED_REPLY_AUDIO_FORMAT or "wav").strip().lower()
    if fmt not in {"wav", "mp3"}:
        fmt = "wav"

    payload: Dict[str, Any] = {
        "audio_base64": audio_b64,
        "mime_type": REQUEST_MIME_TYPE,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "language": lang,
        "mode": mode,
        "voice_style": voice_style,
        "session_id": session_id,
        "channel": "VOICE",
        "client_request_id": client_request_id,

        # Server hints (safe if ignored)
        "reply_audio_format": fmt,
        "reply_voice": (prefer_voice.strip() if isinstance(prefer_voice, str) and prefer_voice.strip() else None),
    }

    payload = {k: v for k, v in payload.items() if v is not None}

    if debug:
        dbg_payload = dict(payload)
        dbg_payload["audio_base64"] = f"<omitted len={len(audio_b64)}>"
        print("[debug] POST /chat_audio payload:")
        print(json.dumps(dbg_payload, indent=2, ensure_ascii=False))

    t0 = time.monotonic()
    resp = post_json_with_retry(url, payload, timeout_sec=timeout_sec, attempts=2)
    latency_ms = int((time.monotonic() - t0) * 1000)

    print(f"[http] Status: {resp.status_code} | latency≈{latency_ms}ms")

    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Response not valid JSON: {e}. Raw text={resp.text!r}") from e

    if resp.status_code != 200:
        print("[error] Non-200 response JSON:")
        safe_data = dict(data) if isinstance(data, dict) else {"raw": data}
        if "audio_base64" in safe_data:
            safe_data["audio_base64"] = "<omitted>"
        print(json.dumps(safe_data, indent=2, ensure_ascii=False))
        raise RuntimeError(f"Server returned status {resp.status_code}")

    # client-side annotations
    if isinstance(data, dict):
        data["_client_request_id"] = client_request_id
        data["_client_latency_ms"] = latency_ms

    if debug and isinstance(data, dict):
        dbg = dict(data)
        if "audio_base64" in dbg:
            dbg["audio_base64"] = f"<omitted len={len(dbg['audio_base64'] or '')}>"
        print("[debug] Response JSON (audio omitted):")
        print(json.dumps(dbg, indent=2, ensure_ascii=False))

    return data


# =============================================================================
# Playback: decoding, ducking, barge-in
# =============================================================================

def _try_import_pydub():
    try:
        from pydub import AudioSegment  # type: ignore
        return AudioSegment
    except Exception:
        return None


def decode_wav_to_float32(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Decode PCM WAV bytes -> (mono float32 [-1..1], sample_rate).
    """
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        ch = wf.getnchannels()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sw != 2:
        raise RuntimeError(f"WAV sampwidth={sw} not supported (expected 2 bytes PCM16).")

    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, int(sr)


def decode_mp3_to_float32(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Decode MP3 using pydub if available; otherwise raise.
    """
    AudioSegment = _try_import_pydub()
    if AudioSegment is None:
        raise RuntimeError("pydub not installed; cannot decode mp3 for duck/stop playback.")
    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    sr = seg.frame_rate
    ch = seg.channels
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)

    # pydub returns interleaved samples if stereo
    if ch > 1:
        samples = samples.reshape(-1, ch).mean(axis=1)

    # Convert to [-1,1] based on sample width
    maxv = float(1 << (8 * seg.sample_width - 1))
    x = samples / maxv
    x = np.clip(x, -1.0, 1.0)
    return x.astype(np.float32), int(sr)


def simple_resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """
    Lightweight resample via linear interpolation (good enough for speech).
    Avoids heavy deps; keeps code portable.
    """
    if sr_in == sr_out:
        return x
    if x.size == 0:
        return x
    ratio = sr_out / float(sr_in)
    n_out = int(round(x.size * ratio))
    if n_out <= 1:
        return x[:1]
    xi = np.linspace(0.0, x.size - 1, num=n_out, dtype=np.float32)
    x0 = np.floor(xi).astype(np.int64)
    x1 = np.minimum(x0 + 1, x.size - 1)
    a = xi - x0.astype(np.float32)
    y = (1.0 - a) * x[x0] + a * x[x1]
    return y.astype(np.float32)


@dataclass
class PlaybackStats:
    method: str
    format: str
    sr: int
    seconds: float
    ducked: bool
    stopped_by_barge: bool
    decode_ok: bool


def _infer_format_from_mime_or_bytes(audio_mime: str, audio_bytes: bytes) -> str:
    m = (audio_mime or "").strip().lower()
    if m in {"audio/wav", "audio/x-wav", "audio/wave"}:
        return "wav"
    if m in {"audio/mpeg", "audio/mp3"}:
        return "mp3"
    # Fallback heuristic
    return "wav" if audio_bytes[:4] == b"RIFF" else "mp3"


class AudioPlayer:
    """
    Non-blocking player that supports:
    - stop()
    - duck(gain, seconds)
    and cooperates with a mic-monitor thread for barge-in.

    Uses sounddevice OutputStream for real control.
    """
    def __init__(self, samplerate: int):
        self.samplerate = int(samplerate)
        self._stop = False
        self._duck_gain = 1.0
        self._duck_until = 0.0

    def stop(self) -> None:
        self._stop = True

    def duck(self, gain: float, seconds: float) -> None:
        self._duck_gain = float(max(0.0, min(1.0, gain)))
        self._duck_until = time.monotonic() + float(max(0.0, seconds))

    def _current_gain(self) -> float:
        if time.monotonic() <= self._duck_until:
            return self._duck_gain
        return 1.0

    def play_blocking(self, x: np.ndarray) -> None:
        """
        Plays mono float32 in [-1..1]. Blocks until completion or stop().
        """
        if x.size == 0:
            return

        block = 1024
        idx = 0

        try:
            with sd.OutputStream(
                samplerate=self.samplerate,
                channels=1,
                dtype="float32",
            ) as stream:
                while idx < x.size and not self._stop:
                    chunk = x[idx: idx + block]
                    g = self._current_gain()
                    out = (chunk * g).astype(np.float32)

                    # Hard limiter to protect ears
                    out = np.clip(out, -0.95, 0.95)

                    stream.write(out.reshape(-1, 1))
                    idx += block
        except Exception as e:
            raise RuntimeError(f"Playback failed (sounddevice): {e}") from e


def monitor_mic_for_barge_in(
    player: AudioPlayer,
    input_device_index: Optional[int],
    base_threshold: float,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Monitors microphone RMS while TARS audio is playing to implement barge-in.

    Policy:
      - On first sustained speech: duck for BARGE_DUCK_SECONDS
      - If speech continues past duck window: stop playback

    Returns diagnostics dict.
    """
    block_size = max(1, int(BARGE_MONITOR_BLOCK_SEC * DEFAULT_SAMPLE_RATE))
    vs = VadState(noise_floor_rms=0.0, threshold_rms=max(base_threshold, base_threshold + THRESHOLD_MARGIN))

    speech_started_at: Optional[float] = None
    first_barge_at: Optional[float] = None
    ducked = False
    stopped = False

    t0 = time.monotonic()

    try:
        with sd.InputStream(
            samplerate=DEFAULT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            device=input_device_index,
        ) as stream:
            while True:
                if player._stop:
                    break

                chunk, overflowed = stream.read(block_size)
                _ = overflowed  # reserved for diagnostics

                rms = compute_rms_int16(chunk.reshape(-1))
                update_vad_threshold(vs, rms)
                cls = vad_classify(vs, rms)

                now = time.monotonic()

                if debug:
                    print(f"[barge] rms={rms:7.1f} thr={vs.threshold_rms:7.1f} cls={cls}")

                if cls == "speech":
                    if speech_started_at is None:
                        speech_started_at = now
                    speech_hold = now - speech_started_at

                    # require sustained speech to count as barge (prevents false triggers)
                    if speech_hold >= BARGE_MIN_SPEECH_HOLD_SEC and first_barge_at is None:
                        first_barge_at = now
                        ducked = True
                        player.duck(BARGE_DUCK_GAIN, BARGE_DUCK_SECONDS)
                        if debug:
                            print(f"[barge] Ducking audio for {BARGE_DUCK_SECONDS:.1f}s (gain={BARGE_DUCK_GAIN:.2f}).")

                    # if barge already happened, check if user continues through duck window
                    if first_barge_at is not None:
                        if (now - first_barge_at) >= BARGE_DUCK_SECONDS:
                            stopped = True
                            player.stop()
                            if debug:
                                print("[barge] Continued speech detected -> stopping playback.")
                            break
                else:
                    speech_started_at = None

                # Safety: don't monitor forever if something goes weird
                if (now - t0) > 60.0:
                    if debug:
                        print("[barge] Monitor safety timeout reached.")
                    break

    except Exception as e:
        # Non-fatal: if mic monitor fails, playback still works.
        return {
            "barge_monitor_error": str(e),
            "ducked": ducked,
            "stopped": stopped,
            "threshold_final": float(vs.threshold_rms),
        }

    return {
        "ducked": ducked,
        "stopped": stopped,
        "threshold_final": float(vs.threshold_rms),
        "noise_floor_est": float(vs.noise_floor_rms / max(1, min(vs.noise_chunks_seen, NOISE_EST_CHUNKS))) if vs.noise_chunks_seen else 0.0,
    }


def play_reply_audio_from_response(
    response_data: Dict[str, Any],
    input_device_index: Optional[int],
    debug: bool = False,
) -> Tuple[bool, PlaybackStats, Dict[str, Any]]:
    """
    Returns (played_ok, playback_stats, barge_stats).

    Behavior:
      - Prefer controlled playback (sounddevice) when we can decode to PCM.
      - If response is MP3 and decoding isn't available, fallback to playsound
        (duck/stop not supported in that fallback path).
    """
    audio_b64 = (response_data.get("audio_base64") or "").strip()
    audio_mime = (response_data.get("audio_mime_type") or "").strip()

    if not audio_b64:
        print("[audio] No audio in response (TTS may have failed).")
        st = PlaybackStats(
            method="none",
            format="unknown",
            sr=0,
            seconds=0.0,
            ducked=False,
            stopped_by_barge=False,
            decode_ok=False,
        )
        return False, st, {}

    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except Exception as e:
        print(f"[warn] Failed to decode audio_base64: {e}")
        st = PlaybackStats(
            method="none",
            format="unknown",
            sr=0,
            seconds=0.0,
            ducked=False,
            stopped_by_barge=False,
            decode_ok=False,
        )
        return False, st, {}

    fmt = _infer_format_from_mime_or_bytes(audio_mime, audio_bytes)

    # Try to decode to PCM float32 so we can duck/stop
    decode_ok = True
    try:
        if fmt == "wav":
            x, sr = decode_wav_to_float32(audio_bytes)
        elif fmt == "mp3":
            x, sr = decode_mp3_to_float32(audio_bytes)
        else:
            raise RuntimeError(f"Unsupported response audio format: {fmt}")
    except Exception as e:
        decode_ok = False
        if debug:
            print(f"[debug] Decode failed for fmt={fmt}: {e}")

        # Fallback path: playsound (blocks; cannot duck/stop)
        suffix = ".wav" if fmt == "wav" else ".mp3"
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="tars_reply_", suffix=suffix)
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)

            print("[play] (fallback) Playing reply audio…")
            playsound(tmp_path)
            try:
                os.remove(tmp_path)
            except Exception:
                pass

            seconds_est = max(0.0, len(audio_bytes) / 16000.0)  # very rough
            st = PlaybackStats(
                method="playsound_fallback",
                format=fmt,
                sr=0,
                seconds=seconds_est,
                ducked=False,
                stopped_by_barge=False,
                decode_ok=False,
            )
            return True, st, {}

        except Exception as e2:
            print(f"[warn] Fallback playback failed: {e2}")
            st = PlaybackStats(
                method="none",
                format=fmt,
                sr=0,
                seconds=0.0,
                ducked=False,
                stopped_by_barge=False,
                decode_ok=False,
            )
            return False, st, {}

    # Controlled playback path
    if sr <= 0:
        sr = PLAYBACK_TARGET_SR
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)

    # Resample to stable target SR for OutputStream if needed
    target_sr = PLAYBACK_TARGET_SR
    if sr != target_sr:
        x = simple_resample_linear(x, sr_in=sr, sr_out=target_sr)
        sr = target_sr

    seconds = float(x.size / float(sr)) if sr > 0 else 0.0

    player = AudioPlayer(samplerate=sr)

    # Start mic monitor in parallel (barge-in)
    barge_stats: Dict[str, Any] = {}
    import threading

    def _monitor():
        nonlocal barge_stats
        barge_stats = monitor_mic_for_barge_in(
            player=player,
            input_device_index=input_device_index,
            base_threshold=BASE_RMS_THRESHOLD,
            debug=debug,
        )

    barge_thread = threading.Thread(target=_monitor, daemon=True)
    barge_thread.start()

    if debug:
        print(f"[debug] Playing decoded audio via sounddevice: fmt={fmt} sr={sr} seconds≈{seconds:.2f} mime={audio_mime!r}")

    ok = True
    try:
        print("[play] Playing reply audio… (barge-in enabled)")
        player.play_blocking(x)
    except Exception as e:
        print(f"[warn] Audio playback failed: {e}")
        ok = False

    # give monitor a moment to exit cleanly
    try:
        player.stop()
    except Exception:
        pass

    try:
        barge_thread.join(timeout=1.0)
    except Exception:
        pass

    ducked = bool(barge_stats.get("ducked", False))
    stopped_by_barge = bool(barge_stats.get("stopped", False))

    st = PlaybackStats(
        method="sounddevice_pcm",
        format=fmt,
        sr=sr,
        seconds=seconds,
        ducked=ducked,
        stopped_by_barge=stopped_by_barge,
        decode_ok=decode_ok,
    )
    return ok, st, barge_stats


def offline_tts_speak(text: str) -> bool:
    """
    Optional local fallback that still produces voice when server TTS fails.
    Requires: pyttsx3 (optional).
    """
    if not ENABLE_OFFLINE_TTS_FALLBACK:
        return False

    try:
        import pyttsx3  # type: ignore
    except Exception:
        print("[audio] Offline fallback requested but pyttsx3 not installed.")
        return False

    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"[audio] Offline TTS fallback failed: {e}")
        return False


# =============================================================================
# Pretty summary
# =============================================================================

def print_clean_summary(response_data: Dict[str, Any]) -> None:
    reply = response_data.get("reply", "")
    mode = response_data.get("mode", "")
    transcript = response_data.get("transcript", "")
    error = response_data.get("error")
    latency_ms = response_data.get("latency_ms")

    client_latency = response_data.get("_client_latency_ms")
    request_id = response_data.get("_client_request_id")

    audio_b64 = response_data.get("audio_base64") or ""
    audio_len = len(audio_b64)
    audio_mime = response_data.get("audio_mime_type") or ""

    print("\n[summary]")
    print(f"  Mode:        {mode!r}")
    print(f"  Transcript:  {transcript!r}")
    print(f"  Reply:       {reply!r}")
    print(f"  Latency(ms): server={latency_ms} | client≈{client_latency}")
    print(f"  Error:       {error}")
    print(f"  Audio:       {'present' if audio_len > 0 else 'none'} (b64len={audio_len})")
    print(f"  AudioMime:   {audio_mime!r}")
    print(f"  ClientReqId: {request_id}")
    print()


# =============================================================================
# Conversational grace (post-TARS listening readiness)
# =============================================================================

def conversational_grace_feedback(grace_sec: float) -> None:
    print(f"[floor] You have ~{grace_sec:.0f}s to respond without rushing.")
    print("[floor] Press ENTER when ready to speak again.\n")


# =============================================================================
# Main
# =============================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Live voice chat client for TARS /chat_audio endpoint (dynamic recording + adaptive silence)."
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"Base URL for TARS API (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Optional STT language hint (e.g. 'en', 'es'). If omitted, STT may auto-detect.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Optional reasoning mode: 'analyst', 'critic', 'synthetic'.",
    )
    parser.add_argument(
        "--voice-style",
        default=None,
        help="Optional voice style: 'brief', 'story', 'technical', 'default'.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_MAX_DURATION_SECONDS,
        help=(
            f"Max recording duration in seconds (default {DEFAULT_MAX_DURATION_SECONDS}). "
            f"Hard max is {HARD_MAX_DURATION_SECONDS}."
        ),
    )
    parser.add_argument(
        "--input-device-index",
        type=int,
        default=None,
        help="Input device index for microphone. Use --list-devices to inspect.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose diagnostics (payload/response; audio omitted).",
    )
    parser.add_argument(
        "--debug-vad",
        action="store_true",
        help="Print per-chunk VAD diagnostics (very verbose).",
    )
    parser.add_argument(
        "--no-health-per-turn",
        action="store_true",
        help="Skip health check before each turn (faster).",
    )
    parser.add_argument(
        "--health-once",
        action="store_true",
        help="Only do health check once at startup (default behavior).",
    )
    parser.add_argument(
        "--prefer-reply-format",
        default=PREFERRED_REPLY_AUDIO_FORMAT,
        help="Hint to server: preferred reply audio format (wav or mp3). Server may ignore safely.",
    )
    parser.add_argument(
        "--prefer-voice",
        default=None,
        help="Hint to server: preferred voice name. If TTS fails, server may fallback.",
    )

    args = parser.parse_args(argv)

    if args.list_devices:
        return list_devices_and_exit()

    api_base = args.api_base
    lang = normalize_lang(args.lang)
    mode = normalize_mode(args.mode)
    voice_style = normalize_voice_style(args.voice_style)

    if args.duration <= 0:
        print("[warn] duration must be > 0. Using default.")
        max_duration = DEFAULT_MAX_DURATION_SECONDS
    else:
        max_duration = min(args.duration, HARD_MAX_DURATION_SECONDS)

    session_id = get_or_create_session_id()
    print(f"[session] session_id={session_id}")

    try:
        h = health_check(api_base)
        print(f"[health] OK. Response: {h}")
    except Exception as e:
        print(f"[fatal] Health check failed: {e}")
        return 1

    print("\n=== Live voice chat with TARS (adaptive recording) ===")
    print(f"API base:      {safe_trim_slash(api_base)}")
    print(f"Language hint: {lang or 'auto-detect'}")
    print(f"Mode hint:     {mode or 'current TARS mode'}")
    print(f"Voice style:   {voice_style or 'server-default'}")
    print(f"Prefer format: {args.prefer_reply_format!r} (server may ignore)")
    if args.input_device_index is not None:
        print(f"Input device:  index {args.input_device_index}")
    else:
        print("Input device:  default (system-selected)")
    print(
        f"Max duration:  {max_duration:.1f}s | Silence stop: {SILENCE_HOLD_SEC:.1f}s | "
        f"Min speech: {MIN_SPEECH_SEC:.2f}s | Chunk: {CHUNK_DURATION_SEC:.2f}s"
    )
    print(f"Barge-in: duck {BARGE_DUCK_SECONDS:.1f}s @ gain={BARGE_DUCK_GAIN:.2f} then stop if speech continues.")
    if ENABLE_OFFLINE_TTS_FALLBACK:
        print("Offline TTS fallback: ENABLED (pyttsx3).")
    else:
        print("Offline TTS fallback: disabled (set TARS_OFFLINE_TTS_FALLBACK=1 to enable).")
    print("\nPress ENTER to start recording a message.")
    print("Type 'q' and press ENTER to quit.\n")

    try:
        while True:
            user_input = input("[prompt] ENTER=Speak | q=Quit: ").strip()
            if user_input.lower() == "q":
                print("[info] Quitting.")
                break

            if not args.no_health_per_turn and not args.health_once:
                try:
                    _ = health_check(api_base)
                except Exception as e:
                    print(f"[warn] Per-turn health check failed: {e}")

            # 1) Record audio with adaptive VAD
            try:
                wav_bytes, rec_stats = record_audio_to_wav_bytes_dynamic(
                    max_duration_seconds=max_duration,
                    samplerate=DEFAULT_SAMPLE_RATE,
                    channels=DEFAULT_CHANNELS,
                    device=args.input_device_index,
                    debug_vad=args.debug_vad,
                )
            except Exception as e:
                print(f"[error] Recording failed: {e}")
                append_trace(session_id, {"type": "record_error", "error": str(e)})
                continue

            if wav_bytes is None:
                append_trace(session_id, {"type": "record_skip", "reason": "too_little_speech_or_silence", "rec_stats": rec_stats})
                continue

            wav_len = len(wav_bytes)
            if wav_len > MAX_WAV_BYTES:
                print(f"[warn] Large WAV payload ({wav_len} bytes). Might increase latency.")
            append_trace(session_id, {"type": "record_ok", "wav_bytes": wav_len, "rec_stats": rec_stats})

            # 2) Call /chat_audio
            print("[tars] Thinking…")
            try:
                response_data = call_chat_audio(
                    api_base=api_base,
                    wav_bytes=wav_bytes,
                    lang=lang,
                    mode=mode,
                    voice_style=voice_style,
                    session_id=session_id,
                    debug=args.debug,
                    timeout_sec=90,
                    prefer_reply_format=args.prefer_reply_format,
                    prefer_voice=args.prefer_voice,
                )
            except Exception as e:
                print(f"[error] /chat_audio failed: {e}")
                append_trace(session_id, {"type": "chat_audio_error", "error": str(e), "rec_stats": rec_stats})
                continue

            # Infer fmt for logging (optional)
            try:
                audio_b64 = (response_data.get("audio_base64") or "").strip()
                audio_mime = (response_data.get("audio_mime_type") or "").strip()
                fmt_inferred = "unknown"
                if audio_b64:
                    ab = base64.b64decode(audio_b64, validate=False)
                    fmt_inferred = _infer_format_from_mime_or_bytes(audio_mime, ab)
            except Exception:
                fmt_inferred = "unknown"

            append_trace(session_id, {
                "type": "chat_audio_ok",
                "transcript": response_data.get("transcript", ""),
                "reply": response_data.get("reply", ""),
                "mode": response_data.get("mode", ""),
                "latency_ms": response_data.get("latency_ms"),
                "client_latency_ms": response_data.get("_client_latency_ms"),
                "error": response_data.get("error"),
                "audio_present": bool((response_data.get("audio_base64") or "").strip()),
                "audio_mime_type": response_data.get("audio_mime_type"),
                "audio_format_inferred": fmt_inferred,
            })

            # 3) Print summary
            print_clean_summary(response_data)

            # 4) Play audio (vocal-first)
            played_ok, pstats, bstats = play_reply_audio_from_response(
                response_data=response_data,
                input_device_index=args.input_device_index,
                debug=args.debug,
            )

            append_trace(session_id, {
                "type": "playback",
                "played_ok": played_ok,
                "playback_method": pstats.method,
                "audio_format": pstats.format,
                "playback_sr": pstats.sr,
                "seconds": pstats.seconds,
                "ducked": pstats.ducked,
                "stopped_by_barge": pstats.stopped_by_barge,
                "decode_ok": pstats.decode_ok,
                "barge_stats": bstats,
            })

            if not played_ok:
                # Fallback: try offline speech (still voice, not text-only)
                reply_text = (response_data.get("reply") or "").strip()
                if reply_text:
                    ok_local = offline_tts_speak(reply_text)
                    if ok_local:
                        print("[audio] Used offline TTS fallback.")
                    else:
                        print("[audio] (degraded) No playable audio, and offline fallback unavailable.")
                else:
                    print("[audio] (degraded) No reply text to speak.")

            # 5) Post-TARS grace window
            conversational_grace_feedback(POST_TARS_GRACE_SEC)

    except KeyboardInterrupt:
        print("\n[info] KeyboardInterrupt received. Exiting…")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
