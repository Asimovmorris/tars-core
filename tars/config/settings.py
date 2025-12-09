# tars/config/settings.py

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

# Expose BASE_DIR for other modules
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

TARS_PROMPT_PATH = BASE_DIR / "tars" / "config" / "tars_system_prompt.txt"


@dataclass
class Settings:
    # Core OpenAI config
    openai_api_key: str
    openai_model: str = "gpt-4.1-mini"

    # NEW: speech-to-text (Whisper) model
    openai_stt_model: str = "whisper-1"

    # NEW: text-to-speech model
    openai_tts_model: str = "gpt-4o-mini-tts"

    # NEW: default TTS voice
    openai_tts_voice: str = "alloy"

    # NEW: default TTS audio format (must match supported formats in audio.service)
    openai_tts_audio_format: str = "mp3"

    # Database path
    db_path: str = str(BASE_DIR / "tars" / "data" / "tars.db")

    # Quality/latency tuning knobs
    openai_timeout_seconds: float = 60.0   # total request timeout
    openai_max_retries: int = 2           # how many times to retry transient failures


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
        # safeguard: enforce positive
        return value if value > 0 else default
    except ValueError:
        return default


def _parse_int_env(name: str, default: int, min_val: int = 0, max_val: int = 10) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        # safeguard: clamp into sane range
        return max(min_val, min(max_val, value))
    except ValueError:
        return default


def load_settings() -> Settings:
    """
    Load configuration from environment variables (and defaults).
    Raises a RuntimeError if required settings are missing.
    Also ensures DB directory exists and normalizes audio model settings.
    """
    # --- Required: API key ---
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env or environment")

    # --- Chat model ---
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"

    # --- STT model ---
    openai_stt_model = os.getenv("OPENAI_STT_MODEL", "whisper-1").strip() or "whisper-1"

    # --- TTS model ---
    openai_tts_model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip() or "gpt-4o-mini-tts"

    # --- TTS voice ---
    openai_tts_voice = os.getenv("OPENAI_TTS_VOICE", "alloy").strip() or "alloy"

    # --- TTS audio format (normalized + safeguarded) ---
    raw_tts_format = os.getenv("OPENAI_TTS_AUDIO_FORMAT", "mp3").strip().lower() or "mp3"
    allowed_formats = {"mp3", "wav"}
    if raw_tts_format not in allowed_formats:
        # Safeguard: fall back to mp3 if unsupported format is configured
        raw_tts_format = "mp3"

    # --- DB path (optional override) ---
    default_db_path = BASE_DIR / "tars" / "data" / "tars.db"
    db_path_env = os.getenv("TARS_DB_PATH", str(default_db_path)).strip() or str(default_db_path)
    db_path = Path(db_path_env)

    # Ensure data directory exists for DB
    data_dir = db_path.parent
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- Quality / speed tuning knobs ---
    timeout_seconds = _parse_float_env("OPENAI_TIMEOUT_SECONDS", 60.0)
    max_retries = _parse_int_env("OPENAI_MAX_RETRIES", 2, min_val=0, max_val=5)

    settings = Settings(
        openai_api_key=api_key,
        openai_model=openai_model,
        openai_stt_model=openai_stt_model,
        openai_tts_model=openai_tts_model,
        openai_tts_voice=openai_tts_voice,
        openai_tts_audio_format=raw_tts_format,
        db_path=str(db_path),
        openai_timeout_seconds=timeout_seconds,
        openai_max_retries=max_retries,
    )

    return settings
