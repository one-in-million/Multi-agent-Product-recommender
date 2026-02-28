"""
Audio transcription using OpenAI Whisper (local, CPU-based).
"""
import whisper

from config import WHISPER_MODEL

# Module-level model cache
_model = None


def _get_model():
    """Load and cache the Whisper model."""
    global _model
    if _model is None:
        print(f"[Transcriber] Loading Whisper '{WHISPER_MODEL}' model (first time may take a moment)...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("[Transcriber] Model loaded successfully.")
    return _model


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file to text using Whisper.

    Args:
        audio_path: Path to the audio file (WAV format).

    Returns:
        Full transcript as a string.
    """
    model = _get_model()
    print(f"[Transcriber] Transcribing audio: {audio_path}")

    result = model.transcribe(
        audio_path,
        language="en",
        fp16=False,  # CPU mode — must disable fp16
    )

    transcript = result.get("text", "").strip()
    print(f"[Transcriber] Transcription complete. Length: {len(transcript)} chars.")
    return transcript
