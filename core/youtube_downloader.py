"""
YouTube audio downloader using yt-dlp.
Downloads audio from a YouTube video and returns the file path.
"""
import os
import yt_dlp

from config import TEMP_AUDIO_DIR


def download_audio(url: str) -> dict:
    """
    Download audio from a YouTube video URL.

    Args:
        url: YouTube video URL.

    Returns:
        dict with keys:
            - audio_path: path to the downloaded audio file
            - title: video title
            - description: video description
            - channel: channel name
            - duration: video duration in seconds
    """
    # Clean up any previous temp audio files
    for f in os.listdir(TEMP_AUDIO_DIR):
        filepath = os.path.join(TEMP_AUDIO_DIR, f)
        if os.path.isfile(filepath):
            os.remove(filepath)

    output_template = os.path.join(TEMP_AUDIO_DIR, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        video_id = info.get("id", "audio")
        audio_path = os.path.join(TEMP_AUDIO_DIR, f"{video_id}.wav")

        # Fallback: find the downloaded file if path doesn't match
        if not os.path.exists(audio_path):
            for f in os.listdir(TEMP_AUDIO_DIR):
                if f.endswith(".wav"):
                    audio_path = os.path.join(TEMP_AUDIO_DIR, f)
                    break

        metadata = {
            "audio_path": audio_path,
            "title": info.get("title", "Unknown"),
            "description": info.get("description", "")[:500],  # Truncate long descriptions
            "channel": info.get("channel", info.get("uploader", "Unknown")),
            "duration": info.get("duration", 0),
            "url": url,
        }

        return metadata


def cleanup_audio(audio_path: str) -> None:
    """Remove the temporary audio file after processing."""
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)
