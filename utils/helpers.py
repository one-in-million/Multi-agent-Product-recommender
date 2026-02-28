"""
Shared utility functions.
"""
import re


def extract_video_id(url: str) -> str | None:
    """
    Extract the YouTube video ID from various URL formats.

    Supports:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://youtube.com/embed/VIDEO_ID
        - https://www.youtube.com/v/VIDEO_ID
        - https://youtube.com/shorts/VIDEO_ID

    Returns:
        Video ID string, or None if not a valid YouTube URL.
    """
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def validate_youtube_url(url: str) -> tuple[bool, str]:
    """
    Validate a YouTube URL.

    Returns:
        Tuple of (is_valid, error_message_or_video_id).
    """
    if not url or not isinstance(url, str):
        return False, "URL cannot be empty."

    url = url.strip()

    if not re.match(r"https?://", url):
        return False, "URL must start with http:// or https://"

    video_id = extract_video_id(url)
    if not video_id:
        return False, "Not a valid YouTube URL. Expected format: https://youtube.com/watch?v=VIDEO_ID"

    return True, video_id


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks for embedding.
    Guarantees forward progress to prevent infinite loops.
    """
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary
        if end < len(text):
            last_break = max(
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end),
            )
            # CRITICAL FIX: Only cut at the sentence if it guarantees we 
            # move forward more than the overlap amount!
            if last_break > start + overlap:
                end = last_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Calculate next start
        next_start = end - overlap
        
        # Absolute safety net: ensure 'start' ALWAYS increases
        if next_start <= start:
            start += chunk_size - overlap  # Force a hard cut
        else:
            start = next_start

    return chunks