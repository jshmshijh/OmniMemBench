"""Media processing helpers for multimodal LLM interactions."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}


def get_mime_type(path: str | Path) -> str:
    """Infer MIME type from file extension."""
    ext = Path(path).suffix.lower()
    mime, _ = mimetypes.guess_type(f"file{ext}")
    if mime:
        return mime
    fallbacks = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
    }
    return fallbacks.get(ext, "application/octet-stream")


def encode_image_base64(path: str | Path) -> str:
    """Encode an image file to base64 data URI."""
    path = Path(path)
    mime = get_mime_type(path)
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def is_url(path: str) -> bool:
    """Check if a string is an HTTP(S) URL."""
    return path.startswith("http://") or path.startswith("https://")


def build_image_content(path_or_url: str) -> dict:
    """Build an OpenAI-format image_url content block."""
    if is_url(path_or_url):
        url = path_or_url
    else:
        url = encode_image_base64(path_or_url)
    return {"type": "image_url", "image_url": {"url": url}}


def build_audio_content(path_or_url: str, format: str = "wav") -> dict:
    """Build an OpenAI-format input_audio content block."""
    if is_url(path_or_url):
        data = path_or_url
    else:
        data = base64.b64encode(Path(path_or_url).read_bytes()).decode("utf-8")
    return {"type": "input_audio", "input_audio": {"data": data, "format": format}}


def build_video_content(path_or_url: str) -> dict:
    """Build a video_url content block (for Gemini-style APIs)."""
    if is_url(path_or_url):
        url = path_or_url
    else:
        url = f"file://{Path(path_or_url).resolve()}"
    return {"type": "video_url", "video_url": {"url": url}}


def modality_from_path(path: str | Path) -> str | None:
    """Detect modality from file extension: 'image', 'audio', 'video', or None."""
    ext = Path(path).suffix.lower()
    if ext in SUPPORTED_IMAGE_EXTS:
        return "image"
    elif ext in SUPPORTED_AUDIO_EXTS:
        return "audio"
    elif ext in SUPPORTED_VIDEO_EXTS:
        return "video"
    return None
