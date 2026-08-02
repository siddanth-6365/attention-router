"""Image understanding and voice-note transcription, cached on disk.

Eight of the 110 messages to route are voice notes with an empty
`message_text`, so transcription is required rather than an enhancement.
Fifteen more carry image posters whose text exists only in pixels.

Both passes write to `cache/media_cache.json`, keyed by `media_id`. A populated
cache is committed with the solution, which means a rerun performs zero media
API calls and sees byte-identical perception input every time. That is the
whole reason the cache exists - it is the one part of the model-facing pipeline
that can be frozen, since `temperature` is not accepted on current models.

Extensions in this corpus are not to be trusted: every image is named `.jpg`,
but the bytes are JPEG, PNG, WebP, and one AVIF. Formats are sniffed from the
file header, and the AVIF - which the vision API does not accept - is rejected
locally rather than after three failed round trips.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from attention_router import config
from attention_router.llm import LLMError, build_client, complete_json

IMAGE_SCHEMA_KEYS = ("transcribed_text", "visual_description", "poster_category")


class UnsupportedMediaError(RuntimeError):
    """The file is a real image, but in a format the vision API rejects."""


def load_cache(path: Path | None = None) -> dict[str, dict]:
    source = path or config.MEDIA_CACHE_JSON
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8"))


def save_cache(cache: dict[str, dict], path: Path | None = None) -> None:
    target = path or config.MEDIA_CACHE_JSON
    target.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys so the committed file has a stable diff between runs.
    target.write_text(
        json.dumps(cache, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def as_text(entry: dict | None) -> str:
    """Flatten a cache entry into the plain text used for search and prompting."""
    if not entry:
        return ""
    if entry.get("kind") == "unavailable":
        return ""
    if entry.get("kind") == "voice":
        transcript = (entry.get("transcript") or "").strip()
        return f"Voice note transcript: {transcript}" if transcript else ""
    parts = []
    if entry.get("transcribed_text"):
        parts.append(f"Text visible in image: {entry['transcribed_text']}")
    if entry.get("visual_description"):
        parts.append(f"Image shows: {entry['visual_description']}")
    if entry.get("poster_category"):
        parts.append(f"Image category: {entry['poster_category']}")
    for key in ("payment_artifacts", "urgency_cues"):
        values = entry.get(key) or []
        if values:
            parts.append(f"{key.replace('_', ' ').title()}: {', '.join(values)}")
    return "\n".join(parts)


def text_index(cache: dict[str, dict]) -> dict[str, str]:
    """media_id -> flattened text, for the retriever and the router."""
    return {media_id: as_text(entry) for media_id, entry in cache.items()}


# --- Image understanding -----------------------------------------------------


def _read_prompt(name: str) -> str:
    return (config.PROMPTS_DIR / name).read_text(encoding="utf-8")


# Several files in dataset/media/images carry a .jpg extension but are
# actually PNG, and the API rejects a mismatched media_type outright. Sniff
# the magic bytes instead of trusting the filename.
IMAGE_SIGNATURES = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)

# What the vision API will accept. Anything else is detected and skipped
# rather than sent and rejected three times over.
SUPPORTED_IMAGE_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)


def detect_image_type(blob: bytes, filename: str = "") -> str:
    """Media type from the file's own header, falling back to its extension."""
    for signature, media_type in IMAGE_SIGNATURES:
        if blob.startswith(signature):
            return media_type
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "image/webp"
    if blob[4:12] == b"ftypavif":
        return "image/avif"  # real format in this corpus; not API-supported
    return mimetypes.guess_type(filename)[0] or "image/jpeg"


def describe_image(client, path: Path) -> dict:
    """One vision call returning structured facts about a poster or screenshot."""
    blob = path.read_bytes()
    media_type = detect_image_type(blob, path.name)
    if media_type not in SUPPORTED_IMAGE_TYPES:
        raise UnsupportedMediaError(
            f"{path.name} is {media_type}, which the vision API does not accept"
        )
    encoded = base64.standard_b64encode(blob).decode("ascii")

    def validate(payload: dict) -> None:
        missing = [key for key in IMAGE_SCHEMA_KEYS if key not in payload]
        if missing:
            raise ValueError(f"Missing required keys: {', '.join(missing)}")

    payload = complete_json(
        client,
        system=_read_prompt("media_image.md"),
        content=[
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": encoded,
                },
            },
            {
                "type": "text",
                "text": (
                    "Describe this image using the JSON schema in your "
                    "instructions. Transcribe all visible text verbatim."
                ),
            },
        ],
        model=config.VISION_MODEL,
        max_tokens=config.VISION_MAX_TOKENS,
        validate=validate,
    )
    payload["kind"] = "image"
    return payload


# --- Voice transcription -----------------------------------------------------


def _asr_provider() -> tuple[str, str, str]:
    """First configured ASR provider: (env var, endpoint, model)."""
    for env_var in config.ASR_API_KEY_ENVS:
        if os.environ.get(env_var, "").strip():
            return env_var, config.ASR_ENDPOINTS[env_var], config.ASR_MODELS[env_var]
    raise LLMError(
        "No ASR key found. Set one of "
        f"{' or '.join(config.ASR_API_KEY_ENVS)} to transcribe voice notes. "
        "Once cache/media_cache.json is populated no key is needed again."
    )


def _multipart(fields: dict[str, str], filename: str, blob: bytes) -> tuple[bytes, str]:
    """Build a multipart/form-data body without pulling in a HTTP library."""
    boundary = f"----hrouter{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/mpeg\r\n\r\n".encode()
    )
    parts.append(blob)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def transcribe_voice(path: Path) -> dict:
    """Transcribe one audio file through the configured Whisper endpoint."""
    env_var, endpoint, model = _asr_provider()
    body, content_type = _multipart(
        {"model": model, "response_format": "json", "temperature": "0"},
        path.name,
        path.read_bytes(),
    )
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {os.environ[env_var].strip()}",
            "Content-Type": content_type,
            # urllib's default agent string trips the provider's bot check
            # (Cloudflare error 1010), so identify the client explicitly.
            "User-Agent": "message-notification-router/1.0",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Surface the provider's message but never the credential.
        raise LLMError(
            f"ASR request failed ({exc.code}): {exc.read().decode('utf-8')[:300]}"
        ) from exc

    return {"kind": "voice", "transcript": (payload.get("text") or "").strip()}


# --- Orchestration -----------------------------------------------------------


def ensure_media(
    data,
    media_ids: set[str],
    *,
    cache: dict[str, dict] | None = None,
    refresh: bool = False,
    verbose: bool = True,
) -> dict[str, dict]:
    """Populate the cache for every requested media id, calling APIs only on misses."""
    cache = load_cache() if cache is None else cache
    pending = sorted(
        media_id
        for media_id in media_ids
        if media_id and (refresh or media_id not in cache)
    )
    if not pending:
        if verbose:
            print(f"media: {len(media_ids)} referenced, all cached, 0 API calls")
        return cache

    client = None
    failures: list[str] = []
    for media_id in pending:
        relative = data.media_paths.get(media_id)
        if not relative:
            if verbose:
                print(f"media: {media_id} has no file path on record, skipping")
            continue
        path = config.DATASET_DIR / relative
        if not path.exists():
            if verbose:
                print(f"media: {path} is missing, skipping")
            continue

        # One unreadable file must not abort the batch. Record the failure so
        # the gap is visible in the cache rather than silently absent, and so
        # a rerun does not keep retrying a file that cannot work.
        try:
            if path.suffix.lower() in {".mp3", ".wav", ".m4a", ".ogg"}:
                cache[media_id] = transcribe_voice(path)
            else:
                client = client or build_client("anthropic")
                cache[media_id] = describe_image(client, path)
        except (UnsupportedMediaError, LLMError) as failure:
            cache[media_id] = {
                "kind": "unavailable",
                "error": str(failure)[:300],
                "source_path": relative,
            }
            failures.append(media_id)
            if verbose:
                print(f"media: SKIPPED {media_id} - {failure}")
            save_cache(cache)
            continue

        cache[media_id]["source_path"] = relative
        if verbose:
            print(f"media: resolved {media_id} ({cache[media_id]['kind']})")
        save_cache(cache)  # checkpoint, so an interruption never loses work

    if failures and verbose:
        # Never let dropped coverage look like full coverage.
        print(f"media: {len(failures)} of {len(pending)} could not be resolved: {failures}")

    return cache


def referenced_media_ids(*message_lists) -> set[str]:
    """Every media_id mentioned across the given message tables."""
    found: set[str] = set()
    for messages in message_lists:
        for row in messages:
            media_id = (row.get("media_id") or "").strip()
            if media_id:
                found.add(media_id)
    return found
