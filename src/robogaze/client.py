"""OpenAI-compatible local VLM client with a small disk cache."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from openai import OpenAI
from PIL import Image

_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def image_block(img: Image.Image) -> dict:
    return {"type": "image_url", "image_url": {"url": _pil_to_data_url(img)}}


def video_block(video_path: str | Path) -> dict:
    path = Path(video_path).expanduser().resolve()
    return {"type": "video_url", "video_url": {"url": f"file://{path}"}}


def _pil_to_data_url(img: Image.Image, quality: int = 85, max_side: int = 1024) -> str:
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


class ModelClient:
    """Single client used for both [LLM] (text) and [VLM] (text+image) calls.

    Cache key = sha256 of (model, messages, response_format, temperature). The
    cached value is the raw response message.content string. Callers are
    responsible for JSON parsing — we cache the string so the cache survives
    schema changes.
    """

    def __init__(
        self,
        model: str | None = None,
        cache_dir: str | Path = "cache",
        temperature: float = 0.0,
    ):
        self.temperature = temperature
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_cache_key: str | None = None
        self.last_cache_hit: bool | None = None
        self.last_response_meta: dict[str, Any] | None = None

        base_url = os.environ.get("LOCAL_BASE_URL", "http://localhost:8000/v1")
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        default = os.environ.get("LOCAL_MODEL", "gemma4")
        self.model = model or default
        thinking = os.environ.get("ROBOGAZE_LOCAL_THINKING", "0") == "1"
        self.extra_body: dict | None = (
            None if thinking else {"chat_template_kwargs": {"enable_thinking": False}}
        )

    def fork(self) -> "ModelClient":
        """Create an independent client handle that shares config and disk cache."""
        child = type(self)(
            model=self.model,
            cache_dir=self.cache_dir,
            temperature=self.temperature,
        )
        child.extra_body = self.extra_body
        return child

    # -- cache --------------------------------------------------------------

    def _cache_key(self, payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_entry(self, key: str) -> dict[str, Any] | None:
        path = self.cache_path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _cache_get(self, key: str) -> str | None:
        entry = self._cache_entry(key)
        if entry is None:
            return None
        content = entry.get("content")
        if content is None and isinstance(entry.get("response"), dict):
            content = entry["response"].get("content")
        if content is None:
            return None
        print(f"[cache hit] {self.cache_path(key)}", flush=True)
        return str(content)

    def _cache_put_request(self, key: str, request: dict) -> None:
        path = self.cache_path(key)
        payload = self._cache_entry(key) or {}
        payload["request"] = request
        _write_json_atomic(path, payload)
        print(f"[cache input] {path}", flush=True)

    def _cache_put(self, key: str, content: str, meta: dict, request: dict | None = None) -> None:
        path = self.cache_path(key)
        existing = self._cache_entry(key) or {}
        payload = {
            "request": request if request is not None else existing.get("request"),
            "response": {
                "content": content,
                "meta": meta,
            },
            "content": content,
            "meta": meta,
        }
        if payload["request"] is None:
            del payload["request"]
        _write_json_atomic(path, payload)

    # -- public API ---------------------------------------------------------

    def call_json(
        self,
        system: str,
        user_blocks: list[dict] | str,
        model: str | None = None,
        max_retries: int = 2,
        max_tokens: int = 4096,
    ) -> dict:
        """Call the chat API and parse a JSON object from the response.

        `user_blocks` may be a plain string (text-only) or a list of OpenAI
        content blocks (text + image_url). Retries up to max_retries times on
        JSON parse failure. Bad responses are written to the cache dir as
        `<key>.failed_attemptN.txt` for inspection.
        """
        if isinstance(user_blocks, str):
            user_content: list[dict] | str = user_blocks
        else:
            user_content = user_blocks

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        model = model or self.model

        # Cache the request signature without the giant base64 blobs.
        cache_payload = {
            "model": model,
            "system": system,
            "user": _strip_for_cache(user_content),
            "temperature": self.temperature,
        }
        key = self._cache_key(cache_payload)
        self.last_cache_key = key
        self.last_cache_hit = False
        self.last_response_meta = None
        logged_user = _strip_for_request_log(user_content)
        request_log = {
            "cache_key": key,
            "input": {
                "system": system,
                "user": logged_user,
            },
            "metadata": {
                "provider": "local",
                "model": model,
                "temperature": self.temperature,
                "media": _media_metadata(logged_user),
            },
        }
        if self.extra_body is not None:
            request_log["metadata"]["extra_body"] = self.extra_body
        self._cache_put_request(key, request_log)
        cached = self._cache_get(key)
        if cached is not None:
            self.last_cache_hit = True
            cached_entry = self._cache_entry(key) or {}
            self.last_response_meta = {
                "model": model,
                "cache_hit": True,
                "cached_meta": cached_entry.get("meta"),
            }
            return _parse_json(cached)

        last_err: Exception | None = None
        last_content = ""
        extra_kwargs: dict = {}
        if self.extra_body is not None:
            extra_kwargs["extra_body"] = self.extra_body
        for attempt in range(max_retries + 1):
            t0 = time.time()
            print(f"[api call] model={model} cache_key={key}", flush=True)
            response = self.client.chat.completions.create(
                model=model,
                temperature=self.temperature,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                **extra_kwargs,
            )
            choice = response.choices[0] if response.choices else None
            content = (choice.message.content if choice else "") or ""
            finish_reason = getattr(choice, "finish_reason", None) if choice else None
            usage = getattr(response, "usage", None)
            last_content = content
            if not content.strip():
                # Empty content is rarely a JSON-parse problem; surface server-side
                # signals (finish_reason='length' = truncation; ='content_filter' =
                # safety; many local VLMs return empty when image count or context
                # length exceeds limits).
                last_err = RuntimeError(
                    f"Empty response (finish_reason={finish_reason!r}, usage={usage})"
                )
                fail_path = self.cache_dir / f"{key}.failed_attempt{attempt}.txt"
                fail_path.write_text(
                    f"<EMPTY> finish_reason={finish_reason!r} usage={usage}\n"
                )
                if attempt < max_retries:
                    continue
                break
            try:
                parsed = _parse_json(content)
                meta = {
                    "model": model,
                    "elapsed_s": round(time.time() - t0, 2),
                    "attempt": attempt,
                    "finish_reason": finish_reason,
                    "usage": _usage_to_dict(usage),
                    "max_tokens": max_tokens,
                }
                self._cache_put(
                    key,
                    content,
                    meta,
                    request_log,
                )
                self.last_response_meta = {**meta, "cache_hit": False}
                return parsed
            except json.JSONDecodeError as e:
                last_err = e
                fail_path = self.cache_dir / f"{key}.failed_attempt{attempt}.txt"
                fail_path.write_text(content)
                if attempt < max_retries:
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your previous response was not valid JSON. Parser error: {e}. "
                                "Output ONLY a single valid JSON object that exactly matches the schema. "
                                "Properly escape every quote, newline, and backslash inside string values. "
                                "Do not wrap the JSON in markdown fences. Do not add any commentary."
                            ),
                        }
                    )
        raise RuntimeError(
            f"Model failed to return valid JSON after {max_retries + 1} attempts: {last_err}. "
            f"Last response saved under cache dir as {key}.failed_attempt*.txt "
            f"(first 300 chars: {last_content[:300]!r})"
        )

def _strip_for_cache(user_content: list[dict] | str) -> Any:
    """Replace media payloads with content hashes so cache keys stay small and correct."""
    if isinstance(user_content, str):
        return user_content
    out = []
    for block in user_content:
        if block.get("type") == "image_url":
            url = block["image_url"]["url"]
            h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            out.append({"type": "image_url", "image_hash": h})
        elif block.get("type") == "video_url":
            url = block["video_url"]["url"]
            info = _file_info_from_url(url, include_hash=True)
            out.append(
                {
                    "type": "video_url",
                    "video_path": info.get("path"),
                    "video_sha256": info.get("sha256"),
                    "size_bytes": info.get("size_bytes"),
                }
            )
        else:
            out.append(block)
    return out


def _strip_for_request_log(user_content: list[dict] | str) -> Any:
    """Return the full debug request text plus compact media metadata."""
    if isinstance(user_content, str):
        return user_content
    out = []
    for block in user_content:
        if block.get("type") == "image_url":
            url = block["image_url"]["url"]
            if url.startswith("data:"):
                out.append(
                    {
                        "type": "image_url",
                        "image_hash": hashlib.sha256(url.encode("utf-8")).hexdigest(),
                        "data_url_prefix": url[:64],
                    }
                )
            else:
                out.append({"type": "image_url", **_file_info_from_url(url, include_hash=True)})
        elif block.get("type") == "video_url":
            url = block["video_url"]["url"]
            info = _file_info_from_url(url, include_hash=True)
            path = info.get("path")
            if path:
                info["ffprobe"] = _ffprobe_video(Path(path))
            out.append({"type": "video_url", **info})
        else:
            out.append(block)
    return out


def _media_metadata(logged_user: Any) -> list[dict[str, Any]]:
    if not isinstance(logged_user, list):
        return []
    media = []
    for index, block in enumerate(logged_user):
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type not in {"image_url", "video_url"}:
            continue
        item = {"index": index, "type": block_type}
        for key in (
            "path",
            "exists",
            "size_bytes",
            "mtime_ns",
            "sha256",
            "image_hash",
            "data_url_prefix",
            "ffprobe",
        ):
            if key in block:
                item[key] = block[key]
        media.append(item)
    return media


def _file_info_from_url(url: str, *, include_hash: bool = False) -> dict[str, Any]:
    info: dict[str, Any] = {"url": url}
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return info
    path = Path(unquote(parsed.path))
    info["path"] = str(path)
    info["exists"] = path.exists()
    if not path.exists():
        return info
    stat = path.stat()
    info["size_bytes"] = stat.st_size
    info["mtime_ns"] = stat.st_mtime_ns
    if include_hash:
        info["sha256"] = _file_sha256(path, stat.st_size, stat.st_mtime_ns)
    return info


def _file_sha256(path: Path, size_bytes: int, mtime_ns: int) -> str:
    key = (str(path.resolve()), size_bytes, mtime_ns)
    cached = _FILE_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[key] = value
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return dict(usage)
    out = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            out[key] = value
    return out or None


def _ffprobe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"available": False}
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,codec_tag_string,pix_fmt,width,height,r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration,size,bit_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"available": True, "error": result.stderr.strip()}
    try:
        return {"available": True, **json.loads(result.stdout)}
    except json.JSONDecodeError as exc:
        return {"available": True, "error": f"invalid ffprobe JSON: {exc}"}


def _parse_json(content: str) -> dict:
    """Extract a JSON object from a model response (tolerant of fence wrapping)."""
    s = content.strip()
    if s.startswith("```"):
        # Strip ``` or ```json fence
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[: -3]
    return json.loads(s.strip())
