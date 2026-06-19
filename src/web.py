import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import io
import ipaddress
import json
import logging
import os
from pathlib import Path
import re
import socket
import ssl
from threading import Lock
import time
from typing import Dict
from typing import List
from typing import Optional
from urllib.parse import unquote
from urllib.parse import urlparse
import uuid
import zipfile

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse
from fastapi.responses import StreamingResponse
import httpx
import openai
from starlette.middleware.trustedhost import TrustedHostMiddleware

if __package__:
    from .conspectum.logger import Logger
    from .conspectum.process import process
    from .conspectum.summary import get_transcription_model
    from .conspectum.summary import guess_audio_suffix
    from .conspectum.summary import is_supported_audio
    from .conspectum.summary import SUPPORTED_AUDIO_EXTENSIONS
else:
    from conspectum.logger import Logger
    from conspectum.process import process
    from conspectum.summary import get_transcription_model
    from conspectum.summary import guess_audio_suffix
    from conspectum.summary import is_supported_audio
    from conspectum.summary import SUPPORTED_AUDIO_EXTENSIONS


load_dotenv(override=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
RESULTS_DIR = os.path.join(STATIC_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
TEMPLATE_PATH = os.path.join(BASE_DIR, "src", "web_template.html")
APP_ENV = os.environ.get("APP_ENV", "development").strip().lower()
IS_PRODUCTION = APP_ENV == "production"
ENABLE_API_DOCS = os.environ.get(
    "ENABLE_API_DOCS",
    "0" if IS_PRODUCTION else "1",
).strip().lower() in {"1", "true", "yes"}
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(512 * 1024 * 1024)))
REMOTE_AUDIO_MAX_BYTES = int(os.environ.get("REMOTE_AUDIO_MAX_BYTES", str(MAX_UPLOAD_BYTES)))
MAX_REMOTE_URL_LENGTH = int(os.environ.get("MAX_REMOTE_URL_LENGTH", "2048"))
MAX_REMOTE_REDIRECTS = int(os.environ.get("MAX_REMOTE_REDIRECTS", "4"))
SECURE_HSTS_SECONDS = int(
    os.environ.get(
        "SECURE_HSTS_SECONDS",
        "31536000" if IS_PRODUCTION else "0",
    )
)
ALLOWED_HOSTS = [host.strip() for host in os.environ.get("ALLOWED_HOSTS", "").split(",") if host.strip()]
PRELOAD_WHISPER = os.environ.get("PRELOAD_WHISPER", "0").strip().lower() in {"1", "true", "yes", "on"}
TRUST_ENV_PROXY = os.environ.get("TRUST_ENV_PROXY", "0").strip().lower() in {"1", "true", "yes", "on"}

VALID_DETAIL_LEVELS = {"brief", "standard", "detailed"}
VALID_LANGUAGES = {"en", "ru"}
UPLOAD_READ_CHUNK_SIZE = 1024 * 1024
SAFE_TEXT_LIMIT = 500
SAFE_FILENAME_LIMIT = 120
PUBLIC_TASK_FIELDS = {
    "task_id",
    "status",
    "messages",
    "progress",
    "stage_code",
    "stage",
    "bundle_url",
    "tex_url",
    "pdf_url",
    "transcript_url",
    "title",
    "language",
    "detail",
    "error",
    "warning",
    "source_mode",
    "source_name",
    "audio_size_bytes",
    "abstract",
    "transcript_preview",
    "transcript_words",
    "abstract_words",
    "created_at",
    "updated_at",
    "completed_at",
    "duration_seconds",
}
STATIC_ASSET_FILENAMES = {"web.css", "web.js"}
GENERATED_ARTIFACT_RE = re.compile(r"^(?:transcript|result)_[0-9a-fA-F-]{36}\.(?:txt|tex|pdf)$")
SUSPICIOUS_DOUBLE_EXTENSIONS = {
    ".apk",
    ".app",
    ".bat",
    ".cmd",
    ".com",
    ".dll",
    ".exe",
    ".hta",
    ".jar",
    ".js",
    ".jse",
    ".msi",
    ".php",
    ".ps1",
    ".py",
    ".rb",
    ".scr",
    ".sh",
    ".svg",
    ".vbs",
    ".wsf",
}
PUBLIC_ERROR_REDACTIONS = [
    (re.compile(r"sk-[A-Za-z0-9_-]+"), "[redacted-api-key]"),
    (re.compile(r"user_[A-Za-z0-9]+"), "[redacted-user-id]"),
]
SECURITY_CSP = "; ".join(
    [
        "default-src 'self'",
        "base-uri 'self'",
        "connect-src 'self'",
        "font-src 'self' data:",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "img-src 'self' data:",
        "media-src 'self'",
        "object-src 'none'",
        "script-src 'self'",
        "style-src 'self' 'unsafe-inline'",
    ]
)

TASK_STAGE_CONFIG = {
    "queued": {"label": "В очереди", "start": 2, "end": 4},
    "starting": {"label": "Началась обработка", "start": 4, "end": 8},
    "prepare_url": {"label": "Подготовка URL", "start": 8, "end": 12},
    "download_audio": {"label": "Скачивание источника", "start": 12, "end": 18},
    "transcribing": {"label": "Распознавание аудио", "start": 18, "end": 38},
    "transcript_ready": {"label": "Транскрипт готов", "start": 38, "end": 42},
    "detect_language": {"label": "Определение языка", "start": 42, "end": 48},
    "summary": {"label": "Сборка конспекта", "start": 48, "end": 60},
    "sections": {"label": "Сборка разделов", "start": 60, "end": 80},
    "postprocess": {"label": "Постобработка", "start": 80, "end": 90},
    "pdf": {"label": "Сборка PDF", "start": 90, "end": 96},
    "pdf_retry": {"label": "Повторная сборка PDF", "start": 94, "end": 98},
    "pdf_problem": {"label": "Проблема с PDF", "start": 92, "end": 96},
    "tex_only": {"label": "Только TEX", "start": 97, "end": 99},
    "done": {"label": "Готово", "start": 100, "end": 100},
    "error": {"label": "Ошибка", "start": 0, "end": 0},
}
DEFAULT_STAGE_CODE = "queued"

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

app = FastAPI(
    title="Conspectum Web",
    description="Premium web interface for turning lecture audio into transcript, LaTeX, and PDF outputs.",
    docs_url="/docs" if ENABLE_API_DOCS else None,
    redoc_url="/redoc" if ENABLE_API_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_API_DOCS else None,
)

if ALLOWED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

tasks: Dict[str, dict] = {}
TASK_TTL = timedelta(hours=float(os.environ.get("TASK_TTL_HOURS", "24")))
RUNNING_TASK_TTL = timedelta(hours=float(os.environ.get("RUNNING_TASK_TTL_HOURS", "12")))
app_logger = logging.getLogger("conspectum.web")

allow_insecure_ssl = os.environ.get("ALLOW_INSECURE_SSL", "").lower() in {
    "1",
    "true",
    "yes",
}
verify: bool | ssl.SSLContext = True
if allow_insecure_ssl:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    verify = ssl_context

http_client = httpx.AsyncClient(
    verify=verify,
    timeout=300.0,
    trust_env=TRUST_ENV_PROXY,
)

ai = openai.AsyncOpenAI(
    base_url=os.environ.get("AI_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.environ["AI_API_KEY"],
    http_client=http_client,
    max_retries=0,
)


async def warm_transcription_model() -> None:
    try:
        await asyncio.to_thread(get_transcription_model)
        app_logger.info("Transcription model warmed up")
    except Exception:
        app_logger.exception("Failed to warm transcription model")


@dataclass(frozen=True)
class RateLimitRule:
    name: str
    limit: int
    window_seconds: int


class InMemoryRateLimiter:
    def __init__(self):
        self._buckets: dict[tuple[str, str], deque[float]] = {}
        self._lock = Lock()

    def check(self, key: str, rule: RateLimitRule) -> Optional[int]:
        now = time.monotonic()
        bucket_key = (rule.name, key)

        with self._lock:
            bucket = self._buckets.setdefault(bucket_key, deque())
            cutoff = now - rule.window_seconds

            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= rule.limit:
                retry_after = max(1, int(rule.window_seconds - (now - bucket[0])))
                return retry_after

            bucket.append(now)

            if not bucket:
                self._buckets.pop(bucket_key, None)

        return None

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()


RATE_LIMITER = InMemoryRateLimiter()
RATE_LIMIT_RULES = {
    ("POST", "/upload"): RateLimitRule("upload", 6, 300),
    ("POST", "/upload-url"): RateLimitRule("upload-url", 4, 300),
    ("GET", "/bundle"): RateLimitRule("bundle", 30, 60),
    ("GET", "/status"): RateLimitRule("status", 180, 60),
}


class WebLogger(Logger):
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.last_progress_bucket: Optional[int] = None

        log_id = str(uuid.uuid4())
        log_dir = os.path.join(LOGS_DIR, log_id)
        os.makedirs(log_dir, exist_ok=True)

        super().__init__(log_dir)

        self.messages: List[str] = []
        tasks[self.task_id]["messages"] = self.messages
        tasks[self.task_id]["progress"] = 0
        set_task_stage(self.task_id, "starting", 0, force=True)

    async def partial_result(self, text: str):
        if self.task_id not in tasks:
            return
        self.messages.append(text)
        tasks[self.task_id]["messages"] = self.messages
        inferred_stage = infer_stage_update_from_message(text)
        if inferred_stage is not None:
            stage_code, stage_progress = inferred_stage
            set_task_stage(self.task_id, stage_code, stage_progress)
        tasks[self.task_id]["updated_at"] = datetime.now(timezone.utc)

    async def progress(self, completed: int, total: int):
        if self.task_id not in tasks:
            return
        if total <= 0:
            percent = 0
        else:
            percent = round(completed / total * 100)

        stage_code = tasks[self.task_id].get("stage_code") or "sections"
        set_task_stage(self.task_id, stage_code, percent)

        bucket = 100 if percent >= 100 else max(0, min(95, percent)) // 5
        if bucket != self.last_progress_bucket:
            progress_text = f"Progress: {percent}%"
            self.messages.append(progress_text)
            tasks[self.task_id]["messages"] = self.messages
            self.last_progress_bucket = bucket

        tasks[self.task_id]["updated_at"] = datetime.now(timezone.utc)

    async def stage(self, stage: str, progress: Optional[int] = None):
        if self.task_id not in tasks:
            return
        set_task_stage(self.task_id, stage, progress)


def request_is_https(request: Request) -> bool:
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    return request.url.scheme == "https" or forwarded_proto.lower() == "https"


def add_security_headers(request: Request, response) -> None:
    response.headers.setdefault("Content-Security-Policy", SECURITY_CSP)
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()")
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")

    if request.url.path == "/" or request.url.path.startswith(("/upload", "/status/", "/bundle/")):
        response.headers.setdefault("Cache-Control", "no-store")

    if SECURE_HSTS_SECONDS > 0 and request_is_https(request):
        response.headers.setdefault(
            "Strict-Transport-Security",
            f"max-age={SECURE_HSTS_SECONDS}; includeSubDomains",
        )


def get_client_identifier(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def get_rate_limit_rule(request: Request) -> Optional[RateLimitRule]:
    path = request.url.path
    if request.method == "GET" and path.startswith("/status/"):
        return RATE_LIMIT_RULES[("GET", "/status")]
    if request.method == "GET" and path.startswith("/bundle/"):
        return RATE_LIMIT_RULES[("GET", "/bundle")]
    return RATE_LIMIT_RULES.get((request.method, path))


def resolve_path_within(base_dir: str, candidate_path: str) -> str:
    base = Path(base_dir).resolve()
    candidate = Path(candidate_path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    return str(candidate)


def is_hidden_or_suspicious_filename(filename: str) -> bool:
    basename = Path(filename).name.strip()
    if not basename or basename in {".", ".."} or basename.startswith("."):
        return True

    suffixes = [suffix.lower() for suffix in Path(basename).suffixes]
    if not suffixes:
        return True

    return any(suffix in SUSPICIOUS_DOUBLE_EXTENSIONS for suffix in suffixes[:-1])


def sanitize_source_name(name: Optional[str], fallback: str = "audio") -> str:
    basename = Path(name or "").name
    cleaned = re.sub(r"[\x00-\x1f\x7f]+", "", basename).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return fallback
    return cleaned[:SAFE_FILENAME_LIMIT]


def summarize_source_url(audio_url: str) -> str:
    parsed = urlparse(audio_url)
    filename = sanitize_source_name(unquote(Path(parsed.path).name), fallback="")
    base = parsed.netloc or "remote-audio"
    if filename:
        return f"{base}/{filename}"[:SAFE_FILENAME_LIMIT]
    return base[:SAFE_FILENAME_LIMIT]


def safe_internal_audio_name(filename: Optional[str], mime_type: Optional[str]) -> str:
    suffix = guess_audio_suffix(filename=filename, mime_type=mime_type)
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        suffix = ".wav"
    return f"upload{suffix}"


def sniff_audio_container(audio_bytes: bytes) -> Optional[str]:
    header = audio_bytes[:64]

    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"
    if header.startswith(b"fLaC"):
        return "flac"
    if header.startswith(b"OggS"):
        return "ogg"
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return "webm"
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return "mp4-family"
    if header.startswith(b"ID3"):
        return "mp3-family"
    if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return "mp3-family"
    return None


def validate_audio_payload(
    *,
    filename: Optional[str],
    mime_type: Optional[str],
    audio_bytes: bytes,
) -> str:
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    if filename and is_hidden_or_suspicious_filename(filename):
        raise HTTPException(status_code=400, detail="Suspicious file name is not allowed.")

    if not is_supported_audio(filename, mime_type):
        raise HTTPException(
            status_code=400,
            detail=("Supported audio formats: .wav, .mp3, .m4a, .ogg, .opus, .flac, .aac, .mp4, .m4b, .webm"),
        )

    sniffed = sniff_audio_container(audio_bytes)
    if sniffed is None:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not look like a supported audio container.",
        )

    suffix = Path(filename or "").suffix.lower()
    mime = (mime_type or "").split(";")[0].strip().lower()

    if sniffed == "wav" and suffix not in {".wav", ""} and mime not in {"", "audio/wav", "audio/x-wav"}:
        raise HTTPException(status_code=400, detail="The uploaded WAV file metadata looks inconsistent.")
    if sniffed == "flac" and suffix not in {".flac", ""} and mime not in {"", "audio/flac", "audio/x-flac"}:
        raise HTTPException(status_code=400, detail="The uploaded FLAC file metadata looks inconsistent.")
    if sniffed == "ogg" and suffix not in {".ogg", ".oga", ".opus", ""} and mime not in {"", "audio/ogg", "audio/opus"}:
        raise HTTPException(status_code=400, detail="The uploaded OGG/Opus file metadata looks inconsistent.")
    if sniffed == "webm" and suffix not in {".webm", ""} and mime not in {"", "audio/webm", "video/webm"}:
        raise HTTPException(status_code=400, detail="The uploaded WebM file metadata looks inconsistent.")
    if sniffed == "mp4-family" and suffix not in {".m4a", ".mp4", ".m4b", ""} and mime not in {"", "audio/mp4"}:
        raise HTTPException(status_code=400, detail="The uploaded MP4/M4A file metadata looks inconsistent.")
    if (
        sniffed == "mp3-family"
        and suffix not in {".mp3", ".aac", ""}
        and mime not in {"", "audio/mpeg", "audio/mp3", "audio/aac"}
    ):
        raise HTTPException(status_code=400, detail="The uploaded MP3/AAC file metadata looks inconsistent.")

    return safe_internal_audio_name(filename, mime_type)


async def read_validated_upload(file: UploadFile) -> tuple[bytes, str, str]:
    original_name = sanitize_source_name(
        file.filename,
        fallback=safe_internal_audio_name(file.filename, file.content_type),
    )
    audio_buffer = bytearray()
    total_bytes = 0

    try:
        while True:
            chunk = await file.read(UPLOAD_READ_CHUNK_SIZE)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Audio file is too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                )
            audio_buffer.extend(chunk)
    finally:
        await file.close()

    audio_bytes = bytes(audio_buffer)
    internal_name = validate_audio_payload(
        filename=original_name,
        mime_type=file.content_type,
        audio_bytes=audio_bytes,
    )
    return audio_bytes, original_name, internal_name


def sanitize_public_error_text(text: str) -> str:
    sanitized = text or "Processing failed."
    sanitized = sanitized.replace(BASE_DIR, "<app>")
    sanitized = sanitized.replace(STATIC_DIR, "<static>")
    sanitized = sanitized.replace(LOGS_DIR, "<logs>")

    for pattern, replacement in PUBLIC_ERROR_REDACTIONS:
        sanitized = pattern.sub(replacement, sanitized)

    sanitized = re.sub(r"[A-Za-z]:\\[^:\n]+", "<path>", sanitized)
    sanitized = re.sub(r"/[^:\n]*/(?:tmp|temp|private|home)[^:\n]*", "<path>", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized[:SAFE_TEXT_LIMIT] or "Processing failed."


def build_public_error_message(exc: Exception) -> str:
    raw_message = sanitize_public_error_text(str(exc))
    raw_message_lower = raw_message.lower()

    if (
        exc.__class__.__name__ in {"APIResponseValidationError", "DecodingError", "JSONDecodeError"}
        or "expecting value" in raw_message_lower
        or "invalid json" in raw_message_lower
        or "failed to decode json" in raw_message_lower
    ):
        return (
            "AI provider returned an invalid response. "
            "The request will usually succeed if retried; check AI_BASE_URL/MODEL_NAME if it repeats."
        )

    if isinstance(exc, (RuntimeError, ValueError, HTTPException)):
        return raw_message

    if exc.__class__.__name__ == "AuthenticationError":
        return "AI provider authentication failed. Check AI_API_KEY and AI_BASE_URL."

    if exc.__class__.__name__ == "RateLimitError":
        return "AI provider rate limit or balance limit was reached. Check the provider account and try again later."

    if exc.__class__.__name__ == "APIStatusError":
        status_code = getattr(exc, "status_code", None)
        if "prompt tokens limit exceeded" in raw_message_lower or "tokens limit exceeded" in raw_message_lower:
            return (
                "AI provider prompt token limit was exceeded for the selected model. "
                "Lower SUMMARY_TRANSCRIPT_CONTEXT_CHARS or use a paid provider key with a larger context limit."
            )
        if (
            status_code == 402
            or "402" in raw_message
            or "credits" in raw_message_lower
            or "can only afford" in raw_message_lower
        ):
            return (
                "AI provider credits are insufficient for the selected model. "
                "Use a cheaper MODEL_NAME or add credits in the provider account."
            )
        return f"AI provider rejected the request: {raw_message}"

    if exc.__class__.__name__ in {
        "APIConnectionError",
        "APITimeoutError",
    }:
        return "The AI provider request failed. Please try again in a moment."

    return "Processing failed due to an internal server error."


def serialize_task(task: dict) -> dict:
    public_task = {}
    for key in PUBLIC_TASK_FIELDS:
        value = task.get(key)
        if isinstance(value, datetime):
            public_task[key] = value.isoformat()
        elif key == "messages":
            public_task[key] = [str(message)[:SAFE_TEXT_LIMIT] for message in (value or [])]
        else:
            public_task[key] = value
    return public_task


def safe_remove_file(file_path: Optional[str]) -> None:
    if not file_path:
        return

    try:
        resolved = resolve_path_within(RESULTS_DIR, file_path)
    except HTTPException:
        app_logger.warning("Skipped deletion for unexpected file path", extra={"path": file_path})
        return

    try:
        os.remove(resolved)
    except FileNotFoundError:
        return
    except OSError:
        app_logger.warning("Failed to remove generated artifact", extra={"path": resolved})


async def ensure_public_hostname(hostname: str, port: int) -> None:
    try:
        addrinfo = await asyncio.to_thread(
            socket.getaddrinfo,
            hostname,
            port,
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise RuntimeError("The remote host could not be resolved.") from exc

    seen_ip = False
    for _, _, _, _, sockaddr in addrinfo:
        host_ip = ipaddress.ip_address(sockaddr[0])
        seen_ip = True
        if (
            host_ip.is_private
            or host_ip.is_loopback
            or host_ip.is_link_local
            or host_ip.is_multicast
            or host_ip.is_reserved
            or host_ip.is_unspecified
        ):
            raise RuntimeError("Private and local network URLs are not allowed for remote audio downloads.")

    if not seen_ip:
        raise RuntimeError("The remote host did not resolve to a public IP address.")


def cleanup_expired_tasks() -> None:
    now = datetime.now(timezone.utc)
    expired_ids = []

    for task_id, task in tasks.items():
        status = task.get("status")
        reference_time = task.get("updated_at") or task.get("created_at") or now
        ttl = RUNNING_TASK_TTL if status == "running" else TASK_TTL

        if now - reference_time > ttl:
            expired_ids.append(task_id)

    for task_id in expired_ids:
        task = tasks.pop(task_id)
        for key in ("tex_path", "pdf_path", "transcript_path"):
            safe_remove_file(task.get(key))


def normalize_detail_value(detail: Optional[str]) -> str:
    if detail is not None and len(detail) > 32:
        raise HTTPException(status_code=400, detail="Detail value is too long.")
    normalized = (detail or "standard").strip().lower()
    if normalized not in VALID_DETAIL_LEVELS:
        raise HTTPException(
            status_code=400,
            detail="Detail must be one of: brief, standard, detailed.",
        )
    return normalized


def normalize_language_value(language: Optional[str]) -> Optional[str]:
    if language is None or not language.strip():
        return None

    if len(language) > 16:
        raise HTTPException(status_code=400, detail="Language value is too long.")

    normalized = language.strip().lower()
    if normalized not in VALID_LANGUAGES:
        raise HTTPException(status_code=400, detail="Language must be en or ru.")
    return normalized


def create_task_state(
    task_id: str,
    detail: str,
    *,
    source_mode: str,
    source_name: Optional[str] = None,
    source_url: Optional[str] = None,
    audio_size_bytes: Optional[int] = None,
) -> dict:
    created_at = datetime.now(timezone.utc)
    return {
        "task_id": task_id,
        "status": "running",
        "messages": [],
        "progress": 0,
        "stage_code": DEFAULT_STAGE_CODE,
        "stage": TASK_STAGE_CONFIG[DEFAULT_STAGE_CODE]["label"],
        "bundle_url": None,
        "tex_url": None,
        "pdf_url": None,
        "transcript_url": None,
        "tex_path": None,
        "pdf_path": None,
        "transcript_path": None,
        "title": None,
        "language": None,
        "detail": detail,
        "error": None,
        "warning": None,
        "source_mode": source_mode,
        "source_name": sanitize_source_name(source_name, fallback="audio") if source_name else None,
        "source_url": source_url,
        "audio_size_bytes": audio_size_bytes,
        "abstract": None,
        "transcript_preview": None,
        "transcript_words": None,
        "abstract_words": None,
        "created_at": created_at,
        "updated_at": created_at,
        "completed_at": None,
        "duration_seconds": None,
    }


def make_preview(text: Optional[str], limit: int = 1800) -> Optional[str]:
    if text is None:
        return None

    normalized = text.strip()
    if len(normalized) <= limit:
        return normalized

    shortened = normalized[:limit].rsplit(" ", 1)[0].rstrip()
    return (shortened or normalized[:limit]).rstrip() + "..."


def count_words(text: Optional[str]) -> int:
    return len((text or "").split())


def safe_download_name(text: Optional[str], fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", (text or "").strip()).strip("-._")
    return normalized[:80] or fallback


def redact_url_for_metadata(source_url: Optional[str]) -> Optional[str]:
    if not source_url:
        return None

    parsed = urlparse(source_url)
    if not parsed.scheme or not parsed.netloc:
        return None

    safe_path = parsed.path or ""
    return f"{parsed.scheme}://{parsed.netloc}{safe_path}"[:SAFE_TEXT_LIMIT]


def get_task_or_404(task_id: str) -> dict:
    try:
        normalized = str(uuid.UUID(task_id))
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc

    task = tasks.get(normalized)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


def get_stage_config(stage_code: Optional[str]) -> dict:
    return TASK_STAGE_CONFIG.get(stage_code or DEFAULT_STAGE_CODE, TASK_STAGE_CONFIG[DEFAULT_STAGE_CODE])


def map_stage_progress(stage_code: Optional[str], stage_progress: Optional[int] = None) -> int:
    config = get_stage_config(stage_code)
    start = int(config["start"])
    end = int(config["end"])

    if stage_progress is None:
        return start

    safe_progress = max(0, min(100, int(stage_progress)))
    return start + round((end - start) * (safe_progress / 100))


def set_task_stage(
    task_id: str,
    stage_code: str,
    stage_progress: Optional[int] = None,
    *,
    force: bool = False,
) -> None:
    task = tasks.get(task_id)
    if not task:
        return

    current_stage_code = task.get("stage_code") or DEFAULT_STAGE_CODE
    current_config = get_stage_config(current_stage_code)
    new_config = get_stage_config(stage_code)

    if force or stage_code == current_stage_code or int(new_config["start"]) >= int(current_config["start"]):
        task["stage_code"] = stage_code
        task["stage"] = str(new_config["label"])
        mapped_progress = map_stage_progress(stage_code, stage_progress)
    else:
        mapped_progress = map_stage_progress(current_stage_code, stage_progress)

    if task.get("status") == "running":
        mapped_progress = min(mapped_progress, 99)

    if force:
        task["progress"] = mapped_progress
    else:
        task["progress"] = max(int(task.get("progress", 0)), mapped_progress)

    task["updated_at"] = datetime.now(timezone.utc)


def infer_stage_update_from_message(text: str) -> Optional[tuple[str, Optional[int]]]:
    normalized = text.lower()

    stage_progress = [
        ("fetching audio", ("download_audio", 10)),
        ("remote audio downloaded", ("download_audio", 100)),
        ("starting transcription", ("transcribing", 5)),
        ("transcription complete", ("transcript_ready", 100)),
        ("detected language", ("detect_language", 100)),
        ("topic of the lecture", ("summary", 70)),
        ("abstract of the lecture", ("summary", 100)),
        ("starting postprocessing", ("postprocess", 10)),
        ("postprocessing complete", ("postprocess", 100)),
        ("postprocessing validation failed", ("postprocess", 100)),
        ("postprocessing warning", ("postprocess", 100)),
        ("retrying pdf generation with a safer latex cleanup pass", ("pdf_retry", 30)),
        ("retrying pdf generation with xelatex unicode support", ("pdf_retry", 45)),
        ("retrying pdf generation with lualatex unicode support", ("pdf_retry", 55)),
        ("retrying pdf generation with a readable fallback layout", ("pdf_retry", 75)),
        ("retrying pdf generation with an ascii-safe transliteration fallback", ("pdf_retry", 88)),
        ("failed to convert latex to pdf", ("pdf_problem", 45)),
        ("pdf generation skipped", ("tex_only", 100)),
    ]

    for marker, stage_data in stage_progress:
        if marker in normalized:
            return stage_data

    return None


async def validate_remote_audio_url(audio_url: str) -> str:
    normalized = audio_url.strip()
    if not normalized:
        raise RuntimeError("Audio URL is required.")
    if len(normalized) > MAX_REMOTE_URL_LENGTH:
        raise RuntimeError("The provided URL is too long.")

    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("Enter a valid direct http:// or https:// link to an audio file.")
    if parsed.username or parsed.password:
        raise RuntimeError("Authenticated URLs are not allowed for remote audio downloads.")

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise RuntimeError("The provided URL contains an invalid port.") from exc

    hostname = parsed.hostname
    if not hostname:
        raise RuntimeError("The provided URL is missing a hostname.")
    if hostname.lower() == "localhost":
        raise RuntimeError("Localhost URLs are not allowed for remote audio downloads.")

    try:
        host_ip = ipaddress.ip_address(hostname)
    except ValueError:
        await ensure_public_hostname(hostname, port)
    else:
        if (
            host_ip.is_private
            or host_ip.is_loopback
            or host_ip.is_link_local
            or host_ip.is_multicast
            or host_ip.is_reserved
            or host_ip.is_unspecified
        ):
            raise RuntimeError("Private and local network URLs are not allowed for remote audio downloads.")

    return normalized


def build_remote_audio_name(audio_url: str, content_type: str | None) -> str:
    parsed = urlparse(audio_url)
    raw_name = os.path.basename(parsed.path)
    filename = sanitize_source_name(unquote(raw_name), fallback="remote-audio")

    if not os.path.splitext(filename)[1]:
        filename += guess_audio_suffix(filename=filename, mime_type=content_type)

    return filename


async def download_audio_from_url(
    audio_url: str,
    logger: WebLogger,
) -> tuple[bytes, str, Optional[str]]:
    normalized_url = await validate_remote_audio_url(audio_url)

    await logger.stage("download_audio", 0)
    await logger.partial_result("Fetching audio from the provided URL...")

    timeout = httpx.Timeout(900.0, connect=30.0)
    request_url = normalized_url
    redirects = 0

    while True:
        async with http_client.stream(
            "GET",
            request_url,
            follow_redirects=False,
            timeout=timeout,
        ) as response:
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    raise RuntimeError("The remote server returned an empty redirect.")
                redirects += 1
                if redirects > MAX_REMOTE_REDIRECTS:
                    raise RuntimeError("Too many redirects while fetching remote audio.")
                request_url = str(response.url.join(location))
                request_url = await validate_remote_audio_url(request_url)
                continue

            if response.status_code >= 400:
                raise RuntimeError(f"Remote audio download failed with HTTP {response.status_code}.")

            content_type_header = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
            filename = build_remote_audio_name(str(response.url), content_type_header)

            if not is_supported_audio(filename, content_type_header):
                raise RuntimeError(
                    "The URL does not look like a supported audio file. "
                    "Use a direct link to .wav, .mp3, .m4a, .ogg, .opus, .flac, .aac, .mp4, or .webm."
                )

            content_length = response.headers.get("content-length")
            expected_bytes = int(content_length) if content_length and content_length.isdigit() else None

            if expected_bytes and expected_bytes > REMOTE_AUDIO_MAX_BYTES:
                max_mb = REMOTE_AUDIO_MAX_BYTES // (1024 * 1024)
                raise RuntimeError(f"Remote audio is too large for the web worker limit ({max_mb} MB).")

            total_bytes = 0
            audio_buffer = bytearray()

            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > REMOTE_AUDIO_MAX_BYTES:
                    max_mb = REMOTE_AUDIO_MAX_BYTES // (1024 * 1024)
                    raise RuntimeError(f"Remote audio is too large for the web worker limit ({max_mb} MB).")
                audio_buffer.extend(chunk)
                if expected_bytes:
                    await logger.progress(total_bytes, expected_bytes)
            break

    audio_bytes = bytes(audio_buffer)
    validate_audio_payload(
        filename=filename,
        mime_type=content_type_header,
        audio_bytes=audio_bytes,
    )

    await logger.partial_result(f"Remote audio downloaded: {round(total_bytes / (1024 * 1024), 1)} MB")
    return audio_bytes, filename, content_type_header or None


def build_task_bundle_bytes_for_task(task: dict) -> bytes:
    archive_bytes = io.BytesIO()

    metadata = {
        "task_id": task.get("task_id"),
        "status": task.get("status"),
        "stage": task.get("stage"),
        "title": task.get("title"),
        "language": task.get("language"),
        "detail": task.get("detail"),
        "source_mode": task.get("source_mode"),
        "source_name": task.get("source_name"),
        "source_url": redact_url_for_metadata(task.get("source_url")),
        "audio_size_bytes": task.get("audio_size_bytes"),
        "warning": task.get("warning"),
        "error": task.get("error"),
        "created_at": task.get("created_at").isoformat() if task.get("created_at") else None,
        "updated_at": task.get("updated_at").isoformat() if task.get("updated_at") else None,
        "completed_at": task.get("completed_at").isoformat() if task.get("completed_at") else None,
        "duration_seconds": task.get("duration_seconds"),
        "transcript_words": task.get("transcript_words"),
        "abstract_words": task.get("abstract_words"),
    }

    summary_text = task.get("abstract") or ""
    files_to_attach = [
        ("transcript.txt", task.get("transcript_path")),
        ("result.tex", task.get("tex_path")),
        ("result.pdf", task.get("pdf_path")),
    ]

    with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        attached_any = False

        for archive_name, path in files_to_attach:
            if path and os.path.exists(path):
                archive.write(resolve_path_within(RESULTS_DIR, path), arcname=archive_name)
                attached_any = True

        if summary_text:
            archive.writestr("summary.txt", summary_text)
            attached_any = True

        archive.writestr(
            "metadata.json",
            json.dumps(metadata, ensure_ascii=False, indent=2),
        )

        if not attached_any:
            raise HTTPException(
                status_code=409,
                detail="This task does not have any downloadable artifacts yet.",
            )

    archive_bytes.seek(0)
    return archive_bytes.getvalue()


def build_task_bundle_bytes(task_id: str) -> bytes:
    return build_task_bundle_bytes_for_task(tasks[task_id])


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    cleanup_expired_tasks()

    rate_limit_rule = get_rate_limit_rule(request)
    if rate_limit_rule is not None:
        client_id = get_client_identifier(request)
        retry_after = RATE_LIMITER.check(client_id, rate_limit_rule)
        if retry_after is not None:
            app_logger.warning(
                "Rate limit exceeded",
                extra={"path": request.url.path, "client": client_id, "bucket": rate_limit_rule.name},
            )
            response = JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down and try again shortly."},
                headers={"Retry-After": str(retry_after)},
            )
            add_security_headers(request, response)
            return response

    response = await call_next(request)
    add_security_headers(request, response)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    app_logger.exception("Unhandled request error on %s", request.url.path)
    if request.url.path == "/":
        response = PlainTextResponse("Internal server error", status_code=500)
    else:
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )
    add_security_headers(request, response)
    return response


@app.on_event("startup")
async def startup_event():
    if PRELOAD_WHISPER:
        asyncio.create_task(warm_transcription_model())


@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(TEMPLATE_PATH, encoding="utf-8", errors="replace") as template_file:
        return template_file.read()


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    detail: Optional[str] = Form("standard"),
):
    normalized_language = normalize_language_value(language)
    normalized_detail = normalize_detail_value(detail)
    try:
        audio_bytes, original_name, internal_name = await read_validated_upload(file)
    except HTTPException as exc:
        app_logger.warning("Rejected upload: %s", sanitize_public_error_text(str(exc.detail)))
        raise

    task_id = str(uuid.uuid4())
    tasks[task_id] = create_task_state(
        task_id,
        normalized_detail,
        source_mode="file",
        source_name=original_name,
        audio_size_bytes=len(audio_bytes),
    )
    app_logger.info("Accepted file upload task %s", task_id)

    logger = WebLogger(task_id)

    asyncio.create_task(
        run_processing(
            task_id=task_id,
            audio_bytes=audio_bytes,
            language=normalized_language,
            detail_level=normalized_detail,
            audio_filename=internal_name,
            audio_mime_type=file.content_type,
            source_name=original_name,
            logger=logger,
        )
    )

    return {"task_id": task_id}


@app.post("/upload-url")
async def upload_url(
    audio_url: str = Form(...),
    language: Optional[str] = Form(None),
    detail: Optional[str] = Form("standard"),
):
    normalized_language = normalize_language_value(language)
    normalized_detail = normalize_detail_value(detail)

    try:
        normalized_url = await validate_remote_audio_url(audio_url)
    except RuntimeError as exc:
        app_logger.warning("Rejected remote audio URL: %s", sanitize_public_error_text(str(exc)))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    task_id = str(uuid.uuid4())
    tasks[task_id] = create_task_state(
        task_id,
        normalized_detail,
        source_mode="url",
        source_name=summarize_source_url(normalized_url),
        source_url=normalized_url,
    )
    app_logger.info("Accepted remote URL task %s", task_id)

    logger = WebLogger(task_id)

    asyncio.create_task(
        run_processing(
            task_id=task_id,
            audio_bytes=None,
            language=normalized_language,
            detail_level=normalized_detail,
            audio_filename=None,
            audio_mime_type=None,
            source_url=normalized_url,
            source_name=normalized_url,
            logger=logger,
        )
    )

    return {"task_id": task_id}


async def run_processing(
    task_id: str,
    audio_bytes: Optional[bytes],
    language: Optional[str],
    detail_level: str,
    audio_filename: Optional[str],
    audio_mime_type: Optional[str],
    logger: WebLogger,
    source_url: Optional[str] = None,
    source_name: Optional[str] = None,
):
    started_at = tasks.get(task_id, {}).get("created_at", datetime.now(timezone.utc))

    try:
        set_task_stage(task_id, "starting", 20)
        resolved_audio_bytes = audio_bytes
        resolved_audio_filename = audio_filename
        resolved_audio_mime_type = audio_mime_type
        resolved_source_name = source_name or audio_filename or "audio"

        if source_url:
            set_task_stage(task_id, "prepare_url", 30)
            (
                resolved_audio_bytes,
                resolved_audio_filename,
                resolved_audio_mime_type,
            ) = await download_audio_from_url(source_url, logger)
            resolved_source_name = resolved_audio_filename or source_url
            tasks[task_id].update(
                {
                    "source_name": sanitize_source_name(resolved_source_name, fallback="remote-audio"),
                    "audio_size_bytes": len(resolved_audio_bytes),
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        elif resolved_audio_bytes is not None:
            tasks[task_id].update(
                {
                    "source_name": sanitize_source_name(resolved_source_name, fallback="audio"),
                    "audio_size_bytes": len(resolved_audio_bytes),
                    "updated_at": datetime.now(timezone.utc),
                }
            )

        if resolved_audio_bytes is None:
            raise RuntimeError("No audio data was provided.")

        result = await process(
            resolved_audio_bytes,
            ai,
            logger,
            language=language,
            detail_level=detail_level,
            audio_filename=resolved_audio_filename,
            audio_mime_type=resolved_audio_mime_type,
        )

        transcript_filename = f"transcript_{uuid.uuid4()}.txt"
        transcript_path = os.path.join(RESULTS_DIR, transcript_filename)

        with open(transcript_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(result.transcript)

        tex_filename = f"result_{uuid.uuid4()}.tex"
        tex_path = os.path.join(RESULTS_DIR, tex_filename)

        with open(tex_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(result.tex)

        pdf_filename = None
        pdf_path = None

        if result.pdf:
            pdf_filename = f"result_{uuid.uuid4()}.pdf"
            pdf_path = os.path.join(RESULTS_DIR, pdf_filename)

            with open(pdf_path, "wb") as file_handle:
                file_handle.write(result.pdf)

        warning_message = result.pdf_warning
        if not pdf_filename and warning_message is None:
            pdf_error = next(
                (
                    message
                    for message in reversed(logger.messages)
                    if message.startswith("Failed to convert LaTeX to PDF:")
                    or message.startswith("PDF generation skipped:")
                ),
                None,
            )
            if pdf_error:
                warning_message = pdf_error
            else:
                warning_message = (
                    "PDF was not generated. The TEX file was created successfully, "
                    "but the exact reason was not captured."
                )

        completed_at = datetime.now(timezone.utc)
        if task_id not in tasks:
            app_logger.warning("Task %s completed after its web state was removed", task_id)
            return

        tasks[task_id].update(
            {
                "status": "done",
                "progress": 100,
                "stage_code": "done",
                "stage": TASK_STAGE_CONFIG["done"]["label"],
                "title": result.title,
                "language": result.language,
                "detail": detail_level,
                "bundle_url": f"/bundle/{task_id}",
                "transcript_url": f"/static/{transcript_filename}",
                "tex_url": f"/static/{tex_filename}",
                "pdf_url": f"/static/{pdf_filename}" if pdf_filename else None,
                "transcript_path": transcript_path,
                "tex_path": tex_path,
                "pdf_path": pdf_path if pdf_filename else None,
                "warning": warning_message,
                "source_name": sanitize_source_name(resolved_source_name, fallback="audio"),
                "audio_size_bytes": len(resolved_audio_bytes),
                "abstract": result.abstract,
                "transcript_preview": make_preview(result.transcript),
                "transcript_words": count_words(result.transcript),
                "abstract_words": count_words(result.abstract),
                "updated_at": completed_at,
                "completed_at": completed_at,
                "duration_seconds": max(
                    0,
                    round((completed_at - started_at).total_seconds()),
                ),
            }
        )
        app_logger.info("Task %s completed successfully", task_id)
    except Exception as exc:
        completed_at = datetime.now(timezone.utc)
        public_error = build_public_error_message(exc)
        app_logger.exception("Task %s failed", task_id)
        if task_id in tasks:
            tasks[task_id].update(
                {
                    "status": "error",
                    "stage_code": "error",
                    "stage": TASK_STAGE_CONFIG["error"]["label"],
                    "error": public_error,
                    "updated_at": completed_at,
                    "completed_at": completed_at,
                    "duration_seconds": max(
                        0,
                        round((completed_at - started_at).total_seconds()),
                    ),
                }
            )
        else:
            app_logger.warning("Task %s failed after its web state was removed: %s", task_id, public_error)


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    return serialize_task(get_task_or_404(task_id))


@app.get("/bundle/{task_id}")
async def get_task_bundle(task_id: str):
    task = get_task_or_404(task_id)
    task_snapshot = dict(task)
    bundle_bytes = await asyncio.to_thread(build_task_bundle_bytes_for_task, task_snapshot)
    archive_name = safe_download_name(task_snapshot.get("title"), f"conspectum-{task_id[:8]}")
    filename = f"{archive_name}.zip"

    return StreamingResponse(
        io.BytesIO(bundle_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/static/{filename}")
async def get_file(filename: str):
    normalized = filename.strip()
    if not normalized or len(normalized) > SAFE_FILENAME_LIMIT or normalized != os.path.basename(normalized):
        raise HTTPException(status_code=404, detail="File not found")

    if normalized in STATIC_ASSET_FILENAMES:
        file_path = resolve_path_within(STATIC_DIR, os.path.join(STATIC_DIR, normalized))
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file_path)

    if not GENERATED_ARTIFACT_RE.fullmatch(normalized):
        raise HTTPException(status_code=404, detail="File not found")

    file_path = resolve_path_within(RESULTS_DIR, os.path.join(RESULTS_DIR, normalized))
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        filename=normalized,
        content_disposition_type="attachment",
        headers={"Cache-Control": "no-store"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
