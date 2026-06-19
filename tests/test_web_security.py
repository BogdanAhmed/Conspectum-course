import asyncio
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch
import uuid

os.environ.setdefault("AI_API_KEY", "test-api-key")
os.environ.setdefault("ENABLE_API_DOCS", "1")
os.environ.setdefault("APP_ENV", "development")

if "faster_whisper" not in sys.modules:
    fake_faster_whisper = types.ModuleType("faster_whisper")

    class DummyWhisperModel:
        def __init__(self, *args, **kwargs):
            return

    fake_faster_whisper.WhisperModel = DummyWhisperModel
    sys.modules["faster_whisper"] = fake_faster_whisper

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi.testclient import TestClient  # noqa: E402

import web  # noqa: E402


def make_minimal_wav_bytes() -> bytes:
    return b"RIFF\x24\x00\x00\x00WAVEfmt "


class WebSecurityTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(web.app)
        self.original_max_upload_bytes = web.MAX_UPLOAD_BYTES
        self.original_rate_limits = dict(web.RATE_LIMIT_RULES)
        self.original_task_ttl = web.TASK_TTL
        self.original_running_task_ttl = web.RUNNING_TASK_TTL
        web.tasks.clear()
        web.RATE_LIMITER.clear()

    def tearDown(self):
        self.client.close()
        web.MAX_UPLOAD_BYTES = self.original_max_upload_bytes
        web.RATE_LIMIT_RULES.clear()
        web.RATE_LIMIT_RULES.update(self.original_rate_limits)
        web.TASK_TTL = self.original_task_ttl
        web.RUNNING_TASK_TTL = self.original_running_task_ttl
        web.tasks.clear()
        web.RATE_LIMITER.clear()

    def test_root_sets_security_headers(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("default-src 'self'", response.headers["content-security-policy"])
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")

    def test_upload_rejects_fake_audio_payload(self):
        response = self.client.post(
            "/upload",
            data={"language": "en", "detail": "standard"},
            files={"file": ("lecture.wav", b"not really audio", "audio/wav")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("does not look like a supported audio container", response.json()["detail"])

    def test_upload_rejects_suspicious_double_extension(self):
        response = self.client.post(
            "/upload",
            data={"detail": "standard"},
            files={"file": ("lecture.exe.mp3", b"ID3fake", "audio/mpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Suspicious file name", response.json()["detail"])

    def test_upload_rejects_oversized_payload(self):
        web.MAX_UPLOAD_BYTES = 8
        response = self.client.post(
            "/upload",
            data={"detail": "standard"},
            files={"file": ("lecture.wav", make_minimal_wav_bytes(), "audio/wav")},
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("too large", response.json()["detail"])

    def test_upload_rejects_invalid_language_value(self):
        response = self.client.post(
            "/upload",
            data={"language": "de", "detail": "standard"},
            files={"file": ("lecture.wav", make_minimal_wav_bytes(), "audio/wav")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Language must be en or ru", response.json()["detail"])

    def test_upload_accepts_valid_audio_and_returns_task_id(self):
        def create_background_task(coro):
            return asyncio.get_running_loop().create_task(coro)

        with patch.object(web, "run_processing", new=AsyncMock(return_value=None)):
            with patch.object(web.asyncio, "create_task", side_effect=create_background_task):
                response = self.client.post(
                    "/upload",
                    data={"language": "en", "detail": "brief"},
                    files={"file": ("lecture.wav", make_minimal_wav_bytes(), "audio/wav")},
                )

        self.assertEqual(response.status_code, 200)
        task_id = response.json()["task_id"]
        self.assertTrue(task_id)
        self.assertIn(task_id, web.tasks)

    def test_status_response_does_not_leak_internal_paths(self):
        task_id = str(uuid.uuid4())
        now = web.datetime.now(web.timezone.utc)
        web.tasks[task_id] = {
            "task_id": task_id,
            "status": "done",
            "messages": ["ok"],
            "progress": 100,
            "stage_code": "done",
            "stage": "Готово",
            "bundle_url": f"/bundle/{task_id}",
            "tex_url": "/static/result_test.tex",
            "pdf_url": "/static/result_test.pdf",
            "transcript_url": "/static/transcript_test.txt",
            "tex_path": "C:/secret/tex.tex",
            "pdf_path": "C:/secret/pdf.pdf",
            "transcript_path": "C:/secret/transcript.txt",
            "source_url": "https://example.com/private?token=secret",
            "title": "Title",
            "language": "en",
            "detail": "standard",
            "error": None,
            "warning": None,
            "source_mode": "file",
            "source_name": "lecture.wav",
            "audio_size_bytes": 123,
            "abstract": "Summary",
            "transcript_preview": "Preview",
            "transcript_words": 10,
            "abstract_words": 5,
            "created_at": now,
            "updated_at": now,
            "completed_at": now,
            "duration_seconds": 12,
        }

        response = self.client.get(f"/status/{task_id}")
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("tex_path", body)
        self.assertNotIn("pdf_path", body)
        self.assertNotIn("transcript_path", body)
        self.assertNotIn("source_url", body)

    def test_provider_prompt_limit_error_is_not_reported_as_authentication_failure(self):
        class APIStatusError(Exception):
            status_code = 402

        public_error = web.build_public_error_message(
            APIStatusError("Error code: 402 - Prompt tokens limit exceeded: 10045 > 8355")
        )

        self.assertIn("prompt token limit", public_error)
        self.assertNotIn("authentication failed", public_error.lower())

    def test_json_decode_error_is_reported_as_ai_provider_response_failure(self):
        public_error = web.build_public_error_message(json.JSONDecodeError("Expecting value", "\n" * 268, 1474))

        self.assertIn("AI provider returned an invalid response", public_error)
        self.assertNotIn("Expecting value", public_error)

    def test_static_route_blocks_traversal_attempt(self):
        response = self.client.get("/static/..%5C.env")
        self.assertEqual(response.status_code, 404)

    def test_upload_url_rejects_private_network_target(self):
        response = self.client.post(
            "/upload-url",
            data={"audio_url": "http://127.0.0.1/audio.mp3", "detail": "standard"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Private and local network URLs are not allowed", response.json()["detail"])

    def test_status_endpoint_rate_limits_excessive_polling(self):
        task_id = str(uuid.uuid4())
        now = web.datetime.now(web.timezone.utc)
        web.tasks[task_id] = {
            "task_id": task_id,
            "status": "running",
            "messages": [],
            "progress": 10,
            "stage_code": "starting",
            "stage": "Началась обработка",
            "bundle_url": None,
            "tex_url": None,
            "pdf_url": None,
            "transcript_url": None,
            "title": None,
            "language": None,
            "detail": "standard",
            "error": None,
            "warning": None,
            "source_mode": "file",
            "source_name": "lecture.wav",
            "audio_size_bytes": 123,
            "abstract": None,
            "transcript_preview": None,
            "transcript_words": None,
            "abstract_words": None,
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
            "duration_seconds": None,
        }
        web.RATE_LIMIT_RULES[("GET", "/status")] = web.RateLimitRule("status-test", 2, 60)
        web.RATE_LIMITER.clear()

        first = self.client.get(f"/status/{task_id}")
        second = self.client.get(f"/status/{task_id}")
        third = self.client.get(f"/status/{task_id}")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(third.status_code, 429)
        self.assertIn("Retry-After", third.headers)

    def test_cleanup_does_not_remove_active_running_task_by_creation_time(self):
        task_id = str(uuid.uuid4())
        now = web.datetime.now(web.timezone.utc)
        web.RUNNING_TASK_TTL = web.timedelta(hours=2)
        web.TASK_TTL = web.timedelta(minutes=1)
        web.tasks[task_id] = {
            "task_id": task_id,
            "status": "running",
            "created_at": now - web.timedelta(days=2),
            "updated_at": now,
            "tex_path": None,
            "pdf_path": None,
            "transcript_path": None,
        }

        web.cleanup_expired_tasks()

        self.assertIn(task_id, web.tasks)

    def test_cleanup_removes_completed_task_after_result_ttl(self):
        task_id = str(uuid.uuid4())
        now = web.datetime.now(web.timezone.utc)
        web.TASK_TTL = web.timedelta(hours=1)
        web.tasks[task_id] = {
            "task_id": task_id,
            "status": "done",
            "created_at": now - web.timedelta(days=2),
            "updated_at": now - web.timedelta(hours=2),
            "completed_at": now - web.timedelta(hours=2),
            "tex_path": None,
            "pdf_path": None,
            "transcript_path": None,
        }

        web.cleanup_expired_tasks()

        self.assertNotIn(task_id, web.tasks)


if __name__ == "__main__":
    unittest.main()
