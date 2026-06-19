import asyncio
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch

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

from conspectum.process import build_chunk_formatting_rules  # noqa: E402
from conspectum.process import extract_adjacent_transcript_context  # noqa: E402
from conspectum.process import get_chunk_ai_model_attempts  # noqa: E402
from conspectum.process import get_chunk_ai_timeout_seconds  # noqa: E402
from conspectum.process import get_chunk_max_tokens  # noqa: E402
from conspectum.process import get_chunk_process_concurrency  # noqa: E402
from conspectum.process import get_chunk_target_chars  # noqa: E402
from conspectum.process import process_chunks  # noqa: E402
from conspectum.process import split_long_sentence  # noqa: E402
from conspectum.summary import build_summary_transcript_context  # noqa: E402
from conspectum.summary import create_chat_completion  # noqa: E402
from conspectum.summary import map_transcription_language_to_output_language  # noqa: E402
from conspectum.summary import trim_middle  # noqa: E402


class ProcessOptimizationTests(unittest.TestCase):
    def test_chunk_rules_are_compact_and_do_not_embed_full_template(self):
        rules = build_chunk_formatting_rules("ru")

        self.assertLess(len(rules), 1200)
        self.assertNotIn("DOCUMENT STRUCTURE GUIDELINES", rules)
        self.assertNotIn("<INSERT TITLE HERE>", rules)
        self.assertIn("Return only LaTeX", rules)

    def test_brief_chunk_rules_are_strictly_compact(self):
        rules = build_chunk_formatting_rules("en", "brief")

        self.assertIn("compact conspectus", rules)
        self.assertIn("55--65% of the detailed-mode length", rules)
        self.assertIn("omit examples", rules)
        self.assertNotIn("not a terse recap", rules)

    def test_brief_mode_uses_fewer_larger_requests(self):
        with patch.dict(
            "os.environ",
            {
                "CHUNK_TARGET_CHARS": "3000",
                "CHUNK_MAX_TOKENS": "1800",
                "CHUNK_PROCESS_CONCURRENCY": "1",
            },
            clear=True,
        ):
            self.assertGreater(get_chunk_target_chars("brief"), get_chunk_target_chars("standard"))
            self.assertLess(get_chunk_max_tokens("brief"), get_chunk_max_tokens("standard"))
            self.assertEqual(get_chunk_target_chars("brief"), 4800)
            self.assertEqual(get_chunk_max_tokens("brief"), 850)
            self.assertEqual(get_chunk_process_concurrency(8, "brief"), 2)
            self.assertEqual(get_chunk_ai_timeout_seconds(), 90.0)
            self.assertEqual(get_chunk_ai_model_attempts(), 3)

    def test_section_progress_starts_at_zero_and_tracks_completed_chunks(self):
        class RecordingLogger:
            def __init__(self):
                self.progress_updates = []

            async def stage(self, *_args):
                return

            async def progress(self, completed, total):
                self.progress_updates.append((completed, total))

            async def partial_result(self, *_args):
                return

            async def file(self, *_args):
                return

        logger = RecordingLogger()
        fake_ai = types.SimpleNamespace()

        with (
            patch("conspectum.process.get_chunk_process_concurrency", return_value=1),
            patch("conspectum.process.process_chunk", new=AsyncMock(return_value="section")),
        ):
            results = asyncio.run(
                process_chunks(
                    chunks=["first", "second"],
                    chunk_rules="rules",
                    ai=fake_ai,
                    language="en",
                    detail_level="brief",
                    logger=logger,
                )
            )

        self.assertEqual(results, ["section", "section"])
        self.assertEqual(logger.progress_updates, [(0, 2), (1, 2), (2, 2)])

    def test_long_sentence_is_split_without_losing_words(self):
        sentence = " ".join(f"word{i}" for i in range(300))
        pieces = split_long_sentence(sentence, 250)

        self.assertGreater(len(pieces), 1)
        self.assertEqual(" ".join(pieces), sentence)
        self.assertTrue(all(len(piece) <= 260 for piece in pieces))

    def test_chunk_concurrency_is_bounded_by_env_and_chunk_count(self):
        with patch.dict("os.environ", {"CHUNK_PROCESS_CONCURRENCY": "99"}):
            self.assertEqual(get_chunk_process_concurrency(3), 3)

        with patch.dict("os.environ", {"CHUNK_PROCESS_CONCURRENCY": "not-a-number"}):
            self.assertEqual(get_chunk_process_concurrency(10), 4)

    def test_adjacent_context_uses_neighbor_transcript_not_generated_chunk(self):
        chunks = ["a" * 900, "current chunk", "b" * 900]

        context = extract_adjacent_transcript_context(chunks, 1)

        self.assertIsNotNone(context)
        self.assertIn("Previous transcript tail", context)
        self.assertIn("Next transcript head", context)
        self.assertIn("current chunk", chunks[1])
        self.assertNotIn("current chunk", context or "")
        self.assertLess(len(context or ""), 1600)

    def test_whisper_language_can_replace_llm_language_detection(self):
        self.assertEqual(map_transcription_language_to_output_language("ru"), "ru")
        self.assertEqual(map_transcription_language_to_output_language("en"), "en")
        self.assertEqual(map_transcription_language_to_output_language("uk"), "ru")
        self.assertIsNone(map_transcription_language_to_output_language("de"))

    def test_trim_middle_respects_configured_character_limit(self):
        text = "a" * 100 + "middle" + "z" * 100

        trimmed = trim_middle(text, 80)

        self.assertLessEqual(len(trimmed), 80)
        self.assertIn("omitted", trimmed)
        self.assertTrue(trimmed.startswith("a"))
        self.assertTrue(trimmed.endswith("z"))

    def test_summary_transcript_context_can_be_limited_for_provider_prompt_limits(self):
        transcript = "a" * 100 + "middle" + "z" * 100

        with patch.dict("os.environ", {"SUMMARY_TRANSCRIPT_CONTEXT_CHARS": "80"}):
            context, was_trimmed = build_summary_transcript_context(transcript)

        self.assertTrue(was_trimmed)
        self.assertLessEqual(len(context), 80)
        self.assertIn("omitted", context)

    def test_chat_completion_tries_fallback_model_after_rate_limit(self):
        class RateLimitError(Exception):
            pass

        class FakeCompletions:
            def __init__(self):
                self.models = []

            async def create(self, **kwargs):
                self.models.append(kwargs["model"])
                if kwargs["model"] == "primary-model":
                    raise RateLimitError("temporarily rate-limited upstream")
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
                )

        completions = FakeCompletions()
        ai = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))

        with patch.dict(
            "os.environ",
            {"MODEL_NAME": "primary-model", "AI_MODEL_FALLBACKS": "fallback-model"},
        ):
            response = asyncio.run(
                create_chat_completion(
                    ai,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=10,
                )
            )

        self.assertEqual(response.choices[0].message.content, "ok")
        self.assertEqual(completions.models, ["primary-model", "fallback-model"])

    def test_chat_completion_tries_fallback_model_after_invalid_json_response(self):
        class FakeCompletions:
            def __init__(self):
                self.models = []

            async def create(self, **kwargs):
                self.models.append(kwargs["model"])
                if kwargs["model"] == "primary-model":
                    raise json.JSONDecodeError("Expecting value", "\n" * 268, 1474)
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
                )

        completions = FakeCompletions()
        ai = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))

        with patch.dict(
            "os.environ",
            {"MODEL_NAME": "primary-model", "AI_MODEL_FALLBACKS": "fallback-model"},
        ):
            response = asyncio.run(
                create_chat_completion(
                    ai,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=10,
                )
            )

        self.assertEqual(response.choices[0].message.content, "ok")
        self.assertEqual(completions.models, ["primary-model", "fallback-model"])

    def test_chat_completion_limits_fallback_attempts(self):
        class RateLimitError(Exception):
            pass

        class FakeCompletions:
            def __init__(self):
                self.models = []

            async def create(self, **kwargs):
                self.models.append(kwargs["model"])
                raise RateLimitError("temporarily rate-limited upstream")

        completions = FakeCompletions()
        ai = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))

        with patch.dict(
            "os.environ",
            {
                "MODEL_NAME": "primary-model",
                "AI_MODEL_FALLBACKS": "fallback-one fallback-two fallback-three",
            },
        ):
            with self.assertRaises(RateLimitError):
                asyncio.run(
                    create_chat_completion(
                        ai,
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=10,
                        max_model_attempts=2,
                    )
                )

        self.assertEqual(completions.models, ["primary-model", "fallback-one"])


if __name__ == "__main__":
    unittest.main()
