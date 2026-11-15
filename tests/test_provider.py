from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from drawingqa.config import config_fingerprint, load_config
from drawingqa.providers import InvalidExtraction, OpenAICompatibleProvider, ProviderError, parse_extraction
from drawingqa.schemas import ModelConfig


def completion(content='{"answer":0,"candidates":[],"complete":true}', **overrides):
    payload = {
        "choices": [{"finish_reason": "stop", "message": {"content": content, "reasoning_content": "private text"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }
    payload.update(overrides)
    return payload


class ExtractionTests(unittest.TestCase):
    def test_fenced_json_and_thinking_wrapper(self):
        answer = parse_extraction('<think>private reasoning</think>\n```json\n{"answer":0,"complete":true}\n```')
        self.assertEqual(answer.answer, 0)
        self.assertTrue(answer.complete)

    def test_multiple_objects_or_prose_are_not_silently_accepted(self):
        for text in ['{"answer":0} {"answer":1}', 'Answer: {"answer":0}', '[{"answer":0}]', '<think>unfinished']:
            with self.subTest(text=text), self.assertRaises(InvalidExtraction):
                parse_extraction(text)

    def test_reject_bad_boxes_nonfinite_extra_fields_and_duplicate_keys(self):
        for text in [
            '{"answer":NaN}', '{"answer":0,"answer":1}', '{"answer":0,"verified":true}',
            '{"candidates":[{"label":"window","bbox":[10,20,30,40]}]}',
            '{"candidates":[{"label":"window","bbox":[0.8,0.1,0.2,0.5]}]}',
        ]:
            with self.subTest(text=text), self.assertRaises(InvalidExtraction):
                parse_extraction(text)

    def test_per_mille_boxes_are_rescaled_and_still_validated(self):
        text = '{"candidates":[{"label":"window","bbox":[135,180,150,725]}]}'
        box = parse_extraction(text, bbox_scale=1000).candidates[0].bbox
        self.assertEqual(box, (0.135, 0.18, 0.15, 0.725))
        with self.assertRaises(InvalidExtraction):
            parse_extraction('{"candidates":[{"label":"window","bbox":[0,0,1200,500]}]}', bbox_scale=1000)


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.image = Path(self.tmp.name) / "view.png"
        Image.new("RGB", (8, 8), "white").save(self.image)

    def tearDown(self):
        self.tmp.cleanup()

    def provider(self, handler, **config):
        self.addCleanup(lambda: p.close())
        p = OpenAICompatibleProvider(ModelConfig(model="test", **config), transport=httpx.MockTransport(handler), sleep=lambda _: None)
        return p

    def test_request_images_output_content_and_usage(self):
        def handler(request):
            self.assertEqual(str(request.url), "http://localhost:8001/v1/chat/completions")
            body = json.loads(request.content)
            self.assertFalse(body["stream"])
            self.assertEqual(body["model"], "test")
            self.assertTrue(body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))
            self.assertEqual(body["response_format"], {"type": "json_object"})
            return httpx.Response(200, json=completion())
        response = self.provider(handler, json_mode=True, input_cost_per_million=1, output_cost_per_million=2).generate("system", "question", [self.image])
        self.assertNotIn("private text", response.content)
        self.assertEqual(response.usage.requests, 1)
        self.assertEqual(response.usage.input_tokens, 100)
        self.assertAlmostEqual(response.usage.estimated_cost_usd, 0.00014)

    def test_retry_rate_limit_then_success_does_not_claim_full_cost(self):
        calls = []
        def handler(request):
            calls.append(1)
            return httpx.Response(429, headers={"Retry-After": "0"}) if len(calls) == 1 else httpx.Response(200, json=completion())
        result = self.provider(handler, max_retries=1).generate("system", "question", [self.image])
        self.assertEqual(result.usage.requests, 2)
        self.assertEqual(result.usage.retries, 1)
        self.assertFalse(result.usage.usage_reported)
        self.assertIsNone(result.usage.estimated_cost_usd)

    def test_exhausted_network_retry_is_redacted(self):
        def handler(request):
            raise httpx.ConnectError("SECRET must not leak", request=request)
        with self.assertRaises(ProviderError) as result:
            self.provider(handler, max_retries=2).generate("s", "u", [])
        self.assertEqual(result.exception.usage.requests, 3)
        self.assertNotIn("SECRET", str(result.exception))

    def test_client_error_not_retried_and_body_redacted(self):
        calls = []
        def handler(request):
            calls.append(1)
            return httpx.Response(401, text="SECRET key and payload")
        with self.assertRaises(ProviderError) as result:
            self.provider(handler).generate("s", "u", [])
        self.assertEqual(len(calls), 1)
        self.assertNotIn("SECRET", str(result.exception))

    def test_missing_environment_key_does_not_send(self):
        def handler(request):
            self.fail("request sent without required credentials")
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ProviderError):
            self.provider(handler, api_key_env="DRAWINGQA_TEST_KEY").generate("s", "u", [])

    def test_truncated_response_preserves_usage(self):
        response = completion()
        response["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(ProviderError) as result:
            self.provider(lambda _: httpx.Response(200, json=response)).generate("s", "u", [])
        self.assertEqual(result.exception.usage.output_tokens, 20)
        self.assertIn("truncated", str(result.exception))

    def test_missing_usage_is_unknown_and_block_content_supported(self):
        response = completion(content=[{"type": "text", "text": '{"answer":1}'}], usage={})
        result = self.provider(lambda _: httpx.Response(200, json=response)).generate("s", "u", [])
        self.assertEqual(result.content, '{"answer":1}')
        self.assertFalse(result.usage.usage_reported)

    def test_async_pending_response_is_explicit(self):
        with self.assertRaisesRegex(ProviderError, "asynchronous"):
            self.provider(lambda _: httpx.Response(202, json={"status": "pending"})).generate("s", "u", [])

    def test_config_hash_never_reads_secret_environment_value(self):
        config = ModelConfig(model="test", api_key_env="DRAWINGQA_TEST_KEY")
        with patch.dict(os.environ, {"DRAWINGQA_TEST_KEY": "first-secret"}):
            first = config_fingerprint(config)
        with patch.dict(os.environ, {"DRAWINGQA_TEST_KEY": "second-secret"}):
            second = config_fingerprint(config)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
