"""Unit tests for the AnswererWrapper Python adapter."""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional

from ocs_toolkit import AnswererWrapperAdapter, AnswererWrapperConfig


class RecordingRequester:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload
        self.last_body: Optional[Dict[str, Any]] = None

    def fetch(self, config: AnswererWrapperConfig, body: Dict[str, Any]) -> bytes:
        self.last_body = body
        return json.dumps(self.payload).encode("utf-8")


class AnswererWrapperAdapterTests(unittest.TestCase):
    def test_run_returns_single_match(self) -> None:
        config = AnswererWrapperConfig.from_mapping(
            {
                "name": "demo",
                "url": "http://example.com",
                "method": "get",
                "contentType": "json",
                "handler": "lambda res: [res['question'], res['answer']] if res.get('code') == 1 else None",
            }
        )
        env = {"title": "1+2", "type": "single", "options": "A\nB"}
        payload = {"code": 1, "question": "1+2", "answer": "3"}

        requester = RecordingRequester(payload)
        adapter = AnswererWrapperAdapter([config], requester=requester)
        result = adapter.run(env)

        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0].question, "1+2")
        self.assertEqual(result.matches[0].answer, "3")
        self.assertEqual(result.matches[0].source, "demo")
        self.assertEqual(result.errors, [])

    def test_data_handler_transforms_payload(self) -> None:
        config = AnswererWrapperConfig.from_mapping(
            {
                "name": "custom-data",
                "url": "http://example.com",
                "method": "post",
                "contentType": "json",
                "data": {
                    "options": {"handler": "lambda env: env['options'].split('\\n')"},
                    "title": "${title}",
                },
                "handler": "lambda res: [res.get('question'), res.get('answer')]",
            }
        )
        env = {"title": "Question", "type": "single", "options": "A. 1\nB. 2"}
        payload = {"code": 1, "question": "Question", "answer": "3"}
        requester = RecordingRequester(payload)

        adapter = AnswererWrapperAdapter([config], requester=requester)
        result = adapter.run(env)

        self.assertEqual(len(result.matches), 1)
        self.assertEqual(requester.last_body, {"options": ["A. 1", "B. 2"], "title": "Question"})

    def test_handler_failure_is_reported(self) -> None:
        config = AnswererWrapperConfig.from_mapping(
            {
                "name": "broken",
                "url": "http://example.com",
                "method": "post",
                "contentType": "json",
                "handler": "lambda res: 'oops'",
            }
        )
        env = {"title": "Question", "type": "single", "options": ""}
        payload = {"code": 1, "question": "Question", "answer": "3"}
        requester = RecordingRequester(payload)

        adapter = AnswererWrapperAdapter([config], requester=requester)
        result = adapter.run(env)

        self.assertEqual(result.matches, [])
        self.assertEqual(len(result.errors), 1)
        self.assertIn("Handler returned", result.errors[0].message)


if __name__ == "__main__":
    unittest.main()
