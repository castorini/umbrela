from __future__ import annotations

import json
import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from textwrap import indent
from typing import Any

import pytest

from umbrela.cli.main import main

pytestmark = pytest.mark.live


@unittest.skipUnless(
    os.getenv("UMBRELA_LIVE_OPENAI_SMOKE") == "1",
    "Set UMBRELA_LIVE_OPENAI_SMOKE=1 to run live OpenAI smoke tests.",
)
class UmbrelaLiveOpenAISmokeTests(unittest.TestCase):
    def _pretty_render(self, payload: dict[str, Any]) -> str:
        artifact = payload["artifacts"][0]
        judgments = artifact["data"]
        model_name = None
        if judgments:
            model_name = judgments[0].get("model")
        lines = [
            "umbrela live smoke result",
            f"model: {model_name or 'unknown'}",
        ]
        for index, judgment in enumerate(judgments, start=1):
            lines.extend(
                [
                    "",
                    f"candidate {index}",
                    f"query: {judgment['query']}",
                    "passage:",
                    indent(str(judgment["passage"]), "  "),
                    f"judgment: {judgment['judgment']}",
                    f"parsed: {bool(judgment['result_status'])}",
                ]
            )
            reasoning = judgment.get("reasoning")
            if reasoning:
                lines.extend(["reasoning:", indent(str(reasoning), "  ")])
        return "\n".join(lines)

    def test_direct_judge_openai_smoke(self) -> None:
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
            self.skipTest("OPENAI_API_KEY or OPENROUTER_API_KEY is required.")

        model = os.getenv("UMBRELA_LIVE_OPENAI_MODEL", "gpt-4o-mini")
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "judge",
                    "--backend",
                    "gpt",
                    "--model",
                    model,
                    "--execution-mode",
                    "async",
                    "--max-concurrency",
                    "4",
                    "--input-json",
                    json.dumps(
                        {
                            "query": "how long is life cycle of flea",
                            "candidates": [
                                "The life cycle of a flea can last anywhere from 20 days to an entire year.",
                                "The life cycle of a flea can last anywhere from 20 days to an entire year, depending on how long it remains in dormant stages such as eggs, larvae, or pupae. Adult fleas may also survive for months when a host is available.",
                                "Flea development includes egg, larva, pupa, and adult stages. Eggs usually hatch in two to six days, and larvae can remain in cocoons for extended periods before emerging.",
                                "Cats and dogs often need regular grooming and parasite prevention. Vacuuming carpets and washing pet bedding can help control pests in the home.",
                            ],
                        }
                    ),
                    "--output",
                    "json",
                ]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["command"], "judge")
        self.assertEqual(payload["status"], "success")
        judgments = payload["artifacts"][0]["data"]
        self.assertEqual(len(judgments), 4)
        self.assertIn("judgment", judgments[0])
        print(self._pretty_render(payload))

    def test_direct_judge_openai_reasoning_smoke(self) -> None:
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
            self.skipTest("OPENAI_API_KEY or OPENROUTER_API_KEY is required.")

        model = os.getenv("UMBRELA_LIVE_OPENAI_REASONING_MODEL", "gpt-5-mini")
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "judge",
                    "--backend",
                    "gpt",
                    "--model",
                    model,
                    "--execution-mode",
                    "async",
                    "--max-concurrency",
                    "2",
                    "--reasoning-effort",
                    "medium",
                    "--include-reasoning",
                    "--input-json",
                    json.dumps(
                        {
                            "query": "how long is life cycle of flea",
                            "candidates": [
                                "The life cycle of a flea can last anywhere from 20 days to an entire year.",
                                "Cats and dogs often need regular grooming and parasite prevention. Vacuuming carpets and washing pet bedding can help control pests in the home.",
                            ],
                        }
                    ),
                    "--output",
                    "json",
                ]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["command"], "judge")
        self.assertEqual(payload["status"], "success")
        judgments = payload["artifacts"][0]["data"]
        self.assertEqual(len(judgments), 2)
        self.assertTrue(
            any((judgment.get("reasoning") or "").strip() for judgment in judgments)
        )
        print(self._pretty_render(payload))

    def test_direct_judge_openrouter_reasoning_smoke(self) -> None:
        if not os.getenv("OPENROUTER_API_KEY"):
            self.skipTest("OPENROUTER_API_KEY is required.")

        model = os.getenv(
            "UMBRELA_LIVE_OPENROUTER_REASONING_MODEL", "openrouter/hunter-alpha"
        )
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "judge",
                    "--backend",
                    "gpt",
                    "--model",
                    model,
                    "--use-openrouter",
                    "--execution-mode",
                    "async",
                    "--max-concurrency",
                    "2",
                    "--reasoning-effort",
                    "medium",
                    "--include-reasoning",
                    "--input-json",
                    json.dumps(
                        {
                            "query": "how long is life cycle of flea",
                            "candidates": [
                                "The life cycle of a flea can last anywhere from 20 days to an entire year.",
                                "Cats and dogs often need regular grooming and parasite prevention. Vacuuming carpets and washing pet bedding can help control pests in the home.",
                            ],
                        }
                    ),
                    "--output",
                    "json",
                ]
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["command"], "judge")
        self.assertEqual(payload["status"], "success")
        judgments = payload["artifacts"][0]["data"]
        self.assertEqual(len(judgments), 2)
        self.assertTrue(
            any((judgment.get("reasoning") or "").strip() for judgment in judgments)
        )
        print(self._pretty_render(payload))
