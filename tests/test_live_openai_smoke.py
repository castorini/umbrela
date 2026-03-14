from __future__ import annotations

import json
import os
import unittest
from contextlib import redirect_stdout
from io import StringIO

from umbrela.cli.main import main


@unittest.skipUnless(
    os.getenv("UMBRELA_LIVE_OPENAI_SMOKE") == "1",
    "Set UMBRELA_LIVE_OPENAI_SMOKE=1 to run live OpenAI smoke tests.",
)
class UmbrelaLiveOpenAISmokeTests(unittest.TestCase):
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
                    "--input-json",
                    json.dumps(
                        {
                            "query": "how long is life cycle of flea",
                            "candidates": [
                                "The life cycle of a flea can last anywhere from 20 days to an entire year."
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
        self.assertEqual(len(judgments), 1)
        self.assertIn("judgment", judgments[0])
