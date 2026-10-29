"""Offline tests: a fake client simulates a model that calls the search tool."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rafeeq.agent import ask  # noqa: E402
from rafeeq.retrieval import Index, load_chunks  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"


class Msg:
    def __init__(self, content=None, tool_query=None):
        self.content = content
        self.tool_calls = None
        if tool_query:
            self.tool_calls = [SimpleNamespace(
                id="c1", function=SimpleNamespace(
                    name="search_docs", arguments=json.dumps({"query": tool_query})))]

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [{"id": "c1", "type": "function",
                                "function": {"name": "search_docs",
                                             "arguments": self.tool_calls[0].function.arguments}}]
        return d


def fake_client(*messages):
    queue = list(messages)

    def create(**_):
        return SimpleNamespace(choices=[SimpleNamespace(message=queue.pop(0))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.index = Index(load_chunks(str(DOCS)))

    def test_valid_citation_is_kept(self):
        client = fake_client(Msg(tool_query="late fine"),
                             Msg(content="The fine is 2 SAR per day [library#3]."))
        ans = ask("What is the late fine?", self.index, client)
        self.assertFalse(ans.refused)
        self.assertEqual([c.id for c in ans.citations], ["library#3"])

    def test_answer_without_valid_citation_is_refused(self):
        client = fake_client(Msg(tool_query="late fine"),
                             Msg(content="The fine is 100 SAR [library#99]."))
        self.assertTrue(ask("What is the late fine?", self.index, client).refused)

    def test_not_found_is_a_refusal(self):
        client = fake_client(Msg(tool_query="printing fees"), Msg(content="NOT_FOUND"))
        self.assertTrue(ask("What are the printing fees?", self.index, client).refused)

    def test_poisoned_chunk_is_quarantined(self):
        client = fake_client(Msg(tool_query="loan system maintenance"), Msg(content="NOT_FOUND"))
        ans = ask("When is maintenance?", self.index, client)
        self.assertEqual(ans.quarantined, ["notice#1"])


if __name__ == "__main__":
    unittest.main()
