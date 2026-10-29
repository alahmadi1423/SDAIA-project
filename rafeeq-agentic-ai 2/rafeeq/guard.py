"""Prompt-injection guard: retrieved document text is data, never instructions."""
import re

from .retrieval import Chunk, normalize

_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+.{0,30}instructions",
    r"(reveal|show|print|leak).{0,30}(system prompt|api[ _-]?key)",
    r"you are now\b",
    r"forget\s+(everything|all)\b",
]
_COMPILED = [re.compile(p) for p in _PATTERNS]


def flag_injection(text: str) -> bool:
    norm = normalize(text)
    return any(p.search(norm) for p in _COMPILED)


def wrap_untrusted(chunk: Chunk) -> str:
    safe = chunk.text.replace("</doc", "< /doc")
    return f'<doc id="{chunk.id}" source="{chunk.source}">\n{safe}\n</doc>'
