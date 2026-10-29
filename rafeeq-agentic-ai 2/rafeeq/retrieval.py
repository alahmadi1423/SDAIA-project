"""Lightweight BM25 retrieval over Markdown/text documents. No model downloads."""
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

K1, B = 1.5, 0.75
_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOP = {"the", "a", "an", "of", "is", "are", "to", "in", "on", "at", "and", "or", "for",
         "how", "what", "when", "do", "does", "can", "many", "much", "i", "my", "it",
         "by", "be", "with", "per", "there", "this", "that"}


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def _stem(tok: str) -> str:
    for suffix in ("ing", "ed", "s"):
        if tok.endswith(suffix) and not tok.endswith("ss") and len(tok) - len(suffix) >= 3:
            return tok[: -len(suffix)]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(normalize(text)) if t not in _STOP]


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    heading: str
    text: str


def _split(text: str, max_chars: int) -> list[tuple[str, str]]:
    """Split a document into (heading, body) pieces; headings (#) start a new piece."""
    sections, heading, buf = [], "", []
    for line in text.splitlines():
        if line.startswith("#"):
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
                buf = []
            heading = line.lstrip("#").strip()
        elif line.strip():
            buf.append(line.strip())
    if buf:
        sections.append((heading, "\n".join(buf).strip()))
    out = []
    for h, body in sections:
        while len(body) > max_chars:
            cut = body.rfind(".", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars - 1
            out.append((h, body[: cut + 1].strip()))
            body = body[cut + 1:].strip()
        if body:
            out.append((h, body))
    return out


def load_chunks(folder: str, max_chars: int = 700) -> list[Chunk]:
    chunks = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        stem = path.stem.replace(" ", "-")
        for n, (heading, body) in enumerate(_split(path.read_text("utf-8"), max_chars), 1):
            chunks.append(Chunk(f"{stem}#{n}", path.name, heading, body))
    return chunks


class Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.docs = [tokenize(c.heading + " " + c.text) for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.avg = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        self.df = Counter(t for d in self.docs for t in set(d))

    def search(self, query: str, k: int = 4) -> list[tuple[float, Chunk]]:
        q, n, scored = set(tokenize(query)), len(self.docs), []
        for i, tf in enumerate(self.tf):
            dl, score = len(self.docs[i]), 0.0
            for t in q:
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                score += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / self.avg))
            if score > 0:
                scored.append((score, self.chunks[i]))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]
