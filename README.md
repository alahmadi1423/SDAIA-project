# Rafeeq - a document-grounded AI agent with citations

Rafeeq answers questions **only** from a folder of documents (Markdown/text), cites the passage behind every
answer, and refuses when the documents don't contain the answer. It runs on free models through OpenRouter.

## How it works
```
question -> agent calls search_docs -> BM25 returns the best passages -> guard quarantines suspicious ones
         -> model writes an answer with [id] citations -> validator: every citation must come from the
            retrieved passages, otherwise the answer becomes a refusal
```

| File | Purpose |
|---|---|
| `rafeeq/retrieval.py` | Heading-aware chunking + BM25 ranking with light stemming |
| `rafeeq/guard.py` | Prompt-injection detection; retrieved text is wrapped as data |
| `rafeeq/agent.py` | Tool-calling loop, model fallbacks, citation validation, refusals |
| `eval/` | Evaluation tasks and an automated runner (non-zero exit on failure, CI-friendly) |
| `tests/` | Offline unit tests using a fake model client |

## Run it
```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...          # free key from openrouter.ai
python cli.py -q "What is the late fine?"
```
Optional: `export RAFEEQ_MODELS="model1,model2"` to change the model and fallback order.

## Evaluate and test
```bash
python eval/run_eval.py --retrieval-only   # offline
python eval/run_eval.py                    # full agent
python -m unittest discover tests
```
Task types: **answerable** (must cite the right passage), **unanswerable** (must refuse),
**injection** (must not be influenced by instructions hidden inside a document).

## Make it yours
1. Replace the sample files in `docs/` with your own documents (`#`/`##` headings become passage boundaries).
2. Rewrite `eval/tasks.json` with questions about your documents. `expect_contains` is text that must appear in the correct passage.
3. Hide a malicious instruction in one of your documents and point `poisoned_contains` at it.

## Known limitations
- BM25 matches words, not meaning, so synonyms can be missed. Natural upgrade: embeddings or hybrid search.
- The guard is pattern-based: it quarantines the whole passage (even useful text in it) and won't catch every injection phrasing.
- Free OpenRouter models are rate-limited and some handle tool calling poorly, hence the fallback list.
- Retrieval stemming is English-only.
