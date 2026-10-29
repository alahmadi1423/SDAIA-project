"""Usage: python cli.py                  (interactive)
       python cli.py -q "your question" (single question)"""
import argparse

from rafeeq.agent import ask, make_client
from rafeeq.retrieval import Index, load_chunks


def show(ans):
    print("\n" + ans.text)
    for c in ans.citations:
        print(f"  [{c.id}] {c.source} - {c.heading}")
    if ans.quarantined:
        print(f"  (quarantined suspicious passages: {', '.join(ans.quarantined)})")
    print()


def main():
    ap = argparse.ArgumentParser(description="Rafeeq - document-grounded assistant")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("-q", "--question")
    args = ap.parse_args()
    index = Index(load_chunks(args.docs))
    client = make_client()
    if args.question:
        return show(ask(args.question, index, client))
    while (q := input("Question (or 'exit'): ").strip()) and q != "exit":
        show(ask(q, index, client))


if __name__ == "__main__":
    main()
