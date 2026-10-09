"""Command-line interface: ``python -m clinical_rag.cli ingest|ask|eval|serve``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from clinical_rag.config import get_settings
from clinical_rag.evaluate import run_evaluation
from clinical_rag.ingest import ingest
from clinical_rag.logging_utils import configure_logging
from clinical_rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clinical_rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="chunk documents and write the chunk store")
    ask_p = sub.add_parser("ask", help="answer a single question")
    ask_p.add_argument("question")
    ask_p.add_argument("--top-k", type=int, default=None)
    sub.add_parser("eval", help="run the gold-set evaluation and write a markdown report")
    serve_p = sub.add_parser("serve", help="run the HTTP API")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)

    try:
        if args.command == "ingest":
            chunks = ingest(
                settings.docs_dir, settings.chunks_path, settings.chunk_size, settings.chunk_overlap
            )
            print(f"Wrote {len(chunks)} chunks to {settings.chunks_path}")
        elif args.command == "ask":
            result = RAGPipeline.from_settings(settings).ask(args.question, top_k=args.top_k)
            out = {
                "answer": result.answer,
                "refused": result.refused,
                "citations": [c.__dict__ for c in result.citations],
                "retrieval": [
                    {"chunk_id": r.chunk.chunk_id, "score": r.score} for r in result.retrieved
                ],
                "redactions": result.redactions,
                "latency_ms": result.latency_ms,
            }
            print(json.dumps(out, indent=2))
        elif args.command == "eval":
            summary = run_evaluation(
                RAGPipeline.from_settings(settings), settings.eval_path, settings.report_path
            )
            for key, value in summary.items():
                shown = f"{value:.3f}" if isinstance(value, float) else str(value)
                print(f"{key:>20}: {shown}")
            print(f"Report written to {settings.report_path}")
        elif args.command == "serve":
            import uvicorn

            uvicorn.run("clinical_rag.api:app", host=args.host, port=args.port)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("command failed", extra={"command": args.command, "error": str(exc)})
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
