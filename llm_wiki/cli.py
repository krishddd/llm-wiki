"""Console entry point for the ``llm-wiki`` package.

Exposes two subcommands so the packaged distribution is runnable without
remembering the uvicorn incantation:

    llm-wiki serve [--host H] [--port P] [--reload]   # FastAPI app (llm_wiki.api:app)
    llm-wiki mcp                                       # agent-facing MCP server

Everything the service actually does still lives in the library modules; this
is a thin wrapper so ``pip install llm-wiki`` yields a usable command.
"""

from __future__ import annotations

import argparse
import sys


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:  # pragma: no cover - uvicorn is a core dep
        print("uvicorn is required to serve the API: pip install 'llm-wiki'", file=sys.stderr)
        return 1
    uvicorn.run(
        "llm_wiki.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _mcp(_args: argparse.Namespace) -> int:
    # Delegates to the module's own __main__ guard.
    from llm_wiki import mcp_server  # noqa: F401

    return mcp_server.main() if hasattr(mcp_server, "main") else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-wiki", description="Local-LLM compounding wiki.")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the FastAPI service.")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    serve.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev).")
    serve.set_defaults(func=_serve)

    mcp = sub.add_parser("mcp", help="Run the agent-facing MCP server.")
    mcp.set_defaults(func=_mcp)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
