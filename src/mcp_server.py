"""LLM-Wiki as a Model Context Protocol (MCP) server.

Exposes the wiki's ingest/query/lint/entities/episodic capabilities as MCP tools
so any MCP-aware client (Claude Code, Cursor, Codex, etc.) can drive the wiki
directly — making LLM-Wiki "agent-native" without code-duplicating the harness.

This is a THIN wrapper around the existing FastAPI service. Two run modes:

1. STDIO (default for local Claude Code):
       python -m src.mcp_server
   Configure in your client's MCP config:
       {"command":"python","args":["-m","src.mcp_server"], "cwd":"/path/to/LLM_Wiki"}

2. SSE / HTTP (for remote agents):
       python -m src.mcp_server --transport sse --port 9000

Requires `mcp` Python SDK:  pip install "mcp[cli]"
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

DEFAULT_API = "http://localhost:8000"


async def _query(api: str, question: str, **kwargs) -> dict:
    async with httpx.AsyncClient(timeout=300.0) as c:
        r = await c.post(f"{api}/query", json={"question": question, **kwargs})
        r.raise_for_status()
        return r.json()


async def _ingest_path(api: str, path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"ok": False, "error": f"file not found: {path}"}
    async with httpx.AsyncClient(timeout=600.0) as c:
        with p.open("rb") as f:
            files = {"files": (p.name, f, "application/octet-stream")}
            r = await c.post(f"{api}/ingest", files=files)
        r.raise_for_status()
        return r.json()


async def _lint(api: str) -> dict:
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.get(f"{api}/lint")
        r.raise_for_status()
        return r.json()


async def _entities(api: str, limit: int = 50) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.get(f"{api}/entities", params={"limit": limit})
        r.raise_for_status()
        return r.json()


def _build_server(api: str):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise SystemExit(
            "MCP SDK not installed. Run: pip install \"mcp[cli]\"\n"
            f"Original error: {e}"
        )

    mcp = FastMCP("llm-wiki")

    @mcp.tool()
    async def query_wiki(question: str, top_k: int = 5, save_back: bool = True) -> dict:
        """Search the LLM-Wiki and return a structured, cited answer.

        Use for any question about content the wiki has ingested. The answer
        comes back with numbered citations [1], [2] mapped to source pages,
        plus follow-up questions.
        """
        return await _query(api, question, top_k=top_k, save_back=save_back)

    @mcp.tool()
    async def ingest_file(path: str) -> dict:
        """Ingest a single document (PDF / DOCX / HTML / MD / PPTX / XLSX / CSV / TXT)
        into the wiki. The pipeline summarises, extracts entities, scores
        confidence, and either publishes to wiki/sources/ or stages in
        wiki/review/.
        """
        return await _ingest_path(api, path)

    @mcp.tool()
    async def lint_wiki() -> dict:
        """Health-check the wiki — orphan pages, stale claims, missing entity
        pages, contradictions. Returns a structured report."""
        return await _lint(api)

    @mcp.tool()
    async def list_entities(limit: int = 50) -> dict:
        """List canonical entities in the wiki, sorted by backlink count."""
        return await _entities(api, limit=limit)

    @mcp.tool()
    async def session_crystallize(correlation_ids: list[str], days: int = 14) -> dict:
        """Phase F1: distil an exploration thread (matched by correlation IDs) into one
        durable wiki page filed at wiki/sources/crystallized-<slug>.md, indexed
        for retrieval. Use when an agent has run a series of related queries and
        wants the lessons saved durably."""
        async with httpx.AsyncClient(timeout=300.0) as c:
            r = await c.post(
                f"{api}/session/crystallize",
                json={"correlation_ids": correlation_ids, "days": days},
            )
            r.raise_for_status()
            return r.json()

    @mcp.tool()
    async def context_start(days: int = 7, top_pages: int = 5) -> dict:
        """Phase F2: session-start briefing for an agent. Returns recent episodic
        timeline + top-N most-accessed pages + open contradictions. Call this
        once at the beginning of any new MCP session."""
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.get(
                f"{api}/context/start",
                params={"days": days, "top_pages": top_pages},
            )
            r.raise_for_status()
            return r.json()

    @mcp.tool()
    async def run_admin_job(job_name: str) -> dict:
        """Manually trigger a registered scheduled job (decay_sweep / episodic_prune /
        promote_episodic / lint_autofix / detect_procedures). Same code path as the
        in-process scheduler — useful for on-demand consolidation."""
        async with httpx.AsyncClient(timeout=600.0) as c:
            r = await c.post(f"{api}/admin/run/{job_name}")
            r.raise_for_status()
            return r.json()

    @mcp.resource("wiki://index")
    async def wiki_index() -> str:
        """The wiki's master catalogue (`wiki/index.md`)."""
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{api}/wiki/index")
            r.raise_for_status()
            return r.json().get("content", "")

    return mcp


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="LLM-Wiki MCP server")
    parser.add_argument("--api", default=DEFAULT_API, help="Backend FastAPI URL")
    parser.add_argument("--transport", choices=("stdio", "sse"), default="stdio")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    mcp = _build_server(args.api)

    if args.transport == "stdio":
        mcp.run()  # default stdio loop
    else:
        # SSE mode (HTTP). FastMCP exposes .run with transport.
        mcp.run(transport="sse", port=args.port)


if __name__ == "__main__":
    main()
