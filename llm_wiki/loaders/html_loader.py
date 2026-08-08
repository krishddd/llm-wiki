"""HTML loader — preserves heading structure, tables, image alt-text, code blocks."""
from __future__ import annotations

from pathlib import Path

from .elements import DocElement, elements_to_markdown, rows_to_markdown


def _html_table_rows(tbl) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in tbl.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        rows.append([c.get_text(" ", strip=True) for c in cells])
    return rows


def load_html_elements(path: Path) -> list[DocElement]:
    from bs4 import BeautifulSoup

    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    body = soup.body or soup
    elements: list[DocElement] = []

    # Walk in document order
    for node in body.descendants:
        name = getattr(node, "name", None)
        if not name:
            continue
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            txt = node.get_text(" ", strip=True)
            if txt:
                elements.append(DocElement(kind="heading", content=txt, meta={"level": int(name[1])}))
        elif name in {"p", "li"}:
            # Skip if this node is inside a table / code block — will be handled by the parent.
            if node.find_parent(["table", "pre", "code"]):
                continue
            txt = node.get_text(" ", strip=True)
            if txt:
                elements.append(DocElement(kind="text", content=txt))
        elif name == "table":
            # Avoid re-processing nested tables
            if node.find_parent("table"):
                continue
            rows = _html_table_rows(node)
            md = rows_to_markdown(rows)
            if md:
                elements.append(DocElement(kind="table", content=md))
        elif name == "pre":
            code = node.get_text("\n", strip=False).strip()
            lang = ""
            code_tag = node.find("code")
            if code_tag and code_tag.get("class"):
                for cls in code_tag.get("class"):
                    if cls.startswith("language-"):
                        lang = cls.replace("language-", "")
                        break
            if code:
                elements.append(DocElement(kind="code", content=code, meta={"lang": lang}))
        elif name == "img":
            alt = node.get("alt") or ""
            src = node.get("src") or ""
            elements.append(DocElement(kind="image", content=alt.strip(), meta={"alt": alt, "path": src}))

    return elements


def load_html(path: Path) -> str:
    return elements_to_markdown(load_html_elements(path))
