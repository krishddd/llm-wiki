"""Source loaders dispatch on file extension.

Two entry points:
- `load_source(path) -> str` — plain Markdown text (backwards-compatible).
- `load_elements(path, *, extract_images, image_out, ocr) -> list[DocElement]` —
  structured elements for multimodal-aware ingestion (preserves tables, tags images).
"""
from __future__ import annotations

from pathlib import Path

from .csv_loader import load_csv_elements
from .docx_loader import load_docx, load_docx_elements
from .elements import DocElement, elements_to_markdown, layout_aware_chunks
from .html_loader import load_html, load_html_elements
from .md_loader import load_md, load_md_elements
from .pdf_loader import load_pdf, load_pdf_elements
from .pptx_loader import load_pptx_elements
from .xlsx_loader import load_xlsx_elements


def load_elements(
    path: str | Path,
    *,
    extract_images: bool = False,
    image_out: Path | None = None,
    ocr: bool = True,
) -> list[DocElement]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return load_pdf_elements(p, extract_images=extract_images, image_out=image_out, ocr=ocr)
    if ext in {".md", ".markdown", ".txt"}:
        return load_md_elements(p)
    if ext in {".html", ".htm"}:
        return load_html_elements(p)
    if ext == ".docx":
        return load_docx_elements(p, extract_images=extract_images, image_out=image_out)
    if ext == ".pptx":
        return load_pptx_elements(p, extract_images=extract_images, image_out=image_out)
    if ext in {".xlsx", ".xlsm"}:
        return load_xlsx_elements(p)
    if ext in {".csv", ".tsv"}:
        return load_csv_elements(p)
    raise ValueError(f"unsupported file type: {ext}")


def load_source(path: str | Path) -> str:
    """Backwards-compatible flat-string entrypoint."""
    return elements_to_markdown(load_elements(path))


__all__ = [
    "DocElement",
    "elements_to_markdown",
    "layout_aware_chunks",
    "load_elements",
    "load_source",
    "load_docx",
    "load_html",
    "load_md",
    "load_pdf",
]
