"""PDF loader.

Uses pdfplumber if available (better text + real table extraction). Falls back to
pypdf for plain text. Optionally: OCR a page via pytesseract when a page has
<50 chars extracted but does contain images (scanned page). Optionally: extract
embedded images and emit DocElement(kind="image") with the saved file path so a
downstream step can caption them with llava.

All optional deps degrade gracefully — missing libs skip the feature, never crash.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .elements import DocElement, elements_to_markdown

log = logging.getLogger(__name__)

_OCR_MIN_CHARS = 50


def _extract_text_pypdf(path: Path) -> list[str]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    out: list[str] = []
    for page in reader.pages:
        try:
            out.append((page.extract_text() or "").strip())
        except Exception:
            out.append("")
    return out


def _ocr_page(path: Path, page_num: int) -> str:
    """Render page → image → pytesseract. Returns "" on any failure."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except Exception as e:
        log.debug("ocr skipped, libs missing", extra={"metadata": {"error": str(e)[:120]}})
        return ""
    try:
        imgs = convert_from_path(str(path), first_page=page_num, last_page=page_num, dpi=200)
        if not imgs:
            return ""
        return (pytesseract.image_to_string(imgs[0]) or "").strip()
    except Exception as e:
        log.warning("ocr failed", extra={"metadata": {"page": page_num, "error": str(e)[:120]}})
        return ""


def _extract_images(path: Path, out_dir: Path) -> list[tuple[int, Path]]:
    """Save embedded images under `out_dir`. Returns [(page_num, image_path), ...]."""
    try:
        from pypdf import PdfReader
    except Exception:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    collected: list[tuple[int, Path]] = []
    try:
        reader = PdfReader(str(path))
        for pi, page in enumerate(reader.pages, start=1):
            imgs = getattr(page, "images", None) or []
            for ii, img in enumerate(imgs):
                try:
                    ext = (img.name.rsplit(".", 1)[-1] if "." in img.name else "png").lower()
                    tgt = out_dir / f"{path.stem}-p{pi}-{ii}.{ext}"
                    tgt.write_bytes(img.data)
                    collected.append((pi, tgt))
                except Exception:
                    continue
    except Exception as e:
        log.debug("pdf image extraction failed", extra={"metadata": {"error": str(e)[:120]}})
    return collected


def load_pdf_elements(path: Path, *, extract_images: bool = False, image_out: Path | None = None, ocr: bool = True) -> list[DocElement]:
    """Return per-page DocElements (text + tables + optional images)."""
    elements: list[DocElement] = []

    # --- try pdfplumber for per-page text + tables ---
    plumber_ok = True
    try:
        import pdfplumber  # type: ignore
    except Exception:
        plumber_ok = False

    if plumber_ok:
        try:
            import pdfplumber  # noqa
            with pdfplumber.open(str(path)) as pdf:
                for pi, page in enumerate(pdf.pages, start=1):
                    txt = (page.extract_text() or "").strip()
                    has_images = bool(getattr(page, "images", None))
                    if len(txt) < _OCR_MIN_CHARS and has_images and ocr:
                        ocr_txt = _ocr_page(path, pi)
                        if ocr_txt:
                            elements.append(DocElement(kind="text", content=ocr_txt, meta={"page": pi, "ocr": True}))
                        elif txt:
                            elements.append(DocElement(kind="text", content=txt, meta={"page": pi}))
                    elif txt:
                        elements.append(DocElement(kind="text", content=txt, meta={"page": pi}))

                    # Tables via pdfplumber
                    try:
                        tables = page.extract_tables() or []
                    except Exception:
                        tables = []
                    for ti, tbl in enumerate(tables):
                        if not tbl:
                            continue
                        from .elements import rows_to_markdown
                        md = rows_to_markdown([[c if c is not None else "" for c in row] for row in tbl])
                        if md:
                            elements.append(DocElement(kind="table", content=md, meta={"page": pi, "idx": ti}))
        except Exception as e:
            log.warning("pdfplumber failed, falling back to pypdf", extra={"metadata": {"error": str(e)[:160]}})
            plumber_ok = False

    if not plumber_ok:
        for pi, txt in enumerate(_extract_text_pypdf(path), start=1):
            if len(txt) < _OCR_MIN_CHARS and ocr:
                ocr_txt = _ocr_page(path, pi)
                if ocr_txt:
                    elements.append(DocElement(kind="text", content=ocr_txt, meta={"page": pi, "ocr": True}))
                    continue
            if txt:
                elements.append(DocElement(kind="text", content=txt, meta={"page": pi}))

    # --- optional image extraction ---
    if extract_images and image_out is not None:
        for page_num, img_path in _extract_images(path, image_out):
            elements.append(
                DocElement(kind="image", content="", meta={"page": page_num, "path": str(img_path)})
            )

    return elements


# ─── backwards-compatible plain-string entrypoints (used by older code paths) ───

def load_pdf(path: Path) -> str:
    return elements_to_markdown(load_pdf_elements(path))


def load_pdf_pages(path: Path) -> list[str]:
    """Kept for compatibility — returns text per page only."""
    by_page: dict[int, list[str]] = {}
    for el in load_pdf_elements(path):
        p = int(el.meta.get("page", 0))
        by_page.setdefault(p, []).append(elements_to_markdown([el]))
    return ["\n\n".join(by_page[k]) for k in sorted(by_page)]
