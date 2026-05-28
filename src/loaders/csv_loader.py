"""CSV / TSV loader — whole file becomes one DocElement(table)."""
from __future__ import annotations

import csv
from pathlib import Path

from .elements import DocElement, elements_to_markdown, rows_to_markdown


def load_csv_elements(path: Path, *, max_rows: int = 500) -> list[DocElement]:
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rdr = csv.reader(f, delimiter=delim)
        for i, row in enumerate(rdr):
            if i >= max_rows:
                break
            rows.append([c for c in row])
    md = rows_to_markdown(rows)
    elements = [DocElement(kind="heading", content=path.stem, meta={"level": 1})]
    if md:
        elements.append(DocElement(kind="table", content=md, meta={"source_file": path.name}))
    return elements


def load_csv(path: Path) -> str:
    return elements_to_markdown(load_csv_elements(path))
