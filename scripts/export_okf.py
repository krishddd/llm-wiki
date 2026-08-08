"""Export the wiki's stable tiers as a standalone OKF v0.1 bundle.

    python scripts/export_okf.py dist/okf-bundle
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--wiki-dir", default="wiki", type=Path)
    ap.add_argument("--keep-existing", action="store_true",
                    help="don't wipe out_dir before exporting")
    args = ap.parse_args()

    from llm_wiki.wiki.okf_export import export_okf_bundle
    stats = export_okf_bundle(args.wiki_dir, args.out_dir, clean=not args.keep_existing)
    print(f"exported {stats['pages_exported']} pages -> {stats['out_dir']}")
    print(f"conformant: {stats['conformant']} "
          f"(skipped non-conformant: {stats['skipped_nonconformant']})")
    for v in stats["violations"]:
        print(f"  violation: {v}")


if __name__ == "__main__":
    main()
