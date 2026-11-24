"""Print a short inventory of the data: files, pages, characters, CSV rows.

Run:  .venv\\Scripts\\python.exe scripts/inventory.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """List every document and table with its size."""
    print("Documents (data/docs):")
    for p in sorted((ROOT / "data" / "docs").iterdir()):
        if p.suffix.lower() == ".pdf":
            reader = PdfReader(p)
            chars = sum(len(pg.extract_text() or "") for pg in reader.pages)
            print(f"  {p.name:40s} {len(reader.pages):3d} pages  {chars:7,d} chars  {p.stat().st_size / 1024:6.0f} KB")
        elif p.suffix.lower() in {".txt", ".md"}:
            print(f"  {p.name:40s}   1 page   {len(p.read_text(encoding='utf-8')):7,d} chars")
    print("Tables (data/tables):")
    for p in sorted((ROOT / "data" / "tables").glob("*.csv")):
        df = pd.read_csv(p)
        print(f"  {p.name:40s} {len(df):3d} rows   columns: {', '.join(df.columns)}")


if __name__ == "__main__":
    main()
