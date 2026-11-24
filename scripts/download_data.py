"""Download the sample PDFs into data/docs/ and verify each one with pypdf.

Run:  .venv\\Scripts\\python.exe scripts/download_data.py
Existing files are skipped, so re-running is cheap. Failed downloads are reported, never replaced.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from pypdf import PdfReader

DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "docs"
USER_AGENT = "Mozilla/5.0 (KnowledgeLens learning project; python-urllib)"

SOURCES: dict[str, str] = {
    "acme_employee_policy_handbook.pdf":
        "https://microsoft.github.io/mcs-labs/labs/mcs-orchestration/company_policies_sample.pdf",
    "rag_lewis_2020.pdf": "https://arxiv.org/pdf/2005.11401",
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762",
}


def download(url: str, dest: Path, timeout: int = 60) -> None:
    """Download `url` to `dest` and check the payload really is a PDF."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if not data.startswith(b"%PDF"):
        raise ValueError(f"response is not a PDF (first bytes: {data[:20]!r})")
    dest.write_bytes(data)


def verify(path: Path) -> tuple[int, str | None]:
    """Return (page_count, first 300 chars of the first page that has text, or None if scanned)."""
    reader = PdfReader(path)
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            return len(reader.pages), text[:300]
    return len(reader.pages), None


def main() -> int:
    """Download every source, verify it, and report failures with their URLs."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str, str]] = []
    for name, url in SOURCES.items():
        dest = DOCS_DIR / name
        if dest.exists():
            print(f"[skip] {name} already exists")
        else:
            try:
                download(url, dest)
                print(f"[ok]   downloaded {name}")
            except Exception as exc:  # report and continue with the rest
                failures.append((name, url, f"{type(exc).__name__}: {exc}"))
                print(f"[FAIL] {name} <- {url}: {exc}")
                continue
        pages, preview = verify(dest)
        if preview is None:
            print(f"       {pages} pages, NO extractable text (scanned?) -> exclude it")
        else:
            print(f"       {pages} pages. First text: {preview!r}\n")
    if failures:
        print("\nDownload these by hand into data/docs/:")
        for name, url, err in failures:
            print(f"  {name}: {url}  ({err})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
