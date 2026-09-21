"""
Law PDF -> Structured JSON extractor
-------------------------------------
Converts a bare-act / statute PDF (e.g. Consumer Protection Act, IT Act, BNS)
into a list of section-level records suitable for chunking + embedding in a
RAG pipeline.

Usage:
    python law_pdf_to_json.py input.pdf "Consumer Protection Act, 2019" output.json

What it does:
    1. Extracts text page-by-page with pdfplumber (layout-aware).
    2. Strips repeated headers/footers (page numbers, gazette headers).
    3. Rejoins lines that were wrapped mid-sentence by the PDF layout.
    4. Splits the cleaned text into per-section records using a regex that
       matches Indian statute section headers (e.g. "5.", "Section 5.",
       "5A.", "12(1)").
    5. Writes out a JSON list: [{"act": ..., "section": ..., "text": ...}, ...]

NOTE: Every bare act PDF is formatted slightly differently. You WILL need to
tweak SECTION_PATTERN and the header/footer regexes below after inspecting
your specific PDF's raw output. Run with --debug first to see raw text before
trusting the structured output.
"""

import sys
import re
import json
import argparse
import pdfplumber

# Matches section headers like: "5.", "5A.", "12.", "Section 5.", "3(1)."
# Adjust this per-PDF after inspecting formatting (--debug mode).
SECTION_PATTERN = re.compile(
    r'^\s*(?:Section\s+)?(\d{1,3}[A-Z]?)\.\s+(.*)$',
    re.MULTILINE
)

# Vertical margins (in PDF points, 1 inch = 72 points) to exclude from the
# TOP and BOTTOM of every page BEFORE extracting any text. This physically
# cuts off the header/footer strip based on its position on the page,
# rather than guessing based on line content — which is what caused real
# section numbers near a page break to get deleted before.
#
# HOW TO TUNE THESE: start with the defaults below. Run --debug and check:
#   - If gazette/header text or page numbers still appear in the cleaned
#     output -> increase the margin (try 60, 70, 80...).
#   - If the first or last line of actual section text is getting cut off
#     -> decrease the margin.
# Different PDFs need different values — this is the one thing you'll need
# to calibrate per document.
TOP_MARGIN_PT = 50
BOTTOM_MARGIN_PT = 40

# Content patterns that are unambiguous enough to strip anywhere in the
# document (unlike bare page numbers, these exact phrases won't ever be
# real section text, so it's safe to match them globally).
GLOBAL_NOISE_PATTERNS = [
    r'^THE GAZETTE OF INDIA.*$',
    r'^\[PART.*\]$',
    r'^www\..*\.gov\.in.*$',
    r'^Ministry of Law and Justice.*$',
]


def extract_raw_text(pdf_path: str) -> list:
    """Extract text from every page, cropping out the header/footer
    margin at the page level (by coordinates) before extraction, so
    headers/footers never make it into the text at all.

    Returns a list of pages, each a list of that page's non-empty lines.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}", file=sys.stderr)
        for i, page in enumerate(pdf.pages):
            bbox = (0, TOP_MARGIN_PT, page.width,
                    page.height - BOTTOM_MARGIN_PT)
            cropped = page.crop(bbox)
            text = cropped.extract_text() or ""
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            pages.append(lines)
            if (i + 1) % 20 == 0:
                print(f"  extracted {i+1} pages...", file=sys.stderr)
    return pages


def strip_global_noise(pages: list) -> list:
    """Flatten pages into one line list, dropping only the unambiguous
    content patterns above — safe to apply anywhere since these exact
    phrases are never real section text."""
    cleaned_lines = []
    for page_lines in pages:
        for line in page_lines:
            if any(re.match(p, line) for p in GLOBAL_NOISE_PATTERNS):
                continue
            cleaned_lines.append(line)
    return cleaned_lines


def clean_text(pages: list) -> str:
    """Strip remaining global noise phrases and rejoin lines that were
    wrapped mid-sentence by the PDF layout."""
    cleaned_lines = strip_global_noise(pages)

    # Rejoin lines that don't end in sentence-ending punctuation and the
    # next line doesn't look like a new section header — likely a
    # PDF-wrapped sentence.
    joined = []
    buffer = ""
    for line in cleaned_lines:
        if buffer and not buffer.endswith((".", ":", ";")) \
           and not SECTION_PATTERN.match(line):
            buffer += " " + line
        else:
            if buffer:
                joined.append(buffer)
            buffer = line
    if buffer:
        joined.append(buffer)

    return "\n".join(joined)


def split_into_sections(cleaned: str, act_name: str) -> list:
    """Split cleaned text into section-level records."""
    matches = list(SECTION_PATTERN.finditer(cleaned))
    records = []
    for idx, m in enumerate(matches):
        section_no = m.group(1)
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        section_text = cleaned[start:end].strip()
        records.append({
            "act": act_name,
            "section": section_no,
            "text": section_text
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("act_name")
    parser.add_argument("output_json")
    parser.add_argument("--debug", action="store_true",
                         help="Print raw + cleaned text instead of writing JSON")
    args = parser.parse_args()

    pages = extract_raw_text(args.pdf_path)
    cleaned = clean_text(pages)

    if args.debug:
        raw_preview = "\n".join(line for page in pages for line in page)
        print("----- RAW (first 2000 chars) -----")
        print(raw_preview[:2000])
        print("\n----- CLEANED (first 2000 chars) -----")
        print(cleaned[:2000])
        return

    records = split_into_sections(cleaned, args.act_name)
    print(f"Found {len(records)} section records.", file=sys.stderr)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
