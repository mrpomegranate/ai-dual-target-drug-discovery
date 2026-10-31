"""Inspect a BindingDB TSV header in-place inside its ZIP; do not extract records.

Usage:
    uv run --locked python scripts/inspect_bindingdb_schema.py \
        --input /path/to/BindingDB_All_202610_tsv.zip \
        --out reports/audits/bindingdb_202610_schema.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

INTERESTING_TERMS = (
    "ic50", "smiles", "uniprot", "target", "organism", "monomer", "reactant",
    "assay", "doi", "pmid", "patent", "source", "chembl", "temp", "ph ",
    "curat", "institution", "ligand", "pubchem",
)


def inspect(archive: Path) -> dict:
    if not archive.is_file():
        raise FileNotFoundError(f"Source ZIP not found: {archive}")
    with zipfile.ZipFile(archive) as zf:
        members = [zi for zi in zf.infolist() if not zi.is_dir()]
        candidates = [zi for zi in members if Path(zi.filename).name == "BindingDB_All.tsv"]
        if len(candidates) != 1:
            raise ValueError(
                "Expected exactly one 'BindingDB_All.tsv' in ZIP, "
                f"found {len(candidates)}. Available: {[x.filename for x in members]}"
            )
        member = candidates[0]
        with (
            zf.open(member, "r") as binary,
            io.TextIOWrapper(
                binary, encoding="utf-8-sig", errors="strict", newline=""
            ) as text,
        ):
            header = next(csv.reader(text, delimiter="\t"))

    if not header:
        raise ValueError("TSV has no header columns")
    names = [name.strip() for name in header]
    counts = Counter(names)
    duplicate_names = sorted(name for name, count in counts.items() if count > 1)
    fields = [{"index_1_based": idx, "name": name} for idx, name in enumerate(names, 1)]
    focus = [field for field in fields if any(token in field["name"].lower() for token in INTERESTING_TERMS)]
    header_bytes = "\t".join(names).encode("utf-8")

    return {
        "source_zip_filename": archive.name,  # Do not record the researcher's private local path.
        "zip_size_bytes": archive.stat().st_size,
        "member_name": member.filename,
        "member_uncompressed_bytes": member.file_size,
        "member_crc32_hex": f"{member.CRC:08x}",
        "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
        "header_column_count": len(fields),
        "duplicate_column_names": duplicate_names,
        "empty_header_positions_1_based": [i for i, name in enumerate(names, 1) if not name],
        "columns": fields,  # All columns, preserving ordinal positions and duplicate names.
        "candidate_scientific_columns": focus,
        "note": "Header inspection only; no assay records processed or counts independently validated.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Existing ZIP archive path")
    parser.add_argument("--out", type=Path, default=Path("reports/audits/bindingdb_202610_schema.json"))
    args = parser.parse_args()
    try:
        result = inspect(args.input)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (FileNotFoundError, ValueError, zipfile.BadZipFile, UnicodeDecodeError, OSError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(f"Archive: {result['source_zip_filename']}")
    print(f"TSV member: {result['member_name']} ({result['member_uncompressed_bytes']:,} bytes)")
    print(f"Header columns: {result['header_column_count']}")
    print(f"Header SHA-256: {result['header_sha256']}")
    if result["duplicate_column_names"]:
        print(f"CAUTION: duplicate column labels: {result['duplicate_column_names']}")
    if result["empty_header_positions_1_based"]:
        print(f"CAUTION: empty header positions: {result['empty_header_positions_1_based']}")
    print("\nCandidate audit columns (position: exact name):")
    for field in result["candidate_scientific_columns"]:
        print(f"  {field['index_1_based']:3}: {field['name']}")
    print(f"\nSaved full header manifest: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
