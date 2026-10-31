
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

TARGETS = {"P00533": "EGFR", "P04626": "HER2"}

FIELDS = {
    "record_id": "BindingDB Reactant_set_id",
    "monomer_id": "BindingDB MonomerID",
    "smiles": "Ligand SMILES",
    "target_name": "Target Name",
    "organism": "Target Source Organism According to Curator or DataSource",
    "ic50_raw": "IC50 (nM)",
    "uniprot": "UniProt (SwissProt) Primary ID of Target Chain 1",
    "chains": "Number of Protein Chains in Target (>1 implies a multichain complex)",
    "doi": "Article DOI",
    "pmid": "PMID",
    "source": "Curation/DataSource",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-target", type=int, default=3)
    args = parser.parse_args()

    if args.per_target < 1:
        parser.error("--per-target must be positive")

    examples = {"EGFR": [], "HER2": []}
    rows_scanned = 0

    with (
        zipfile.ZipFile(args.input) as archive,
        archive.open("BindingDB_All.tsv") as binary,
        io.TextIOWrapper(
                binary,
                encoding="utf-8-sig",
                errors="replace",
                newline=""
        ) as stream,
    ):
        csv.field_size_limit(10_000_000)

        reader = csv.DictReader(stream, delimiter="\t")

        missing = [
            col for col in FIELDS.values()
            if col not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        for row in reader:
            rows_scanned += 1

            uniprot = row[FIELDS["uniprot"]].strip()
            chains = row[FIELDS["chains"]].strip()
            ic50 = row[FIELDS["ic50_raw"]].strip()

            if uniprot not in TARGETS:
                continue
            if chains != "1" or not ic50:
                continue

            target = TARGETS[uniprot]
            if len(examples[target]) >= args.per_target:
                continue

            examples[target].append({
                key: row[col].strip()
                for key, col in FIELDS.items()
            })

            if all(
                len(v) >= args.per_target
                for v in examples.values()
            ):
                break

    result = {
        "source_archive": Path(args.input).name,
        "rows_scanned": rows_scanned,
        "selection": "Single-chain EGFR/HER2, nonempty IC50",
        "samples": examples,
        "note": "Raw examples only; no curation performed",
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    print(f"Rows scanned: {rows_scanned:,}")
    for target, records in examples.items():
        print(f"{target}: {len(records)} examples")
        for record in records:
            print(
                f"  Record {record['record_id']}: "
                f"IC50 = {record['ic50_raw']} nM"
            )

    print(f"Saved to: {output}")

if __name__ == "__main__":
    main()
