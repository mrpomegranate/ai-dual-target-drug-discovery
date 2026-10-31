"""Stream BindingDB ZIP into an EGFR/HER2 observation table and counts.

This is an evidence-extraction audit, NOT a model-ready dataset builder.
- Single-chain SwissProt accession matches only (chain 1).
- Missing IC50 records excluded from observation CSV, but counted.
- Unparseable IC50 records retained and flagged, never silently discarded.
- Duplicate molecule/assay observations are intentionally not collapsed.
"""

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from dual_target.ic50 import parse_ic50

TARGETS = {"P00533": "EGFR", "P04626": "HER2"}
MEMBER = "BindingDB_All.tsv"
FIELDS = {
    "reactant_set_id": "BindingDB Reactant_set_id",
    "monomer_id": "BindingDB MonomerID",
    "smiles": "Ligand SMILES",
    "inchi_key": "Ligand InChI Key",
    "target_name": "Target Name",
    "organism": "Target Source Organism According to Curator or DataSource",
    "uniprot": "UniProt (SwissProt) Primary ID of Target Chain 1",
    "chain_count": "Number of Protein Chains in Target (>1 implies a multichain complex)",
    "ic50_raw": "IC50 (nM)",
    "source": "Curation/DataSource",
    "article_doi": "Article DOI",
    "bindingdb_doi": "BindingDB Entry DOI",
    "pmid": "PMID",
    "pubchem_aid": "PubChem AID",
    "patent_number": "Patent Number",
    "ph": "pH",
    "temperature_c": "Temp (C)",
}
OUTPUT_COLUMNS = [
    "source_record_number",
    *FIELDS.keys(),
    "target",
    "measurement_status",
    "ic50_relation",
    "ic50_nm",
    "pic50",
    "activity_at_1000_nm",
    "parse_error",
    "organism_flag",
    "smiles_flag",
]


def audit(input_zip: Path, observations_csv: Path, summary_json: Path, progress_every: int = 250_000):
    """Audit a local ZIP. Writes final outputs only after successful completion."""
    if not input_zip.is_file():
        raise FileNotFoundError(input_zip)
    if observations_csv.resolve() == summary_json.resolve():
        raise ValueError("Observation and summary paths must differ")

    observations_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    partial_csv = observations_csv.with_name(observations_csv.name + ".partial")

    totals = Counter()
    per_target = {name: Counter() for name in TARGETS.values()}
    sample_parse_errors = []

    with zipfile.ZipFile(input_zip, "r") as archive:
        member_info = archive.getinfo(MEMBER)
        zip_size = input_zip.stat().st_size
        with (
            archive.open(member_info, "r") as binary,
            partial_csv.open("w", encoding="utf-8", newline="") as dest,
            io.TextIOWrapper(binary, encoding="utf-8-sig", errors="replace", newline="") as stream,
        ):
            csv.field_size_limit(10_000_000)
            reader = csv.reader(stream, delimiter="\t")
            header = next(reader)
            duplicates = [key for key, n in Counter(header).items() if n > 1]
            if duplicates:
                raise ValueError(f"Duplicate headers: {duplicates[:10]}")
            positions = {col: idx for idx, col in enumerate(header)}
            missing = [name for name in FIELDS.values() if name not in positions]
            if missing:
                raise ValueError(f"Missing required BindingDB columns: {missing}")
            indices = {key: positions[column] for key, column in FIELDS.items()}
            header_sha256 = hashlib.sha256(
                "\t".join(name.strip() for name in header).encode("utf-8")
            ).hexdigest()
            key_uniprot = indices["uniprot"]
            writer = csv.DictWriter(dest, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()

            for record_number, cells in enumerate(reader, start=1):
                totals["data_records_scanned"] += 1
                if progress_every and record_number % progress_every == 0:
                    print(
                        f"Scanned {record_number:,} records; "
                        f"wrote {totals['observation_rows_written']:,} observations",
                        file=sys.stderr,
                        flush=True,
                    )
                if len(cells) != len(header):
                    totals["malformed_width_records"] += 1
                    continue

                accession = cells[key_uniprot].strip()
                target = TARGETS.get(accession)
                if target is None:
                    continue
                t = per_target[target]
                totals["chain1_target_accession_records"] += 1
                t["accession_matched_records"] += 1

                chain_count = cells[indices["chain_count"]].strip()
                if chain_count != "1":
                    t["excluded_not_single_chain"] += 1
                    totals["excluded_not_single_chain"] += 1
                    continue
                t["eligible_single_chain_records"] += 1

                raw = cells[indices["ic50_raw"]].strip()
                if not raw:
                    t["excluded_missing_ic50"] += 1
                    totals["excluded_missing_ic50"] += 1
                    continue

                fields = {name: cells[idx].strip() for name, idx in indices.items()}
                organism = fields["organism"].casefold()
                if organism in {"homo sapiens", "human"}:
                    organism_flag = "human"
                elif not organism:
                    organism_flag = "missing"
                else:
                    organism_flag = "nonhuman_or_unexpected"
                t[f"organism_{organism_flag}"] += 1

                smiles_flag = "present" if fields["smiles"] else "missing"
                t[f"smiles_{smiles_flag}"] += 1

                parse_error = ""
                try:
                    measured = parse_ic50(raw)
                    # Nonempty raw values should never return None.
                    if measured is None:
                        raise ValueError("Nonempty IC50 returned no parse result")
                except ValueError as exc:
                    measured = None
                    parse_error = str(exc)
                    status = "invalid"
                    if len(sample_parse_errors) < 10:
                        sample_parse_errors.append(
                            {"record_number": record_number, "raw_ic50": raw, "error": parse_error}
                        )
                else:
                    status = "exact" if measured.relation == "=" else "censored"

                t[f"ic50_{status}"] += 1
                totals[f"ic50_{status}"] += 1

                record = {
                    "source_record_number": record_number,
                    **fields,
                    "target": target,
                    "measurement_status": status,
                    "ic50_relation": measured.relation if measured else "",
                    "ic50_nm": measured.value_nm if measured else "",
                    "pic50": measured.pic50 if measured and measured.pic50 is not None else "",
                    "activity_at_1000_nm": measured.activity if measured else "indeterminate",
                    "parse_error": parse_error,
                    "organism_flag": organism_flag,
                    "smiles_flag": smiles_flag,
                }
                writer.writerow(record)
                t["observation_rows_written"] += 1
                totals["observation_rows_written"] += 1


    summary = {
        "audit_stage": "M1.4_observation_extraction_not_curated_labels",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "archive_filename": input_zip.name,
            "archive_size_bytes": zip_size,
            "member_name": MEMBER,
            "member_uncompressed_bytes": member_info.file_size,
            "member_crc32_hex": f"{member_info.CRC:08x}",
            "header_column_count": len(header),
            "header_sha256": header_sha256,
        },
        "target_policy": {
            "egfr": "P00533",
            "her2": "P04626",
            "match_field": FIELDS["uniprot"],
            "chain_count_required": "1",
            "activity_threshold_nm": 1000,
            "organism_policy": "Record observed metadata; flag but do not silently exclude mismatches",
            "invalid_ic50_policy": "Retain flagged row without pIC50",
            "duplicates_policy": "Preserve original observations without aggregation",
        },
        "totals": dict(sorted(totals.items())),
        "per_target": {name: dict(sorted(c.items())) for name, c in per_target.items()},
        "first_parse_errors": sample_parse_errors,
        "outputs": {"observations_csv": str(observations_csv)},
        "limitations": [
            "This stage matches chain-1 SwissProt primary accessions only.",
            "Multichain complexes are excluded from this observation extract and counted.",
            "No canonicalization, molecule merging, assay reconciliation, or duplicate removal.",
            "The organism flag is metadata, not a strict inclusion criterion.",
            "UTF-8 decoding uses replacement characters for invalid source bytes.",
            "The source ZIP SHA-256 was verified separately and is not recalculated by this script.",
            "All counts here are observation-level, not unique molecule or paired-molecule counts.",
        ],
    }
    # A failed scan never overwrites an earlier completed observation CSV.
    partial_csv.replace(observations_csv)
    partial_json = summary_json.with_name(summary_json.name + ".partial")
    partial_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    partial_json.replace(summary_json)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="BindingDB October 2026 ZIP")
    parser.add_argument("--observations", type=Path, required=True, help="Output observation CSV")
    parser.add_argument("--summary", type=Path, required=True, help="Output audit JSON")
    parser.add_argument("--progress-every", type=int, default=250_000)
    args = parser.parse_args(argv)
    result = audit(args.input, args.observations, args.summary, args.progress_every)
    print(f"Scanned: {result['totals'].get('data_records_scanned', 0):,}")
    print(f"Observations written: {result['totals'].get('observation_rows_written', 0):,}")
    for name, counts in result["per_target"].items():
        print(
            f"{name}: written={counts.get('observation_rows_written', 0):,}, "
            f"exact={counts.get('ic50_exact', 0):,}, "
            f"censored={counts.get('ic50_censored', 0):,}, "
            f"invalid={counts.get('ic50_invalid', 0):,}"
        )
    print(f"Summary: {args.summary}")
    print(f"Observations: {args.observations}")


if __name__ == "__main__":
    main()
