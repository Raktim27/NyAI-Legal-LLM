#!/usr/bin/env python3
"""Convert IPC CSV to constitution_qa JSON format.

Produces a list of {"question": ..., "answer": ...} entries.

Mapping used per CSV row:
 - Question: "What does {Section} of the Indian Penal Code state?"
   Answer: Description column (full text)
 - Question: "What is the offence under {Section}?"
   Answer: Offense column
 - Question: "What is the punishment under {Section}?"
   Answer: Punishment column

Usage:
  python scripts/convert_ipc_to_constitution_qa.py --input <csv> --output <json>
"""
import argparse
import csv
import json
from pathlib import Path


def normalize_section(section_raw: str) -> str:
    # Prefer a nicer label: use the raw Section value (e.g. IPC_140) but
    # replace underscores with spaces and uppercase 'IPC' to 'Section' when suitable.
    if not section_raw:
        return "IPC Section"
    s = section_raw.strip()
    # If like IPC_140 -> Section 140
    if s.upper().startswith("IPC_"):
        return "Section " + s.split("_")[-1]
    return s


def row_to_qas(row: dict) -> list:
    desc = (row.get("Description") or "").strip()
    offense = (row.get("Offense") or "").strip()
    punishment = (row.get("Punishment") or "").strip()
    section_raw = (row.get("Section") or "").strip()

    section_label = normalize_section(section_raw)

    qas = []
    if desc:
        q = f"What does {section_label} of the Indian Penal Code state?"
        a = desc
        qas.append({"question": q, "answer": a})

    if offense:
        q = f"What is the offence under {section_label}?"
        a = offense
        qas.append({"question": q, "answer": a})

    if punishment:
        q = f"What is the punishment under {section_label}?"
        a = punishment
        qas.append({"question": q, "answer": a})

    return qas


def convert(input_path: Path, output_path: Path) -> int:
    results = []
    with input_path.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader, start=1):
            try:
                qas = row_to_qas(row)
                results.extend(qas)
            except Exception as e:
                print(f"Warning: failed to process row {i}: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as out:
        json.dump(results, out, ensure_ascii=False, indent=4)

    return len(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to ipc CSV")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    total = convert(input_path, output_path)
    print(f"Wrote {total} QA entries to {output_path}")


if __name__ == '__main__':
    main()
