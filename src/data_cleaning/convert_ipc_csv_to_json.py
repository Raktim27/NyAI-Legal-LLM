"""
--------------------------------------------------------------
 Convert IPC CSV → JSON (Improved Parsing for Multi-line Fields)
--------------------------------------------------------------
Author: Raktim
Purpose:
    - Convert IPC CSV (with multiline text) into structured JSON
    - Extract full legal description, ignoring "in Simple Words" part
--------------------------------------------------------------
"""

import csv
import json
import re

# --------------------------------------------------------------
# Step 1. File paths
# --------------------------------------------------------------
CSV_FILE = "data/raw/Ipc/ipc_sections.csv"   # Input CSV file
OUTPUT_JSON = "data/processed/ipc_sections.json"  # Output JSON file

# --------------------------------------------------------------
# Step 2. Parse CSV and extract structured text
# --------------------------------------------------------------
data = []

with open(CSV_FILE, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if not row or len(row) == 0:
            continue

        # The first column contains the full text (multi-line)
        text = row[0].strip()

        # Find section number using regex
        match = re.search(r"Description of IPC Section\s*(\d+)", text)
        if not match:
            continue  # skip rows that don't follow this format
        section_num = match.group(1)

        # Extract everything after 'According to section ...' but before 'IPC xxx in Simple Words'
        parts = re.split(r"IPC\s*\d+\s*in\s*Simple\s*Words", text, flags=re.IGNORECASE)
        main_text = parts[0].strip()

        # Remove the "Description of IPC Section X" title line if present
        main_text = re.sub(r"Description of IPC Section\s*\d+", "", main_text, flags=re.IGNORECASE).strip()

        if len(main_text) < 20:
            continue  # skip junk rows

        # Build JSON record
        record = {
            "question": f"What is IPC Section {section_num}?",
            "answer": main_text
        }
        data.append(record)

# --------------------------------------------------------------
# Step 3. Write to JSON
# --------------------------------------------------------------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"✅ Created {OUTPUT_JSON} with {len(data)} valid IPC entries.")
