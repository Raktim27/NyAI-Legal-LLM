"""
--------------------------------------------------------------
 IPC JSON Cleaner & Quality Checker
--------------------------------------------------------------
Author: Raktim
Purpose:
    - Load IPC sections JSON file
    - Detect and remove duplicates / invalid / incomplete entries
    - Provide summary statistics
    - Save cleaned file for downstream RAG usage
--------------------------------------------------------------
"""

import json
import re

# --------------------------------------------------------------
# Step 1. File paths
# --------------------------------------------------------------
INPUT_FILE = "data/processed/ipc_sections.json"
OUTPUT_FILE = "data/processed/ipc_sections_cleaned.json"

# --------------------------------------------------------------
# Step 2. Load JSON
# --------------------------------------------------------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"🔍 Loaded {len(data)} entries from {INPUT_FILE}")

# --------------------------------------------------------------
# Step 3. Initialize helpers
# --------------------------------------------------------------
seen_questions = set()
seen_answers = set()
clean_data = []
duplicates = 0
short_entries = 0
invalid_sections = 0

# --------------------------------------------------------------
# Step 4. Process entries
# --------------------------------------------------------------
for item in data:
    q = item.get("question", "").strip()
    a = item.get("answer", "").strip()

    # ---- Check for empty fields ----
    if not q or not a:
        continue

    # ---- Check if question format is valid ----
    match = re.search(r"IPC\s*Section\s*(\d+)", q, flags=re.IGNORECASE)
    if not match:
        invalid_sections += 1
        continue

    # ---- Remove duplicates ----
    if q in seen_questions or a in seen_answers:
        duplicates += 1
        continue

    # ---- Check for too short text ----
    if len(a.split()) < 10:
        short_entries += 1
        continue

    seen_questions.add(q)
    seen_answers.add(a)

    clean_data.append({
        "question": q,
        "answer": a
    })

# --------------------------------------------------------------
# Step 5. Save cleaned file
# --------------------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(clean_data, f, indent=4, ensure_ascii=False)

# --------------------------------------------------------------
# Step 6. Summary
# --------------------------------------------------------------
print("\n✅ Cleaning Completed!")
print(f"📘 Original entries: {len(data)}")
print(f"🧹 Cleaned entries: {len(clean_data)}")
print(f"🚫 Duplicates removed: {duplicates}")
print(f"⚠️ Invalid sections skipped: {invalid_sections}")
print(f"✂️ Too short entries skipped: {short_entries}")
print(f"💾 Cleaned file saved as: {OUTPUT_FILE}")
