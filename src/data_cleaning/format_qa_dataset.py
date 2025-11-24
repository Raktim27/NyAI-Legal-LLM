"""
--------------------------------------------------------------
 Legal Q&A Dataset Cleaning and Preparation Script
--------------------------------------------------------------
Author: Raktim
Purpose:
    - Combine multiple Indian Legal Q&A JSON files
    - Clean and standardize question-answer pairs
    - Export clean JSONs for downstream NLP applications:
        (a) RAG (Retrieval-Augmented Generation)
        (b) LLM Fine-Tuning (Instruction format)
--------------------------------------------------------------
Expected Directory Structure:
    data/
        raw/
            legal_qa/
                constitution_qa.json
                crpc_qa.json
                ipc_qa.json
        processed/
--------------------------------------------------------------
"""

import os
import json
import re
import pandas as pd


# --------------------------------------------------------------
# Step 1. Define Paths
# --------------------------------------------------------------
RAW_DIR = "data/raw/legal_qa/"
PROCESSED_DIR = "data/processed/"
FILES = ["constitution_qa.json", "crpc_qa.json", "ipc_qa.json"]

# Ensure processed folder exists
os.makedirs(PROCESSED_DIR, exist_ok=True)


# --------------------------------------------------------------
# Step 2. Load and Merge All Q&A Files
# --------------------------------------------------------------
def load_all_files(file_list):
    """Load and merge all JSON files into a single list of dicts."""
    all_records = []
    for file in file_list:
        file_path = os.path.join(RAW_DIR, file)
        if not os.path.exists(file_path):
            print(f"⚠️ Skipping missing file: {file}")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_records.extend(data)
            print(f"✅ Loaded {len(data)} records from {file}")
    return all_records


# Load all files
records = load_all_files(FILES)
print(f"\n🔹 Total combined records before cleaning: {len(records)}")

# Convert to DataFrame
df = pd.DataFrame(records)
print("Columns available:", list(df.columns))


# --------------------------------------------------------------
# Step 3. Basic Cleaning
# --------------------------------------------------------------
def clean_text(text):
    """
    Clean the input text:
      - Normalize spaces
      - Remove non-alphanumeric symbols except punctuation
      - Fix spacing before punctuation
    """
    text = str(text)
    text = re.sub(r'\s+', ' ', text)  # normalize multiple spaces
    text = re.sub(r'[^\w\s.,;:!?\'"-]', '', text)  # remove odd chars
    text = re.sub(r'\s([?.!"](?:\s|$))', r'\1', text)  # fix space before punctuation
    return text.strip()


# Drop invalid or duplicate Q&A pairs
df = df.dropna(subset=["question", "answer"]).drop_duplicates(subset=["question", "answer"])

# Apply cleaning
df["question_clean"] = df["question"].apply(clean_text)
df["answer_clean"] = df["answer"].apply(clean_text)

print(f"✅ After cleaning: {len(df)} records remain.")


# --------------------------------------------------------------
# Step 4. Save Cleaned Q&A Data for RAG Applications
# --------------------------------------------------------------
cleaned_path = os.path.join(PROCESSED_DIR, "legal_qa_clean.json")
df_clean = df[["question_clean", "answer_clean"]].rename(columns={
    "question_clean": "question",
    "answer_clean": "answer"
})
df_clean.to_json(cleaned_path, orient="records", indent=2, force_ascii=False)
print(f"💾 Saved cleaned dataset for RAG: {cleaned_path}")


# --------------------------------------------------------------
# Step 5. Prepare Fine-Tuning Dataset (Instruction Format)
# --------------------------------------------------------------
def to_instruction_format(row):
    """Convert question–answer pairs into instruction–response format."""
    return {
        "instruction": row["question"],
        "input": "",
        "output": row["answer"]
    }

finetune_data = [to_instruction_format(row) for _, row in df_clean.iterrows()]

finetune_path = os.path.join(PROCESSED_DIR, "legal_qa_finetune.json")
with open(finetune_path, "w", encoding="utf-8") as f:
    json.dump(finetune_data, f, indent=2, ensure_ascii=False)
print(f"💾 Saved fine-tuning dataset: {finetune_path}")


# --------------------------------------------------------------
# Step 6. Final Summary
# --------------------------------------------------------------
print("\n🎯 Data Preprocessing Completed Successfully!")
print(f"   ➤ Cleaned Q&A pairs: {len(df_clean)}")
print(f"   ➤ Files generated:")
print(f"       1. {cleaned_path}")
print(f"       2. {finetune_path}")
print("--------------------------------------------------------------")
