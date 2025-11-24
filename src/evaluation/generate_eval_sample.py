import json, random, os

INPUT_FILE = "data/processed/legal_qa_clean.json"
OUTPUT_FILE = "data/processed/legal_eval_sample.json"

os.makedirs("data/processed", exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Take random 50 samples (or fewer if dataset smaller)
sample = random.sample(data, min(50, len(data)))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(sample, f, indent=4, ensure_ascii=False)

print(f"✅ Created evaluation sample file: {OUTPUT_FILE}")
print(f"📊 Total samples: {len(sample)}")
