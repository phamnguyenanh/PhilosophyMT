import json
import re
from pathlib import Path

dataset_dir = Path("blank_dataset")
output_file = Path("dataset.jsonl")

files = sorted(
    dataset_dir.glob("*.jsonl"),
    key=lambda p: int(re.search(r"\d+", p.name).group())
                  if re.search(r"\d+", p.name) else 0
)

count = 0
with open(output_file, "w", encoding="utf-8") as out:
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.write(line + "\n")
                    count += 1

print(f"✓ Đã gộp {len(files)} file — {count} records → {output_file}")
