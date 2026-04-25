import json
from pathlib import Path

json_path = Path("output/image 4/image 4_res.json")
if not json_path.is_file():
    raise FileNotFoundError(f"JSON file not found: {json_path}")

with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

texts = data["rec_texts"]
raw_text = "\n".join(texts)

output_path = json_path.parent / "raw_text.txt"
with output_path.open("w", encoding="utf-8") as f:
    f.write(raw_text)

print(f"Raw text saved successfully: {output_path}")