import json
from pathlib import Path


RAW_TEXT_FILE_NAME = "raw_text.txt"


def extract_ocr_text(
    json_path: Path,
    output_path: Path | None = None,
) -> str:
    json_path = Path(json_path)
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    texts = data["rec_texts"]
    raw_text = "\n".join(texts)

    output_path = output_path or json_path.parent / RAW_TEXT_FILE_NAME
    with output_path.open("w", encoding="utf-8") as f:
        f.write(raw_text)

    print(f"Raw text saved successfully: {output_path}")
    return raw_text


def main() -> str:
    raise SystemExit("Run this project from app.py so it can provide the OCR JSON path.")


if __name__ == "__main__":
    main()
