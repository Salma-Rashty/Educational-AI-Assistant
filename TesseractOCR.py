import re
import pytesseract
from pathlib import Path
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

RAW_TEXT_FILE_NAME = "raw_text.txt"


def _clean_line(text: str) -> str:
    text = re.sub(r'[\d]+[\s\d]*[\d]+', '..........', text)
    text = re.sub(r'[a-zA-Z]{1,3}[\s\d]+[a-zA-Z\d]+', '..........', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def run_ocr(
    selected_filenames: list[str],
    image_dir: Path,
    output_dir: Path,
) -> list[Path]:
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")
    if not selected_filenames:
        raise ValueError("No images selected.")

    image_paths = []
    for filename in selected_filenames:
        path = image_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        image_paths.append(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_text_paths = []

    for image_path in image_paths:
        print(f"Processing: {image_path}")
        image = Image.open(image_path)
        raw_text = pytesseract.image_to_string(image, lang='ara')

        lines = [_clean_line(line) for line in raw_text.splitlines()]
        cleaned_text = "\n".join(line for line in lines if line)

        name = image_path.stem
        image_output_dir = output_dir / name
        image_output_dir.mkdir(parents=True, exist_ok=True)

        raw_text_path = image_output_dir / RAW_TEXT_FILE_NAME
        raw_text_path.write_text(cleaned_text, encoding="utf-8")

        print(f"Raw text saved: {raw_text_path}")
        raw_text_paths.append(raw_text_path)

    return raw_text_paths


def main() -> None:
    raise SystemExit("Run this project from app.py")


if __name__ == "__main__":
    main()