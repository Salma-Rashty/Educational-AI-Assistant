import re
import cv2
import numpy as np
import pytesseract
from pathlib import Path
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

RAW_TEXT_FILE_NAME = "raw_text.txt"


def _preprocess_image(image: Image.Image) -> Image.Image:
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Adaptive threshold: finds more text but merges fill-in-blank dot sequences
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10,
    )

    # CLAHE + Otsu: keeps dot gaps white but can miss text near large whitespace areas
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    _, otsu = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OR combination: white if either image says white.
    # Where adaptive merged the dots into a bar (black) but Otsu kept gaps (white) → OR=white
    # Where both agree on text (both black) → OR=black, all text preserved
    combined = cv2.bitwise_or(adaptive, otsu)

    return Image.fromarray(combined)


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
        image = _preprocess_image(Image.open(image_path))
        raw_text = pytesseract.image_to_string(image, lang='ara', config='--psm 4 --oem 1')

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
