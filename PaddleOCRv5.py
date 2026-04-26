import os
from pathlib import Path

os.environ.setdefault("PADDLE_DISABLE_ONE_DNN", "1")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR


def create_ocr() -> PaddleOCR:
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        enable_mkldnn=False,
    )


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
    ocr = create_ocr()
    json_paths = []

    for image_path in image_paths:
        print(f"Processing: {image_path}")
        result = ocr.predict(input=str(image_path))
        name = image_path.stem
        output_prefix = output_dir / name

        for res in result:
            res.print()
            res.save_to_img(str(output_prefix))
            res.save_to_json(str(output_prefix))

        json_paths.append(output_prefix / f"{name}_res.json")

    return json_paths


def main() -> list[Path]:
    raise SystemExit("Run this project from app.py so it can provide the image and folder names.")


if __name__ == "__main__":
    main()
