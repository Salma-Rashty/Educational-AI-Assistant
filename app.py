from pathlib import Path

from PaddleOCRv5 import run_ocr
from extract_ocr_text import RAW_TEXT_FILE_NAME, extract_ocr_text
from test_llama3 import run_llama3_from_file


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_FOLDER_NAME = "test images"
OUTPUT_FOLDER_NAME = "output"
SELECTED_IMAGE_FILENAME = "image 3.png"


def main() -> None:
    try:
        image_dir = PROJECT_DIR / IMAGE_FOLDER_NAME
        output_dir = PROJECT_DIR / OUTPUT_FOLDER_NAME
        selected_filenames = [SELECTED_IMAGE_FILENAME]

        ocr_json_paths = run_ocr(
            selected_filenames=selected_filenames,
            image_dir=image_dir,
            output_dir=output_dir,
        )
        if not ocr_json_paths:
            raise RuntimeError("OCR completed but did not return any JSON output paths.")

        ocr_json_path = ocr_json_paths[0]
        image_output_dir = Path(ocr_json_path).parent
        raw_text_path = image_output_dir / RAW_TEXT_FILE_NAME

        extract_ocr_text(ocr_json_path, raw_text_path)
        run_llama3_from_file(raw_text_path)

        print("Pipeline completed successfully.")
    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        raise


if __name__ == "__main__":
    main()
