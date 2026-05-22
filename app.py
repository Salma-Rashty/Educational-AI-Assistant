from pathlib import Path

from english.ocr import RAW_TEXT_FILE_NAME, extract_ocr_text
from english.llm import run_llama3_from_file
from arabic.llm import run_qwen3_from_file


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_FOLDER_NAME = "test images"
OUTPUT_FOLDER_NAME = "output"
SELECTED_IMAGE_FILENAME = "image 1.png"
LANGUAGE = "ar"


def _get_ocr_runner(language: str):
    if language == "ar":
        from arabic.ocr import run_ocr
    else:
        from english.ocr import run_ocr
    return run_ocr


def main() -> None:
    try:
        image_dir = PROJECT_DIR / IMAGE_FOLDER_NAME / LANGUAGE
        output_dir = PROJECT_DIR / OUTPUT_FOLDER_NAME / LANGUAGE
        selected_filenames = [SELECTED_IMAGE_FILENAME]

        run_ocr = _get_ocr_runner(LANGUAGE)
        output_paths = run_ocr(
            selected_filenames=selected_filenames,
            image_dir=image_dir,
            output_dir=output_dir,
        )
        if not output_paths:
            raise RuntimeError("OCR completed but did not return any output paths.")

        if LANGUAGE == "ar":
            raw_text_path = output_paths[0]
        else:
            ocr_json_path = output_paths[0]
            raw_text_path = Path(ocr_json_path).parent / RAW_TEXT_FILE_NAME
            extract_ocr_text(ocr_json_path, raw_text_path)

        if LANGUAGE == "ar":
            run_qwen3_from_file(raw_text_path)
        else:
            run_llama3_from_file(raw_text_path)

        print("Pipeline completed successfully.")
    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        raise


if __name__ == "__main__":
    main()
