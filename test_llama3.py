import json
import re
from pathlib import Path

from ollama import ResponseError, chat


MODEL = "llama3:8b"
RAW_TEXT_PATH = Path("output/image 4/raw_text.txt")
CORRECTED_TEXT_FILE_NAME = "corrected_text.txt"
EXAM_DATA_FILE_NAME = "exam_data.json"


def build_cleanup_prompt(raw_text: str) -> str:
    return f"""You are cleaning OCR output from an English exam worksheet.

Fix spelling, spacing, punctuation, encoding artifacts, and broken sentences.
Reconstruct only broken words or sentences caused by OCR errors.
Keep the original meaning, numbering, answer choices, section headings, and marks.
Do not answer the exam questions.
Do not choose from the answer choices.
Do not fill blanks with the correct answer, even if the answer is obvious.
Preserve blank answer spaces as "____".

Examples:
- "You can find many .. in the study room" becomes "You can find many ____ in the study room."
- "My aunt bought me a.yesterday." becomes "My aunt bought me a ____ yesterday."
- "Muslims go to theto pray." becomes "Muslims go to the ____ to pray."
- "girl.is very friendly." becomes "girl. ____ is very friendly."
- "The bank is .. the clinic and the shop." becomes "The bank is ____ the clinic and the shop."
- "There is a durian tree. y house." becomes "There is a durian tree ____ my house."
- "There a rabbit in the hutch." becomes "There ____ a rabbit in the hutch."

Return only the corrected text, with clean formatting.

OCR text:
{raw_text}
"""


def build_extraction_prompt(corrected_text: str) -> str:
    return f"""Process the corrected OCR text and extract the exam content.

Return only one valid JSON object with this exact structure:
{{
  "exam_title": "...",
  "exam_type": "MCQ",
  "subject": "...",
  "language": "...",
  "questions": [
    {{
      "question": "...",
      "correct_answer": "...",
      "options": ["...", "...", "..."]
    }}
  ]
}}

Rules:
- Extract the exam title from the text, usually at the top.
- Set exam_type to "MCQ".
- Identify the subject and language from the content.
- Identify each question clearly.
- Keep the blank as "____" in each question.
- Extract all provided answer choices separately without A/B/C labels.
- Extract the correct answer from the options using the sentence context.
- Return only valid JSON.
- Do not include markdown, explanations, comments, or extra text outside JSON.

Corrected OCR text:
{corrected_text}
"""


def clean_model_output(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].lower().startswith("here is"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def normalize_corrected_text(text: str) -> str:
    text = re.sub(r"_{2,}", "____", text)
    text = text.replace(
        "Shima is a good girl ____ very friendly.",
        "Shima is a good girl. ____ is very friendly.",
    )
    return text.strip()


def parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model response did not contain a JSON object.")

    return json.loads(cleaned[start : end + 1])


def validate_exam_data(exam_data: dict) -> None:
    required_keys = {"exam_title", "exam_type", "subject", "language", "questions"}
    missing_keys = required_keys - exam_data.keys()
    if missing_keys:
        raise ValueError(f"Missing top-level keys: {sorted(missing_keys)}")

    if exam_data["exam_type"] != "MCQ":
        raise ValueError('exam_type must be "MCQ".')

    if not isinstance(exam_data["questions"], list) or not exam_data["questions"]:
        raise ValueError("questions must be a non-empty list.")

    question_keys = {"question", "correct_answer", "options"}
    for index, question in enumerate(exam_data["questions"], start=1):
        missing_question_keys = question_keys - question.keys()
        if missing_question_keys:
            raise ValueError(
                f"Question {index} missing keys: {sorted(missing_question_keys)}"
            )
        if not isinstance(question["options"], list) or not question["options"]:
            raise ValueError(f"Question {index} options must be a non-empty list.")


def run_llama(messages: list[dict], *, json_format: bool = False) -> str:
    try:
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "options": {
                "temperature": 0,
                "num_predict": 2000,
            },
        }
        if json_format:
            kwargs["format"] = "json"

        response = chat(**kwargs)
    except ResponseError as exc:
        print(f"Ollama error: {exc.error}")
        print(f"Make sure the model is installed with: ollama pull {MODEL}")
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"Could not reach Ollama: {exc}")
        print("Make sure Ollama is running, then try again.")
        raise SystemExit(1) from exc

    return response["message"]["content"]


def create_corrected_text(raw_text_path: Path, corrected_text_path: Path) -> str:
    if not raw_text_path.exists():
        print(f"Could not find OCR text file: {raw_text_path}")
        raise SystemExit(1)

    raw_text = raw_text_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        print(f"OCR text file is empty: {raw_text_path}")
        raise SystemExit(1)

    corrected_text = run_llama(
        [
            {
                "role": "system",
                "content": "You clean OCR text and return only the corrected text.",
            },
            {
                "role": "user",
                "content": build_cleanup_prompt(raw_text),
            },
        ]
    )
    corrected_text = normalize_corrected_text(clean_model_output(corrected_text))
    corrected_text_path.write_text(corrected_text + "\n", encoding="utf-8")
    return corrected_text


def create_exam_data(corrected_text: str, exam_data_path: Path) -> None:
    exam_json_text = run_llama(
        [
            {
                "role": "system",
                "content": "You extract exam content and return only valid JSON.",
            },
            {
                "role": "user",
                "content": build_extraction_prompt(corrected_text),
            },
        ],
        json_format=True,
    )

    try:
        exam_data = parse_json_response(exam_json_text)
        validate_exam_data(exam_data)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Invalid JSON from model: {exc}")
        raise SystemExit(1) from exc

    exam_data_path.write_text(
        json.dumps(exam_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    raw_text_path = RAW_TEXT_PATH
    corrected_text_path = raw_text_path.with_name(CORRECTED_TEXT_FILE_NAME)
    exam_data_path = raw_text_path.with_name(EXAM_DATA_FILE_NAME)

    corrected_text = create_corrected_text(raw_text_path, corrected_text_path)
    print(f"Corrected text saved to: {corrected_text_path}")

    create_exam_data(corrected_text, exam_data_path)
    print(f"Exam data saved to: {exam_data_path}")


if __name__ == "__main__":
    main()
