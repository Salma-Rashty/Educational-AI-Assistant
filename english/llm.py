import json
import re
from pathlib import Path

from ollama import ResponseError, chat


MODEL = "llama3:8b"
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

Return only one valid JSON object with this structure:
{{
  "exam_title": "...",
  "subject": "...",
  "language": "...",
  "exercises": [
    {{
      "exercise_type": "<type>",
      "questions": [ ... ]
    }}
  ]
}}

How to determine exercise_type — use ONLY these three rules in order:
1. "MCQ" — the question has a list of answer choices (A/B/C or numbered options). Every question with options is MCQ, even if the question itself contains a blank.
2. "True/False" — the section is explicitly labeled "True or False", "Correct or Incorrect", or similar in the text AND the question has NO answer choices.
3. "Fill in the blank" — the section is explicitly labeled "Fill in the blank" or similar AND the question has NO answer choices.

IMPORTANT: Do NOT invent exercise types. If a question has answer choices (A/B/C), it is ALWAYS MCQ — never True/False or Fill in the blank. Only create a True/False or Fill in the blank exercise if the text has an explicit section heading saying so.

Question structure by type:
- MCQ: {{"question": "...", "correct_answer": "...", "options": ["...", "...", "..."]}}
- True/False: {{"question": "...", "correct_answer": "True"}} or {{"question": "...", "correct_answer": "False"}}
- Fill in the blank: {{"question": "...", "correct_answer": "..."}}

Additional rules:
- Extract the exam title from the text, usually at the top.
- Identify the subject and language from the content.
- Keep the blank as "____" in questions.
- For MCQ: extract answer choices without A/B/C labels. Infer the correct answer from context.
- Return only valid JSON. No markdown, no explanations, no text outside the JSON object.

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


VALID_EXERCISE_TYPES = {"MCQ", "True/False", "Fill in the blank"}


def validate_exam_data(exam_data: dict) -> None:
    for key in ("exam_title", "subject", "language"):
        if key not in exam_data:
            print(f"Warning: missing '{key}' — defaulting to 'Unknown'.")
            exam_data[key] = "Unknown"

    if not isinstance(exam_data.get("exercises"), list):
        print("Warning: 'exercises' missing or not a list — defaulting to empty.")
        exam_data["exercises"] = []
        return

    valid_exercises = []
    for ex_index, exercise in enumerate(exam_data["exercises"], start=1):
        if "exercise_type" not in exercise:
            print(f"Warning: skipping exercise {ex_index} — missing 'exercise_type'.")
            continue
        if exercise["exercise_type"] not in VALID_EXERCISE_TYPES:
            print(f"Warning: skipping exercise {ex_index} — unknown type '{exercise['exercise_type']}'.")
            continue
        if not isinstance(exercise.get("questions"), list) or not exercise["questions"]:
            print(f"Warning: skipping exercise {ex_index} — no questions.")
            continue

        ex_type = exercise["exercise_type"]
        valid_questions = []
        for q_index, question in enumerate(exercise["questions"], start=1):
            if "question" not in question or "correct_answer" not in question:
                print(f"Warning: skipping exercise {ex_index}, question {q_index} — missing 'question' or 'correct_answer'.")
                continue
            if ex_type == "MCQ" and (not isinstance(question.get("options"), list) or not question["options"]):
                print(f"Warning: skipping exercise {ex_index}, question {q_index} — MCQ missing options.")
                continue
            valid_questions.append(question)

        if not valid_questions:
            print(f"Warning: skipping exercise {ex_index} — no valid questions after filtering.")
            continue

        exercise["questions"] = valid_questions
        valid_exercises.append(exercise)

    if not valid_exercises:
        print("Warning: no valid exercises found in extracted data.")

    exam_data["exercises"] = valid_exercises


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


def create_corrected_text_from_text(raw_text: str) -> str:
    raw_text = raw_text.strip()
    if not raw_text:
        print("OCR text is empty.")
        raise SystemExit(1)

    corrected_text = run_llama(
        [
            {"role": "system", "content": "You clean OCR text and return only the corrected text."},
            {"role": "user", "content": build_cleanup_prompt(raw_text)},
        ]
    )
    corrected_text = normalize_corrected_text(clean_model_output(corrected_text))
    if not corrected_text:
        print("Warning: cleanup model returned empty text — falling back to raw OCR text.")
        corrected_text = normalize_corrected_text(raw_text)
    return corrected_text


def create_corrected_text(raw_text_path: Path, corrected_text_path: Path) -> str:
    if not raw_text_path.exists():
        print(f"Could not find OCR text file: {raw_text_path}")
        raise SystemExit(1)

    raw_text = raw_text_path.read_text(encoding="utf-8").strip()
    corrected_text = create_corrected_text_from_text(raw_text)
    corrected_text_path.write_text(corrected_text + "\n", encoding="utf-8")
    return corrected_text


def create_exam_data_json(corrected_text: str) -> str:
    messages = [
        {"role": "system", "content": "You extract exam content and return only valid JSON."},
        {"role": "user", "content": build_extraction_prompt(corrected_text)},
    ]

    exam_data = None
    for attempt in range(1, 3):
        exam_json_text = run_llama(messages, json_format=True)
        try:
            exam_data = parse_json_response(exam_json_text)
            break
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"Warning: could not parse model response (attempt {attempt}/2): {exc}")

    if exam_data is None:
        print("Warning: model did not return valid JSON after 2 attempts — writing empty skeleton.")
        exam_data = {"exam_title": "Unknown", "subject": "Unknown", "language": "Unknown", "exercises": []}
    else:
        validate_exam_data(exam_data)

    return json.dumps(exam_data, ensure_ascii=False, indent=2)


def create_exam_data(corrected_text: str, exam_data_path: Path) -> str:
    exam_json = create_exam_data_json(corrected_text)
    exam_data_path.write_text(exam_json + "\n", encoding="utf-8")
    return exam_json


def run_llama3(raw_text: str) -> str:
    corrected_text = create_corrected_text_from_text(raw_text)
    return create_exam_data_json(corrected_text)


def run_llama3_from_file(raw_text_path: Path) -> str:
    raw_text_path = Path(raw_text_path)
    corrected_text_path = raw_text_path.with_name(CORRECTED_TEXT_FILE_NAME)
    exam_data_path = raw_text_path.with_name(EXAM_DATA_FILE_NAME)

    corrected_text = create_corrected_text(raw_text_path, corrected_text_path)
    print(f"Corrected text saved to: {corrected_text_path}")

    exam_json = create_exam_data(corrected_text, exam_data_path)
    print(f"Exam data saved to: {exam_data_path}")
    return exam_json


def main() -> None:
    raise SystemExit("Run this project from app.py so it can provide the raw text path.")


if __name__ == "__main__":
    main()
