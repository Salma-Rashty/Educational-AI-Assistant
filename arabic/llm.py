import json
import re
import subprocess
import sys
from pathlib import Path

from ollama import ResponseError, chat


MODEL = "qwen3:8b"
CORRECTED_TEXT_FILE_NAME = "corrected_text.txt"
EXAM_DATA_FILE_NAME = "exam_data.json"


def install_model() -> None:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if MODEL not in result.stdout:
        print(f"Pulling {MODEL}...")
        pull = subprocess.run(["ollama", "pull", MODEL])
        if pull.returncode != 0:
            print(f"Failed to pull {MODEL}. Make sure Ollama is running.")
            sys.exit(1)
        print(f"{MODEL} installed successfully.")


def build_cleanup_prompt(raw_text: str) -> str:
    return f"""أنت تقوم بتنظيف نص عربي استُخرج من ورقة امتحان عبر تقنية OCR.

قواعد صارمة:
- احتفظ بالنص باللغة العربية تماماً كما هو. لا تترجم أي شيء إلى الإنجليزية أو أي لغة أخرى.
- صحح فقط أخطاء OCR الواضحة: المسافات الزائدة، علامات الترقيم المكسورة، الأحرف المشوهة.
- أعد تركيب الكلمات أو الجمل المكسورة بسبب أخطاء OCR فقط.
- احتفظ بالترقيم الأصلي وخيارات الإجابة والعناوين والدرجات.
- لا تجب على أسئلة الامتحان.
- لا تختر من بين خيارات الإجابة.
- لا تملأ الفراغات بالإجابة الصحيحة حتى لو كانت واضحة.
- احتفظ بمساحات الإجابة الفارغة كـ "____".

أعد النص المُصحَّح فقط بتنسيق نظيف باللغة العربية.

نص OCR:
{raw_text}
"""


def build_extraction_prompt(corrected_text: str) -> str:
    return f"""Process the corrected OCR text and extract the exam content.

Return only one valid JSON object with this structure:
{{
  "exam_title": "...",
  "subject": "...",
  "language": "Arabic",
  "exercises": [
    {{
      "exercise_type": "<type>",
      "questions": [ ... ]
    }}
  ]
}}

How to determine exercise_type — use ONLY these three rules in order:
1. "MCQ" — the question has a list of answer choices (أ/ب/ج or numbered options). Every question with options is MCQ, even if the question itself contains a blank.
2. "True/False" — the section is explicitly labeled "صح أو خطأ", "صواب أو خطأ", or similar in the text AND the question has NO answer choices.
3. "Fill in the blank" — the section is explicitly labeled "أكمل" or "املأ الفراغ" or similar AND the question has NO answer choices.

IMPORTANT: Do NOT invent exercise types. If a question has answer choices (أ/ب/ج), it is ALWAYS MCQ — never True/False or Fill in the blank. Only create a True/False or Fill in the blank exercise if the text has an explicit section heading saying so.

Question structure by type:
- MCQ: {{"question": "...", "correct_answer": "...", "options": ["...", "...", "..."]}}
- True/False: {{"question": "...", "correct_answer": "صح"}} or {{"question": "...", "correct_answer": "خطأ"}}
- Fill in the blank: {{"question": "...", "correct_answer": "..."}}

Additional rules:
- Extract the exam title from the text, usually at the top.
- Set language to "Arabic". Do not change this value.
- Keep all questions, options, and answers in Arabic exactly as they appear. Do not translate anything.
- Keep the blank as "____" in questions.
- For MCQ: extract answer choices without أ/ب/ج labels. Infer the correct answer from context.
- Return only valid JSON. No markdown, no explanations, no text outside the JSON object.

Corrected OCR text:
{corrected_text}
"""


def clean_model_output(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].lower().startswith("here is"):
        lines = lines[1:]
    text = "\n".join(lines).strip()
    # Qwen3 thinking-mode models wrap responses in <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def normalize_corrected_text(text: str) -> str:
    text = re.sub(r"_{2,}", "____", text)
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
            print(f"Warning: missing '{key}' — defaulting to 'غير محدد'.")
            exam_data[key] = "غير محدد"

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


def run_qwen3(messages: list[dict], *, json_format: bool = False, think: bool = True) -> str:
    try:
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "options": {
                "temperature": 0,
                "num_predict": 2000,
                "think": think,
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

    corrected_text = run_qwen3(
        [
            {"role": "system", "content": "أنت تنظف نص OCR عربي وتعيد النص المصحح فقط باللغة العربية."},
            {"role": "user", "content": build_cleanup_prompt(raw_text)},
        ],
        think=False,
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
        exam_json_text = run_qwen3(messages, json_format=True)
        try:
            exam_data = parse_json_response(exam_json_text)
            break
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"Warning: could not parse model response (attempt {attempt}/2): {exc}")

    if exam_data is None:
        print("Warning: model did not return valid JSON after 2 attempts — writing empty skeleton.")
        exam_data = {"exam_title": "غير محدد", "subject": "غير محدد", "language": "Arabic", "exercises": []}
    else:
        validate_exam_data(exam_data)

    return json.dumps(exam_data, ensure_ascii=False, indent=2)


def create_exam_data(corrected_text: str, exam_data_path: Path) -> str:
    exam_json = create_exam_data_json(corrected_text)
    exam_data_path.write_text(exam_json + "\n", encoding="utf-8")
    return exam_json


def run_qwen3_pipeline(raw_text: str) -> str:
    corrected_text = create_corrected_text_from_text(raw_text)
    return create_exam_data_json(corrected_text)


def run_qwen3_from_file(raw_text_path: Path) -> str:
    raw_text_path = Path(raw_text_path)
    corrected_text_path = raw_text_path.with_name(CORRECTED_TEXT_FILE_NAME)
    exam_data_path = raw_text_path.with_name(EXAM_DATA_FILE_NAME)

    corrected_text = create_corrected_text(raw_text_path, corrected_text_path)
    print(f"Corrected text saved to: {corrected_text_path}")

    exam_json = create_exam_data(corrected_text, exam_data_path)
    print(f"Exam data saved to: {exam_data_path}")
    return exam_json


def main() -> None:
    install_model()
    raise SystemExit("Run this project from app.py so it can provide the raw text path.")


if __name__ == "__main__":
    main()
