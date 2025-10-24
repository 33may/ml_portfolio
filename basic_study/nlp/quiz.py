import json
from typing import List, Dict, Tuple
from loguru import logger

from process_video import structured_to_text
from rag import similarity_search, client, CHAT_MODEL

def generate_quiz(
    structured: Dict,
    n: int,
    model: str = CHAT_MODEL,
) -> List[Dict]:
    """
    Make n questions:
      • ~66 % one-correct (answers len == 1)
      • ~33 % multiple-correct (answers len > 1)
    JSON schema: {question, choices, answers}
    """
    logger.info(f"Generating {n}-question quiz (single/multi)")

    context = structured_to_text(structured)

    schema = {
        "type": "object",
        "properties": {
        "quiz": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "choices": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3
                    },
                    "answers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1
                    }
                },
                "required": ["question", "choices", "answers"],
                "additionalProperties": False
            }
        }
    },
    "required": ["quiz"],
    "additionalProperties": False
}

    prompt = (
        "You are an experienced examiner. Using the context, write "
        f"{n} assessment questions: about two-thirds must have exactly ONE correct choice, "
        "and about one-third must have SEVERAL correct choices. "
        "Provide 3–6 options per question (label them A, B, C …). "
        "Return **JSON only**, matching the schema; do NOT add explanations.\n\n"
        f"Context:\n\"\"\"\n{context}\n\"\"\""
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "quizSchema",
                "schema": schema,
            },
        },
    )
    quiz = json.loads(resp.choices[0].message.content)
    logger.success("Quiz generated")
    return quiz["quiz"]


def grade_quiz(
    quiz: List[Dict],
    user_answers: List[str],
    kb: List[Dict],
    model: str = CHAT_MODEL,
) -> Tuple[int, str]:
    """
    Check answers; build feedback with RAG explanations.
    """
    correct_cnt = 0
    feedback_lines: List[str] = []

    for idx, (q, ua_raw) in enumerate(zip(quiz, user_answers), 1):
        ua_set = {s.strip().upper() for s in ua_raw.split(",") if s.strip()}
        ans_set = {a.strip().upper() for a in q["answers"]}

        if ua_set == ans_set:
            correct_cnt += 1
            feedback_lines.append(f"**{idx}. Correct!**")
            continue

        # explanation via RAG
        context = "\n\n".join(similarity_search(
            f"{q['question']} {' '.join(q['choices'])}", kb, top_k=32
        ))
        exp_prompt = (
            "Based on the provided context be a teacher that explains the student why their answer is wron "
            "give the correct answer(s).\n\n"
            f"Question: {q['question']}\n"
            f"Options: {', '.join(q['choices'])}\n"
            f"Correct: {', '.join(ans_set)}\n"
            f"User: {', '.join(ua_set)}\n\n"
            f"Context:\n\"\"\"\n{context}\n\"\"\""
        )

        try:
            exp_resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": exp_prompt}],
                temperature=0.2,
                max_tokens=200,
            )
            explanation = exp_resp.choices[0].message.content.strip()
        except Exception as e:
            explanation = f"(could not generate explanation: {e})"

        feedback_lines.append(
            f"**{idx}. Incorrect.**\n"
            f"Your answer: `{', '.join(ua_set) or '—'}`\n\n"
            f"Correct answer(s): `{', '.join(ans_set)}`\n\n{explanation}"
        )

    header = f"### Score: {correct_cnt}/{len(quiz)}"
    return correct_cnt, header + "\n\n" + "\n\n".join(feedback_lines)
