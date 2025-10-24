from typing import List, Dict

from loguru import logger

from rag import similarity_search, client, CHAT_MODEL


def answer_question(
    question: str,
    kb: List[Dict],
    top_k: int = 32,
    model: str = CHAT_MODEL,
) -> str:
    """
    Return an answer built strictly from `kb`.
    Logs every step; if KB is empty or embeddings fail, returns a fallback.
    """
    logger.info(f"User question: {question!r}")

    if not question.strip():
        logger.warning("Empty question received")
        return "Please ask something"

    try:
        context_chunks: List[str] = similarity_search(question, kb, top_k)
    except Exception as exc:
        logger.exception("Similarity search failed")
        return f"Error during search: {exc}"

    if not context_chunks:
        logger.warning("No similar chunks found")
        return "Sorry, I couldn't find relevant information in the lesson."

    context = "\n\n".join(context_chunks)
    logger.debug(f"Supporting text: {context}")

    system_msg = (
        "You are an expert tutor. Use the provided context to understand and answer the question. "
    )
    user_msg = f"Context:\n\"\"\"\n{context}\n\"\"\"\n\nQuestion: {question}"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        answer = resp.choices[0].message.content.strip()
        logger.info("Answer generated successfully")
        return answer
    except Exception as exc:
        logger.exception("LLM call failed")
        return f"Error: {exc}"