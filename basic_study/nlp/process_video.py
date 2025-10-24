import re
from typing import List, Dict, Tuple
from urllib.parse import urlparse, parse_qs

from loguru import logger

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptList, FetchedTranscript

from rag import client, build_embeddings, CHAT_MODEL, chunk_text

ytt_api = YouTubeTranscriptApi()

class TranscriptException(Exception):
    text = ""

def get_id_from_url(url: str) -> str:
    """
    Extract the YouTube video ID from a standard or shortened URL.
    """
    parsed = urlparse(url)

    if parsed.hostname and parsed.hostname.endswith("youtu.be"):
        return parsed.path.lstrip("/")

    if parsed.hostname and "youtube.com" in parsed.hostname:
        query = parse_qs(parsed.query)
        if "v" in query:
            return query["v"][0]

    raise ValueError("Unsupported YouTube URL format")

def format_ytt_list_response(response: TranscriptList) -> List[Dict]:
    return [{"name": lang.language, "code" : lang.language_code} for lang in response._translation_languages]

def format_ytt_transcript_response(response: FetchedTranscript) -> str:
    result = ""

    for snippet in response.snippets:
        result += f"{snippet.text} "

    return result

def process_video(url: str) -> str:
    video_id = get_id_from_url(url)
    logger.info(f"Fetching transcript for {video_id}")
    # try:
    tr = ytt_api.fetch(video_id=video_id, languages=["en"])
    # except Exception as exc:
    #     logger.exception("Transcript fetch failed")
    #     raise TranscriptException(str(exc)) from exc
    return format_ytt_transcript_response(tr)

def call_router(
    transcript: str,
    model: str = "chatgpt-4o-latest",
    language: str = "english",
) -> str:
    """
    Ask OpenAI to transform the transcript into a detailed Markdown lesson.
    """
    logger.info("Requesting Markdown lesson from LLM")
    prompt = (
        "You are a knowledgeable tutor. Use the following transcript to build a large, "
        "in-depth lesson in Markdown. Do NOT include a title, table of contents, or '---' "
        "fences—only the content itself. Use # for top-level sections and ## for subsections. "
        "Add '(Additional)' to headings that contain extra material beyond the transcript. "
        "Focus on explanatory depth, technical terms, and possible test material. "
        "Expand the transcript three-fold and use **bold** to highlight key terms. "
        "Every section and subsection must be at least three sentences long. "
        f"Answer only in {language}.\n\nTranscript:\n\"\"\"\n{transcript}\n\"\"\""
    )

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    logger.debug("LLM Markdown generation complete")
    return response.choices[0].message.content


def parse_markdown_to_sections(md: str) -> Dict[str, List[Dict]]:
    """
    Convert Markdown headings into a nested section tree.
    """
    lines = md.splitlines()
    sections: List[Dict] = []
    stack: List[Dict] = [{"level": 0, "node": {"subsections": sections}}]
    header_re = re.compile(r"^(#{1,6})\s+(.*)")

    for line in lines:
        m = header_re.match(line)
        if m:
            level = len(m.group(1))
            header = m.group(2).strip()
            node = {"header": header, "content": "", "subsections": []}

            while stack and stack[-1]["level"] >= level:
                stack.pop()
            stack[-1]["node"]["subsections"].append(node)
            stack.append({"level": level, "node": node})
        else:
            if stack:
                cur = stack[-1]["node"]
                cur["content"] += ("\n" if cur["content"] else "") + line

    logger.debug("Markdown parsed into nested tree")
    return {"sections": sections}


def structured_to_text(structured: Dict) -> str:
    """Convert a section tree into indented plain text (debug helper)."""
    def _recurse(sec: Dict, lvl: int = 0) -> str:
        indent = "  " * lvl
        lines = [f"{indent}{sec['header']}"]
        for ln in sec["content"].splitlines():
            if ln.strip():
                lines.append(f"{indent}- {ln.strip()}")
        for sub in sec["subsections"]:
            lines.append(_recurse(sub, lvl + 1))
        return "\n".join(lines)

    return "\n".join(_recurse(t) for t in structured["sections"])


def generate_structure(
    video_url: str,
    language: str = "english",
    model: str = CHAT_MODEL
) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    """
    Full pipeline: YouTube URL → (section tree, RAG embeddings).
    """
    logger.info(f"Starting pipeline for {video_url}")

    transcript = process_video(video_url)
    markdown = call_router(transcript, model=model, language=language)
    structured = parse_markdown_to_sections(markdown)

    chunks = chunk_text(markdown)
    chunk_embeddings = build_embeddings(chunks)

    logger.success("Pipeline complete")
    return structured, chunk_embeddings