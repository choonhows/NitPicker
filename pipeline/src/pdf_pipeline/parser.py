from __future__ import annotations

import re
from collections.abc import Iterable

from .models import ParsedQuestion

QUESTION_WITH_Q_RE = re.compile(r"^Q\s*(\d{1,4})(?:[\).:-])?\s+(.*)$", re.IGNORECASE)
QUESTION_BARE_RE = re.compile(r"^(\d{1,4})[\).:-]\s+(.*)$", re.IGNORECASE)
OPTION_RE = re.compile(r"^([A-E])[\).]\s+(.*)$", re.IGNORECASE)
ANSWER_RE = re.compile(r"^(?:answer|ans|correct\s*answer)\s*[:=-]\s*([A-E]|\d{1,4})\b", re.IGNORECASE)
EXPLANATION_RE = re.compile(r"^(?:explanation|rationale|solution)\s*[:=-]\s*(.*)$", re.IGNORECASE)
INLINE_QUESTION_BREAK_RE = re.compile(r"\s+(Q\s*\d{1,4}(?:[\).:-])?\s+)", re.IGNORECASE)
INLINE_OPTION_RE = re.compile(
    r"(?:^|\s)([A-Ea-e])[\).]\s*([^\n]+?)(?=(?:\s+[A-Ea-e][\).]\s)|$)",
    re.IGNORECASE,
)
PAGE_NOISE_RE = re.compile(r"^[-–]?\s*\d+\s*[-–]?$")
OPTION_LABEL_TOKEN_RE = re.compile(r"([A-Ea-e])[\).]")
PAGE_FOOTER_RE = re.compile(r"\s*[–-]\s*\d+\s*[–-]\s*$")


def _clean_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_multiline(text: str) -> str:
    lines = [_clean_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _strip_page_footer(text: str) -> str:
    lines = text.splitlines()
    while lines and PAGE_FOOTER_RE.search(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def _prepare_page_text(text: str) -> str:
    """Insert question boundaries for merged text chunks like '... Q10 ...'."""
    return INLINE_QUESTION_BREAK_RE.sub(r"\n\1", text)


def _match_question_start(line: str) -> tuple[str, str] | None:
    q_match = QUESTION_WITH_Q_RE.match(line)
    if q_match:
        return q_match.group(1), q_match.group(2)

    bare_match = QUESTION_BARE_RE.match(line)
    if bare_match:
        return bare_match.group(1), bare_match.group(2)

    return None


def _extract_inline_options(question_text: str) -> tuple[str, list[str]]:
    """Extract options embedded in one line: 'a) ... b) ... c) ... d) ...'."""
    matches = list(INLINE_OPTION_RE.finditer(question_text))
    if len(matches) < 3:
        return question_text, []

    first = matches[0]
    prompt = _clean_line(question_text[: first.start()])
    options: list[str] = []
    for match in matches:
        letter = match.group(1).upper()
        text = _clean_line(match.group(2))
        if text:
            options.append(f"{letter}. {text}")

    return prompt or question_text, options


def _strip_inline_options_from_text(text: str) -> str:
    """Remove trailing inline option blobs from question text if any remain."""
    matches = list(INLINE_OPTION_RE.finditer(text))
    if len(matches) < 3:
        return text
    return _clean_multiline(text[: matches[0].start()])


def _is_probable_option_continuation(line: str) -> bool:
    if not line:
        return False
    if _match_question_start(line):
        return False
    if OPTION_RE.match(line) or ANSWER_RE.match(line) or EXPLANATION_RE.match(line):
        return False
    if PAGE_NOISE_RE.match(line):
        return False
    return True


def _normalize_reconstructed_options(
    question_text: str,
    options: list[str],
) -> tuple[str, list[str]]:
    """Repair malformed option parsing in compact OCR outputs."""
    normalized = [_clean_line(opt) for opt in options if _clean_line(opt)]
    if len(normalized) >= 3:
        return question_text, normalized

    blob_parts = [question_text] + normalized
    blob = _clean_line(" ".join(blob_parts))
    matches = list(INLINE_OPTION_RE.finditer(blob))
    if len(matches) >= 3:
        rebuilt = [f"{m.group(1).upper()}. {_clean_line(m.group(2))}" for m in matches]
        stem = _clean_multiline(blob[: matches[0].start()])
        return stem or question_text, rebuilt

    # Last-resort: if labels exist but values are collapsed, split by labels.
    labels = [m.group(1).upper() for m in OPTION_LABEL_TOKEN_RE.finditer(blob)]
    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)

    if len(unique_labels) >= 3:
        # Extract trailing numeric tokens commonly used in fractional options.
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", blob)
        if len(numbers) >= len(unique_labels):
            tail = numbers[-len(unique_labels):]
            rebuilt = [
                f"{label}. {value}"
                for label, value in zip(unique_labels, tail, strict=False)
            ]
            stem = OPTION_LABEL_TOKEN_RE.split(blob)[0]
            return _clean_multiline(stem), rebuilt

        rebuilt = [f"{label}." for label in unique_labels]
        stem = OPTION_LABEL_TOKEN_RE.split(blob)[0]
        return _clean_multiline(stem), rebuilt

    return question_text, normalized


def _fix_fraction_style_options(
    question_text: str,
    options: list[str],
) -> tuple[str, list[str]]:
    """
    Recover OCR-split choices like:
    stem: "... 1 1 1 1"
    options: A. 168 / B. 42 / C. 35 / D. 7
    -> A. 1/168 / B. 1/42 / C. 1/35 / D. 1/7
    """
    if len(options) < 3:
        return question_text, options

    # Require explicit detached numerator pattern like "... 1 1 1 1".
    if not re.search(r"(?:^|\s)1(?:\s+1){2,}(?:\s|$)", question_text):
        return question_text, options

    parsed: list[tuple[str, str]] = []
    for opt in options:
        m = re.match(r"^([A-E])\.\s*(\d+)$", opt)
        if not m:
            return question_text, options
        parsed.append((m.group(1), m.group(2)))

    one_tokens = re.findall(r"\b1\b", question_text)
    if len(one_tokens) < len(options):
        return question_text, options

    fixed = [f"{label}. 1/{value}" for label, value in parsed]

    # Remove trailing detached numerator tokens from question text.
    original_text = question_text.strip()
    cleaned_text = re.sub(r"(?:\s+1){3,}\s*$", "", question_text).strip()
    # Guardrail: never replace a meaningful stem with an empty/near-empty string.
    if len(cleaned_text) < 20 and len(original_text) >= 20:
        cleaned_text = original_text
    return cleaned_text, fixed


def _score_question_candidate(question: ParsedQuestion) -> float:
    text = (question.question_text or "").lower()
    option_count = len(question.options)
    page = question.page_start or 0

    score = 0.0
    score += min(option_count, 5) * 24.0
    score += min(len(question.question_text), 900) / 40.0
    if "which of the following" in text:
        score += 25.0
    if "sample answer" in text:
        score -= 30.0
    if page and page <= 2:
        score -= 20.0
    return score


def _dedupe_numeric_questions(questions: list[ParsedQuestion]) -> list[ParsedQuestion]:
    numeric = [q for q in questions if q.question_number and str(q.question_number).isdigit()]
    if len(numeric) < 20:
        return questions

    max_q = max(int(q.question_number or "0") for q in numeric)
    if max_q < 30:
        return questions

    grouped: dict[str, list[ParsedQuestion]] = {}
    for q in questions:
        if not q.question_number or not str(q.question_number).isdigit():
            continue
        grouped.setdefault(str(q.question_number), []).append(q)

    selected: list[ParsedQuestion] = []
    for qnum, group in grouped.items():
        best = max(group, key=_score_question_candidate)
        selected.append(best)

    selected.sort(key=lambda item: int(item.question_number or "0"))
    return selected


def parse_questions(pages: Iterable[dict[str, str | int]], pdf_name: str) -> list[ParsedQuestion]:
    """
    Parse question blocks from page text with lightweight heuristics:
    - question starts: 12. / 12) / Q12:
    - options: A. / B) ...
    - answer line: Answer: C
    - explanation line: Explanation: ...
    """
    questions: list[ParsedQuestion] = []
    current: ParsedQuestion | None = None
    collecting_explanation = False

    def flush_current() -> None:
        nonlocal current, collecting_explanation
        if current and (current.question_text or current.options):
            original_question_text = _clean_multiline(current.question_text)
            current.question_text = _clean_multiline(current.question_text)
            if not current.options:
                prompt, inline_options = _extract_inline_options(current.question_text)
                if prompt:
                    current.question_text = prompt
                current.options.extend(inline_options)
            current.question_text = _strip_inline_options_from_text(current.question_text)
            current.options = [_clean_line(opt) for opt in current.options if _clean_line(opt)]
            current.question_text, current.options = _normalize_reconstructed_options(
                current.question_text,
                current.options,
            )
            current.question_text, current.options = _fix_fraction_style_options(
                current.question_text,
                current.options,
            )
            current.question_text = _strip_page_footer(current.question_text)
            if not _clean_line(current.question_text):
                current.question_text = _strip_page_footer(original_question_text)
            if current.explanation:
                current.explanation = _clean_multiline(current.explanation)
            questions.append(current)
        current = None
        collecting_explanation = False

    for page in pages:
        page_number = int(page["page_number"])
        text = _prepare_page_text(str(page["text"]))
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            question_match = _match_question_start(line)
            if question_match:
                flush_current()
                qnum, remainder = question_match
                current = ParsedQuestion(
                    pdf_name=pdf_name,
                    page_start=page_number,
                    question_number=qnum,
                    question_text=remainder,
                )
                continue

            if current is None:
                continue

            option_match = OPTION_RE.match(line)
            if option_match:
                letter, option_text = option_match.groups()
                current.options.append(f"{letter.upper()}. {_clean_line(option_text)}")
                collecting_explanation = False
                continue

            answer_match = ANSWER_RE.match(line)
            if answer_match:
                current.answer = answer_match.group(1).upper()
                collecting_explanation = False
                continue

            explanation_match = EXPLANATION_RE.match(line)
            if explanation_match:
                current.explanation = _clean_line(explanation_match.group(1))
                collecting_explanation = True
                continue

            # Keep appending multi-line question or explanation content.
            if collecting_explanation:
                existing = current.explanation or ""
                current.explanation = f"{existing}\n{line}".strip()
            elif current.options and _is_probable_option_continuation(line):
                current.options[-1] = f"{current.options[-1]} {_clean_line(line)}".strip()
            else:
                current.question_text = f"{current.question_text}\n{line}".strip()

    flush_current()
    return _dedupe_numeric_questions(questions)
