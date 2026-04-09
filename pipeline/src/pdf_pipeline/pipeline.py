from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
import re

try:
    import fitz
except ModuleNotFoundError:
    fitz = None

from .extractor import extract_images, extract_question_region_images, extract_text_by_page
from .models import PipelineResult
from .parser import parse_questions
from .writers import write_json, write_question_answer_mapping


def _bbox_height(bbox: Iterable[float] | None) -> float:
    if not bbox:
        return 0.0
    values = list(bbox)
    if len(values) != 4:
        return 0.0
    return max(0.0, float(values[3]) - float(values[1]))


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0) * (by1 - by0))
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _is_option_image_layout_reliable(question) -> bool:
    bboxes = question.option_image_bboxes or {}
    labels = sorted(bboxes.keys())
    if len(labels) < 2:
        return False

    rects: list[list[float]] = []
    for label in labels:
        bbox = bboxes.get(label)
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        rect = [float(x) for x in bbox]
        rects.append(rect)

    heights = [_bbox_height(rect) for rect in rects]
    # Extremely tall crops usually indicate spillover and unreliable option segmentation.
    if sum(1 for h in heights if h > 260.0) >= 2:
        return False

    # If many option boxes overlap too much, they are likely not true per-choice crops.
    high_overlap_pairs = 0
    total_pairs = 0
    for idx in range(len(rects)):
        for jdx in range(idx + 1, len(rects)):
            total_pairs += 1
            if _bbox_iou(rects[idx], rects[jdx]) > 0.55:
                high_overlap_pairs += 1

    if total_pairs > 0 and high_overlap_pairs / total_pairs >= 0.5:
        return False

    return True


def _parse_answer_pdf(answer_pdf_path: Path) -> dict[str, str]:
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is required for answer mapping. Install it with: pip install PyMuPDF"
        )

    answer_map: dict[str, str] = {}
    with fitz.open(answer_pdf_path) as doc:
        text = "\n".join(page.get_text("text") for page in doc)

    # Capture patterns like: "1 d", "2 b" from answer-key tables.
    for number, letter in re.findall(r"\b(\d{1,3})\s*([A-Da-d])\b", text):
        if number not in answer_map:
            answer_map[number] = letter.upper()
    return answer_map


def _categorize_and_map_answers(
    questions: list,
    extracted_images: list,
    pdf_name: str,
    answers_dir: Path | None,
) -> None:
    answer_map: dict[str, str] = {}
    answer_source: str | None = None
    if answers_dir:
        candidate = answers_dir / pdf_name
        if candidate.exists():
            answer_map = _parse_answer_pdf(candidate)
            answer_source = str(candidate)

    image_qnums: set[str] = set()
    for image in extracted_images:
        if not image.question_number:
            continue
        if image.classification in {"question_figure", "possible_option_figure"}:
            image_qnums.add(str(image.question_number))

    table_re = re.compile(r"\b(table|matrix|truth table|rows?|columns?|sql\s+command)\b", re.IGNORECASE)

    for question in questions:
        qnum = str(question.question_number) if question.question_number is not None else ""
        text = question.question_text or ""

        has_table = bool(table_re.search(text))
        has_image = bool(question.option_image_paths) or (qnum in image_qnums)

        if has_table:
            question.category = "table"
        elif has_image:
            question.category = "image"
        else:
            question.category = "plain_text"

        if qnum and qnum in answer_map:
            question.mapped_answer = answer_map[qnum]
            question.answer_source_file = answer_source


def _backfill_image_choice_placeholders(questions: list) -> None:
    image_choice_keywords = (
        "diagram",
        "figure",
        "graph",
        "curve",
        "shown below",
        "logical circuit",
    )

    for question in questions:
        labels = sorted((question.option_image_paths or {}).keys())
        if len(labels) < 2:
            continue

        layout_reliable = _is_option_image_layout_reliable(question)

        normalized: dict[str, str] = {}
        for option in question.options:
            if not option:
                continue
            prefix = option[:1].upper()
            if prefix in {"A", "B", "C", "D"} and option.startswith(f"{prefix}."):
                normalized[prefix] = option

        # Keep text choices when all labels are present and content looks usable.
        has_complete_text = all(label in normalized for label in labels)

        text_values = [
            normalized[label].split(".", 1)[1].strip()
            for label in labels
            if label in normalized
        ]
        short_or_numeric_count = sum(
            1
            for value in text_values
            if (len(value) <= 3 or value.replace(".", "", 1).isdigit())
        )
        stem_text = (question.question_text or "").lower()
        stem_suggests_image_choices = any(keyword in stem_text for keyword in image_choice_keywords)
        alpha_count = sum(1 for value in text_values if re.search(r"[A-Za-z]", value))

        likely_image_based = (
            len(labels) >= 3
            and stem_suggests_image_choices
            and short_or_numeric_count >= max(2, len(text_values) - 1)
            and alpha_count <= 1
            and layout_reliable
        )

        # If text choices are incomplete, or the question likely uses image-only choices, use placeholders.
        use_placeholders = (not has_complete_text) or likely_image_based
        if use_placeholders:
            question.options = [f"{label}. [Image choice]" for label in labels]
        elif has_complete_text:
            question.options = [normalized[label] for label in labels]

        # For normal text-choice questions, suppress noisy option-image outputs.
        if not use_placeholders:
            question.option_image_paths = {}
            question.option_image_bboxes = {}


def discover_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.pdf"))
    return []


def _to_web_path(path_value: str | None, base_dir: Path) -> str:
    if not path_value:
        return ""

    raw = str(path_value)
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw

    try:
        rel = os.path.relpath(raw, start=str(base_dir))
        return rel.replace("\\", "/")
    except Exception:
        return raw.replace("\\", "/")


def _normalize_result_paths_for_web(questions: list, images: list, base_dir: Path) -> None:
    for question in questions:
        if question.region_image_path:
            question.region_image_path = _to_web_path(question.region_image_path, base_dir)

        option_paths = question.option_image_paths or {}
        if option_paths:
            question.option_image_paths = {
                label: _to_web_path(path, base_dir)
                for label, path in option_paths.items()
            }

    for image in images:
        image.file_path = _to_web_path(image.file_path, base_dir)


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    image_dir: Path,
    extract_images_enabled: bool = True,
    answers_dir: Path | None = None,
) -> tuple[list[PipelineResult], dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    pdfs = discover_pdfs(input_path)
    results: list[PipelineResult] = []

    for pdf_path in pdfs:
        pages = extract_text_by_page(pdf_path)
        questions = parse_questions(pages, pdf_path.name)
        if extract_images_enabled:
            extract_question_region_images(pdf_path, image_dir, questions=questions)
        _backfill_image_choice_placeholders(questions)
        images = (
            extract_images(pdf_path, image_dir, questions=questions)
            if extract_images_enabled
            else []
        )
        _categorize_and_map_answers(
            questions=questions,
            extracted_images=images,
            pdf_name=pdf_path.name,
            answers_dir=answers_dir,
        )
        _normalize_result_paths_for_web(
            questions=questions,
            images=images,
            base_dir=output_dir,
        )
        results.append(
            PipelineResult(
                pdf_path=str(pdf_path),
                questions=questions,
                images=images,
            )
        )

    artifacts = {
        "json": str(write_json(results, output_dir)),
        "qa_mapping": str(write_question_answer_mapping(results, output_dir)),
    }
    return results, artifacts
