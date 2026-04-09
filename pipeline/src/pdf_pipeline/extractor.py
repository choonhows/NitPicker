from __future__ import annotations

from pathlib import Path
import re
from typing import Any

try:
    import fitz
except ModuleNotFoundError:
    fitz = None

from .models import ExtractedImage
from .models import ParsedQuestion

QUESTION_START_RE = re.compile(r"^(?:Q\s*)?(\d{1,4})[\).:-]\s+", re.IGNORECASE)
OPTION_RE = re.compile(r"^([A-Ea-e])[\).]\s+", re.IGNORECASE)
INLINE_QUESTION_BREAK_RE = re.compile(r"\s+(Q\s*\d{1,4}(?:[\).:-])?\s+)", re.IGNORECASE)


def _require_fitz() -> None:
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is required for PDF extraction. Install it with: pip install PyMuPDF"
        )


def _prepare_text_for_parsing(text: str) -> str:
    # Some PDFs collapse multiple questions in one line; force a newline before Qxx markers.
    return INLINE_QUESTION_BREAK_RE.sub(r"\n\1", text)


def _extract_question_line_markers(page: Any) -> list[dict[str, float | str]]:
    markers: list[dict[str, float | str]] = []
    data = page.get_text("dict")

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            line_text = "".join(str(span.get("text", "")) for span in spans).strip()
            if not line_text:
                continue

            match = QUESTION_START_RE.match(line_text)
            if not match:
                continue

            _, y0, _, y1 = line.get("bbox", (0.0, 0.0, 0.0, 0.0))
            markers.append(
                {
                    "question_number": match.group(1),
                    "y0": float(y0),
                    "y1": float(y1),
                }
            )

    markers.sort(key=lambda item: float(item["y0"]))

    # Fallback/augmentation: detect explicit Q-number tokens directly from page words.
    page_width = float(page.rect.width)
    try:
        words = page.get_text("words", sort=True)
    except TypeError:
        words = page.get_text("words")

    seen_word_keys: set[tuple[str, int]] = set()
    for word in words:
        if len(word) < 5:
            continue
        x0, y0, x1, y1, token = word[:5]
        # Question numbers usually appear near the left side of the page.
        if float(x0) > page_width * 0.35:
            continue
        token_text = str(token).strip()
        token_match = re.match(r"^(?:Q\s*(\d{1,4})[\).:-]?|(\d{1,4})[\).:-])$", token_text, re.IGNORECASE)
        if not token_match:
            continue

        qnum = token_match.group(1) or token_match.group(2)
        key = (qnum, int(float(y0)))
        if key in seen_word_keys:
            continue
        seen_word_keys.add(key)
        markers.append(
            {
                "question_number": qnum,
                "y0": float(y0),
                "y1": float(y1),
            }
        )

    markers.sort(key=lambda item: float(item["y0"]))
    return markers


def _extract_option_line_markers(page: Any) -> list[dict[str, float | str]]:
    markers: list[dict[str, float | str]] = []
    data = page.get_text("dict")
    page_width = float(page.rect.width)

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            line_text = "".join(str(span.get("text", "")) for span in spans).strip()
            if not line_text:
                continue

            match = OPTION_RE.match(line_text)
            if not match:
                continue

            x0, y0, _, y1 = line.get("bbox", (0.0, 0.0, 0.0, 0.0))
            if float(x0) > page_width * 0.72:
                continue
            if len(line_text) < 6:
                continue
            markers.append(
                {
                    "choice_label": match.group(1).upper(),
                    "y0": float(y0),
                    "y1": float(y1),
                }
            )

    # Fallback from word tokens for layouts where options are split from text lines.
    try:
        words = page.get_text("words", sort=True)
    except TypeError:
        words = page.get_text("words")

    bins: dict[int, list[tuple[str, float, float]]] = {}
    for word in words:
        if len(word) < 5:
            continue
        x0, y0, x1, y1, token = word[:5]
        if float(x0) > page_width * 0.72:
            continue

        token_text = str(token).strip()
        m = re.match(r"^([A-Ea-e])[\).]$", token_text)
        if not m:
            continue

        label = m.group(1).upper()
        y_bin = int(float(y0) // 40)
        bins.setdefault(y_bin, []).append((label, float(y0), float(y1)))

    for items in bins.values():
        distinct = {label for label, _, _ in items}
        # Consider this an option row only if several choice labels co-occur.
        if len(distinct) < 3:
            continue
        seen_labels: set[str] = set()
        for label, y0, y1 in items:
            if label in seen_labels:
                continue
            seen_labels.add(label)
            markers.append(
                {
                    "choice_label": label,
                    "y0": y0,
                    "y1": y1,
                }
            )

    markers.sort(key=lambda item: float(item["y0"]))
    return markers


def _select_marker_for_question(
    question_number: str,
    markers: list[dict[str, float | str]],
    used_indices: set[int],
) -> tuple[int | None, dict[str, float | str] | None]:
    # Prefer exact q-number marker not yet used.
    for idx, marker in enumerate(markers):
        if idx in used_indices:
            continue
        if str(marker["question_number"]) == str(question_number):
            return idx, marker

    # Fallback: first unused marker in reading order.
    for idx, marker in enumerate(markers):
        if idx not in used_indices:
            return idx, marker
    return None, None


def _extract_option_label_positions(
    page: Any,
    y_min: float,
    y_max: float,
) -> list[dict[str, float | str]]:
    try:
        words = page.get_text("words", sort=True)
    except TypeError:
        words = page.get_text("words")

    labels: list[dict[str, float | str]] = []
    seen: set[str] = set()
    for word in words:
        if len(word) < 5:
            continue
        x0, y0, x1, y1, token = word[:5]
        y0f = float(y0)
        if y0f < y_min or y0f > y_max:
            continue

        token_text = str(token).strip()
        match = re.match(r"^([A-Da-d])[\).]$", token_text)
        if not match:
            continue

        label = match.group(1).upper()
        if label in seen:
            continue
        seen.add(label)
        labels.append(
            {
                "label": label,
                "x0": float(x0),
                "x1": float(x1),
                "y0": y0f,
                "y1": float(y1),
            }
        )

    labels.sort(key=lambda item: (float(item["y0"]), float(item["x0"])))
    return labels


def _attach_option_image_crops(
    page: Any,
    question: ParsedQuestion,
    option_destination: Path,
    pdf_stem: str,
    page_idx: int,
    y_start: float,
    question_end_y: float,
    zoom: float,
) -> None:
    labels = _extract_option_label_positions(page, y_min=y_start + 20.0, y_max=question_end_y - 4.0)
    if len(labels) < 2:
        return

    page_width = float(page.rect.width)
    row_centers: list[float] = []
    rows: list[list[dict[str, float | str]]] = []
    for item in labels:
        yc = (float(item["y0"]) + float(item["y1"])) / 2.0
        matched_row = -1
        for idx, center in enumerate(row_centers):
            if abs(center - yc) <= 24.0:
                row_centers[idx] = (center + yc) / 2.0
                matched_row = idx
                break
        if matched_row < 0:
            row_centers.append(yc)
            rows.append([item])
        else:
            rows[matched_row].append(item)

    row_order = sorted(range(len(row_centers)), key=lambda i: row_centers[i])
    row_centers = [row_centers[i] for i in row_order]
    rows = [rows[i] for i in row_order]
    for row in rows:
        row.sort(key=lambda item: float(item["x0"]))

    row_index_by_id = {
        id(item): row_idx
        for row_idx, row in enumerate(rows)
        for item in row
    }

    option_destination.mkdir(parents=True, exist_ok=True)
    qnum_safe = str(question.question_number or "0").zfill(3)

    for item in labels:
        label = str(item["label"])
        row_idx = row_index_by_id.get(id(item), 0)
        row = rows[row_idx]
        col_idx = next((i for i, it in enumerate(row) if it is item), 0)
        next_row_y = row_centers[row_idx + 1] - 8.0 if row_idx + 1 < len(row_centers) else question_end_y - 4.0

        # Split horizontal space by neighboring option labels in the same row.
        # This avoids cutting image-based choices that are laid out in 3-4 columns.
        overlap_pad = 42.0

        if col_idx == 0:
            x0 = 8.0
        else:
            prev_x = float(row[col_idx - 1]["x0"])
            curr_x = float(item["x0"])
            x0 = max(8.0, (prev_x + curr_x) / 2.0 - overlap_pad)

        if col_idx + 1 >= len(row):
            x1 = page_width - 8.0
        else:
            curr_x = float(item["x0"])
            next_x = float(row[col_idx + 1]["x0"])
            x1 = min(page_width - 8.0, (curr_x + next_x) / 2.0 + overlap_pad)

        y0 = max(0.0, float(item["y0"]) - 8.0)
        y1 = min(question_end_y - 2.0, next_row_y)
        if y1 - y0 < 40.0:
            y1 = min(question_end_y - 2.0, y0 + 180.0)
        if y1 <= y0 + 8.0:
            continue

        clip = fitz.Rect(x0, y0, x1, y1)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)

        out_name = f"{pdf_stem}_q{qnum_safe}_p{page_idx:03d}_opt_{label}.png"
        out_path = option_destination / out_name
        pix.save(str(out_path))

        question.option_image_paths[label] = str(out_path)
        question.option_image_bboxes[label] = [float(clip.x0), float(clip.y0), float(clip.x1), float(clip.y1)]


def extract_question_region_images(
    pdf_path: Path,
    image_output_dir: Path,
    questions: list[ParsedQuestion],
    zoom: float = 2.2,
) -> None:
    """
    Render each question's page region to a PNG image.
    This captures vector diagrams/tables that are not embedded raster images.
    """
    _require_fitz()
    pdf_stem = pdf_path.stem
    destination = image_output_dir / pdf_stem / "question_regions"
    option_destination = image_output_dir / pdf_stem / "option_regions"
    destination.mkdir(parents=True, exist_ok=True)

    questions_by_page: dict[int, list[ParsedQuestion]] = {}
    for question in questions:
        if question.page_start is None:
            continue
        questions_by_page.setdefault(int(question.page_start), []).append(question)

    if not questions_by_page:
        return

    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            page_questions = questions_by_page.get(page_idx, [])
            if not page_questions:
                continue

            markers = _extract_question_line_markers(page)
            option_markers = _extract_option_line_markers(page)
            wanted_qnums = {
                str(q.question_number)
                for q in page_questions
                if q.question_number is not None
            }
            if wanted_qnums:
                markers = [m for m in markers if str(m["question_number"]) in wanted_qnums]

            page_rect = page.rect
            used_markers: set[int] = set()

            if not markers:
                # Last-resort fallback: split the page into equal bands by question count.
                sorted_questions = sorted(
                    [q for q in page_questions if q.question_number],
                    key=lambda q: int(str(q.question_number)),
                )
                if not sorted_questions:
                    continue

                top = 8.0
                bottom = page_rect.height - 8.0
                band_height = max(40.0, (bottom - top) / len(sorted_questions))

                for idx, question in enumerate(sorted_questions):
                    y_start = top + (idx * band_height)
                    y_end = min(bottom, y_start + band_height)
                    clip = fitz.Rect(8.0, y_start, page_rect.width - 8.0, y_end)
                    matrix = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)

                    safe_qnum = str(question.question_number).zfill(3)
                    out_name = f"{pdf_stem}_q{safe_qnum}_p{page_idx:03d}.png"
                    out_path = destination / out_name
                    pix.save(str(out_path))

                    question.region_image_path = str(out_path)
                    question.region_bbox = [float(clip.x0), float(clip.y0), float(clip.x1), float(clip.y1)]
                continue

            for question in page_questions:
                if not question.question_number:
                    continue

                marker_idx, marker = _select_marker_for_question(
                    question_number=str(question.question_number),
                    markers=markers,
                    used_indices=used_markers,
                )
                if marker_idx is None or marker is None:
                    continue

                used_markers.add(marker_idx)

                y_start = max(0.0, float(marker["y0"]) - 2.0)
                # Region ends at next question marker if available, otherwise to page bottom.
                next_y = page_rect.height - 8.0
                found_next_question = False
                for next_marker in markers:
                    if float(next_marker["y0"]) > float(marker["y0"]) + 1.0:
                        next_y = float(next_marker["y0"]) - 8.0
                        found_next_question = True
                        break

                question_end_y = next_y

                # Keep full question body until the next question marker.
                # This avoids truncating large tables and image-based choice rows.
                if not found_next_question:
                    # Prevent runaway crops when no reliable delimiter exists.
                    fallback_height = 560.0 if len(question.options) >= 3 else 460.0
                    next_y = min(next_y, y_start + fallback_height)

                # Avoid oversized crops only when we do not have clear delimiters.
                if not found_next_question:
                    max_region_height = page_rect.height * 0.82
                    if next_y - y_start > max_region_height:
                        next_y = y_start + max_region_height

                if next_y <= y_start + 12.0:
                    next_y = min(page_rect.height - 8.0, y_start + 220.0)

                clip = fitz.Rect(8.0, y_start, page_rect.width - 8.0, next_y)
                if clip.width <= 1 or clip.height <= 1:
                    continue

                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)

                safe_qnum = str(question.question_number).zfill(3)
                out_name = f"{pdf_stem}_q{safe_qnum}_p{page_idx:03d}.png"
                out_path = destination / out_name
                pix.save(str(out_path))

                question.region_image_path = str(out_path)
                question.region_bbox = [float(clip.x0), float(clip.y0), float(clip.x1), float(clip.y1)]
                _attach_option_image_crops(
                    page=page,
                    question=question,
                    option_destination=option_destination,
                    pdf_stem=pdf_stem,
                    page_idx=page_idx,
                    y_start=y_start,
                    question_end_y=question_end_y,
                    zoom=zoom,
                )


def _extract_layout_markers(page: Any) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    question_markers: list[dict[str, float | str]] = []
    option_markers: list[dict[str, float | str]] = []
    data = page.get_text("dict")

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            line_text = "".join(str(span.get("text", "")) for span in spans).strip()
            if not line_text:
                continue

            x0, y0, x1, y1 = line.get("bbox", (0.0, 0.0, 0.0, 0.0))

            q_match = QUESTION_START_RE.match(line_text)
            if q_match:
                question_markers.append(
                    {
                        "question_number": q_match.group(1),
                        "y0": float(y0),
                        "y1": float(y1),
                    }
                )

            option_match = OPTION_RE.match(line_text)
            if option_match:
                option_markers.append(
                    {
                        "choice_label": option_match.group(1).upper(),
                        "y0": float(y0),
                        "y1": float(y1),
                    }
                )

    question_markers.sort(key=lambda item: float(item["y0"]))
    option_markers.sort(key=lambda item: float(item["y0"]))
    return question_markers, option_markers


def _guess_question_number(
    image_mid_y: float,
    question_markers: list[dict[str, float | str]],
) -> tuple[str | None, float]:
    if not question_markers:
        return None, 0.2

    candidate: dict[str, float | str] | None = None
    for marker in question_markers:
        if float(marker["y0"]) <= image_mid_y:
            candidate = marker
        else:
            break

    if candidate is None:
        first = question_markers[0]
        distance = abs(image_mid_y - float(first["y0"]))
        confidence = 0.45 if distance < 200 else 0.3
        return str(first["question_number"]), confidence

    distance = abs(image_mid_y - float(candidate["y0"]))
    confidence = 0.9 if distance < 120 else 0.75 if distance < 240 else 0.6
    return str(candidate["question_number"]), confidence


def _guess_choice_label(
    image_mid_y: float,
    option_markers: list[dict[str, float | str]],
    max_distance: float = 80.0,
) -> tuple[str | None, float]:
    if not option_markers:
        return None, 0.0

    nearest: dict[str, float | str] | None = None
    nearest_distance = float("inf")
    for marker in option_markers:
        marker_mid = (float(marker["y0"]) + float(marker["y1"])) / 2.0
        distance = abs(image_mid_y - marker_mid)
        if distance < nearest_distance:
            nearest = marker
            nearest_distance = distance

    if nearest is None or nearest_distance > max_distance:
        return None, 0.0

    confidence = 0.95 if nearest_distance < 24 else 0.8 if nearest_distance < 50 else 0.65
    return str(nearest["choice_label"]), confidence


def _classify_image(
    width: int | None,
    height: int | None,
    choice_label: str | None,
) -> tuple[str, float]:
    w = width or 0
    h = height or 0
    area = w * h

    if w and h and (w <= 44 or h <= 44 or area <= 2500):
        return "decorative_or_icon", 0.92

    if choice_label and area <= 30000:
        return "possible_option_figure", 0.78

    if w and h and (area >= 16000 or w >= 160 or h >= 120):
        return "question_figure", 0.84

    return "unclassified", 0.45


def extract_text_by_page(pdf_path: Path) -> list[dict[str, str | int]]:
    """Extract plain text from each page while preserving page index metadata."""
    _require_fitz()
    pages: list[dict[str, str | int]] = []
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc, start=1):
            page_text = page.get_text("text", sort=True)
            pages.append(
                {
                    "page_number": idx,
                    "text": _prepare_text_for_parsing(page_text),
                }
            )
    return pages


def extract_images(
    pdf_path: Path,
    image_output_dir: Path,
    questions: list[ParsedQuestion] | None = None,
) -> list[ExtractedImage]:
    """Extract embedded images with page-layout classification metadata."""
    _require_fitz()
    extracted: list[ExtractedImage] = []
    pdf_stem = pdf_path.stem
    destination = image_output_dir / pdf_stem
    destination.mkdir(parents=True, exist_ok=True)

    valid_questions_by_page: dict[int, list[str]] = {}
    for q in questions or []:
        if q.page_start is None or not q.question_number:
            continue
        valid_questions_by_page.setdefault(int(q.page_start), []).append(str(q.question_number))

    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            question_markers, option_markers = _extract_layout_markers(page)
            image_list = page.get_images(full=True)
            for image_idx, img in enumerate(image_list, start=1):
                xref = img[0]
                base = doc.extract_image(xref)
                image_bytes = base["image"]
                ext = base.get("ext", "bin")
                out_name = f"{pdf_stem}_p{page_idx:03d}_{image_idx:03d}.{ext}"
                out_path = destination / out_name

                if not out_path.exists():
                    with out_path.open("wb") as handle:
                        handle.write(image_bytes)

                rects = page.get_image_rects(xref)
                if not rects:
                    rects = [fitz.Rect(0.0, 0.0, 0.0, 0.0)]

                for occurrence_idx, rect in enumerate(rects, start=1):
                    image_mid_y = (float(rect.y0) + float(rect.y1)) / 2.0
                    question_number, question_conf = _guess_question_number(image_mid_y, question_markers)
                    valid_for_page = valid_questions_by_page.get(page_idx, [])
                    if valid_for_page:
                        if question_number not in valid_for_page:
                            # If we cannot confidently map, prefer unknown over a wrong question id.
                            question_number = valid_for_page[0] if len(valid_for_page) == 1 else None
                            question_conf = 0.55 if len(valid_for_page) == 1 else 0.25

                    width = base.get("width")
                    height = base.get("height")
                    area = (width or 0) * (height or 0)

                    # Large figures are unlikely to be option glyphs.
                    if area > 30000:
                        choice_label, choice_conf = None, 0.0
                    else:
                        choice_label, choice_conf = _guess_choice_label(image_mid_y, option_markers)

                    classification, base_conf = _classify_image(
                        width=width,
                        height=height,
                        choice_label=choice_label,
                    )

                    final_conf = max(base_conf, question_conf, choice_conf)

                    extracted.append(
                        ExtractedImage(
                            pdf_name=pdf_path.name,
                            page_number=page_idx,
                            image_index=image_idx,
                            image_occurrence_index=occurrence_idx,
                            width=width,
                            height=height,
                            bbox=[float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                            question_number=question_number,
                            classification=classification,
                            choice_label=choice_label,
                            classification_confidence=round(final_conf, 3),
                            ext=ext,
                            file_path=str(out_path),
                        )
                    )

    return extracted
