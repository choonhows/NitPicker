from __future__ import annotations

from dataclasses import field
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractedImage:
    pdf_name: str
    page_number: int
    image_index: int
    image_occurrence_index: int = 1
    width: int | None = None
    height: int | None = None
    bbox: list[float] | None = None
    question_number: str | None = None
    classification: str | None = None
    choice_label: str | None = None
    classification_confidence: float | None = None
    ext: str = ""
    file_path: str = ""


@dataclass
class ParsedQuestion:
    pdf_name: str
    page_start: int | None = None
    question_number: str | None = None
    question_text: str = ""
    options: list[str] = field(default_factory=list)
    answer: str | None = None
    explanation: str | None = None
    region_image_path: str | None = None
    region_bbox: list[float] | None = None
    option_image_paths: dict[str, str] = field(default_factory=dict)
    option_image_bboxes: dict[str, list[float]] = field(default_factory=dict)
    category: str | None = None
    mapped_answer: str | None = None
    answer_source_file: str | None = None


@dataclass
class PipelineResult:
    pdf_path: str
    questions: list[ParsedQuestion] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)

    @property
    def pdf_name(self) -> str:
        return Path(self.pdf_path).name
