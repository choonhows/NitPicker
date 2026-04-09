from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import PipelineResult


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(results: list[PipelineResult], out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    output = out_dir / "pipeline_results.json"
    serializable = [asdict(result) for result in results]
    output.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def write_image_manifest(results: list[PipelineResult], out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    output = out_dir / "images_manifest.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for result in results:
            for image in result.images:
                handle.write(json.dumps(asdict(image), ensure_ascii=False) + "\n")
    return output


def _collect_question_answer_mapping(
    node: Any,
    output: dict[str, dict[str, str]],
    current_pdf_name: str | None = None,
) -> None:
    if isinstance(node, list):
        for item in node:
            _collect_question_answer_mapping(item, output, current_pdf_name)
        return

    if not isinstance(node, dict):
        return

    next_pdf_name = current_pdf_name
    if isinstance(node.get("pdf_name"), str):
        next_pdf_name = node["pdf_name"]
    elif isinstance(node.get("pdf_path"), str):
        next_pdf_name = Path(node["pdf_path"]).name

    question_number = node.get("question_number")
    if question_number is not None and next_pdf_name:
        answer_value = node.get("mapped_answer") or node.get("answer")
        if isinstance(answer_value, str) and answer_value.strip():
            per_pdf = output.setdefault(next_pdf_name, {})
            per_pdf[str(question_number)] = answer_value.strip().upper()

    for value in node.values():
        if isinstance(value, (dict, list)):
            _collect_question_answer_mapping(value, output, next_pdf_name)


def write_question_answer_mapping(results: list[PipelineResult], out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    output = out_dir / "question_answer_mapping.json"

    serializable = [asdict(result) for result in results]
    mapping: dict[str, dict[str, str]] = {}
    _collect_question_answer_mapping(serializable, mapping)

    ordered_mapping = {
        pdf_name: {
            question: answers[question]
            for question in sorted(answers.keys(), key=lambda value: int(value) if value.isdigit() else value)
        }
        for pdf_name, answers in sorted(mapping.items())
    }

    output.write_text(json.dumps(ordered_mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
