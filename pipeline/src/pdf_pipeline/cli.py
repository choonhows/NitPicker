from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="philnits-pipeline",
        description="Extract PhilNITS questions/answers from PDFs and optionally extract embedded images.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a PDF file or a directory containing PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for structured output file (pipeline_results.json).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("outputs/images"),
        help="Directory where extracted images are saved.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Disable image extraction.",
    )
    parser.add_argument(
        "--answers-dir",
        type=Path,
        default=None,
        help="Optional directory containing answer-key PDFs (matched by PDF filename).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results, artifacts = run_pipeline(
        input_path=args.input_path,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        extract_images_enabled=not args.no_images,
        answers_dir=args.answers_dir,
    )

    total_questions = sum(len(item.questions) for item in results)
    total_images = sum(len(item.images) for item in results)

    print(f"Processed PDFs: {len(results)}")
    print(f"Extracted questions: {total_questions}")
    print(f"Extracted images: {total_images}")
    print("Artifacts:")
    for name, path in artifacts.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
