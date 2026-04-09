# PhilNITS PDF Content Pipeline

A Python pipeline for PDFs containing PhilNITS review questions and answers.

## Features

- Extracts page text from PDF files.
- Parses question/choice/answer/explanation blocks using PhilNITS-friendly heuristics.
- Extracts embedded images from PDFs.
- Renders per-question region images (useful for diagrams/tables that are not embedded raster images).
- Extracts per-option image crops for image-based choices when detectable.
- Maps answer keys from a matching answers PDF folder.
- Writes a single structured artifact for downstream workflows:
  - `pipeline_results.json`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Repository Hygiene

Do not commit your local virtual environment.

- Keep `.venv/` in `.gitignore`.
- Commit source code plus dependency files (`requirements.txt`, `pyproject.toml`).
- Do not commit generated `outputs/` if your frontend repo is the deployment target.

## Usage

Run on a single PDF:

```bash
philnits-pipeline "D:/philnits/questions/2024October_A.pdf" --output-dir outputs --image-dir outputs/images
```

Run on a directory recursively:

```bash
philnits-pipeline "D:/philnits/questions" --output-dir outputs --image-dir outputs/images
```

Disable image extraction:

```bash
philnits-pipeline "D:/philnits/questions" --no-images
```

Map answers from a folder containing matching answer-key PDFs:

```bash
philnits-pipeline "D:/philnits/questions/2024October_A.pdf" --output-dir outputs --image-dir outputs/images --answers-dir "D:/philnits/answers"
```

Run recursively and map answers in one command (recommended):

```bash
philnits-pipeline "D:/philnits/questions" --output-dir outputs --image-dir outputs/images --answers-dir "D:/philnits/answers"
```

The pipeline automatically matches each question PDF to an answer PDF by filename, so you do not need to run the command per year/file.

Export directly to a frontend static folder (no backend required):

```bash
philnits-pipeline "D:/philnits/questions/2024October_A.pdf" --output-dir "D:/path/to/frontend/public/ai-data" --image-dir "D:/path/to/frontend/public/ai-data/images" --answers-dir "D:/philnits/answers"
```

## Output

The pipeline writes:

- `outputs/pipeline_results.json`
- image assets under `outputs/images/...`

`pipeline_results.json` includes question text, options, answer/explanation fields, category (`plain_text`, `image`, `table`), mapped answers (if available), region image paths, and option image paths.

Image paths are written as web-safe relative paths (for example `images/2024October_A/...`) so static frontend apps can load assets directly.

## Frontend Integration (No Backend)

Use this project as an offline/CI data generator and publish only generated artifacts in your frontend app.

Recommended frontend structure:

- `public/ai-data/pipeline_results.json`
- `public/ai-data/images/...`

In your website, fetch JSON from:

- `/ai-data/pipeline_results.json`

Then resolve image URLs directly from fields in the JSON (`region_image_path`, `option_image_paths`, `images[].file_path`).

If you want to preview locally before integrating, serve your frontend and open the page that consumes `/ai-data/pipeline_results.json`.


## Parsing Notes

The parser uses regex heuristics and works best when the PDF text has patterns like:

- Question start: `12. ...`, `12) ...`, `Q12: ...`
- Options: `A. ...`, `B) ...`
- Answer line: `Answer: C`
- Explanation line: `Explanation: ...`

If your materials have a different format, update regexes in `src/pdf_pipeline/parser.py`.
