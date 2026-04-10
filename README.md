# NitPicker

NitPicker is a React + TypeScript web app for Philnits FE exam preparation. 

## Warning
this is a rush side project, there might be bugs but I intend to fix later, but for now, 
just have fun and study <3

## Features
It includes:
- Mock exam generation from the question vault
- Configurable exam setup (timed/untimed, question count, topic filters)
- Instant answer reveal mode with explanation support
- Results analytics (score, percentage, pass/fail, category breakdown, wrong-question review)
- Previous exams browser grouped by year, with dropdown question review

## Data Source

Questions are loaded from markdown files under philnits-vault. The app parses each markdown file and extracts:
- Question text
- Choices (A-D)
- Correct answer
- Explanation text
- Inferred category/topic

Markdown and math in explanations are rendered in HTML using:
- react-markdown
- remark-gfm
- remark-math
- rehype-katex
- katex

## Main Routes

- / : Home page
- /notes : Notes page
- /mockexamprep : Mock exam setup
- /mockexam : Exam-taking page
- /mockexamresults : Results page
- /previousexams : Previous exam explorer

## Tech Stack

- React 19
- TypeScript
- Vite
- Tailwind CSS
- Phosphor Icons

## Getting Started

1. Install dependencies

```bash
npm install
```

2. Run development server

```bash
npm run dev
```

3. Build for production

```bash
npm run build
```

4. Preview production build

```bash
npm run preview
```

## Project Structure

- src/pages : App pages
- src/components : Shared UI components
- src/exam : Exam models and question-bank logic
- src/assets : App assets used in components
- public : Static files served directly (including favicon)
- philnits-vault : Markdown question source files
