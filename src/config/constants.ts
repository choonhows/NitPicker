export const MOCK_EXAM_IMAGE_BASE_PATH =
  import.meta.env.VITE_MOCK_EXAM_IMAGE_BASE_PATH ?? '/pipeline/outputs-all/'

export const UI_LIMITS = {
  maxDurationMinutes: 180,
  maxQuestionCount: 100,
  defaultDurationMinutes: 90,
  defaultQuestionCount: 80,
} as const

// Google Form "NitPicker question reports" (owned by the maintainers); the question code field is pre-filled.
export const reportQuestionUrl = (questionCode: string) =>
  `https://docs.google.com/forms/d/e/1FAIpQLSdKyqKRs2PNqJqqh9SQGQAvkijtYfoHusgoNO9x6uU4W8d1AA/viewform?usp=pp_url&entry.1221462421=${encodeURIComponent(questionCode)}`
