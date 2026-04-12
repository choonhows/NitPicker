export const MOCK_EXAM_IMAGE_BASE_PATH =
  import.meta.env.VITE_MOCK_EXAM_IMAGE_BASE_PATH ?? '/pipeline/outputs-all/'

export const UI_LIMITS = {
  maxDurationMinutes: 180,
  maxQuestionCount: 100,
  defaultDurationMinutes: 90,
  defaultQuestionCount: 80,
} as const
