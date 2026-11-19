export const MOCK_EXAM_IMAGE_BASE_PATH =
  import.meta.env.VITE_MOCK_EXAM_IMAGE_BASE_PATH ?? '/pipeline/outputs-all/'

export const UI_LIMITS = {
  maxDurationMinutes: 180,
  maxQuestionCount: 100,
  defaultDurationMinutes: 90,
  defaultQuestionCount: 80,
} as const

// Supabase project for question reports. The publishable key is meant for browser code;
// the question_reports table only lets visitors insert (see the RLS policy in Supabase).
export const SUPABASE_URL = "https://oawmeieyurmqdmyvqrbe.supabase.co"
export const SUPABASE_PUBLISHABLE_KEY = "sb_publishable_d1Mkk5hKwFw3KG6nAS6XcA_j0VzqHWN"
