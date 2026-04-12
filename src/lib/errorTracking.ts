import * as Sentry from '@sentry/react'

export type ErrorMetadata = Record<string, string | number | boolean | null | undefined>

const enabled = () => import.meta.env.VITE_ENABLE_ERROR_TRACKING === 'true'
const dsn = import.meta.env.VITE_SENTRY_DSN

export const initErrorTracking = () => {
  if (!enabled() || !dsn) {
    return
  }

  Sentry.init({
    dsn,
    environment: import.meta.env.MODE,
    tracesSampleRate: 0.1,
  })
}

export const trackError = (error: unknown, metadata?: ErrorMetadata) => {
  if (!enabled()) {
    return
  }

  const normalized = error instanceof Error ? error : new Error(String(error))

  if (dsn) {
    Sentry.captureException(normalized, {
      extra: metadata,
    })
    return
  }

  console.error('[error-tracking:fallback]', {
    message: normalized.message,
    stack: normalized.stack,
    metadata,
  })
}
