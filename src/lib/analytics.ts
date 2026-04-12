type AnalyticsEventPayload = Record<string, string | number | boolean>

declare global {
  interface Window {
    dataLayer?: unknown[]
    gtag?: (...args: unknown[]) => void
  }
}

const enabled = () => import.meta.env.VITE_ENABLE_ANALYTICS === 'true'
const measurementId = import.meta.env.VITE_GA_MEASUREMENT_ID

const ensureGtag = () => {
  if (!enabled() || !measurementId || typeof window === 'undefined') {
    return false
  }

  window.dataLayer = window.dataLayer ?? []
  window.gtag =
    window.gtag ??
    function gtag(...args: unknown[]) {
      window.dataLayer?.push(args)
    }

  return true
}

export const initAnalytics = () => {
  if (!ensureGtag()) {
    return
  }

  window.gtag?.('js', new Date())
  window.gtag?.('config', measurementId, { send_page_view: false })
}

export const trackPageView = (path: string) => {
  if (!ensureGtag()) {
    return
  }

  window.gtag?.('event', 'page_view', {
    page_path: path,
  })
}

export const trackEvent = (name: string, payload?: AnalyticsEventPayload) => {
  if (!ensureGtag()) {
    return
  }

  window.gtag?.('event', name, payload)
}
