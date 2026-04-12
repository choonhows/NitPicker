import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App.tsx'
import { ErrorBoundary } from './components/ErrorBoundary'
import RouteAnalytics from './components/RouteAnalytics'
import { initAnalytics } from './lib/analytics'
import { initErrorTracking } from './lib/errorTracking'

initErrorTracking()
initAnalytics()

if ('serviceWorker' in navigator && import.meta.env.PROD) {
  void navigator.serviceWorker.register('/sw.js')
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <BrowserRouter>
        <RouteAnalytics />
        <App />
      </BrowserRouter>
    </ErrorBoundary>
  </StrictMode>,
)
