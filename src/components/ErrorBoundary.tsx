import { Component, type ReactNode } from 'react'
import { trackError } from '../lib/errorTracking'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error) {
    trackError(error, { source: 'ErrorBoundary' })
    console.error('ErrorBoundary caught error:', error)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '100vh',
            backgroundColor: 'var(--bg)',
            color: 'var(--text-h)',
            fontFamily: 'var(--sans)',
            padding: '20px',
          }}
        >
          <div
            style={{
              maxWidth: '500px',
              textAlign: 'center',
            }}
          >
            <h1 style={{ fontSize: '24px', marginBottom: '16px' }}>
              Oops! Something went wrong
            </h1>
            <p style={{ marginBottom: '24px', color: 'var(--text)' }}>
              We encountered an unexpected error. Please try refreshing the page.
            </p>
            {import.meta.env.DEV && this.state.error && (
              <pre
                style={{
                  backgroundColor: 'var(--code-bg)',
                  padding: '16px',
                  borderRadius: '4px',
                  overflow: 'auto',
                  textAlign: 'left',
                  fontSize: '12px',
                  marginBottom: '16px',
                }}
              >
                {this.state.error.toString()}
              </pre>
            )}
            <button
              onClick={() => window.location.reload()}
              style={{
                backgroundColor: 'var(--accent)',
                color: '#fff',
                border: 'none',
                padding: '10px 20px',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '16px',
              }}
            >
              Refresh Page
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
