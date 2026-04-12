type LoadingSpinnerProps = {
  label?: string
  className?: string
}

export default function LoadingSpinner({ label = 'Loading...', className = '' }: LoadingSpinnerProps) {
  return (
    <div className={`flex items-center gap-3 ${className}`} role="status" aria-live="polite" aria-label={label}>
      <span className="h-5 w-5 rounded-full border-2 border-black/30 dark:border-white/30 border-t-black dark:border-t-white animate-spin" />
      <span className="text-sm opacity-75">{label}</span>
    </div>
  )
}
