type SkeletonLoaderProps = {
  lines?: number
}

export default function SkeletonLoader({ lines = 4 }: SkeletonLoaderProps) {
  return (
    <div className="w-full animate-pulse flex flex-col gap-2" aria-hidden="true">
      {Array.from({ length: lines }).map((_, index) => (
        <div
          key={`skeleton-line-${index}`}
          className={`h-4 rounded bg-black/10 dark:bg-white/10 ${index === lines - 1 ? 'w-2/3' : 'w-full'}`}
        />
      ))}
    </div>
  )
}
