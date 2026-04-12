import type { ReactNode } from 'react'
import NavBar from './navbar'
import Footer from './footer'

type LayoutProps = {
  children: ReactNode
  className?: string
}

export default function Layout({ children, className = '' }: LayoutProps) {
  return (
    <div className={`min-h-screen flex w-full flex-col items-center bg-white dark:bg-zinc-950 ${className}`}>
      <NavBar />
      {children}
      <Footer />
    </div>
  )
}
