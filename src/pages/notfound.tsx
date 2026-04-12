import { Link } from 'react-router-dom'
import Layout from '../components/Layout'

export default function NotFoundPage() {
  return (
    <Layout className="py-10 md:py-0 dark:text-white">
      <main className="flex-1 w-full flex items-center justify-center px-6 md:px-12">
        <section className="w-full max-w-2xl border-2 border-black dark:border-white p-8 md:p-12 text-center flex flex-col gap-5">
          <p className="text-xs uppercase tracking-[0.25em] opacity-60">Error 404</p>
          <h1 className="text-4xl md:text-6xl font-bold">Page not found</h1>
          <p className="text-sm md:text-base opacity-75">
            The page you are looking for does not exist or may have been moved.
          </p>
          <div className="flex justify-center">
            <Link
              to="/"
              className="border-2 border-black dark:border-white px-6 py-3 font-bold hover:bg-black hover:text-white dark:hover:bg-white dark:hover:text-black transition-colors duration-200"
            >
              GO HOME
            </Link>
          </div>
        </section>
      </main>
    </Layout>
  )
}
