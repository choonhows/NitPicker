import { Route, Routes } from 'react-router-dom'
import { Suspense, lazy } from 'react'
import './App.css'
import LoadingSpinner from './components/LoadingSpinner'

const HomePage = lazy(() => import('./pages/homepage'))
const NotesPage = lazy(() => import('./pages/notes'))
const MockExamPage = lazy(() => import('./pages/mockexam'))
const MockExamPrepPage = lazy(() => import('./pages/mockexamprep'))
const MockExamResultsPage = lazy(() => import('./pages/mockexamresults'))
const PreviousExamsPage = lazy(() => import('./pages/previousexam'))
const NotFoundPage = lazy(() => import('./pages/notfound'))

function App() {
  return (
    <>
      <Suspense
        fallback={(
          <div className="min-h-screen flex items-center justify-center bg-white dark:bg-zinc-950 dark:text-white">
            <LoadingSpinner label="Loading page" />
          </div>
        )}
      >
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/notes" element={<NotesPage />} />
          <Route path="/mockexam" element={<MockExamPage />} />
          <Route path="/mockexamprep" element={<MockExamPrepPage />} />
          <Route path='/mockexamresults' element={<MockExamResultsPage/>} />
          <Route path='/previousexams' element={<PreviousExamsPage />} />
          <Route path='*' element={<NotFoundPage />} />
        </Routes>
      </Suspense>
    </>
  )
}

export default App
