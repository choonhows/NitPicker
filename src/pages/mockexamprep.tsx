import { useState } from "react";
import NavBar from "../components/navbar";
import Footer from "../components/footer";

const categoryTopics = {
  Technology: [
    "Number Systems & Data Representation",
    "Applied Mathematics",
    "Discrete Math & Algorithms",
    "Computer Architecture & Hardware",
    "Operating Systems",
    "Digital Logic",
    "Computer Graphics",
    "Databases",
    "Networking",
    "Cybersecurity",
    "Software Engineering & Design",
    "Software Testing",
    "Emerging Technologies",
  ],
  Management: [
    "Project Management",
    "IT Service Management (ITSM)",
    "System Auditing",
    "Quality Management",
    "Corporate Finance",
  ],
  Strategy: [
    "Business Strategy",
    "System Strategy",
    "IT Governance & Compliance",
    "Law & Intellectual Property",
    "Digital Trends",
  ],
} as const;

const allTopics = Object.values(categoryTopics).flat();
type CategoryName = keyof typeof categoryTopics;

// This is a temporary page to show the exam maker page
export default function MockExamPrepPage() {
  const [examType, setExamType] = useState<"AM EXAM" | "PM EXAM">("AM EXAM");
  const [isTimed, setIsTimed] = useState(true);
  const [durationMinutes, setDurationMinutes] = useState(90);
  const [questionCount, setQuestionCount] = useState(80);
  const [instantAnswers, setInstantAnswers] = useState(false);
  const [selectedTopics, setSelectedTopics] = useState<string[]>(allTopics);

  const toggleTopic = (topic: string) => {
    setSelectedTopics((currentTopics) => {
      const isSelected = currentTopics.includes(topic);

      if (isSelected) {
        // At least one topic should remain selected.
        if (currentTopics.length === 1) {
          return currentTopics;
        }

        return currentTopics.filter((currentTopic) => currentTopic !== topic);
      }

      return [...currentTopics, topic];
    });
  };

  const toggleCategory = (categoryName: CategoryName) => {
    const topicsInCategory: readonly string[] = categoryTopics[categoryName];

    setSelectedTopics((currentTopics) => {
      const hasAllTopics = topicsInCategory.every((topic) => currentTopics.includes(topic));

      if (hasAllTopics) {
        const remainingTopics = currentTopics.filter((topic) => !topicsInCategory.includes(topic));

        // Keep at least one selected topic across all categories.
        if (remainingTopics.length === 0) {
          return currentTopics;
        }

        return remainingTopics;
      }

      return [...new Set([...currentTopics, ...topicsInCategory])];
    });
  };

  const renderCategoryFieldset = (categoryName: CategoryName) => {
    const topics = categoryTopics[categoryName];
    const isCategoryFullySelected = topics.every((topic) => selectedTopics.includes(topic));

    return (
      <fieldset key={categoryName} className="border-2 p-3 lg:p-4">
        <div className="mb-3">
          <label className="inline-flex items-center gap-2 text-base lg:text-lg font-medium">
            <input
              type="checkbox"
              checked={isCategoryFullySelected}
              onChange={() => toggleCategory(categoryName)}
            />
            <span>{categoryName}</span>
          </label>
        </div>
        <div className="grid grid-cols-1 gap-2">
          {topics.map((topic) => (
            <label key={topic} className="flex items-center gap-2 text-sm lg:text-base">
              <input
                type="checkbox"
                checked={selectedTopics.includes(topic)}
                onChange={() => toggleTopic(topic)}
              />
              <span>{topic}</span>
            </label>
          ))}
        </div>
      </fieldset>
    );
  };

  return (
    <>
      <div className="min-h-screen flex w-full flex-col items-center bg-white gap-10 select-none">
        <NavBar></NavBar>

        <div className="min-h-[78vh] w-full flex flex-col lg:flex-row items-center  justify-between px-6 lg:px-20 gap-10">
          <div className="flex flex-col gap-10 w-full lg:w-auto">
            <h1 className="-tracking-widest text-6xl lg:text-8xl">You ready?</h1>
            <button className="flex py-6 lg:py-8 items-center justify-center border-2 duration-300 group hover:bg-black cursor-pointer">
              <h1 className="text-2xl group-hover:text-white font-bold group-hover:font-light duration-300">
                LETS GO
              </h1>
            </button>
          </div>

          <div className="border-2 h-[70vh] w-full lg:w-[52vw] flex flex-col overflow-hidden">
            <div className="border-b-2 p-4 lg:p-6 flex flex-col gap-4">
              <h1 className="font-light text-3xl">Exam Type</h1>
              <div className="w-full flex gap-3">
                <button
                  type="button"
                  onClick={() => setExamType("AM EXAM")}
                  className={`font-medium text-base lg:text-xl px-4 py-2 border-2 flex-1 duration-200 cursor-pointer ${
                    examType === "AM EXAM" ? "bg-black text-white" : "bg-white text-black"
                  }`}
                >
                  AM EXAM
                </button>
                <button
                  type="button"
                  onClick={() => setExamType("PM EXAM")}
                  className={`font-medium text-base lg:text-xl px-4 py-2 border-2 flex-1 duration-200 cursor-pointer ${
                    examType === "PM EXAM" ? "bg-black text-white" : "bg-white text-black"
                  }`}
                >
                  PM EXAM
                </button>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-4 lg:p-6 flex flex-col gap-7">
              <div className="flex flex-col lg:flex-row gap-6 p-4">
                {/* TIMER */}
                <section className="flex flex-col gap-3 lg:flex-1">
                  <h2 className="text-2xl font-light">Timer</h2>
                  <div className="flex gap-5 items-center">
                    <label className="flex items-center gap-2 text-sm lg:text-base">
                      <input
                        type="radio"
                        name="timed-mode"
                        checked={isTimed}
                        onChange={() => setIsTimed(true)}
                      />
                      Timed
                    </label>
                    <label className="flex items-center gap-2 text-sm lg:text-base">
                      <input
                        type="radio"
                        name="timed-mode"
                        checked={!isTimed}
                        onChange={() => setIsTimed(false)}
                      />
                      Not timed
                    </label>
                  </div>
                  
                {/* NUMBER OF QUESTIONS */}
                  <label className="flex flex-col gap-2 w-1/2">
                    <span className="text-sm lg:text-base">Duration (minutes)</span>
                    <input
                      type="number"
                      min={1}
                      max={180}
                      disabled={!isTimed}
                      value={durationMinutes}
                      onChange={(event) => {
                        const parsedMinutes = Number(event.target.value);
                        if (Number.isNaN(parsedMinutes)) {
                          return;
                        }
                        setDurationMinutes(Math.min(180, Math.max(1, parsedMinutes)));
                      }}
                      className="border-2 px-3 py-2 disabled:opacity-45"
                    />
                  </label>
                  <p className="text-xs opacity-70">Default: 90 minutes. Maximum: 180 minutes.</p>
                </section>
                
                {/* QUESTIONS */}
                <section className="flex flex-col gap-3 lg:flex-1">
                  <h2 className="text-2xl font-light">Questions</h2>
                  <label className="flex flex-col gap-2 max-w-xs">
                    <span className="text-sm lg:text-base">How many questions?</span>
                    <input
                      type="number"
                      min={1}
                      max={100}
                      value={questionCount}
                      onChange={(event) => {
                        const parsedCount = Number(event.target.value);
                        if (Number.isNaN(parsedCount)) {
                          return;
                        }
                        setQuestionCount(Math.min(100, Math.max(1, parsedCount)));
                      }}
                      className="border-2 px-3 py-2"
                    />
                  </label>
                  <p className="text-xs opacity-70">Default: 80 questions. Maximum: 100 questions.</p>
                </section>
              </div>
              
              <section className="flex flex-col gap-3 pb-2">
                <h2 className="text-2xl font-light">Answer Reveal</h2>
                <label className="flex items-center gap-2 text-sm lg:text-base">
                  <input
                    type="checkbox"
                    checked={instantAnswers}
                    onChange={(event) => setInstantAnswers(event.target.checked)}
                  />
                  Show the correct answer instantly after each question
                </label>
              </section>

              <section className="flex flex-col gap-4">
                <h2 className="text-2xl font-light">Categories & Topics</h2>
                <p className="text-xs opacity-70">
                  Default: all topics selected. At least one topic must stay selected.
                </p>

                <div className="flex flex-col lg:grid lg:grid-cols-[1.2fr_1fr] gap-4 items-start">
                  <div className="w-full">{renderCategoryFieldset("Technology")}</div>
                  <div className="w-full flex flex-col gap-4">
                    {renderCategoryFieldset("Management")}
                    {renderCategoryFieldset("Strategy")}
                  </div>
                </div>
              </section>

              
            </div>
          </div>
        </div>

        <Footer></Footer>
      </div>
    </>
  );
}
