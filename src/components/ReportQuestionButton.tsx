import { useRef, useState } from "react";
import { TriangleAlert, X } from "lucide-react";
import { SUPABASE_PUBLISHABLE_KEY, SUPABASE_URL } from "../config/constants";

const REASONS = [
  ["wrong_answer", "Wrong answer key"],
  ["image_problem", "Image missing, cut off, or not needed"],
  ["garbled_text", "Garbled text or symbols"],
  ["broken_choices", "Choices broken or duplicated"],
  ["not_a_real_question", "Not a real exam question"],
  ["other", "Something else"],
] as const;

type State = "idle" | "sending" | "sent" | "error";

/** "Report" badge + dialog that saves a row to Supabase `question_reports` (insert-only for visitors). */
export default function ReportQuestionButton({ questionCode, className = "" }: { questionCode: string; className?: string }) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [reason, setReason] = useState("");
  const [details, setDetails] = useState("");
  const [state, setState] = useState<State>("idle");

  const open = () => {
    setState("idle");
    setReason("");
    setDetails("");
    dialogRef.current?.showModal();
  };
  const close = () => dialogRef.current?.close();

  const send = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!reason || state === "sending") return;
    setState("sending");
    try {
      const response = await fetch(`${SUPABASE_URL}/rest/v1/question_reports`, {
        method: "POST",
        headers: {
          apikey: SUPABASE_PUBLISHABLE_KEY,
          Authorization: `Bearer ${SUPABASE_PUBLISHABLE_KEY}`,
          "Content-Type": "application/json",
          Prefer: "return=minimal",
        },
        body: JSON.stringify({
          question_code: questionCode,
          reason,
          details: details.trim().slice(0, 1000) || null,
          page: window.location.pathname.slice(0, 200),
        }),
      });
      setState(response.ok ? "sent" : "error");
    } catch {
      setState("error");
    }
  };

  return (
    <>
      <button
        type="button"
        onClick={open}
        title="Report a problem with this question"
        className={`badge-brut shrink-0 cursor-pointer bg-white dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 hover:border-zinc-950 dark:hover:border-zinc-300 ${className}`}
      >
        <TriangleAlert className="w-3 h-3" />
        <span>REPORT</span>
      </button>

      <dialog
        ref={dialogRef}
        aria-labelledby={`report-title-${questionCode}`}
        onClick={(event) => event.target === dialogRef.current && close()}
        // Keep typing and Space/Enter in here away from the exam's keyboard shortcuts.
        onKeyDown={(event) => event.stopPropagation()}
        className="modal-panel m-auto p-0 backdrop:bg-black/60 text-zinc-950 dark:text-zinc-100"
      >
        <div className="p-6 md:p-8 flex flex-col gap-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h2 id={`report-title-${questionCode}`} className="text-xl font-bold font-heading">
                Report a problem
              </h2>
              <p className="font-mono text-xs text-zinc-500 dark:text-zinc-400 mt-1">{questionCode}</p>
            </div>
            <button type="button" onClick={close} aria-label="Close" className="p-1 cursor-pointer">
              <X className="h-5 w-5" />
            </button>
          </div>

          {state === "sent" ? (
            <div className="motion-rise flex flex-col gap-4" role="status">
              <p className="text-sm leading-relaxed">Thanks. The maintainers will check this question and fix it.</p>
              <button type="button" onClick={close} className="btn-solid px-4 py-2.5 text-xs font-mono uppercase self-start">
                Done
              </button>
            </div>
          ) : (
            <form onSubmit={send} className="flex flex-col gap-5">
              <fieldset className="flex flex-col gap-2">
                <legend className="text-sm font-bold mb-2">What's wrong with this question?</legend>
                {REASONS.map(([value, label]) => (
                  <label key={value} className="flex items-center gap-3 text-sm cursor-pointer min-h-9">
                    <input
                      type="radio"
                      name={`reason-${questionCode}`}
                      value={value}
                      checked={reason === value}
                      onChange={() => setReason(value)}
                      required
                      className="h-4 w-4 accent-blue-600"
                    />
                    {label}
                  </label>
                ))}
              </fieldset>

              <label className="flex flex-col gap-2 text-sm font-bold">
                <span>
                  Details <span className="font-normal text-zinc-500 dark:text-zinc-400">(optional)</span>
                </span>
                <textarea
                  value={details}
                  onChange={(event) => setDetails(event.target.value)}
                  maxLength={1000}
                  rows={3}
                  placeholder="e.g. the table is cut off, the answer should be c)"
                  className="form-input font-sans text-base font-normal resize-y"
                />
              </label>

              {state === "error" ? (
                <p className="text-sm text-red-600 dark:text-red-400" role="alert">
                  The report didn't send. Check your connection and try again.
                </p>
              ) : null}

              <div className="flex gap-3">
                <button
                  type="submit"
                  disabled={!reason || state === "sending"}
                  className="btn-solid px-4 py-2.5 text-xs font-mono uppercase disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {state === "sending" ? "Sending…" : "Send report"}
                </button>
                <button type="button" onClick={close} className="btn-outline px-4 py-2.5 text-xs font-mono uppercase">
                  Cancel
                </button>
              </div>
            </form>
          )}
        </div>
      </dialog>
    </>
  );
}
