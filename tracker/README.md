# Progress Tracker System

Living documents for MLOps Assignment 2 (50 marks, 5 modules, 21 days).

Same system that ran Assignment 1, with the emphasis shifted to the two things that cost marks last
time: **visible cross-validation** and **thick CI/CD**.

## Files

| File | Purpose | Update cadence |
|------|---------|----------------|
| `PROGRESS.md` | 21-day dashboard with day-by-day checklists | Daily |
| `TASKS.md` | 5 graded modules mapped to deliverables and rubric | Weekly |
| `DAILY_LOG.md` | Append-only journal of work, blockers, decisions | Daily |
| `EVIDENCE.md` | Checklist of screenshots/artifacts needed for the video + submission | When evidence captured |
| `DECISIONS.md` | Architecture/design decisions and rationale (ADR-style) | When deciding |
| `video_script.md` | Demo video shot list (created Day 20) | Day 20-21 |

## Daily workflow

1. **Start of day** — Open `PROGRESS.md`, find today's entry, review goal and sub-tasks.
2. **During work** — As you finish each sub-task, change `[ ]` to `[x]`. If blocked, mark `[!]` and
   note it in `DAILY_LOG.md`.
3. **When deciding architecture/tools** — Add an entry to `DECISIONS.md` BEFORE you implement.
   Captures original thinking (academic integrity).
4. **When you capture a screenshot** — Tick it off in `EVIDENCE.md` so nothing is missing when you
   record the video.
5. **End of day** — Append an entry to `DAILY_LOG.md` (what done, what worked, what didn't, time
   spent, tomorrow's plan, commit SHAs).
6. **End of week** — Update `TASKS.md` status columns and run the weekly checkpoint in `PROGRESS.md`.

## Status legend

- `[ ]` not started
- `[/]` in progress
- `[x]` done
- `[!]` blocked (see latest entry in `DAILY_LOG.md`)
- `[~]` deferred or partially done

## The two standing rules

Carried from the Assignment 1 grader comment — *"No cross-validation, thin CI/CD"*:

1. **Cross-validation is a deliverable, not a technique.** If a grader skimming the repo for 60
   seconds cannot find per-fold numbers, it does not count. Every CV artifact is listed in
   `EVIDENCE.md`.
2. **Every CI/CD stage must be demonstrable.** Registry push, automated deploy, smoke-test gate,
   and a proven red run each need a screenshot and a moment in the video.

## Overall progress

**Days complete:** 0 / 21
**Modules complete:** 0 / 5 (0 / 50 self-assessed)
**Last updated:** 2026-08-15 (Day 0 — planning done; tracker seeded; Day 1 starts next session)
