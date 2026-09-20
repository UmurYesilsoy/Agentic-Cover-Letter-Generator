# Agentic Cover Letter System

A LangGraph system that turns **a job posting + a CV + past cover letters** into a tailored cover
letter — with company research, evidence selection, deterministic constraint checking, a
rubric-based revision loop, and a human approval gate.

Everything lives in [`cover_letter_agent.ipynb`](cover_letter_agent.ipynb).

## Quick start

```bash
uv venv --python 3.13
uv pip install langgraph langchain-anthropic python-dotenv ipykernel pypdf python-docx
```

Put your key in `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Drop your documents into `inputs/`, then open the notebook and select the `.venv` kernel.

```
inputs/
├── job_posting.txt        # or .pdf / .docx / .md
├── cv.txt                 # or .pdf / .docx
└── past_letters/          # any number of letters you wrote before
    ├── 01_elsevier.txt
    └── ...
```

The repo ships with a worked example (a Coolblue data-science posting) so the notebook runs
unmodified on first open.

## What the graph does

```
START
  │
  ▼
redact ──────── one call lists the employers in your CV; Python rewrites the text once,
  │             so no later stage can leak a name it never received
  ▼
brief ───────── posting + redacted CV → ranked requirements, which evidence answers
  │             each one, the gaps, and what to lead with
  ▼
research ────── web search → facts about the company, each carrying a source URL;
  │             Python discards anything it cannot attribute
  ▼
write ───────── brief + research + your voice → the letter
  │
  ▼
review ◄────┐   Python rules + one critique call → issues, three scores, edits
  │         │
  ├─ revise ┘   applies the edits
  │
  ▼
approve ─────── interrupt(): approve, or send feedback in your own words
  │
  ▼
 END           writes outputs/<date>_<company>-<role>.md
```

7 nodes, ~5 model calls, linear. Almost everything passed between stages is markdown — only
three things are typed (the employer list, the brief's company/role, the review scores), because
those are the only three Python itself reads.

## The four problems it solves

A single prompt asks one model call to do four incompatible jobs at once. Each stage here exists
because one of them fails in a specific, repeatable way.

| Failure | Cause | Fix |
|---|---|---|
| Wrong evidence chosen | Recency bias — the newest CV entry beats the most *relevant* one | `brief` matches evidence against weighted requirements and has to justify the choice |
| Recycled phrasing | The writer's own tics resurface in every letter | phrases recurring across two or more past letters are banned outright |
| Leaked employer names | An instruction the model can silently drop under revision pressure | `redact` rewrites the CV before any writing stage sees it, plus a regex gate on the output |
| Unfalsifiable flattery | Nothing grounds "your mission resonates with me" | `research` discards any claim without a visited source URL |

## Configuration

All in the `Config` dataclass:

| Setting | Default | Notes |
|---|---|---|
| `nameable_employers` | `()` | Anonymisation allowlist. Everything not listed becomes a generic descriptor. |
| `output_language` | `"English"` | Independent of the posting's language. |
| `min_words` / `max_words` | 280 / 400 | Enforced deterministically, not by asking nicely. |
| `address_hard_gaps` | `False` | Whether unmet hard requirements are named in the letter. |
| `max_revisions` | 3 | |
| `pass_threshold` | 4.0 | Mean of the three review scores, out of 5. |
| `model` | `claude-opus-5` | Used by every node. |

## Outputs

`outputs/` accumulates one file per application: `YYYY-MM-DD_<company>-<role>.md`, the letter.

## Testing

```bash
.venv/bin/python run_headless.py --dry-run   # structure only, no API calls, free
.venv/bin/python run_headless.py             # full run, auto-approves at the review gate
```

## Cost

Roughly five model calls per run, plus web search. About three minutes end to end.
