"""Cover letter writing agent - state, prompts, nodes and the compiled graph, all in one
importable module. This is the single source of truth for the pipeline: api.py runs it in
production, graph.py re-exports it for LangGraph Studio, and it can be imported directly from a
notebook or script for interactive use.

Usage from a notebook or script:

    from agent import load_inputs, run

    inputs = load_inputs()      # reads inputs/ next to this file
    result = run(inputs)        # runs through `langgraph dev` if it's up, else in-process
    print(result["final_letter"])
"""
import html
import inspect
import json
import re
import uuid
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Optional, TypedDict

import httpx
from dotenv import load_dotenv

# must run before the langchain/langgraph imports below: langsmith caches whether tracing is
# enabled the first time anything reads the env var, so importing those packages first makes
# LANGSMITH_TRACING silently no-op even once .env is loaded
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import os
assert os.environ.get("ANTHROPIC_API_KEY"), f"Put ANTHROPIC_API_KEY in {BASE_DIR / '.env'}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: bool) -> bool:
    """Reads a boolean from the environment, defaulting to `default` when unset - lets local/CLI
    use keep today's behavior with zero config, while a deployment (Render's dashboard env vars)
    can flip it off explicitly. See dump_prompts/persist_letters below: both default on, since
    they're genuinely useful for local/notebook use, but neither belongs in a stateless API
    deployment writing personal content (CVs, letters) to the server's own disk for no reason."""
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    model: str = "claude-opus-5"
    output_language: str = "English"

    # per-section word budgets
    intro_words: int = 95
    body_words: int = 235
    close_words: int = 45
    max_words: int = 390        # hard ceiling for the assembled letter

    web_search_max_uses: int = 6
    # write every rendered prompt to outputs/.prompts/ (includes CV/job ad/letter text verbatim,
    # since that's what's interpolated into most prompts) - set DUMP_PROMPTS=false in a
    # deployment's environment to turn this off
    dump_prompts: bool = _env_flag("DUMP_PROMPTS", True)
    # write the finished letter to outputs/ - set PERSIST_LETTERS=false in a deployment's
    # environment; the API already returns the letter in its response, so a second copy on the
    # server's own disk serves no purpose there and needlessly retains personal content
    persist_letters: bool = _env_flag("PERSIST_LETTERS", True)

    # evaluate/revise loop (automatic, pre-human_review only - see route_after_evaluate): passes
    # once overall_score >= this AND coherent/flows_well/grounded/specific are all true AND
    # there are no unsupported claims, or once max_auto_revisions passes are spent, whichever
    # comes first - either way, control then goes to human_review, not straight to END
    eval_score_threshold: int = 4
    max_auto_revisions: int = 3

    # anchored to this file's directory, not the caller's cwd, so a notebook importing this
    # module from anywhere still reads/writes the same inputs/outputs as the notebook version
    inputs_dir: Path = BASE_DIR / "inputs"
    outputs_dir: Path = BASE_DIR / "outputs"


cfg = Config()
cfg.outputs_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    if suffix == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    raise ValueError(f"Unsupported file type: {path.name}")


def find(stem: str) -> Optional[Path]:
    matches = [p for p in cfg.inputs_dir.glob(f"{stem}.*")
               if p.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}]
    return matches[0] if matches else None


LOGIN_WALL = re.compile(r"(sign in|log in|create an account|enable javascript|captcha|"
                        r"access denied|403 forbidden)", re.I)


READER_URL = "https://r.jina.ai/"


def fetch_job_from_url(url: str) -> Optional[str]:
    """Fetch a posting via Jina AI's Reader API - a JS-rendering proxy that handles the
    JavaScript-heavy job boards (LinkedIn, Workday, Greenhouse) that Claude's own web_fetch tool
    reliably failed on, since that tool never executes JavaScript at all. Returns None when the
    page is unusable (fetch failed, login wall, or too short to be a real posting). Set
    JINA_API_KEY in .env for higher rate limits; works without one for light use."""
    headers = {"Accept": "text/plain"}
    if os.environ.get("JINA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['JINA_API_KEY']}"
    try:
        response = httpx.get(READER_URL + url, headers=headers, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        print(f"[inputs] fetch failed: {type(exc).__name__}")
        return None

    text = response.text
    if len(text.split()) < 150 or LOGIN_WALL.search(text[:600]):
        print("[inputs] the URL did not yield a usable posting")
        return None
    return text


def load_inputs() -> dict:
    posting = None
    url_file = cfg.inputs_dir / "job_url.txt"
    if url_file.exists() and url_file.read_text().strip().startswith("http"):
        url = url_file.read_text().strip()
        print(f"[inputs] fetching {url}")
        posting = fetch_job_from_url(url)

    if posting is None:
        path = find("job_posting")
        if path is None:
            raise FileNotFoundError("No usable job_url.txt and no job_posting.(txt|md|pdf|docx)")
        posting = read_document(path)
        print(f"[inputs] using {path.name}")

    letters_dir = cfg.inputs_dir / "past_letters"
    letter_paths = sorted(p for p in letters_dir.glob("*")
                          if p.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}) if letters_dir.exists() else []

    data = {"job_ad": posting,
            "cv": read_document(find("cv")),
            "past_letters": [read_document(p) for p in letter_paths]}

    print(f"job ad  : {len(data['job_ad'].split())} words")
    print(f"cv      : {len(data['cv'].split())} words")
    print(f"letters : {len(letter_paths)}")
    return data


# ---------------------------------------------------------------------------
# Schemas, state and helpers
# ---------------------------------------------------------------------------

class Qualification(BaseModel):
    qualification: str = Field(description="education, experience, skill or knowledge")
    evidence: str = Field(description="the specific thing on the CV that establishes it")
    value_to_team: str = Field(description="what it would let this particular team do, or do "
                                           "better - phrased from their side, not the candidate's")
    relevance: int = Field(ge=1, le=5)


class Qualifications(BaseModel):
    items: list[Qualification]


class ResearchSummary(BaseModel):
    company: str = Field(description="the hiring company's name, exactly as it appears")
    role: str = Field(description="the role's title, exactly as it appears")
    reasons: list[str] = Field(description="the candidate motivations/reasons listed, exactly as "
                                           "given, most probable first - do not summarize, "
                                           "reword or invent any")


class Assembled(BaseModel):
    letter: str = Field(description="the final cover letter - the given opening and body, plus "
                                    "the closing paragraph you write, with every listed problem "
                                    "fixed and duplication/coherence issues resolved")
    changes: list[str] = Field(description="a short list of what was changed and why, for a "
                                           "human reviewing the output")


class LetterEvaluation(BaseModel):
    grounded: bool = Field(description="every claim about the candidate is supported by the CV "
                                       "or the candidate's past cover letters")
    unsupported_claims: list[str] = Field(description="claims that aren't grounded, quoted")
    specific: bool = Field(description="clearly about this role and company - if you could swap "
                                       "in a competitor's name and it would still read sensibly, "
                                       "this fails")
    coherent: bool = Field(description="the three paragraphs don't repeat the same hook, fact or "
                                       "phrase - they were written independently of each other "
                                       "and may have converged on the same point")
    flows_well: bool = Field(description="no sudden topic changes - each paragraph and each "
                                         "sentence follows naturally from what came before, "
                                         "rather than reading as unconnected blocks")
    issues: list[str] = Field(description="concrete problems to fix, specific enough that "
                                          "someone revising the letter would know exactly what "
                                          "to change")
    overall_score: int = Field(ge=1, le=5, description="1 = needs major rework, 5 = no changes "
                                                        "needed")


class RevisedLetter(BaseModel):
    letter: str = Field(description="the corrected cover letter")
    changes: list[str] = Field(description="what was changed and why")


class State(TypedDict, total=False):
    job_ad: str
    cv: str
    past_letters: list

    closings: list

    research_notes: str
    sources: list

    # candidate_reasons: research()'s ranked list, offered as a menu by select_reasons().
    # selected_reasons: what the human actually picked there (plus anything they typed
    # themselves) - this, not candidate_reasons, is what para_intro writes from.
    candidate_reasons: list
    selected_reasons: list

    # qualifications: assess_qualifications()'s ranked list, offered as a menu by
    # select_qualifications().
    # selected_qualifications: what was picked there (plus any custom entries) - this, not
    # qualifications, is what para_body writes from.
    qualifications: list
    selected_qualifications: list
    para_intro: str
    para_body: str

    final_letter: str
    changes: list

    evaluation_issues: list
    evaluation_unsupported_claims: list
    evaluation_score: int
    evaluation_grounded: bool
    evaluation_specific: bool
    evaluation_coherent: bool
    evaluation_flows_well: bool
    revision_count: int

    # set by the caller - each gates one node independently, so any combination of the three can
    # be interactive while the rest run automatically
    select_reasons_in_the_loop: bool          # select_reasons
    select_qualifications_in_the_loop: bool   # select_qualifications
    human_in_the_loop: bool                   # human_review
    human_action: str

    # set by research() (see its docstring for why) - used by finalize()/revise() via
    # save_letter() to name the output file. Not otherwise exposed: every other node still reads
    # job_ad directly rather than these pre-extracted copies of the same information
    company: str
    role: str


def log_prompt(system: str, user: str, node: str = None) -> None:
    """Write the fully rendered prompt to outputs/.prompts/<node>.md.

    The node name is taken from the call stack by default - log_prompt <- ask/prose <- the node -
    so no call site has to pass it, and a new node gets prompt logging for free. Pass `node`
    explicitly when logging from the node function itself rather than through ask/prose."""
    if not cfg.dump_prompts:
        return
    node = node or inspect.stack()[2].function
    directory = cfg.outputs_dir / ".prompts"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{node}.md").write_text(
        f"# {node}\n\n## SYSTEM\n\n{system}\n\n## USER\n\n{user}\n", encoding="utf-8")


def ask(schema, system: str, user: str, attempts: int = 3, max_tokens: int = 16000):
    """A structured call, retried on a malformed response.

    Keep max_tokens generous - Claude Opus 5 thinks by default and those tokens come out of the
    same budget, so starving it truncates the tool call rather than the prose."""
    log_prompt(system, user)
    chain = ChatAnthropic(model=cfg.model, max_tokens=max_tokens).with_structured_output(schema)
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    for attempt in range(1, attempts + 1):
        try:
            return chain.invoke(messages)
        except Exception as exc:
            if attempt == attempts:
                raise
            print(f"    ({schema.__name__} attempt {attempt} failed: {type(exc).__name__}, retrying)")


def prose(system: str, user: str, max_tokens: int = 12000) -> str:
    log_prompt(system, user)
    message = ChatAnthropic(model=cfg.model, max_tokens=max_tokens).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)])
    text = message.text if isinstance(message.text, str) else message.text()
    if not text.strip():
        raise RuntimeError("Model returned no text - thinking consumed the budget; raise max_tokens.")
    return text.strip()


STYLE_RULES = """- Language: {language}.
- Output the paragraph text only. No heading, no preamble, no commentary."""


def style() -> str:
    return STYLE_RULES.format(language=cfg.output_language)


# ---------------------------------------------------------------------------
# Node: extract_closings
# ---------------------------------------------------------------------------

SIGNOFF = re.compile(r"^(kind regards|yours sincerely|yours faithfully|best regards|sincerely|"
                     r"regards|many thanks|thank you,)", re.I | re.M)


def extract_closing(letter: str) -> Optional[str]:
    """The last real paragraph before the sign-off."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", letter) if p.strip()]
    while paragraphs and (SIGNOFF.match(paragraphs[-1]) or len(paragraphs[-1].split()) <= 5):
        paragraphs.pop()
    return paragraphs[-1] if paragraphs else None


def extract_closings(state: State) -> dict:
    """Pulls a model closing paragraph out of each past letter, for finalize() to write the new
    closing from - pure Python, no model call needed to find the end of a letter."""
    letters = state.get("past_letters", [])
    closings = [c for c in (extract_closing(letter) for letter in letters) if c]
    print(f"[extract_closings] {len(closings)} closing paragraph(s) extracted")
    return {"closings": closings}


# ---------------------------------------------------------------------------
# Node: research
# ---------------------------------------------------------------------------

RESEARCH_PROMPT = """Read this job advertisement, identify the hiring company and the team the role sits in (if mentioned in the job advertisement), then research both using the web_search tool.

In light of your research and the information given in the job advertisement, find specific, checkable facts rather than marketing adjectives:
1. What the company does, its scale and market position.
2. How the team's part of the business actually works day to day - its operations, its named
   services, the problems it owns.
3. Concrete recent developments: expansion, technology investment, published engineering work.
   Prefer the last two years.
4. Documented engineering or data practice - tech blogs, conference talks, reports.
5. Stated culture and working practices, in the company's own words where possible.

You may call web_search directly, or from code execution (e.g. to filter/summarize a large result
before it enters your context) - both draw from the same shared budget of {max_uses} calls total,
so don't spend it on exploratory or repeat queries; once it's used up, further calls (either way)
will fail with a search-quota error rather than a result. If a call does fail for another reason,
only that specific call failed; before concluding search is unavailable, check back through every
web_search result you already received in this same response - do not contradict or discard
results you already have when writing your answer.

JOB ADVERTISEMENT:
{ad}

Once you are done researching, write your answer as four sections, in this order:

COMPANY: the hiring company's name, exactly as the job advertisement gives it.

ROLE: the role's title, exactly as the job advertisement gives it.

FINDINGS:
- one checkable fact per line, each followed by its source URL in square brackets, e.g. "runs
  its own delivery fleet and installs appliances in the home [https://source.url]". Use only
  URLs you actually retrieved via web_search in this response - discard anything you cannot
  attribute to one of them, including things you happen to know independently. A fact is never
  an adjective: "runs its own delivery fleet" is a fact; "is an innovative company" is not.

REASONS:
- why a candidate might genuinely want to work at this specific company or in this specific team (his/her
  motivations), considering BOTH the findings above AND what the job advertisement itself says
  about the company and the team. What can make this specific company or team attractive for the candidate? Reasons should be from the candidate's perspective. Rank them, most probable first.
"""


EXTRACT_SUMMARY_SYSTEM = """Extract three things from this research answer about a job
advertisement's hiring company and team:

- `company`: the hiring company's name, exactly as it appears.
- `role`: the role's title, exactly as it appears.
- `reasons`: the candidate motivations/reasons listed, as a list of strings, most probable first,
  exactly as given - do not summarize, reword, merge or invent any."""


def research(state: State) -> dict:
    """Search and ground facts in one call, returning them as a single text field, plus the
    hiring company, role and candidate reasons. Extracted via a small structured-output call
    (below) rather than by regex on the model's own COMPANY:/ROLE:/REASONS: heading formatting -
    a regex silently comes back empty if the model doesn't format a heading exactly as asked
    (different casing, markdown emphasis, no trailing blank line), and since this feeds a
    human-facing picker (select_reasons) it needs to actually be reliable, not just usually
    right. ask()'s built-in retry-on-malformed-response covers the rest."""
    searcher = ChatAnthropic(model=cfg.model, max_tokens=16000).bind_tools(
        [{"type": "web_search_20260209", "name": "web_search", "max_uses": cfg.web_search_max_uses}])
    prompt = RESEARCH_PROMPT.format(ad=state["job_ad"], max_uses=cfg.web_search_max_uses)
    log_prompt("(single combined message - no separate system prompt)", prompt, node="research")

    result = searcher.invoke([HumanMessage(content=prompt)])
    text = result.text if isinstance(result.text, str) else result.text()
    if not text.strip():
        raise RuntimeError("research returned no text - thinking consumed the budget; raise max_tokens.")

    summary = ask(ResearchSummary, EXTRACT_SUMMARY_SYSTEM, f"RESEARCH ANSWER:\n{text}")
    company, role, candidate_reasons = summary.company, summary.role, list(summary.reasons)

    # best-effort cosmetic trim of the COMPANY:/ROLE:/REASONS: headings out of research_notes -
    # the actual data is already captured above via the structured call, so this not matching
    # exactly just leaves those lines sitting in research_notes harmlessly, not a functional bug
    text = re.sub(r"^(COMPANY|ROLE):.*\n?", "", text, flags=re.M | re.I)
    text = re.split(r"^\s*#{0,3}\s*\**REASONS\**:?\s*$", text, maxsplit=1, flags=re.M | re.I)[0]

    sources, seen = [], set()
    for block in result.content:
        if not isinstance(block, dict) or block.get("type") != "web_search_tool_result":
            continue
        content = block.get("content")
        if not isinstance(content, list):          # an error object rather than results
            print(f"[research] search error: {content}")
            continue
        for item in content:
            if item.get("url") and item["url"] not in seen:
                seen.add(item["url"])
                sources.append({"title": item.get("title", ""), "url": item["url"]})

    # drop any line citing a URL that was never actually retrieved - this check is agnostic to
    # which section (FINDINGS vs REASONS) a line belongs to, since it only fires on lines that
    # cite a URL at all; REASONS lines never do, so they pass through untouched
    kept, dropped = [], 0
    for line in text.splitlines():
        cited = re.findall(r"https?://[^\s\]]+", line)
        if cited and not any(url in seen for url in cited):
            dropped += 1
            continue
        kept.append(line)

    notes = html.unescape("\n".join(kept)).strip()

    print(f"[research] {company!r} / {role!r} - {len(sources)} sources, "
          f"{len(candidate_reasons)} candidate reasons, {dropped} ungrounded line(s) dropped")

    return {"company": company, "role": role, "research_notes": notes,
            "candidate_reasons": candidate_reasons, "sources": sources}


# ---------------------------------------------------------------------------
# Node: select_reasons
# ---------------------------------------------------------------------------

def select_reasons(state: State) -> dict:
    """Pauses so a human picks which of research()'s candidate reasons para_intro should build
    the opening paragraph on, and optionally adds their own. Only when
    select_reasons_in_the_loop is set (independent of select_qualifications_in_the_loop - see
    State); otherwise falls back to the top 4 by rank, since research() already ranks them
    most-probable-first.

    Resume with Command(resume=...) where the payload is {"selected": [0, 2], "custom": ["..."]}
    - `selected` are indices into the `reasons` list the interrupt showed, any number; `custom`
    is any number of the human's own reasons, added on top."""
    reasons = state.get("candidate_reasons", [])
    if not state.get("select_reasons_in_the_loop"):
        return {"selected_reasons": reasons[:4]}

    print("[select_reasons] waiting for a selection...")
    decision = interrupt({
        "gate": "select_reasons",
        "reasons": reasons,
    }) or {}

    picked = [reasons[i] for i in decision.get("selected", []) if 0 <= i < len(reasons)]
    custom = [r.strip() for r in decision.get("custom", []) if r and r.strip()]

    print(f"[select_reasons] {len(picked)} selected, {len(custom)} custom")
    return {"selected_reasons": picked + custom}


# ---------------------------------------------------------------------------
# Node: para_intro
# ---------------------------------------------------------------------------

INTRO_SYSTEM = """Write the opening paragraph of my cover letter for the below job advertisement,
grounded in the research notes about the company/team and built on my selected
reasons for wanting this role. Its single job is to answer why I want to work for
this company and this team/role (my motivation).

- Around {words} words. One paragraph.
- Build the paragraph on my selected reasons given below - these were chosen by me, so use all of them if they fit naturally within the word budget,
  otherwise prioritise the ones that fit best together.
- Do not use long sentences.
- Show understanding of what the company/team actually does.
- Begin with a salutation on its own line ("Dear Hiring Team,"), then a blank line, then the paragraph.

""" + "{style}"


def para_intro(state: State) -> dict:
    reasons = "\n".join(f"- {r}" for r in state.get("selected_reasons", [])) or "(none selected)"
    text = prose(
        INTRO_SYSTEM.format(words=cfg.intro_words, style=style()),
        f"JOB ADVERTISEMENT:\n{state['job_ad']}\n\n"
        f"RESEARCH NOTES (background on the company/team, not reasons to state directly):\n"
        f"{state['research_notes']}\n\n"
        f"CANDIDATE'S SELECTED REASONS (build the paragraph on these):\n{reasons}")

    print(f"[para_intro] {len(text.split())} words")
    return {"para_intro": text}


# ---------------------------------------------------------------------------
# Node: assess_qualifications
# ---------------------------------------------------------------------------

ASSESS_QUALIFICATIONS_SYSTEM = """Using my CV and past cover letters, identify the qualifications that matter most for this job advertisement, and explain the specific value each qualification would bring to this team.

For each entry:
- `qualification`: education, experience, skill or knowledge. It needs to be a full sentence.
- `evidence`: the specific thing in the CV or past letters that establishes it. Quote or closely
  paraphrase. If you cannot point to something concrete, the qualification does not belong here.
- `value_to_team`: what this qualification would let THIS team do, or do better. Write it from their side, in
  terms of their problems - not as a restatement of the my experience. This is the most
  important field; a generic benefit that would apply to any team means the entry is weak.
- `relevance`: 1-5 against what the advertisement actually emphasises.

Return eight entries, strongest first."""


def assess_qualifications(state: State) -> dict:
    letters = "\n\n--- letter ---\n\n".join(state.get("past_letters", [])[:3]) or "(none)"
    result = ask(Qualifications, ASSESS_QUALIFICATIONS_SYSTEM,
                 f"JOB ADVERTISEMENT:\n{state['job_ad']}\n\n"
                 f"CV:\n{state['cv']}\n\n"
                 f"PAST COVER LETTERS (for additional detail about the candidate's work):\n{letters}")

    items = sorted((q.model_dump() for q in result.items), key=lambda q: -q["relevance"])
    print(f"[assess_qualifications] {len(items)} qualifications")
    for item in items[:4]:
        print(f"    [{item['relevance']}] {item['qualification']}")
        print(f"        value: {item['value_to_team'][:95]}")
    return {"qualifications": items}


# ---------------------------------------------------------------------------
# Node: select_qualifications
# ---------------------------------------------------------------------------

def select_qualifications(state: State) -> dict:
    """Pauses so a human picks which of assess_qualifications()'s ranked qualifications
    para_body should build the body paragraph on, and optionally adds their own (qualification +
    value_to_team only - no evidence field, since that's meant to point at something in the CV;
    a custom entry gets marked as candidate-supplied instead, and evaluate()'s grounded check is
    the safety net if it turns out unsupported). No selection cap - para_body's own prompt
    already says to build on two or three properly rather than listing everything.

    Only when select_qualifications_in_the_loop is set (independent of
    select_reasons_in_the_loop - see State); otherwise falls back to the top 6 by relevance,
    exactly what para_body used to take directly before this gate existed.

    Resume with Command(resume=...) where the payload is {"selected": [0, 2],
    "custom": [{"qualification": "...", "value_to_team": "..."}]} - `selected` are indices into
    the `qualifications` list the interrupt showed."""
    qualifications = state.get("qualifications", [])
    if not state.get("select_qualifications_in_the_loop"):
        return {"selected_qualifications": qualifications[:6]}

    print("[select_qualifications] waiting for a selection...")
    decision = interrupt({
        "gate": "select_qualifications",
        "qualifications": qualifications,
    }) or {}

    picked = [qualifications[i] for i in decision.get("selected", []) if 0 <= i < len(qualifications)]
    custom = []
    for entry in decision.get("custom", []):
        qualification = (entry.get("qualification") or "").strip()
        value_to_team = (entry.get("value_to_team") or "").strip()
        if qualification and value_to_team:
            custom.append({"qualification": qualification,
                            "evidence": "(candidate-supplied, not drawn from the CV)",
                            "value_to_team": value_to_team, "relevance": 5})

    print(f"[select_qualifications] {len(picked)} selected, {len(custom)} custom")
    return {"selected_qualifications": picked + custom}


# ---------------------------------------------------------------------------
# Node: para_body
# ---------------------------------------------------------------------------

BODY_SYSTEM = """Write the body of my cover letter: why I am a good fit for this role.

- Two or three paragraphs, {words} words in total.
- Use my selected qualifications given below. These were chosen by me, so use all of them if they fit naturally within the word budget, otherwise prioritise the ones that fit best together. Invent no metric,
  date, tool or responsibility. evidence field shows the specific thing in the CV or past cover letters that establishes the qualification.
- Mention something I actually did, then connect it to what the team
  needs - the `value to team` line tells you what that connection is. The point of the paragraph
  is what the team gets, not what I have or what I did.


""" + "{style}"


def para_body(state: State) -> dict:
    chosen = state.get("selected_qualifications", [])
    rendered = "\n\n".join(
        f"[{q['relevance']}/5] {q['qualification']}\n"
        f"    evidence: {q['evidence']}\n"
        f"    value to team: {q['value_to_team']}" for q in chosen)

    text = prose(
        BODY_SYSTEM.format(words=cfg.body_words, style=style()),
        f"JOB ADVERTISEMENT:\n{state['job_ad']}\n\n"
        f"QUALIFICATIONS:\n{rendered}")

    print(f"[para_body] {len(text.split())} words")
    return {"para_body": text}


# ---------------------------------------------------------------------------
# Node: finalize
# ---------------------------------------------------------------------------

REVISE_SYSTEM = """You are given my cover letter's opening and body paragraphs - written
independently of each other - my own past cover letters' closings (for style), my CV for context, and a list of problems found by deterministic checks.

Write the closing paragraph, then produce the final, corrected letter:
- The closing: around {close_words} words, short, forward-looking, no new claims about my experience or qualifications. Match the structure and register of my
  own past closings, given below - their length, their level of formality, how they make the ask
  - without copying their sentences word for word; this is a different application. End with a
  sign-off line matching the one the past letters use ("Kind regards," or similar) on its own
  line, then my name on the line after it. Both are required.
- Keep the total letter, opening and body included, to at most {max_words} words.
- Fix every problem listed.
- Because the opening and body were written without seeing each other, they may repeat a hook,
  fact or phrase, or read disjointedly at the boundary. Find and fix this: cut or merge repeated
  points, smooth transitions, and keep the tone consistent throughout, into the closing you write.
- You may lightly edit wording for flow and coherence. Do not invent new claims, facts, metrics
  or responsibilities that are not already present in the letter.
- The CV is background context only, to help you edit accurately and consistently - you are not
  checking the letter's claims against it or removing anything for lack of CV support.
- Language: {language}.

Report a short list of what you changed and why."""

PLACEHOLDER_RE = re.compile(r"(\[[A-Za-z][^\]]{0,40}\]|\{\{.*?\}\}|\bTODO\b|\bXXXX?\b)")


def save_letter(company: str, role: str, letter: str) -> Optional[Path]:
    """Shared by finalize() and revise() - revise() overwrites the same file finalize() already
    wrote, under the same name, since a revision doesn't change the company/role it's filed
    under. Skipped entirely when cfg.persist_letters is off (see Config)."""
    if not cfg.persist_letters:
        return None
    slug = re.sub(r"[^a-z0-9]+", "-", f"{company}-{role}".lower()).strip("-")[:60]
    path = cfg.outputs_dir / f"{date.today().isoformat()}_{slug}_v2.md"
    path.write_text(letter, encoding="utf-8")
    return path


def finalize(state: State) -> dict:
    intro_and_body = f"{state['para_intro']}\n\n{state['para_body']}".strip()
    intro_and_body = re.sub(r"\n{3,}", "\n\n", intro_and_body)

    def deterministic(text: str) -> list:
        # LENGTH and SIGN-OFF aren't checked here - the closing (which decides both) doesn't
        # exist yet at this point, it's written by the same call these problems feed into. They're
        # checked afterwards instead, against revised.letter - see below.
        problems = []
        for match in set(PLACEHOLDER_RE.findall(text)):
            problems.append(f"PLACEHOLDER: unfilled {match!r}")
        if not re.match(r"^(dear|to whom)", text.strip(), re.I):
            problems.append("SALUTATION: the letter does not open with one")
        return problems

    problems = deterministic(intro_and_body)
    print(f"[finalize] {len(intro_and_body.split())} words (opening+body), {len(problems)} problem(s)")
    for problem in problems:
        print(f"    {problem}")

    examples = "\n\n--- past closing ---\n\n".join(state.get("closings", [])) or "(none supplied)"

    revised = ask(Assembled, REVISE_SYSTEM.format(close_words=cfg.close_words, max_words=cfg.max_words,
                                                   language=cfg.output_language),
                  f"CANDIDATE'S CV (background context only):\n{state['cv']}\n\n"
                  f"THE CANDIDATE'S OWN PAST CLOSINGS, to model the new closing on:\n{examples}\n\n"
                  f"PROBLEMS FOUND:\n" + ("\n".join(f"- {p}" for p in problems) or "(none)") + "\n\n"
                  f"LETTER SO FAR (para_intro and para_body, written independently - write the "
                  f"closing paragraph to follow them):\n"
                  f"---\n{intro_and_body}\n---")

    print(f"[finalize] revised -> {len(revised.letter.split())} words")
    for change in revised.changes:
        print(f"    - {change}")

    words = len(revised.letter.split())
    if words > cfg.max_words:
        print(f"    ! LENGTH: {words} words, limit {cfg.max_words} (post-check, not auto-fixed)")
    if not SIGNOFF.search("\n".join(revised.letter.strip().splitlines()[-3:])):
        print("    ! SIGN-OFF: no sign-off line before the name (post-check, not auto-fixed)")

    path = save_letter(state["company"], state["role"], revised.letter)
    if path:
        print(f"[finalize] -> {path}")

    return {"final_letter": revised.letter, "changes": revised.changes}


# ---------------------------------------------------------------------------
# Nodes: evaluate / revise
# ---------------------------------------------------------------------------

EVALUATE_SYSTEM = """Judge a finished cover letter against the candidate's CV and their past
cover letters (both are background material the letter's claims should be traceable to - a claim
grounded in a past letter is as valid as one grounded in the CV) and against the job
advertisement.

Check:
- `grounded`: every claim about the candidate is supported by the CV or the past letters. List
  anything that isn't in `unsupported_claims`, quoted.
- `specific`: the letter is clearly about this role and company - if you could swap in a
  competitor's name and it would still read sensibly, it fails this check.
- `coherent`: the three paragraphs don't repeat the same hook, fact or phrase - they were written
  independently of each other and may have converged on the same opening move or point.
- `flows_well`: no sudden topic changes - each paragraph, and each sentence within it, should
  follow naturally from what came before. A letter that reads as three unconnected blocks stapled
  together fails this even if each block is individually fine.

List concrete problems in `issues` - specific enough that someone fixing the letter would know
exactly what to change. Score `overall_score` 1-5, where 5 means no changes needed."""


def evaluate(state: State) -> dict:
    letters = "\n\n--- past letter ---\n\n".join(state.get("past_letters", [])) or "(none supplied)"
    result = ask(LetterEvaluation, EVALUATE_SYSTEM,
                 f"JOB ADVERTISEMENT:\n{state['job_ad']}\n\n"
                 f"CV:\n{state['cv']}\n\n"
                 f"CANDIDATE'S PAST COVER LETTERS:\n{letters}\n\n"
                 f"LETTER TO JUDGE:\n---\n{state['final_letter']}\n---")

    print(f"[evaluate] score {result.overall_score}/5 - grounded={result.grounded} "
          f"specific={result.specific} coherent={result.coherent} flows_well={result.flows_well}")
    for issue in result.issues:
        print(f"    - {issue}")

    return {"evaluation_issues": result.issues,
            "evaluation_unsupported_claims": result.unsupported_claims,
            "evaluation_score": result.overall_score,
            "evaluation_grounded": result.grounded,
            "evaluation_specific": result.specific,
            "evaluation_coherent": result.coherent,
            "evaluation_flows_well": result.flows_well}


REVISE_LOOP_SYSTEM = """You are given a cover letter, an evaluator's findings about it, the
candidate's CV and past cover letters for grounding, and the job advertisement.

Fix exactly what the evaluator flagged - the unsupported claims and the listed issues. Do not
invent new claims, facts, metrics or responsibilities. You may lightly edit wording where that's
needed for flow or coherence, but leave everything else as it is.

Report a short list of what you changed and why."""


def revise(state: State) -> dict:
    letters = "\n\n--- past letter ---\n\n".join(state.get("past_letters", [])) or "(none supplied)"
    findings = [f"UNSUPPORTED: {c}" for c in state.get("evaluation_unsupported_claims", [])] \
        + list(state.get("evaluation_issues", []))

    result = ask(RevisedLetter, REVISE_LOOP_SYSTEM,
                 f"JOB ADVERTISEMENT:\n{state['job_ad']}\n\n"
                 f"CV:\n{state['cv']}\n\n"
                 f"CANDIDATE'S PAST COVER LETTERS:\n{letters}\n\n"
                 f"EVALUATOR'S FINDINGS:\n" + ("\n".join(f"- {f}" for f in findings) or "(none)") + "\n\n"
                 f"LETTER:\n---\n{state['final_letter']}\n---")

    print(f"[revise] -> {len(result.letter.split())} words")
    for change in result.changes:
        print(f"    - {change}")

    path = save_letter(state["company"], state["role"], result.letter)
    if path:
        print(f"[revise] -> {path}")

    return {"final_letter": result.letter,
            "changes": state.get("changes", []) + result.changes,
            "revision_count": state.get("revision_count", 0) + 1}


def route_after_evaluate(state: State) -> str:
    """Keeps revising automatically, before a human ever sees the letter, until every check
    passes - overall_score >= eval_score_threshold, coherent/flows_well/grounded/specific all
    true, and no unsupported claims - or cfg.max_auto_revisions passes are spent, whichever comes
    first. Either way, control then goes to human_review rather than straight to END - a human
    gets final say instead of just the threshold.

    Once a human has actually been through human_review, state["human_action"] is set (see its
    docstring - this happens every time it runs, whether they approved, edited or asked for a
    revise) and this always returns "human_review" instead: a human-requested revise gets exactly
    one more evaluate() pass, then goes straight back to them, rather than re-entering the
    automatic loop above and its cap - that cap only ever governs the pre-human_review phase."""
    if state.get("human_action"):
        return "human_review"

    good_enough = (state.get("evaluation_score", 5) >= cfg.eval_score_threshold
                   and state.get("evaluation_coherent", True)
                   and state.get("evaluation_flows_well", True)
                   and state.get("evaluation_grounded", True)
                   and state.get("evaluation_specific", True)
                   and not state.get("evaluation_unsupported_claims"))
    if good_enough or state.get("revision_count", 0) >= cfg.max_auto_revisions:
        return "human_review"
    return "revise"


def human_review(state: State) -> dict:
    """Pauses for a human decision on the finished letter via interrupt() - only when the caller
    set human_in_the_loop (see State). Resume with Command(resume=...) where the payload is
    {"action": "approve"} | {"action": "edit", "letter": "..."} | {"action": "revise", "notes":
    "..." (optional)}. A human-requested revise isn't capped the way the automatic loop is: it
    routes to revise() same as an automatic pass, which loops back to evaluate() and then here
    again - but since this function always sets human_action below, route_after_evaluate sees it
    set and always lands back on human_review rather than re-entering its own automatic loop."""
    if not state.get("human_in_the_loop"):
        return {"human_action": "approve"}

    print("[human_review] waiting for a decision...")
    decision = interrupt({
        "gate": "human_review",
        "letter": state["final_letter"],
        "score": state.get("evaluation_score"),
        "issues": state.get("evaluation_issues", []),
        "unsupported_claims": state.get("evaluation_unsupported_claims", []),
    }) or {}

    action = decision.get("action", "approve")
    print(f"[human_review] -> {action}")

    if action == "edit":
        return {"final_letter": decision["letter"], "human_action": "edit"}
    if action == "revise":
        issues = list(state.get("evaluation_issues", []))
        if decision.get("notes"):
            issues = issues + [f"HUMAN NOTE: {decision['notes']}"]
        return {"evaluation_issues": issues, "human_action": "revise"}
    return {"human_action": "approve"}


def route_after_human_review(state: State) -> str:
    return "revise" if state.get("human_action") == "revise" else END


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def load(state: State) -> dict:
    """Reads inputs/ from disk when the caller didn't already supply job_ad/cv (e.g. an empty
    `{}` input, as Studio uses)."""
    if state.get("job_ad") and state.get("cv"):
        return {}
    return load_inputs()


builder = StateGraph(State)

for name, fn in [("load", load), ("extract_closings", extract_closings), ("research", research),
                 ("select_reasons", select_reasons), ("para_intro", para_intro),
                 ("assess_qualifications", assess_qualifications),
                 ("select_qualifications", select_qualifications),
                 ("para_body", para_body),
                 ("finalize", finalize),
                 ("evaluate", evaluate), ("revise", revise), ("human_review", human_review)]:
    builder.add_node(name, fn)

builder.add_edge(START, "load")

# extract_closings, research and assess_qualifications all only need load's output, not each
# other's - they run concurrently. assess_qualifications reads job_ad/cv/past_letters directly
# (not extract_closings's closings), so it gets its own edge from load rather than sitting
# downstream of extract_closings - that lets its LLM call overlap with research's slow web
# searches from the start, instead of only starting once research's shared tick with
# extract_closings has already cleared
builder.add_edge("load", "extract_closings")
builder.add_edge("load", "research")
builder.add_edge("load", "assess_qualifications")

# para_intro and para_body are written independently of each other - each only depends on its own
# upstream branch, so both run concurrently. The closing is no longer a third parallel branch: it's
# written by finalize() itself (see REVISE_SYSTEM), since finalize already needs to read both
# paragraphs to merge them - writing the closing there for free avoids a dedicated LLM call for it
builder.add_edge("research", "select_reasons")
builder.add_edge("select_reasons", "para_intro")
builder.add_edge("assess_qualifications", "select_qualifications")
builder.add_edge("select_qualifications", "para_body")

# finalize is the join: a *list* of start nodes in one add_edge call is what actually makes
# LangGraph wait for ALL of them - separate single-source add_edge calls use OR semantics under
# the hood (finalize becomes eligible as soon as any one finishes), which races the others.
# extract_closings is included since finalize reads its closings (see REVISE_SYSTEM) - harmless
# for timing, since extract_closings (no LLM call) always finishes long before para_intro/
# para_body do, but without an edge here extract_closings would dangle straight to END with no
# visible link to where its output is used
builder.add_edge(["para_intro", "para_body", "extract_closings"], "finalize")

builder.add_edge("finalize", "evaluate")
builder.add_conditional_edges("evaluate", route_after_evaluate,
                               {"revise": "revise", "human_review": "human_review"})
builder.add_edge("revise", "evaluate")
builder.add_conditional_edges("human_review", route_after_human_review, {"revise": "revise", END: END})

# a checkpointer is required for the interrupt()/Command(resume=...) gates to work at all -
# InMemorySaver is fine here since a thread only needs to survive one process's lifetime (the
# notebook kernel, or one `langgraph dev` server run); nothing here needs it to outlive that
graph = builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Calling the graph
# ---------------------------------------------------------------------------

def _prompt_select_reasons(payload: dict) -> dict:
    """Render select_reasons's interrupt payload and collect a selection from stdin."""
    reasons = payload.get("reasons", [])
    print("\n" + "=" * 70)
    print("SELECT REASONS - pick as many as you like for the opening paragraph to build on")
    for i, r in enumerate(reasons):
        print(f"  [{i}] {r}")
    print("-" * 70)

    raw = input("Numbers, comma-separated (blank for none): ").strip()
    selected = [int(x) for x in raw.split(",") if x.strip().isdigit()] if raw else []

    print("Add your own reasons too, if you like - one per line, blank line to finish:")
    custom = []
    while (line := input()) != "":
        custom.append(line)

    return {"selected": selected, "custom": custom}


def _prompt_select_qualifications(payload: dict) -> dict:
    """Render select_qualifications's interrupt payload and collect a selection from stdin."""
    qualifications = payload.get("qualifications", [])
    print("\n" + "=" * 70)
    print("SELECT QUALIFICATIONS - pick as many as you think are strongest")
    for i, q in enumerate(qualifications):
        print(f"  [{i}] [{q['relevance']}/5] {q['qualification']}")
        print(f"       evidence: {q['evidence']}")
        print(f"       value to team: {q['value_to_team']}")
    print("-" * 70)

    raw = input("Numbers, comma-separated (blank for none): ").strip()
    selected = [int(x) for x in raw.split(",") if x.strip().isdigit()] if raw else []

    custom = []
    print("Add your own qualifications too, if you like - blank qualification to stop.")
    while True:
        qualification = input("Qualification (blank to stop): ").strip()
        if not qualification:
            break
        value_to_team = input("What would this be worth to this team? ").strip()
        custom.append({"qualification": qualification, "value_to_team": value_to_team})

    return {"selected": selected, "custom": custom}


def _prompt_human_review(payload: dict) -> dict:
    """Render human_review's interrupt payload and collect a decision from stdin - the
    notebook's synchronous equivalent of resuming a paused thread in the Studio UI."""
    print("\n" + "=" * 70)
    print(f"HUMAN REVIEW - score {payload.get('score')}/5")
    for issue in payload.get("issues", []):
        print(f"  issue: {issue}")
    for claim in payload.get("unsupported_claims", []):
        print(f"  unsupported: {claim}")
    print("-" * 70)
    print(payload.get("letter", ""))
    print("-" * 70)

    action = input("approve / edit / revise? [approve] ").strip().lower() or "approve"
    if action == "edit":
        print("Paste the replacement letter, then an empty line to finish:")
        lines = []
        while (line := input()) != "":
            lines.append(line)
        return {"action": "edit", "letter": "\n".join(lines)}
    if action == "revise":
        notes = input("Notes for the reviser (optional): ").strip()
        return {"action": "revise", "notes": notes} if notes else {"action": "revise"}
    return {"action": "approve"}


_PROMPT_BY_GATE = {
    "select_reasons": _prompt_select_reasons,
    "select_qualifications": _prompt_select_qualifications,
    "human_review": _prompt_human_review,
}


def _invoke_in_process(payload: dict, config: dict) -> dict:
    """graph.invoke(), resuming past any interrupt(s) with stdin prompts matched to whichever
    gate(s) raised them. More than one can be pending at once now that assess_qualifications runs off load
    directly (see the graph-wiring comment) - select_reasons and select_qualifications can land
    in the same tick, in which case LangGraph requires each answer keyed by its own interrupt id
    rather than one bare value (a single interrupt still accepts a bare value, but keying by id
    works either way, so this always does the latter)."""
    result = graph.invoke(payload, config)
    while "__interrupt__" in result:
        resume_payload = {i.id: _PROMPT_BY_GATE[i.value["gate"]](i.value)
                           for i in result["__interrupt__"]}
        result = graph.invoke(Command(resume=resume_payload), config)
    return result


def run(payload: dict, studio_url: str = "http://127.0.0.1:2024") -> dict:
    """Invoke the graph, routing through a running `langgraph dev` server (so the run shows up
    as a thread in Studio) when one is reachable, falling back to an in-process `graph.invoke`
    otherwise. Every call gets its own thread_id - required by the checkpointer even for runs
    that never hit an interrupt (payload sets none of the three *_in_the_loop flags - see State).

    If a gate does pause and this goes through Studio, the pause is Studio's own built-in
    interrupt UI - resume the thread there, not from this function. The in-process fallback
    instead resumes via stdin prompts (see _invoke_in_process)."""
    config = {"recursion_limit": 30, "configurable": {"thread_id": str(uuid.uuid4())}}

    try:
        from langgraph_sdk import get_sync_client
    except ImportError:
        return _invoke_in_process(payload, config)

    client = get_sync_client(url=studio_url)
    try:
        thread = client.threads.create(graph_id="cover_letter")
    except Exception as exc:
        print(f"[agent] dev server not reachable ({type(exc).__name__}); running in-process")
        return _invoke_in_process(payload, config)

    print(f"[agent] thread {thread['thread_id']} - open Studio to watch it run")
    return client.runs.wait(thread["thread_id"], "cover_letter",
                             input=payload, config={"recursion_limit": 30})


def _shape(thread_id: str, result: dict) -> dict:
    """Always a list under "reviews" - even a single pending gate - so callers have one code path
    regardless of how many interrupts happen to be pending at once. Each entry carries its own
    interrupt_id, which resume() needs back to say which one a decision answers once there's more
    than one (see resume()'s docstring)."""
    interrupts = result.get("__interrupt__")
    if interrupts:
        return {"status": "pending_review", "thread_id": thread_id,
                "reviews": [dict(i.value, interrupt_id=i.id) for i in interrupts]}
    return {"status": "completed", "thread_id": thread_id,
            **{k: result[k] for k in ("final_letter", "changes", "company", "role") if k in result}}


def start(payload: dict) -> dict:
    """Start a fresh thread, running until completion or the first interrupt - each of
    select_reasons, select_qualifications and human_review pauses only when its own
    *_in_the_loop flag is set (see State). For a programmatic caller (api.py) that drives each
    gate itself via resume() below, rather than the stdin-prompt loop run()/_invoke_in_process
    use for interactive/local use. Always a plain in-process invoke - no Studio routing, since a
    deployed server has no Studio to route to."""
    thread_id = str(uuid.uuid4())
    config = {"recursion_limit": 30, "configurable": {"thread_id": thread_id}}
    result = graph.invoke(payload, config)
    return _shape(thread_id, result)


def resume(thread_id: str, decision: dict, interrupt_id: str = None) -> dict:
    """Resume one pending gate with a decision - see select_reasons()/select_qualifications()/
    human_review()'s docstrings for what `decision` should contain for each. Most runs only ever
    have a single interrupt pending, in which case `interrupt_id` can be omitted. If more than
    one gate is pending at once (select_reasons and select_qualifications can land in the same
    tick - see the graph-wiring comment), LangGraph needs to know which one this decision answers;
    pass the `interrupt_id` from the specific entry in _shape()'s "reviews" list being responded
    to. Resuming one still leaves any others pending, to be resumed separately.

    Raises ValueError (api.py maps this to a 404) if the thread doesn't exist, isn't currently
    paused, or interrupt_id is required but missing/doesn't match a pending one."""
    config = {"recursion_limit": 30, "configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state.next:
        raise ValueError(f"thread {thread_id!r} has nothing pending to resume")

    pending = [i for t in state.tasks for i in t.interrupts]
    pending_ids = {i.id for i in pending}
    if len(pending) > 1 and interrupt_id is None:
        gates = [i.value.get("gate") for i in pending]
        raise ValueError(f"thread {thread_id!r} has {len(pending)} pending gates {gates} - "
                          f"resume() needs 'interrupt_id' to say which one this decision answers")
    if interrupt_id is not None and interrupt_id not in pending_ids:
        raise ValueError(f"thread {thread_id!r} has no pending interrupt with id {interrupt_id!r}")

    resume_payload = {interrupt_id: decision} if interrupt_id else decision
    result = graph.invoke(Command(resume=resume_payload), config)
    return _shape(thread_id, result)
