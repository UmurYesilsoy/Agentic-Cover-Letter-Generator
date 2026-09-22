"""Cover letter writing agent - a standalone module extracted from cover_letter_v2.ipynb so it
can be imported from other notebooks/scripts instead of only running inside that notebook.

cover_letter_v2.ipynb remains untouched and keeps its own copy of this same pipeline for
interactive editing; this file does not import from it. Keep the two in sync by hand if you
change prompts/nodes in one and want the change reflected in the other.

Usage from another notebook:

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

@dataclass
class Config:
    model: str = "claude-opus-5"
    output_language: str = "English"

    # per-section word budgets
    intro_words: int = 90
    body_words: int = 190
    close_words: int = 45
    max_words: int = 400        # hard ceiling for the assembled letter

    # employers NOT listed here are replaced by generic descriptors before anything is written
    nameable_employers: tuple = ("ASML", "Philips")
    redact_employers: bool = True

    web_search_max_uses: int = 6
    dump_prompts: bool = True   # write every rendered prompt to outputs/.prompts/

    # evaluate/revise loop: a score at or above this passes; below it triggers exactly one
    # revise pass (bounded by State's revision_count, not by this) before returning regardless
    eval_score_threshold: int = 4

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

class Employer(BaseModel):
    real: str = Field(description="the organisation name exactly as written in the CV")
    generic: str = Field(description="a descriptor precise about industry, region and scale but "
                                     "never identifying, e.g. 'a Dutch telecommunications operator'")


class EmployerList(BaseModel):
    employers: list[Employer]


class Qualification(BaseModel):
    qualification: str = Field(description="education, experience, skill or knowledge")
    evidence: str = Field(description="the specific thing on the CV that establishes it")
    value_to_team: str = Field(description="what it would let this particular team do, or do "
                                           "better - phrased from their side, not the candidate's")
    relevance: int = Field(ge=1, le=5)


class Qualifications(BaseModel):
    items: list[Qualification]


class Assembled(BaseModel):
    company: str = Field(description="the hiring company's name, exactly as the ad gives it")
    role: str = Field(description="the role's title, exactly as the ad gives it")
    letter: str = Field(description="the final cover letter, with every listed problem fixed and "
                                    "duplication/coherence issues across paragraphs resolved")
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

    redactions: dict
    cv_clean: str
    letters_clean: list
    closings: list

    research_notes: str
    sources: list

    qualifications: list
    para_intro: str
    para_body: str
    para_close: str

    final_letter: str
    changes: list

    evaluation_issues: list
    evaluation_unsupported_claims: list
    evaluation_score: int
    revision_count: int

    # set by the caller, not by any node - human_review only pauses on interrupt() when this is
    # true. The notebook's interactive run sets it; api.py's run_pipeline() doesn't, so the
    # deployed API keeps auto-approving here and its single-request-response contract is unchanged
    human_in_the_loop: bool
    human_action: str

    # not otherwise exposed (every other node reads job_ad directly rather than a pre-extracted
    # copy - see research()/para_intro()/etc.) - kept here only so revise() can re-save the
    # letter under the same filename assemble() already picked, without a second extraction call
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
- Every claim carries its evidence in the same sentence. No adjective stands alone as a
  qualification.
- Concrete nouns and verbs. If a sentence would survive swapping in a different company or a
  different candidate, it is filler - cut it.
- Refer to organisations exactly as the source material names them. Some are described
  generically on purpose; never substitute a real name or guess at one.
- Output the paragraph text only. No heading, no preamble, no commentary."""


def style() -> str:
    return STYLE_RULES.format(language=cfg.output_language)


# ---------------------------------------------------------------------------
# Node: prepare
# ---------------------------------------------------------------------------

EMPLOYER_SYSTEM = """List every organisation named in this CV as an employer, client, or project
host - including universities the person studied at.

For each, give the name exactly as written, plus a generic descriptor capturing industry, region
and scale precisely enough to be meaningful in a cover letter but never identifying. Good: "a
Dutch telecommunications operator". Bad: "a company", "a well-known tech firm"."""

SIGNOFF = re.compile(r"^(kind regards|yours sincerely|yours faithfully|best regards|sincerely|"
                     r"regards|many thanks|thank you,)", re.I | re.M)


def extract_closing(letter: str) -> Optional[str]:
    """The last real paragraph before the sign-off."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", letter) if p.strip()]
    while paragraphs and (SIGNOFF.match(paragraphs[-1]) or len(paragraphs[-1].split()) <= 5):
        paragraphs.pop()
    return paragraphs[-1] if paragraphs else None


def prepare(state: State) -> dict:
    letters = state.get("past_letters", [])
    closings = [c for c in (extract_closing(letter) for letter in letters) if c]

    if not cfg.redact_employers:
        print(f"[prepare] redaction off; {len(closings)} closing paragraph(s) extracted")
        return {"cv_clean": state["cv"], "letters_clean": letters,
                "closings": closings, "redactions": {}}

    found = ask(EmployerList, EMPLOYER_SYSTEM, f"CV:\n\n{state['cv']}")
    allow = {name.strip().lower() for name in cfg.nameable_employers}
    mapping = {e.real: e.generic for e in found.employers if e.real.strip().lower() not in allow}

    def scrub(text: str) -> str:
        # longest first, so "Bridgestone Mobility Solutions" goes before "Bridgestone"
        for real in sorted(mapping, key=len, reverse=True):
            text = re.sub(rf"\b{re.escape(real)}\b", mapping[real], text, flags=re.IGNORECASE)
        return text

    print(f"[prepare] {len(found.employers)} organisations, {len(mapping)} redacted, "
          f"{len(closings)} closing paragraph(s) extracted")
    return {"cv_clean": scrub(state["cv"]),
            "letters_clean": [scrub(letter) for letter in letters],
            "closings": [scrub(c) for c in closings],
            "redactions": mapping}


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

Call the web_search tool directly for each query. Do not invoke it indirectly through code
execution or any scripting tool - that path is not supported and will error. If a direct
web_search call itself errors, only that specific call failed; before concluding search is
unavailable, check back through every web_search result you already received in this same
response - do not contradict or discard results you already have when writing your answer.

JOB ADVERTISEMENT:
{ad}

Once you are done researching, write your answer as two sections, in this order:

FINDINGS:
- one checkable fact per line, each followed by its source URL in square brackets, e.g. "runs
  its own delivery fleet and installs appliances in the home [https://source.url]". Use only
  URLs you actually retrieved via web_search in this response - discard anything you cannot
  attribute to one of them, including things you happen to know independently. A fact is never
  an adjective: "runs its own delivery fleet" is a fact; "is an innovative company" is not.

REASONS:
- why a candidate might genuinely want this specific company or this specific team (their
  motivations), considering BOTH the findings above AND what the job advertisement itself says
  about the company and the team. Rank them, most probable first.
"""


def research(state: State) -> dict:
    """Search and ground facts in one call, returning them as a single text field. Company, role
    and team are not extracted here - the job ad already names them, and every downstream node
    reads the job ad directly instead of a pre-extracted copy of the same information."""
    searcher = ChatAnthropic(model=cfg.model, max_tokens=16000).bind_tools(
        [{"type": "web_search_20260209", "name": "web_search", "max_uses": cfg.web_search_max_uses}])
    prompt = RESEARCH_PROMPT.format(ad=state["job_ad"])
    log_prompt("(single combined message - no separate system prompt)", prompt, node="research")

    result = searcher.invoke([HumanMessage(content=prompt)])
    text = result.text if isinstance(result.text, str) else result.text()
    if not text.strip():
        raise RuntimeError("research returned no text - thinking consumed the budget; raise max_tokens.")

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

    print(f"[research] {len(sources)} sources, {dropped} ungrounded line(s) dropped")

    return {"research_notes": notes, "sources": sources}


# ---------------------------------------------------------------------------
# Node: para_intro
# ---------------------------------------------------------------------------

INTRO_SYSTEM = """Write the opening paragraph of a cover letter for the below job advertisement in light of the research findings about the company/team/role and possible candidate reasons (motivations). Its single job is to answer why
this candidate wants to work for this company and this team/role (his/her motivation).

- Around {words} words. One paragraph.
- Build on two strongest of the possible candidate reasons given below. One reason must be
  about the company and the other must be about the team or the role.
- Do not use long sentences.
- Show understanding of what the company/team actually does.
- Begin with a salutation on its own line ("Dear Hiring Team," unless the advertisement names
  someone), then a blank line, then the paragraph.

""" + "{style}"


def para_intro(state: State) -> dict:
    text = prose(
        INTRO_SYSTEM.format(words=cfg.intro_words, style=style()),
        f"JOB ADVERTISEMENT:\n{state['job_ad']}\n\n"
        f"RESEARCH NOTES AND CANDIDATE MOTIVATIONS (the only facts and reasons you may state):\n"
        f"{state['research_notes']}")

    print(f"[para_intro] {len(text.split())} words")
    return {"para_intro": text}


# ---------------------------------------------------------------------------
# Node: qualify
# ---------------------------------------------------------------------------

QUALIFY_SYSTEM = """Identify the candidate's qualifications that matter most for this role, and
what each would be worth to this specific team.

For each entry:
- `qualification`: education, experience, skill or knowledge. It needs to be a full sentence.
- `evidence`: the specific thing in the CV or past letters that establishes it. Quote or closely
  paraphrase. If you cannot point to something concrete, the qualification does not belong here.
- `value_to_team`: what this would let THIS team do, or do better. Write it from their side, in
  terms of their problems - not as a restatement of the candidate's experience. This is the most
  important field; a generic benefit that would apply to any team means the entry is weak.
- `relevance`: 1-5 against what the advertisement actually emphasises.

Return six to ten entries, strongest first. Include only what you can evidence - do not pad the
list with things the candidate might plausibly know. A short honest list produces a better letter
than a long hopeful one.

Refer to organisations exactly as the source material names them; some are deliberately
generic."""


def qualify(state: State) -> dict:
    letters = "\n\n--- letter ---\n\n".join(state.get("letters_clean", [])[:3]) or "(none)"
    result = ask(Qualifications, QUALIFY_SYSTEM,
                 f"JOB ADVERTISEMENT:\n{state['job_ad']}\n\n"
                 f"CV:\n{state['cv_clean']}\n\n"
                 f"PAST COVER LETTERS (for additional detail about the candidate's work):\n{letters}")

    items = sorted((q.model_dump() for q in result.items), key=lambda q: -q["relevance"])
    print(f"[qualify] {len(items)} qualifications")
    for item in items[:4]:
        print(f"    [{item['relevance']}] {item['qualification']}")
        print(f"        value: {item['value_to_team'][:95]}")
    return {"qualifications": items}


# ---------------------------------------------------------------------------
# Node: para_body
# ---------------------------------------------------------------------------

BODY_SYSTEM = """Write the body of a cover letter: why this candidate is a good fit for this role.

- Two or three paragraphs, {words} words in total.
- Use the qualifications supplied, and the evidence given with them. Invent no metric,
  date, tool or responsibility. evidence field shows the specific thing in the CV or past cover letters that establishes the qualification.
- Mention something the candidate actually did, then connect it to what the team
  needs - the `value to team` line tells you what that connection is. The point of the paragraph
  is what the team gets, not what the candidate has.
- Build on two or three qualifications properly rather than listing all of them.


""" + "{style}"


def para_body(state: State) -> dict:
    chosen = state["qualifications"][:6]
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
# Node: para_close
# ---------------------------------------------------------------------------

CLOSE_SYSTEM = """Write the closing paragraph of a cover letter.

- Around {words} words. Short.
- Match the structure and register of the candidate's own past closings, which are given to you.
  Follow their shape - their length, their level of formality, how they make the ask. Do not copy
  their sentences word for word; this is a different application.
- Forward-looking.
- No new claims about the candidate's experience.
- End with a sign-off line matching the one the past letters use ("Kind regards," or similar) on
  its own line, then the candidate's name on the line after it. Both are required.

""" + "{style}"


def para_close(state: State) -> dict:
    examples = "\n\n--- past closing ---\n\n".join(state.get("closings", [])) or "(none supplied)"
    text = prose(
        CLOSE_SYSTEM.format(words=cfg.close_words, style=style()),
        f"CANDIDATE: {state['cv_clean'].strip().splitlines()[0]}\n\n"
        f"JOB ADVERTISEMENT:\n{state['job_ad']}\n\n"
        f"THE CANDIDATE'S OWN PAST CLOSINGS, to model:\n{examples}")

    print(f"[para_close] {len(text.split())} words")
    return {"para_close": text}


# ---------------------------------------------------------------------------
# Node: assemble
# ---------------------------------------------------------------------------

REVISE_SYSTEM = """You are given a cover letter assembled from three paragraphs - an opening, a
body and a closing - written independently of each other, plus the candidate's CV for context and
a list of problems found by deterministic checks.

Produce a final, corrected letter:
- Fix every problem listed.
- Because the three paragraphs were written without seeing each other, they may repeat a hook,
  fact or phrase, or read disjointedly at the paragraph boundaries. Find and fix this: cut or
  merge repeated points, smooth transitions, keep the tone consistent throughout.
- You may lightly edit wording for flow and coherence across paragraph boundaries. Do not invent
  new claims, facts, metrics or responsibilities that are not already present in the letter.
- The CV is background context only, to help you edit accurately and consistently - you are not
  checking the letter's claims against it or removing anything for lack of CV support.

Also extract the hiring company's name and the role's title from the job advertisement, and
report a short list of what you changed and why."""

PLACEHOLDER_RE = re.compile(r"(\[[A-Za-z][^\]]{0,40}\]|\{\{.*?\}\}|\bTODO\b|\bXXXX?\b)")


def save_letter(company: str, role: str, letter: str) -> Path:
    """Shared by assemble() and revise() - revise() overwrites the same file assemble() already
    wrote, under the same name, since a revision doesn't change the company/role it's filed
    under."""
    slug = re.sub(r"[^a-z0-9]+", "-", f"{company}-{role}".lower()).strip("-")[:60]
    path = cfg.outputs_dir / f"{date.today().isoformat()}_{slug}_v2.md"
    path.write_text(letter, encoding="utf-8")
    return path


def assemble(state: State) -> dict:
    letter = f"{state['para_intro']}\n\n{state['para_body']}\n\n{state['para_close']}".strip()
    letter = re.sub(r"\n{3,}", "\n\n", letter)

    def deterministic(text: str) -> list:
        problems = []
        words = len(text.split())
        if words > cfg.max_words:
            problems.append(f"LENGTH: {words} words, limit {cfg.max_words} - cut {words - cfg.max_words}")
        for real, generic in state.get("redactions", {}).items():
            if re.search(rf"\b{re.escape(real)}\b", text, flags=re.IGNORECASE):
                problems.append(f"REDACTION: '{real}' must not be named - use '{generic}'")
        for match in set(PLACEHOLDER_RE.findall(text)):
            problems.append(f"PLACEHOLDER: unfilled {match!r}")
        if not re.match(r"^(dear|to whom)", text.strip(), re.I):
            problems.append("SALUTATION: the letter does not open with one")
        if not SIGNOFF.search("\n".join(text.strip().splitlines()[-3:])):
            problems.append("SIGN-OFF: no sign-off line before the name")
        return problems

    problems = deterministic(letter)
    print(f"[assemble] {len(letter.split())} words, {len(problems)} problem(s)")
    for problem in problems:
        print(f"    {problem}")

    revised = ask(Assembled, REVISE_SYSTEM,
                  f"JOB ADVERTISEMENT:\n{state['job_ad']}\n\n"
                  f"CANDIDATE'S CV (background context only):\n{state['cv_clean']}\n\n"
                  f"PROBLEMS FOUND:\n" + ("\n".join(f"- {p}" for p in problems) or "(none)") + "\n\n"
                  f"LETTER (para_intro, para_body and para_close, written independently):\n"
                  f"---\n{letter}\n---")

    print(f"[assemble] revised -> {len(revised.letter.split())} words")
    for change in revised.changes:
        print(f"    - {change}")

    path = save_letter(revised.company, revised.role, revised.letter)
    print(f"[assemble] -> {path}")

    return {"final_letter": revised.letter, "changes": revised.changes,
            "company": revised.company, "role": revised.role}


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
    letters = "\n\n--- past letter ---\n\n".join(state.get("letters_clean", [])) or "(none supplied)"
    result = ask(LetterEvaluation, EVALUATE_SYSTEM,
                 f"JOB ADVERTISEMENT:\n{state['job_ad']}\n\n"
                 f"CV:\n{state['cv_clean']}\n\n"
                 f"CANDIDATE'S PAST COVER LETTERS:\n{letters}\n\n"
                 f"LETTER TO JUDGE:\n---\n{state['final_letter']}\n---")

    print(f"[evaluate] score {result.overall_score}/5 - grounded={result.grounded} "
          f"specific={result.specific} coherent={result.coherent} flows_well={result.flows_well}")
    for issue in result.issues:
        print(f"    - {issue}")

    return {"evaluation_issues": result.issues,
            "evaluation_unsupported_claims": result.unsupported_claims,
            "evaluation_score": result.overall_score}


REVISE_LOOP_SYSTEM = """You are given a cover letter, an evaluator's findings about it, the
candidate's CV and past cover letters for grounding, and the job advertisement.

Fix exactly what the evaluator flagged - the unsupported claims and the listed issues. Do not
invent new claims, facts, metrics or responsibilities. You may lightly edit wording where that's
needed for flow or coherence, but leave everything else as it is.

Report a short list of what you changed and why."""


def revise(state: State) -> dict:
    letters = "\n\n--- past letter ---\n\n".join(state.get("letters_clean", [])) or "(none supplied)"
    findings = [f"UNSUPPORTED: {c}" for c in state.get("evaluation_unsupported_claims", [])] \
        + list(state.get("evaluation_issues", []))

    result = ask(RevisedLetter, REVISE_LOOP_SYSTEM,
                 f"JOB ADVERTISEMENT:\n{state['job_ad']}\n\n"
                 f"CV:\n{state['cv_clean']}\n\n"
                 f"CANDIDATE'S PAST COVER LETTERS:\n{letters}\n\n"
                 f"EVALUATOR'S FINDINGS:\n" + ("\n".join(f"- {f}" for f in findings) or "(none)") + "\n\n"
                 f"LETTER:\n---\n{state['final_letter']}\n---")

    print(f"[revise] -> {len(result.letter.split())} words")
    for change in result.changes:
        print(f"    - {change}")

    path = save_letter(state["company"], state["role"], result.letter)
    print(f"[revise] -> {path}")

    return {"final_letter": result.letter,
            "changes": state.get("changes", []) + result.changes,
            "revision_count": state.get("revision_count", 0) + 1}


def route_after_evaluate(state: State) -> str:
    """One AUTOMATIC revise pass at most - the same 'one corrective pass, not a loop' rule
    assemble() already follows, so this can't become the two-gates-arguing failure mode the V1
    experiment hit. Once the score passes or that one pass is spent, control goes to
    human_review rather than straight to END - a human gets final say instead of just the
    threshold, and can still request further revisions themselves from there."""
    if state.get("evaluation_score", 5) >= cfg.eval_score_threshold or state.get("revision_count", 0) >= 1:
        return "human_review"
    return "revise"


def human_review(state: State) -> dict:
    """Pauses for a human decision on the finished letter via interrupt() - only when the caller
    set human_in_the_loop (see State). Resume with Command(resume=...) where the payload is
    {"action": "approve"} | {"action": "edit", "letter": "..."} | {"action": "revise", "notes":
    "..." (optional)}. A human-requested revise isn't bounded the way the automatic one is: it
    routes to revise() same as the automatic pass, which loops back to evaluate() and then here
    again - by then revision_count is >= 1, so route_after_evaluate always lands back on
    human_review rather than auto-revising a second time."""
    if not state.get("human_in_the_loop"):
        return {"human_action": "approve"}

    print("[human_review] waiting for a decision...")
    decision = interrupt({
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

for name, fn in [("load", load), ("prepare", prepare), ("research", research),
                 ("para_intro", para_intro), ("qualify", qualify), ("para_body", para_body),
                 ("para_close", para_close), ("assemble", assemble),
                 ("evaluate", evaluate), ("revise", revise), ("human_review", human_review)]:
    builder.add_node(name, fn)

builder.add_edge(START, "load")

# prepare and research both only need load's output, not each other's - they run concurrently
builder.add_edge("load", "prepare")
builder.add_edge("load", "research")

# para_intro, para_body and para_close are written independently of each other and of one
# another's output - each only depends on its own upstream branch, so all three run concurrently
builder.add_edge("research", "para_intro")
builder.add_edge("prepare", "qualify")
builder.add_edge("qualify", "para_body")
builder.add_edge("prepare", "para_close")

# assemble is the join: a *list* of start nodes in one add_edge call is what actually makes
# LangGraph wait for ALL of them - three separate single-source add_edge calls use OR semantics
# under the hood (assemble becomes eligible as soon as any one finishes), which races para_body
builder.add_edge(["para_intro", "para_body", "para_close"], "assemble")

builder.add_edge("assemble", "evaluate")
builder.add_conditional_edges("evaluate", route_after_evaluate,
                               {"revise": "revise", "human_review": "human_review"})
builder.add_edge("revise", "evaluate")
builder.add_conditional_edges("human_review", route_after_human_review, {"revise": "revise", END: END})

# a checkpointer is required for human_review's interrupt()/Command(resume=...) to work at all -
# InMemorySaver is fine here since a thread only needs to survive one process's lifetime (the
# notebook kernel, or one `langgraph dev` server run); nothing here needs it to outlive that
graph = builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Calling the graph
# ---------------------------------------------------------------------------

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


def _invoke_in_process(payload: dict, config: dict) -> dict:
    """graph.invoke(), resuming past any human_review interrupt with a stdin prompt. Only
    payloads with human_in_the_loop=True ever actually pause - a plain run just runs straight
    through, same as before this was added."""
    result = graph.invoke(payload, config)
    while "__interrupt__" in result:
        decision = _prompt_human_review(result["__interrupt__"][0].value)
        result = graph.invoke(Command(resume=decision), config)
    return result


def run(payload: dict, studio_url: str = "http://127.0.0.1:2024") -> dict:
    """Invoke the graph, routing through a running `langgraph dev` server (so the run shows up
    as a thread in Studio) when one is reachable, falling back to an in-process `graph.invoke`
    otherwise. Every call gets its own thread_id - required by the checkpointer even for runs
    that never hit human_review's interrupt().

    If payload sets human_in_the_loop=True and this goes through Studio, the pause is Studio's
    own built-in interrupt UI - resume the thread there, not from this function. The in-process
    fallback instead resumes via stdin prompts (see _invoke_in_process)."""
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
    if "__interrupt__" in result:
        return {"status": "pending_review", "thread_id": thread_id,
                "review": result["__interrupt__"][0].value}
    return {"status": "completed", "thread_id": thread_id,
            **{k: result[k] for k in ("final_letter", "changes", "company", "role") if k in result}}


def start(payload: dict) -> dict:
    """Start a fresh thread, running until completion or the first human_review interrupt (only
    reachable when payload sets human_in_the_loop=True - see State). For a programmatic caller
    (api.py) that drives the review itself via resume() below, rather than the stdin-prompt loop
    run()/_invoke_in_process use for interactive/local use. Always a plain in-process invoke - no
    Studio routing, since a deployed server has no Studio to route to."""
    thread_id = str(uuid.uuid4())
    config = {"recursion_limit": 30, "configurable": {"thread_id": thread_id}}
    result = graph.invoke(payload, config)
    return _shape(thread_id, result)


def resume(thread_id: str, decision: dict) -> dict:
    """Resume a thread paused at human_review with a decision - see human_review's docstring for
    the decision shape. Raises ValueError (api.py maps this to a 404) if the thread doesn't
    exist or isn't currently paused - both look identical to LangGraph (get_state().next is
    empty either way), which is what's checked here."""
    config = {"recursion_limit": 30, "configurable": {"thread_id": thread_id}}
    if not graph.get_state(config).next:
        raise ValueError(f"thread {thread_id!r} has nothing pending to resume")
    result = graph.invoke(Command(resume=decision), config)
    return _shape(thread_id, result)
