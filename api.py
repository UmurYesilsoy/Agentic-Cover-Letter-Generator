import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from agent import fetch_job_from_url, read_document, run
from agent import resume as agent_resume
from agent import start as agent_start

APP_API_KEY = os.environ["APP_API_KEY"]


def verify_api_key(x_api_key: str = Header()):
    if x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Same 422 shape FastAPI's default handler produces, minus each error's `input` and `ctx`
    fields. `input`: for a model-level check like "provide job_ad or job_ad_url, not both" that
    has no single field to point at, the default handler dumps the *entire* request body
    (including cv text) into that field - noisy and pointlessly echoes back what was just sent.
    `ctx`: embeds the raw ValueError object from the validator, which isn't JSON-serializable
    once we've dropped out of FastAPI's own default handler (which has a custom encoder for
    it) - dropping it here isn't optional, unlike `input`, or this handler itself crashes."""
    errors = [{k: v for k, v in e.items() if k not in ("input", "ctx")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})

# word-count floors, not character counts - matches the convention agent.py's
# fetch_job_from_url already uses (len(text.split()) < 150) to judge a job ad as unusable.
# Supplied fields are validated when present; omitting a field (None) is still valid and means
# "read from inputs/ on disk", handled by the graph's `load` node, not by these checks.
MIN_JOB_AD_WORDS = 150
MIN_CV_WORDS = 100
MIN_PAST_LETTER_WORDS = 50


class GenerateRequest(BaseModel):
    job_ad: Optional[str] = None
    job_ad_url: Optional[str] = None
    cv: Optional[str] = None
    past_letters: Optional[list[str]] = None
    # off by default - when true, the run stops at agent.py's human_review gate instead of
    # returning a finished letter, and the caller finishes it via POST /generate/resume. Existing
    # callers that never set this keep getting today's single-call, always-finished response.
    human_in_the_loop: bool = False

    @field_validator("job_ad")
    @classmethod
    def job_ad_substantial(cls, v):
        if v is not None and len(v.split()) < MIN_JOB_AD_WORDS:
            raise ValueError(f"job_ad looks too short to be a real posting "
                              f"({len(v.split())} words, need {MIN_JOB_AD_WORDS}+)")
        return v

    @field_validator("cv")
    @classmethod
    def cv_substantial(cls, v):
        if v is not None and len(v.split()) < MIN_CV_WORDS:
            raise ValueError(f"cv looks too short to be a real CV "
                              f"({len(v.split())} words, need {MIN_CV_WORDS}+)")
        return v

    @field_validator("past_letters")
    @classmethod
    def past_letters_substantial(cls, v):
        if v is not None:
            for i, letter in enumerate(v):
                if len(letter.split()) < MIN_PAST_LETTER_WORDS:
                    raise ValueError(f"past_letters[{i}] looks too short "
                                      f"({len(letter.split())} words, need {MIN_PAST_LETTER_WORDS}+)")
        return v

    @model_validator(mode="after")
    def not_both_job_ad_forms(self):
        if self.job_ad and self.job_ad_url:
            raise ValueError("provide either job_ad or job_ad_url, not both")
        return self


def resolve_job_ad(job_ad: Optional[str], job_ad_url: Optional[str]) -> Optional[str]:
    """Fetch job_ad_url via agent.fetch_job_from_url when a URL was given instead of text.
    Raises a 422 - not a silent fallback - when the fetch fails, so the caller finds out to
    paste the text instead of quietly getting a letter built from missing content."""
    if not job_ad_url:
        return job_ad
    text = fetch_job_from_url(job_ad_url)
    if text is None:
        raise HTTPException(
            status_code=422,
            detail=f"couldn't extract a usable job posting from {job_ad_url} - the page may be "
                   f"behind a login wall or otherwise unreadable. Try pasting the text directly.")
    return text


def run_pipeline(job_ad: Optional[str], cv: Optional[str], past_letters: Optional[list[str]],
                  human_in_the_loop: bool = False) -> dict:
    payload = {k: v for k, v in
               {"job_ad": job_ad, "cv": cv, "past_letters": past_letters}.items()
               if v is not None}
    if not human_in_the_loop:
        return run(payload)                    # unchanged: always finishes in this one call
    payload["human_in_the_loop"] = True
    return agent_start(payload)                 # may come back "pending_review" instead


def extract_upload_text(upload: UploadFile) -> str:
    """Bridge an in-memory UploadFile to agent.read_document(), which expects a real file path
    (it dispatches on the file extension). Written to a temp file and cleaned up immediately."""
    suffix = Path(upload.filename or "").suffix or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(upload.file.read())
        tmp_path = Path(tmp.name)
    try:
        return read_document(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "cover-letter-agent",
        "version": "1.0",
    }


@app.post("/generate")
def generate(request: GenerateRequest = GenerateRequest(), x_api_key: str = Header(..., alias="X-API-Key")):
    """Run the cover letter pipeline. job_ad can be pasted text or job_ad_url (not both) - a URL
    is fetched via agent.fetch_job_from_url before running. Omit job_ad/cv entirely to read from
    inputs/ on disk instead - the graph's `load` node falls back to that automatically, same as
    invoking it with `{}` in Studio. A plain `def` (not `async def`) so FastAPI runs this
    multi-minute, blocking call in its threadpool instead of on the event loop."""
    verify_api_key(x_api_key)
    job_ad = resolve_job_ad(request.job_ad, request.job_ad_url)
    return run_pipeline(job_ad, request.cv, request.past_letters, request.human_in_the_loop)


@app.post("/generate/upload")
def generate_from_upload(
    cv: UploadFile = File(...),
    job_ad: Optional[str] = Form(None),
    job_ad_url: Optional[str] = Form(None),
    past_letter_1: Optional[UploadFile] = File(None),
    past_letter_2: Optional[UploadFile] = File(None),
    past_letter_3: Optional[UploadFile] = File(None),
    human_in_the_loop: bool = Form(False),
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    """Same pipeline as /generate: cv and past letters are uploaded files (.txt/.md/.pdf/.docx -
    whatever agent.read_document() supports); job_ad is pasted text or job_ad_url (not both) -
    never a file, since job postings are normally pasted or linked rather than saved as a
    document. Past letters are three separate optional slots rather than a true file array:
    Swagger UI doesn't reliably render a "choose file" control for an array-of-files field (it
    fell back to plain text-array inputs), and qualify() only ever reads the first three of
    whatever's supplied anyway, so three named slots lose nothing. Text extracted from each file
    is validated through the same GenerateRequest checks /generate uses, so a garbled/empty
    upload is rejected with a 422 the same way a too-short pasted string is."""
    verify_api_key(x_api_key)

    cv_text = extract_upload_text(cv)
    past_letters = [f for f in (past_letter_1, past_letter_2, past_letter_3) if f is not None]
    letter_texts = [extract_upload_text(f) for f in past_letters] or None

    try:
        request = GenerateRequest(cv=cv_text, job_ad=job_ad, job_ad_url=job_ad_url,
                                   past_letters=letter_texts, human_in_the_loop=human_in_the_loop)
    except ValidationError as exc:
        # include_context=False - the default errors() embeds the raw ValueError object in
        # each entry's ctx, which isn't JSON-serializable when passed to HTTPException.detail
        # manually like this (FastAPI's own automatic validation handler has a custom encoder
        # for that case; we're bypassing it here since this validation happens after upload
        # parsing, not during it). include_input=False - same reason as the RequestValidationError
        # handler above: a model-level check has no single field to blame, so the default would
        # dump the entire extracted cv/job_ad text back into the response.
        raise HTTPException(status_code=422,
                             detail=exc.errors(include_url=False, include_context=False,
                                                include_input=False))

    job_ad_final = resolve_job_ad(request.job_ad, request.job_ad_url)
    return run_pipeline(job_ad_final, request.cv, request.past_letters, request.human_in_the_loop)


class ResumeRequest(BaseModel):
    thread_id: str
    action: str                     # "approve" | "edit" | "revise"
    letter: Optional[str] = None    # required when action == "edit"
    notes: Optional[str] = None     # optional, used when action == "revise"

    @model_validator(mode="after")
    def valid_decision(self):
        if self.action not in {"approve", "edit", "revise"}:
            raise ValueError("action must be 'approve', 'edit' or 'revise'")
        if self.action == "edit" and not (self.letter and self.letter.strip()):
            raise ValueError("action 'edit' requires a non-empty 'letter'")
        return self


@app.post("/generate/resume")
def generate_resume(request: ResumeRequest, x_api_key: str = Header(..., alias="X-API-Key")):
    """Resume a thread that /generate or /generate/upload left "pending_review" (human_in_the_loop
    was true and the run reached agent.py's human_review gate). Same response shape as the start
    call: either "completed", or "pending_review" again if the decision was itself 'revise' -
    revise() re-runs evaluate() afterwards, which always lands back on human_review."""
    verify_api_key(x_api_key)
    decision = {"action": request.action}
    if request.action == "edit":
        decision["letter"] = request.letter
    elif request.action == "revise" and request.notes:
        decision["notes"] = request.notes

    try:
        return agent_resume(request.thread_id, decision)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
