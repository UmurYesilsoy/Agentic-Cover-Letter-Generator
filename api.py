from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from agent import run

app = FastAPI()


class GenerateRequest(BaseModel):
    job_ad: Optional[str] = None
    cv: Optional[str] = None
    past_letters: Optional[list[str]] = None


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/generate")
def generate(request: GenerateRequest = GenerateRequest()):
    """Run the cover letter pipeline. Omit job_ad/cv to read from inputs/ on disk instead - the
    graph's `load` node falls back to that automatically, same as invoking it with `{}` in
    Studio. A plain `def` (not `async def`) so FastAPI runs this multi-minute, blocking call in
    its threadpool instead of on the event loop."""
    payload = {k: v for k, v in request.model_dump().items() if v is not None}
    return run(payload)
