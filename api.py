from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from agent import run

import os

APP_API_KEY = os.environ["APP_API_KEY"]

def verify_api_key(x_api_key: str = Header()):
    if x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

app = FastAPI()

class GenerateRequest(BaseModel):
    job_ad: Optional[str] = None
    cv: Optional[str] = None
    past_letters: Optional[list[str]] = None


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/generate")
def generate(request: GenerateRequest = GenerateRequest(), x_api_key: str = Header(..., alias="X-API-Key"),):
    """Run the cover letter pipeline. Omit job_ad/cv to read from inputs/ on disk instead - the
    graph's `load` node falls back to that automatically, same as invoking it with `{}` in
    Studio. A plain `def` (not `async def`) so FastAPI runs this multi-minute, blocking call in
    its threadpool instead of on the event loop."""

    verify_api_key(x_api_key)
    
    payload = {k: v for k, v in request.model_dump().items() if v is not None}
    return run(payload)
