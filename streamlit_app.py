"""Simple frontend for the deployed cover-letter API (api.py, running on Render).

Password-gated via STREAMLIT_PASSWORD - deliberately a *separate* secret from APP_API_KEY.
APP_API_KEY stays a server-side secret this app injects itself (in call_generate below) and
never shows to whoever's using the app; STREAMLIT_PASSWORD is the one you hand out to friends
and recruiters. Reusing APP_API_KEY as the password would mean anyone you give it to could call
the real API directly, bypassing this app entirely.

Local setup: create .streamlit/secrets.toml (gitignored - never commit this) with:
    APP_API_KEY = "..."         # same value as the Render deployment's APP_API_KEY
    STREAMLIT_PASSWORD = "..."  # a password you choose and share with friends/recruiters

When deployed to Streamlit Community Cloud, set the same two keys in that app's own Secrets
panel instead - the code doesn't change either way.

No usage limiting here yet (per-person or global) - deliberately out of scope for now.
"""
import httpx
import streamlit as st

API_BASE_URL = st.secrets.get("API_BASE_URL", "https://cover-letter-agent-fjps.onrender.com")
MAX_PAST_LETTERS = 3
# generous: the real pipeline run takes several minutes on its own, and Render's free tier can
# take up to a minute more to wake from an idle "cold start" before the request even starts
REQUEST_TIMEOUT = 600

st.set_page_config(page_title="Cover Letter Agent", page_icon="✉️")


def check_password() -> bool:
    """Shared-password gate, kept separate from APP_API_KEY (see module docstring). Renders the
    password prompt and returns False until the correct password has been entered."""
    if st.session_state.get("authenticated"):
        return True

    st.title("Cover Letter Agent")
    password = st.text_input("Password", type="password")
    if st.button("Unlock"):
        if password == st.secrets.get("STREAMLIT_PASSWORD"):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def call_generate(cv_file, job_ad_text, job_ad_url, letter_files, select_reasons_in_the_loop,
                   select_qualifications_in_the_loop, human_in_the_loop) -> dict:
    """POST to /generate/upload - see api.py for the actual contract this mirrors. Comes back
    {"status": "pending_review", ...} if any of the three flags pauses its own gate (reasons,
    qualifications, or the final letter), otherwise a finished
    {"status": "completed", "final_letter": ..., ...} in this one call."""
    files = {"cv": (cv_file.name, cv_file.getvalue())}
    for i, letter in enumerate(letter_files[:MAX_PAST_LETTERS], start=1):
        files[f"past_letter_{i}"] = (letter.name, letter.getvalue())

    data = {"select_reasons_in_the_loop": str(select_reasons_in_the_loop).lower(),
            "select_qualifications_in_the_loop": str(select_qualifications_in_the_loop).lower(),
            "human_in_the_loop": str(human_in_the_loop).lower()}
    if job_ad_text:
        data["job_ad"] = job_ad_text
    if job_ad_url:
        data["job_ad_url"] = job_ad_url

    headers = {"X-API-Key": st.secrets["APP_API_KEY"]}
    response = httpx.post(f"{API_BASE_URL}/generate/upload", files=files, data=data,
                           headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def call_resume(thread_id: str, interrupt_id: str = None, **decision) -> dict:
    """POST to /generate/resume - continues one pending gate of a thread call_generate left
    "pending_review". The decision's shape depends on which gate is being resumed -
    action/letter/notes for human_review, selected/custom(_qualifications) for the other two (see
    api.py's ResumeRequest). `interrupt_id` is only required when more than one gate was pending
    at once - see render_review()."""
    body = {"thread_id": thread_id, **decision}
    if interrupt_id:
        body["interrupt_id"] = interrupt_id
    headers = {"X-API-Key": st.secrets["APP_API_KEY"]}
    response = httpx.post(f"{API_BASE_URL}/generate/resume", json=body,
                           headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def render_result(result: dict) -> None:
    st.success("Done!")
    st.markdown(result["final_letter"])
    if result.get("changes"):
        with st.expander("What was changed while assembling the letter"):
            for change in result["changes"]:
                st.write(f"- {change}")
    if st.button("Start a new letter"):
        st.session_state.pop("final_result", None)
        st.rerun()


def request_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            return exc.response.json().get("detail", exc.response.text)
        except Exception:
            return exc.response.text
    return str(exc)


def resume_and_store(thread_id: str, interrupt_id: str = None, **decision) -> None:
    """Send a review decision for one gate, then replace pending_reviews with whatever the server
    says is still pending - could be empty (falls through to final_result), the same list minus
    the one just answered, or with a newly-reached gate added (e.g. human_review, once
    select_reasons/select_qualifications are both done)."""
    with st.spinner("Working..."):
        try:
            result = call_resume(thread_id, interrupt_id, **decision)
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            st.error(f"Couldn't send your decision: {request_error(exc)}")
            return

    st.session_state.pop("pending_reviews", None)
    if result.get("status") == "pending_review":
        st.session_state["pending_reviews"] = [
            {"thread_id": result["thread_id"], **r} for r in result["reviews"]]
    else:
        st.session_state["final_result"] = result
    st.rerun()


def render_select_reasons(review: dict) -> None:
    """The select_reasons gate, surfaced client-side: pick up to 3 of research()'s candidate
    reasons for the opening paragraph to build on, and optionally add your own."""
    thread_id, interrupt_id = review["thread_id"], review["interrupt_id"]
    reasons = review.get("reasons", [])

    st.title("Why do you want this role?")
    st.caption("Pick up to 3 reasons for the opening paragraph to build on, or write your own.")

    selected_labels = st.multiselect("Candidate reasons (from research)", options=reasons,
                                      max_selections=3, key=f"reason_choices_{interrupt_id}")
    custom_text = st.text_area("Add your own reasons (optional, one per line)",
                                key=f"custom_reasons_{interrupt_id}")

    if st.button("Continue", type="primary", key=f"reasons_continue_{interrupt_id}"):
        selected = [reasons.index(label) for label in selected_labels]
        custom = [line.strip() for line in custom_text.splitlines() if line.strip()]
        if not selected and not custom:
            st.error("Pick at least one reason, or write your own.")
        else:
            resume_and_store(thread_id, interrupt_id, selected=selected, custom=custom)


def render_select_qualifications(review: dict) -> None:
    """The select_qualifications gate, surfaced client-side: pick freely from qualify()'s ranked
    list for the body paragraph to build on (no cap - para_body's own prompt already says to
    build on two or three properly rather than listing everything), and optionally add one of
    your own (qualification + what it's worth to the team - no evidence field, since that's
    meant to point at something in the CV; evaluate()'s grounded check is the safety net if a
    custom entry turns out unsupported)."""
    thread_id, interrupt_id = review["thread_id"], review["interrupt_id"]
    quals = review.get("qualifications", [])

    st.title("Which qualifications should the letter lead with?")
    st.caption("Pick as many as you think are strongest - the letter will build on 2-3 of them.")

    with st.expander("See evidence and value-to-team for each"):
        for q in quals:
            st.markdown(f"**[{q['relevance']}/5] {q['qualification']}**")
            st.write(f"Evidence: {q['evidence']}")
            st.write(f"Value to team: {q['value_to_team']}")
            st.divider()

    labels = [f"[{q['relevance']}/5] {q['qualification']}" for q in quals]
    chosen_labels = st.multiselect("Qualifications (from your CV)", options=labels,
                                    key=f"qual_choices_{interrupt_id}")

    with st.expander("Add a qualification of your own (optional)"):
        custom_qual = st.text_input("Qualification", key=f"custom_qual_title_{interrupt_id}")
        custom_value = st.text_area("What would this be worth to this team?",
                                     key=f"custom_qual_value_{interrupt_id}")

    if st.button("Continue", type="primary", key=f"quals_continue_{interrupt_id}"):
        selected = [i for i, label in enumerate(labels) if label in chosen_labels]
        custom_qualifications = []
        if custom_qual.strip() and custom_value.strip():
            custom_qualifications = [{"qualification": custom_qual.strip(),
                                       "value_to_team": custom_value.strip()}]
        if not selected and not custom_qualifications:
            st.error("Pick at least one qualification, or add your own.")
        else:
            resume_and_store(thread_id, interrupt_id, selected=selected,
                              custom_qualifications=custom_qualifications)


def render_human_review(review: dict) -> None:
    """The human_review gate, surfaced client-side: the letter plus the evaluator's score and
    issues, with three ways to respond - mirrors the notebook's own stdin-driven version of this
    same interrupt() payload (see cover_letter_v2.ipynb, "Run it")."""
    thread_id, interrupt_id = review["thread_id"], review["interrupt_id"]

    st.title("Review your letter")
    st.caption(f"Evaluator score: {review.get('score')}/5")
    if review.get("issues"):
        with st.expander("Issues the evaluator flagged", expanded=True):
            for issue in review["issues"]:
                st.write(f"- {issue}")
    if review.get("unsupported_claims"):
        with st.expander("Claims that aren't grounded in your CV or past letters"):
            for claim in review["unsupported_claims"]:
                st.write(f"- {claim}")

    st.markdown(review.get("letter", ""))

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Approve", type="primary", key=f"approve_{interrupt_id}"):
            resume_and_store(thread_id, interrupt_id, action="approve")

    with col2:
        with st.popover("Ask for another revision"):
            notes = st.text_area("Notes for the reviser (optional)",
                                  key=f"revise_notes_{interrupt_id}")
            if st.button("Send for revision", key=f"revise_{interrupt_id}"):
                resume_and_store(thread_id, interrupt_id, action="revise", notes=notes or None)

    with col3:
        with st.popover("Edit directly"):
            edited = st.text_area("Replacement letter", value=review.get("letter", ""),
                                   height=300, key=f"edit_letter_{interrupt_id}")
            if st.button("Save edited letter", key=f"save_edit_{interrupt_id}"):
                if not edited.strip():
                    st.error("The letter can't be empty.")
                else:
                    resume_and_store(thread_id, interrupt_id, action="edit", letter=edited)


RENDER_BY_GATE = {
    "select_reasons": render_select_reasons,
    "select_qualifications": render_select_qualifications,
    "human_review": render_human_review,
}


def render_review() -> None:
    """Renders every currently pending gate - normally just one, but select_reasons and
    select_qualifications can both be ready at once (see agent.py's graph-wiring comment), in
    which case both show up on the page together, each independently answerable."""
    reviews = st.session_state["pending_reviews"]
    for i, review in enumerate(reviews):
        RENDER_BY_GATE[review["gate"]](review)
        if i < len(reviews) - 1:
            st.divider()


def render_generate_form() -> None:
    st.title("Cover Letter Agent")
    st.caption("Upload your CV, tell us about the role, and get a tailored cover letter.")

    cv_file = st.file_uploader("Your CV", type=["txt", "md", "pdf", "docx"])

    job_ad_mode = st.radio("Job advertisement", ["Paste the text", "Provide a link"],
                            horizontal=True)
    job_ad_text, job_ad_url = None, None
    if job_ad_mode == "Paste the text":
        job_ad_text = st.text_area("Job ad text", height=200)
    else:
        job_ad_url = st.text_input("Job ad URL")

    letter_files = st.file_uploader(
        f"Past cover letters (optional, up to {MAX_PAST_LETTERS})",
        type=["txt", "md", "pdf", "docx"], accept_multiple_files=True)
    if len(letter_files) > MAX_PAST_LETTERS:
        st.warning(f"Only the first {MAX_PAST_LETTERS} will be used.")

    select_reasons_in_the_loop = st.checkbox(
        "Choose your own reasons",
        help="Pause after research so you can pick which candidate reasons open the letter, "
             "instead of the top-ranked ones being used automatically.")
    select_qualifications_in_the_loop = st.checkbox(
        "Choose your own qualifications",
        help="Pause after research so you can pick which qualifications the body paragraph "
             "builds on, instead of the top-ranked ones being used automatically.")
    human_in_the_loop = st.checkbox(
        "Review the final letter before it's done",
        help="Pause once the letter is finished so you can approve it, edit it directly, or ask "
             "for another revision pass - otherwise it finalizes automatically.")

    if st.button("Generate my cover letter", type="primary"):
        if cv_file is None:
            st.error("Please upload your CV.")
            return
        if not job_ad_text and not job_ad_url:
            st.error("Please paste the job ad text or provide a link.")
            return

        with st.spinner("Generating your cover letter - this takes a few minutes "
                         "(the server may also need a minute to wake up first)..."):
            try:
                result = call_generate(cv_file, job_ad_text, job_ad_url, letter_files,
                                        select_reasons_in_the_loop,
                                        select_qualifications_in_the_loop, human_in_the_loop)
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                st.error(f"Couldn't generate a letter: {request_error(exc)}")
                return

        if result.get("status") == "pending_review":
            st.session_state["pending_reviews"] = [
                {"thread_id": result["thread_id"], **r} for r in result["reviews"]]
        else:
            st.session_state["final_result"] = result
        st.rerun()


def main():
    if not check_password():
        return

    # pending review(s) or a finished result take over the whole page until resolved, so the
    # generation form and the review/result views never end up rendered at the same time
    if st.session_state.get("pending_reviews"):
        render_review()
    elif st.session_state.get("final_result"):
        render_result(st.session_state["final_result"])
    else:
        render_generate_form()


main()
