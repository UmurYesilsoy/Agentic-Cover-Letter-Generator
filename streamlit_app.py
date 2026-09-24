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


def call_resume(thread_id: str, **decision) -> dict:
    """POST to /generate/resume - continues a thread call_generate left "pending_review". The
    decision's shape depends on which gate is being resumed - action/letter/notes for
    human_review, selected/custom for select_reasons (see api.py's ResumeRequest)."""
    body = {"thread_id": thread_id, **decision}
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


def resume_and_store(thread_id: str, **decision) -> None:
    """Send a review decision, then stash whatever comes back for the next rerun - another
    pending_review (the next gate, or a further revise) or a final_result (approve/edit)."""
    with st.spinner("Working..."):
        try:
            result = call_resume(thread_id, **decision)
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            st.error(f"Couldn't send your decision: {request_error(exc)}")
            return

    st.session_state.pop("pending_review", None)
    if result.get("status") == "pending_review":
        st.session_state["pending_review"] = {"thread_id": result["thread_id"], **result["review"]}
    else:
        st.session_state["final_result"] = result
    st.rerun()


def render_select_reasons() -> None:
    """The select_reasons gate, surfaced client-side: pick up to 3 of research()'s candidate
    reasons for the opening paragraph to build on, and optionally add your own."""
    review = st.session_state["pending_review"]
    thread_id = review["thread_id"]
    reasons = review.get("reasons", [])

    st.title("Why do you want this role?")
    st.caption("Pick up to 3 reasons for the opening paragraph to build on, or write your own.")

    selected_labels = st.multiselect("Candidate reasons (from research)", options=reasons,
                                      max_selections=3, key="reason_choices")
    custom_text = st.text_area("Add your own reasons (optional, one per line)",
                                key="custom_reasons")

    if st.button("Continue", type="primary"):
        selected = [reasons.index(label) for label in selected_labels]
        custom = [line.strip() for line in custom_text.splitlines() if line.strip()]
        if not selected and not custom:
            st.error("Pick at least one reason, or write your own.")
        else:
            resume_and_store(thread_id, selected=selected, custom=custom)


def render_select_qualifications() -> None:
    """The select_qualifications gate, surfaced client-side: pick freely from qualify()'s ranked
    list for the body paragraph to build on (no cap - para_body's own prompt already says to
    build on two or three properly rather than listing everything), and optionally add one of
    your own (qualification + what it's worth to the team - no evidence field, since that's
    meant to point at something in the CV; evaluate()'s grounded check is the safety net if a
    custom entry turns out unsupported)."""
    review = st.session_state["pending_review"]
    thread_id = review["thread_id"]
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
                                    key="qual_choices")

    with st.expander("Add a qualification of your own (optional)"):
        custom_qual = st.text_input("Qualification", key="custom_qual_title")
        custom_value = st.text_area("What would this be worth to this team?",
                                     key="custom_qual_value")

    if st.button("Continue", type="primary"):
        selected = [i for i, label in enumerate(labels) if label in chosen_labels]
        custom_qualifications = []
        if custom_qual.strip() and custom_value.strip():
            custom_qualifications = [{"qualification": custom_qual.strip(),
                                       "value_to_team": custom_value.strip()}]
        if not selected and not custom_qualifications:
            st.error("Pick at least one qualification, or add your own.")
        else:
            resume_and_store(thread_id, selected=selected,
                              custom_qualifications=custom_qualifications)


def render_human_review() -> None:
    """The human_review gate, surfaced client-side: the letter plus the evaluator's score and
    issues, with three ways to respond - mirrors the notebook's own stdin-driven version of this
    same interrupt() payload (see cover_letter_v2.ipynb, "Run it")."""
    review = st.session_state["pending_review"]
    thread_id = review["thread_id"]

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
        if st.button("Approve", type="primary"):
            resume_and_store(thread_id, action="approve")

    with col2:
        with st.popover("Ask for another revision"):
            notes = st.text_area("Notes for the reviser (optional)", key="revise_notes")
            if st.button("Send for revision"):
                resume_and_store(thread_id, action="revise", notes=notes or None)

    with col3:
        with st.popover("Edit directly"):
            edited = st.text_area("Replacement letter", value=review.get("letter", ""),
                                   height=300, key="edit_letter")
            if st.button("Save edited letter"):
                if not edited.strip():
                    st.error("The letter can't be empty.")
                else:
                    resume_and_store(thread_id, action="edit", letter=edited)


def render_review() -> None:
    """Dispatches to whichever gate's UI is currently pending."""
    gate = st.session_state["pending_review"].get("gate")
    if gate == "select_reasons":
        render_select_reasons()
    elif gate == "select_qualifications":
        render_select_qualifications()
    else:
        render_human_review()


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
            st.session_state["pending_review"] = {"thread_id": result["thread_id"], **result["review"]}
        else:
            st.session_state["final_result"] = result
        st.rerun()


def main():
    if not check_password():
        return

    # a paused review or a finished result takes over the whole page until it's resolved, so the
    # generation form and the review/result views never end up rendered at the same time
    if st.session_state.get("pending_review"):
        render_review()
    elif st.session_state.get("final_result"):
        render_result(st.session_state["final_result"])
    else:
        render_generate_form()


main()
