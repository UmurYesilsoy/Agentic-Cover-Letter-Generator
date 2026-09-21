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


def call_generate(cv_file, job_ad_text, job_ad_url, letter_files) -> dict:
    """POST to /generate/upload - see api.py for the actual contract this mirrors."""
    files = {"cv": (cv_file.name, cv_file.getvalue())}
    for i, letter in enumerate(letter_files[:MAX_PAST_LETTERS], start=1):
        files[f"past_letter_{i}"] = (letter.name, letter.getvalue())

    data = {}
    if job_ad_text:
        data["job_ad"] = job_ad_text
    if job_ad_url:
        data["job_ad_url"] = job_ad_url

    headers = {"X-API-Key": st.secrets["APP_API_KEY"]}
    response = httpx.post(f"{API_BASE_URL}/generate/upload", files=files, data=data,
                           headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def main():
    if not check_password():
        return

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
                result = call_generate(cv_file, job_ad_text, job_ad_url, letter_files)
            except httpx.HTTPStatusError as exc:
                try:
                    detail = exc.response.json().get("detail", exc.response.text)
                except Exception:
                    detail = exc.response.text
                st.error(f"Couldn't generate a letter: {detail}")
                return
            except httpx.RequestError as exc:
                st.error(f"Couldn't reach the server: {exc}")
                return

        st.success("Done!")
        st.markdown(result["final_letter"])

        if result.get("changes"):
            with st.expander("What was changed while assembling the letter"):
                for change in result["changes"]:
                    st.write(f"- {change}")


main()
