"""Unit tests for agent.py's own logic - pure functions taking plain dicts, no FastAPI, no real
LangGraph invocation, no LLM calls. These pin down the routing/shaping logic that's been hand-
verified with throwaway scripts throughout development; a real end-to-end test (mocking the LLM
calls so the actual graph/checkpointer/interrupt cycle still runs) belongs in a separate file."""
import os
import types

os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-key-for-tests")

import pytest
from langgraph.graph import END

import agent


# ---------------------------------------------------------------------------
# route_after_evaluate
# ---------------------------------------------------------------------------

def _eval_state(**overrides) -> dict:
    """A state that would pass every check outright, unless overridden."""
    state = {
        "evaluation_score": agent.cfg.eval_score_threshold,
        "evaluation_coherent": True,
        "evaluation_flows_well": True,
        "evaluation_grounded": True,
        "evaluation_specific": True,
        "evaluation_unsupported_claims": [],
        "revision_count": 0,
    }
    state.update(overrides)
    return state


def test_route_after_evaluate_good_enough_goes_to_human_review():
    assert agent.route_after_evaluate(_eval_state()) == "human_review"


def test_route_after_evaluate_low_score_revises():
    state = _eval_state(evaluation_score=agent.cfg.eval_score_threshold - 1)
    assert agent.route_after_evaluate(state) == "revise"


@pytest.mark.parametrize("field", ["evaluation_coherent", "evaluation_flows_well",
                                    "evaluation_grounded", "evaluation_specific"])
def test_route_after_evaluate_any_single_failing_criterion_revises(field):
    state = _eval_state(**{field: False})
    assert agent.route_after_evaluate(state) == "revise"


def test_route_after_evaluate_unsupported_claims_revises_despite_high_score():
    state = _eval_state(evaluation_unsupported_claims=["some claim"])
    assert agent.route_after_evaluate(state) == "revise"


def test_route_after_evaluate_stops_once_cap_is_reached():
    state = _eval_state(evaluation_score=1, revision_count=agent.cfg.max_auto_revisions)
    assert agent.route_after_evaluate(state) == "human_review"


def test_route_after_evaluate_keeps_revising_below_the_cap():
    state = _eval_state(evaluation_score=1, revision_count=agent.cfg.max_auto_revisions - 1)
    assert agent.route_after_evaluate(state) == "revise"


def test_route_after_evaluate_never_reenters_automatic_loop_after_human_review():
    """Once human_review has run once (human_action is always set by then, see its docstring),
    a human-requested revise must get exactly one more evaluate() pass then go straight back -
    not fall into the automatic cap-bounded loop, even if the automatic cap was never reached."""
    state = _eval_state(evaluation_score=1, revision_count=0, human_action="revise")
    assert agent.route_after_evaluate(state) == "human_review"


# ---------------------------------------------------------------------------
# route_after_human_review
# ---------------------------------------------------------------------------

def test_route_after_human_review_revise():
    assert agent.route_after_human_review({"human_action": "revise"}) == "revise"


@pytest.mark.parametrize("action", ["approve", "edit", None])
def test_route_after_human_review_otherwise_ends(action):
    assert agent.route_after_human_review({"human_action": action}) == END


# ---------------------------------------------------------------------------
# select_reasons / select_qualifications - automatic fallback (loop flag off)
# ---------------------------------------------------------------------------

def test_select_reasons_fallback_takes_top_four():
    reasons = [f"reason {i}" for i in range(6)]
    result = agent.select_reasons({"candidate_reasons": reasons})
    assert result == {"selected_reasons": reasons[:4]}


def test_select_reasons_fallback_with_fewer_than_four():
    reasons = ["only one"]
    result = agent.select_reasons({"candidate_reasons": reasons})
    assert result == {"selected_reasons": reasons}


def test_select_qualifications_fallback_takes_top_six():
    quals = [{"qualification": f"q{i}", "relevance": 5} for i in range(8)]
    result = agent.select_qualifications({"qualifications": quals})
    assert result == {"selected_qualifications": quals[:6]}


# ---------------------------------------------------------------------------
# _shape
# ---------------------------------------------------------------------------

def _fake_interrupt(value: dict, id_: str):
    return types.SimpleNamespace(value=value, id=id_)


def test_shape_completed_result():
    result = {"final_letter": "Dear...", "changes": ["x"], "company": "Acme", "role": "Engineer",
              "revision_count": 1}
    shaped = agent._shape("thread-1", result)
    assert shaped == {"status": "completed", "thread_id": "thread-1",
                       "final_letter": "Dear...", "changes": ["x"],
                       "company": "Acme", "role": "Engineer"}


def test_shape_pending_review_single_interrupt():
    result = {"__interrupt__": [_fake_interrupt({"gate": "select_reasons", "reasons": []}, "id-1")]}
    shaped = agent._shape("thread-2", result)
    assert shaped == {"status": "pending_review", "thread_id": "thread-2",
                       "reviews": [{"gate": "select_reasons", "reasons": [], "interrupt_id": "id-1"}]}


def test_shape_pending_review_multiple_interrupts():
    result = {"__interrupt__": [
        _fake_interrupt({"gate": "select_reasons", "reasons": []}, "id-1"),
        _fake_interrupt({"gate": "select_qualifications", "qualifications": []}, "id-2"),
    ]}
    shaped = agent._shape("thread-3", result)
    assert shaped["status"] == "pending_review"
    assert len(shaped["reviews"]) == 2
    assert {r["interrupt_id"] for r in shaped["reviews"]} == {"id-1", "id-2"}


# ---------------------------------------------------------------------------
# resume - the one error path that's cheap to hit without a real pending interrupt
# ---------------------------------------------------------------------------

def test_resume_raises_for_a_thread_with_nothing_pending():
    with pytest.raises(ValueError, match="nothing pending"):
        agent.resume("thread-that-never-started", {})


# ---------------------------------------------------------------------------
# extract_closing
# ---------------------------------------------------------------------------

def test_extract_closing_strips_signoff_and_name():
    letter = (
        "Dear Hiring Team,\n\n"
        "This is the opening paragraph.\n\n"
        "This is the real closing paragraph, forward-looking and short.\n\n"
        "Kind regards,\n"
        "Jamie Doe"
    )
    assert agent.extract_closing(letter) == \
        "This is the real closing paragraph, forward-looking and short."


def test_extract_closing_returns_none_for_no_real_paragraph():
    assert agent.extract_closing("Kind regards,\nJamie Doe") is None


# ---------------------------------------------------------------------------
# PLACEHOLDER_RE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Dear [Hiring Manager],",
    "We use {{company_name}} internally.",
    "TODO fill this in",
    "reference XXXX in your reply",
])
def test_placeholder_re_matches_known_placeholder_styles(text):
    assert agent.PLACEHOLDER_RE.findall(text)


@pytest.mark.parametrize("text", [
    "Dear Hiring Team,",
    "I led a team of five engineers.",
    "The project shipped in Q4.",
])
def test_placeholder_re_does_not_false_positive_on_normal_prose(text):
    assert not agent.PLACEHOLDER_RE.findall(text)
