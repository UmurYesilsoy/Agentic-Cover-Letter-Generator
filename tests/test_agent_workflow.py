"""End-to-end tests of the real graph, checkpointer, and interrupt()/Command(resume=...) cycle -
the one thing tests/test_agent.py's pure unit tests can't exercise. Only the LLM calls are faked
(by monkeypatching agent.ChatAnthropic), so everything else - graph wiring, tick/interrupt
semantics, multi-interrupt resume-by-id - runs for real. This is deliberately what would have
caught the multi-interrupt and empty-picker bugs found earlier in this project's history."""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-key-for-tests")

import agent


class _FakeMessage:
    """Stands in for whatever ChatAnthropic().invoke(...) returns when there's no
    with_structured_output involved - prose() and research()'s raw search call both just read
    .text; research() also reads .content looking for web_search_tool_result blocks, so an empty
    list there is a legitimate "no sources found" response, not a broken one."""
    def __init__(self, text, content=None):
        self.text = text
        self.content = content if content is not None else []


_FAKE_TEXT = ("This is a fake but well-formed paragraph of plausible length, standing in for "
              "whatever the real model would have written, with no URLs in it to avoid tripping "
              "the ungrounded-line filter in research().")


def _fake_instance_for(schema):
    """One plausible, schema-valid instance per structured-output type ask() ever requests -
    see agent.py's ResearchSummary/Qualifications/Assembled/LetterEvaluation/RevisedLetter."""
    if schema is agent.ResearchSummary:
        return schema(company="Acme Corp", role="Senior Engineer",
                       reasons=["Fake reason one", "Fake reason two", "Fake reason three"])
    if schema is agent.Qualifications:
        return schema(items=[
            agent.Qualification(qualification=f"Fake qualification {i}",
                                 evidence=f"Fake evidence {i}",
                                 value_to_team=f"Fake value {i}", relevance=5)
            for i in range(1, 9)
        ])
    if schema is agent.Assembled:
        return schema(letter="Dear Hiring Team,\n\nFake opening paragraph.\n\n"
                              "Fake body paragraph.\n\nFake closing paragraph.\n\n"
                              "Kind regards,\nTest Candidate",
                       changes=["Merged the independently-written paragraphs"])
    if schema is agent.LetterEvaluation:
        return schema(grounded=True, specific=True, coherent=True, flows_well=True,
                       unsupported_claims=[], issues=[], overall_score=5)
    if schema is agent.RevisedLetter:
        return schema(letter="Dear Hiring Team,\n\nRevised fake letter.\n\nKind regards,\n"
                              "Test Candidate",
                       changes=["Fixed the flagged issue"])
    raise AssertionError(f"test fake asked for an unexpected schema: {schema!r}")


class _FakeStructuredChain:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        return _fake_instance_for(self.schema)


class FakeChatAnthropic:
    """Replaces agent.ChatAnthropic for the duration of a test. Supports every call shape agent.py
    actually uses: plain .invoke() (prose(), research()'s search call), .with_structured_output()
    (ask()), and .bind_tools() (research()'s search call, which chains straight to .invoke())."""
    def __init__(self, *args, **kwargs):
        pass

    def bind_tools(self, *args, **kwargs):
        return self

    def with_structured_output(self, schema):
        return _FakeStructuredChain(schema)

    def invoke(self, messages):
        return _FakeMessage(text=_FAKE_TEXT)


def _payload(**overrides):
    payload = {
        "job_ad": "Fake job ad text, standing in for a real posting.",
        "cv": "Fake CV text, standing in for a real one.",
        "past_letters": [],
        "select_reasons_in_the_loop": True,
        "select_qualifications_in_the_loop": True,
        "human_in_the_loop": False,
    }
    payload.update(overrides)
    return payload


def test_full_workflow_start_partial_resume_then_completed(monkeypatch):
    monkeypatch.setattr(agent, "ChatAnthropic", FakeChatAnthropic)

    result = agent.start(_payload())

    assert result["status"] == "pending_review"
    assert {r["gate"] for r in result["reviews"]} == {"select_reasons", "select_qualifications"}
    thread_id = result["thread_id"]

    reasons_review = next(r for r in result["reviews"] if r["gate"] == "select_reasons")
    quals_review = next(r for r in result["reviews"] if r["gate"] == "select_qualifications")
    assert reasons_review["reasons"]          # research()'s fake reasons made it through
    assert quals_review["qualifications"]     # assess_qualifications()'s fake items made it through

    # answering only one of two pending gates must leave exactly the other one pending
    partial = agent.resume(thread_id, {"selected": [0], "custom": []},
                            interrupt_id=reasons_review["interrupt_id"])
    assert partial["status"] == "pending_review"
    assert len(partial["reviews"]) == 1
    assert partial["reviews"][0]["gate"] == "select_qualifications"
    assert partial["reviews"][0]["interrupt_id"] == quals_review["interrupt_id"]

    # answering the second gate should now run the rest of the pipeline to completion
    final = agent.resume(thread_id, {"selected": [0, 2], "custom": []},
                          interrupt_id=quals_review["interrupt_id"])
    assert final["status"] == "completed"
    assert final["thread_id"] == thread_id
    assert final["company"] == "Acme Corp"
    assert final["role"] == "Senior Engineer"
    assert "final_letter" in final and final["final_letter"]


def test_full_workflow_with_all_gates_off_completes_in_one_call(monkeypatch):
    """No human_in_the_loop flags at all - every gate takes its automatic fallback, so start()
    should run straight through to a completed result with no interrupts at all."""
    monkeypatch.setattr(agent, "ChatAnthropic", FakeChatAnthropic)

    result = agent.start(_payload(select_reasons_in_the_loop=False,
                                   select_qualifications_in_the_loop=False,
                                   human_in_the_loop=False))

    assert result["status"] == "completed"
    assert result["company"] == "Acme Corp"
    assert result["role"] == "Senior Engineer"
    assert "final_letter" in result and result["final_letter"]
