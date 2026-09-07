import json
from orchestrator.llm import FakeLLM
from orchestrator.orchestrator import default_decompose


def test_default_decompose_parses_json_array():
    llm = FakeLLM([json.dumps([{"description": "A"}, {"description": "B"}])])
    subs = default_decompose(llm, "do the thing")
    assert [s.description for s in subs] == ["A", "B"]
    assert [s.index for s in subs] == [0, 1]


def test_default_decompose_tolerates_surrounding_prose():
    llm = FakeLLM(['Here you go:\n[{"description": "only"}]\nthat is all'])
    subs = default_decompose(llm, "x")
    assert len(subs) == 1
    assert subs[0].description == "only"
