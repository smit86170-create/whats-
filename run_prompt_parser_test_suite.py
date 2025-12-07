"""Ad-hoc test suite for prompt parser.
Covers scheduling, alternation, grouping, formatting, weighting, and parsing paths.
"""
from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

MODULE_PATH = Path('test_new/prompt_parser_patched_superhybrid_2.8_1.2(вернул обычные and).py')

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str


def load_module(module_name: str, *, additive_delta: bool = False):
    env_key = "ATTENTION_DELTA_ADDITIVE"
    if additive_delta:
        os.environ[env_key] = "1"
    else:
        os.environ.pop(env_key, None)
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def get_text_at_step(schedule: List[List], step: int) -> str:
    for end_step, text in schedule:
        if step <= end_step:
            return text
    return ""


def run_test(name: str, func: Callable[[], None], results: List[TestResult]):
    try:
        func()
    except AssertionError as exc:
        results.append(TestResult(name, False, f"Assertion failed: {exc}"))
    except Exception as exc:  # pragma: no cover - defensive
        results.append(TestResult(name, False, f"Error: {exc}"))
    else:
        results.append(TestResult(name, True, "OK"))


def main():
    mod = load_module("prompt_parser_under_test", additive_delta=False)
    mod_delta = load_module("prompt_parser_under_test_delta", additive_delta=True)

    results: List[TestResult] = []

    # Scheduling tests
    def test_standard_schedule():
        schedule = mod.get_schedule("[from:to:5]", steps=10, use_scheduling=True, seed=123)
        assert get_text_at_step(schedule, 1) == "from"
        assert get_text_at_step(schedule, 6) == "to"
    run_test("Scheduling: standard [from:to:step]", test_standard_schedule, results)

    def test_fast_path_schedule():
        schedule = mod.get_schedule("[foo]:3", steps=10, use_scheduling=True, seed=123)
        assert get_text_at_step(schedule, 2) == ""
        assert get_text_at_step(schedule, 4) == "foo"
    run_test("Scheduling: fast-path [text]:N", test_fast_path_schedule, results)

    def test_range_schedule():
        schedule = mod.get_schedule("[foo]:3 1-4, 6-8", steps=10, use_scheduling=True, seed=123)
        assert get_text_at_step(schedule, 2) == "foo"
        assert get_text_at_step(schedule, 5) == ""
        assert get_text_at_step(schedule, 7) == "foo"
    run_test("Scheduling: explicit ranges", test_range_schedule, results)

    def test_percent_schedule():
        schedule = mod.get_schedule("[foo]:3 10%-50%", steps=10, use_scheduling=True, seed=123)
        assert get_text_at_step(schedule, 1) == "foo"
        assert get_text_at_step(schedule, 6) == ""
    run_test("Scheduling: percent ranges", test_percent_schedule, results)

    def test_reverse_schedule():
        schedule = mod.get_schedule("[x:y:z]:5 reverse", steps=10, use_scheduling=True, seed=123)
        assert get_text_at_step(schedule, 1) == "z"
        assert get_text_at_step(schedule, 3) == "y"
        assert get_text_at_step(schedule, 6) == "x"
    run_test("Scheduling: reverse flag", test_reverse_schedule, results)

    # Alternation tests
    def test_alternation_cycle():
        schedule = mod.get_schedule("[a|b|c]", steps=10, use_scheduling=True, seed=123)
        assert get_text_at_step(schedule, 1) == "a"
        assert get_text_at_step(schedule, 2) == "b"
        assert get_text_at_step(schedule, 3) == "c"
    run_test("Alternation: cyclic [a|b|c]", test_alternation_cycle, results)

    def test_alternation_distinct():
        schedule = mod.get_schedule("[a|b|c]!", steps=10, use_scheduling=True, seed=123)
        assert len(schedule) == 1
        assert get_text_at_step(schedule, 10) in {"a", "b", "c"}
    run_test("Alternation: distinct [a|b|c]!, seeded", test_alternation_distinct, results)

    def test_alternation_simple():
        schedule = mod.get_schedule("a|b", steps=4, use_scheduling=True, seed=123)
        assert get_text_at_step(schedule, 1) == "a"
        assert get_text_at_step(schedule, 2) == "b"
    run_test("Alternation: simple a|b", test_alternation_simple, results)

    def test_empty_variants():
        schedule1 = mod.get_schedule("[a|]", steps=4, use_scheduling=True, seed=123)
        assert get_text_at_step(schedule1, 2) == ""
        schedule2 = mod.get_schedule("[|b]", steps=4, use_scheduling=True, seed=123)
        assert get_text_at_step(schedule2, 1) == ""
    run_test("Alternation: empty variants", test_empty_variants, results)

    def test_numbered_selection():
        schedule = mod.get_schedule("2[a|b|c]", steps=1, use_scheduling=True, seed=123)
        text = get_text_at_step(schedule, 1)
        parts = [p.strip() for p in text.split(',') if p.strip()]
        assert len(parts) == 2
    run_test("Alternation: numbered selection", test_numbered_selection, results)

    def test_numbered_distinct():
        schedule = mod.get_schedule("2![a|b|c]", steps=1, use_scheduling=True, seed=123)
        text = get_text_at_step(schedule, 1)
        parts = [p.strip() for p in text.split(',') if p.strip()]
        assert len(parts) == 2
        assert len(parts) == len(set(parts))
    run_test("Alternation: numbered distinct", test_numbered_distinct, results)

    # Grouping
    def test_curly_grouping():
        schedule = mod.get_schedule("{a,b,c}", steps=1, use_scheduling=True, seed=123)
        text = get_text_at_step(schedule, 1)
        assert "a" in text and "b" in text and "c" in text
    run_test("Grouping: curly braces", test_curly_grouping, results)

    # Sequencing & formatting
    def test_owner_tag():
        schedule = mod.get_schedule("owner::tag", steps=1, use_scheduling=True, seed=123)
        assert get_text_at_step(schedule, 1) == "owner: tag"
    run_test("Sequencing: owner::tag", test_owner_tag, results)

    def test_owner_nested_tags():
        schedule = mod.get_schedule("owner::tag::tag2!!", steps=1, use_scheduling=True, seed=123)
        text = get_text_at_step(schedule, 1)
        assert "owner: tag" in text and "tag2" in text
    run_test("Sequencing: owner::tag::tag2!!", test_owner_nested_tags, results)

    def test_top_level_sequence3():
        schedule = mod.get_schedule("owner:::tag1!!!", steps=1, use_scheduling=True, seed=123)
        text = get_text_at_step(schedule, 1)
        assert "owner" in text and "tag1" in text
    run_test("Sequencing: top-level sequence3", test_top_level_sequence3, results)

    def test_compound_and_escape():
        schedule = mod.get_schedule("word_word", steps=1, use_scheduling=True, seed=123)
        assert get_text_at_step(schedule, 1) == "word_word"
        schedule2 = mod.get_schedule(r"\[escaped\]", steps=1, use_scheduling=True, seed=123)
        assert "[escaped]" in get_text_at_step(schedule2, 1)
    run_test("Sequencing: compounds and escaping", test_compound_and_escape, results)

    # Weighting & attention
    def test_round_bracket_weight():
        tokens = mod.parse_prompt_attention("(word)")
        assert math.isclose(tokens[0][1], 1.1, rel_tol=1e-6)
    run_test("Attention: (word)", test_round_bracket_weight, results)

    def test_explicit_weight():
        tokens = mod.parse_prompt_attention("(word:1.5)")
        assert math.isclose(tokens[0][1], 1.5, rel_tol=1e-6)
    run_test("Attention: (word:1.5)", test_explicit_weight, results)

    def test_inline_weight():
        tokens = mod.parse_prompt_attention("word:1.5")
        assert math.isclose(tokens[0][1], 1.5, rel_tol=1e-6)
    run_test("Attention: inline word:1.5", test_inline_weight, results)

    def test_square_weight():
        tokens = mod.parse_prompt_attention("[word]")
        assert math.isclose(tokens[0][1], 1/1.1, rel_tol=1e-6)
    run_test("Attention: [word]", test_square_weight, results)

    def test_nested_weight():
        tokens = mod.parse_prompt_attention("((word))")
        assert math.isclose(tokens[0][1], 1.21, rel_tol=1e-6)
    run_test("Attention: nested ((word))", test_nested_weight, results)

    def test_delta_weights():
        tokens = mod_delta.parse_prompt_attention("word +0.5")
        assert math.isclose(tokens[0][1], 1.5, rel_tol=1e-6)
        tokens = mod_delta.parse_prompt_attention("word -0.5")
        assert math.isclose(tokens[0][1], 0.5, rel_tol=1e-6)
    run_test("Attention: delta weights", test_delta_weights, results)

    # Composition AND handling
    def test_and_operator():
        tokens = mod.parse_prompt_attention("a AND b")
        assert any("&" in t for t, _ in tokens), "AND should normalize to & in tokens"
    run_test("Composition: AND operator", test_and_operator, results)

    # Fast-path vs Lark complexity check
    def test_complexity_detection():
        assert mod._needs_complex_parse("simple prompt", "simple prompt") is False
        complex_prompt = "start (a [b|c])"
        assert mod._needs_complex_parse(complex_prompt, complex_prompt) is True
    run_test("Parser routing: fast-path vs Lark", test_complexity_detection, results)

    def test_complex_prompt_parsed():
        schedule = mod.get_schedule("pre [a:(b|c):5] post", steps=5, use_scheduling=True, seed=123, use_visitor=True)
        assert schedule and isinstance(schedule, list)
    run_test("Parser routing: complex prompt via Lark", test_complex_prompt_parsed, results)

    # Report
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}: {r.message}")
    print(f"\nTotal: {passed} passed, {failed} failed, {len(results)} total")


if __name__ == "__main__":
    main()
