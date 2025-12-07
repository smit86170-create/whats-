"""Utility script to exercise get_schedule on sample prompts.

The script imports `get_schedule` from
`test_new/prompt_parser_patched_superhybrid_2.8_1.2(вернул обычные and).py`
and prints schedule segments plus selected steps for several prompts.
"""
from __future__ import annotations

from importlib import util
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

MODULE_PATH = (
    Path(__file__).resolve().parent
    / "test_new"
    / "prompt_parser_patched_superhybrid_2.8_1.2(вернул обычные and).py"
)


def _load_get_schedule():
    spec = util.spec_from_file_location("prompt_parser_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {MODULE_PATH}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "get_schedule")


def text_at_step(step: int, schedule: Sequence[Sequence[int | str]]) -> str:
    for end_step, text in schedule:
        if step <= int(end_step):
            return str(text)
    return ""


def run_case(name: str, prompt: str, steps: int, checkpoints: Iterable[int]) -> None:
    get_schedule = _load_get_schedule()
    schedule: List[Tuple[int, str]] = get_schedule(
        prompt, steps, use_scheduling=True, seed=42, use_visitor=True
    )

    print(f"=== {name} ===")
    print(f"Prompt: {prompt!r}, steps={steps}")
    print("Schedule:", schedule)

    for step in checkpoints:
        if 1 <= step <= steps:
            print(f"Step {step}: {text_at_step(step, schedule)!r}")
        else:
            print(f"Step {step}: skipped (outside 1..{steps})")
    print()


def main() -> None:
    checkpoints = (1, 6, 11, 16)
    cases = [
        (
            "Multiple delayed starts / Scenario A",
            "[fire:5] and [water:10] and [earth:15]",
            20,
        ),
        (
            "Range gaps",
            "base, [obj:1.0 5-8, 12-15]",
            20,
        ),
        (
            "Multiple ranges Scenario B",
            "[star]:1.0 1-2, 5-6 and [moon]:1.0 8-10",
            20,
        ),
        (
            "Combo-wombo",
            "[A:B:5] mixed with [C:D:10] reverse",
            15,
        ),
        (
            "Multiple brackets fix",
            "[cat:dog:5] [day:night:10]",
            15,
        ),
    ]

    for name, prompt, steps in cases:
        run_case(name, prompt, steps, checkpoints)


if __name__ == "__main__":
    main()
