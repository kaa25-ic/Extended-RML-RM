"""Runtime-compatible monitor-state encoding utilities."""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Iterable

import numpy as np


def normalize_monitor_state(state: str) -> str:
    """Remove runtime-generated variable suffixes from a monitor-state string."""
    return re.sub(r"_[0-9]+", "", state)


def _strip_leading_eps_prefix(state: str) -> str:
    if state.startswith("(eps*"):
        return state[len("(eps*") :]
    return state


def split_top_level_factors(state: str) -> list[str]:
    """Split a monitor-state expression at top-level ``*`` factors only."""
    normalized = normalize_monitor_state(state).replace("@", "")
    normalized = _strip_leading_eps_prefix(normalized)

    factors: list[str] = []
    current: list[str] = []
    paren_depth = 0
    bracket_depth = 0

    for character in normalized:
        if character == "(":
            paren_depth += 1
        elif character == ")":
            paren_depth = max(paren_depth - 1, 0)
        elif character == "[":
            bracket_depth += 1
        elif character == "]":
            bracket_depth = max(bracket_depth - 1, 0)

        if character == "*" and paren_depth == 0 and bracket_depth == 0:
            factor = "".join(current).strip()
            if factor:
                factors.append(factor)
            current = []
            continue

        current.append(character)

    trailing = "".join(current).strip()
    if trailing:
        factors.append(trailing)

    return factors


def replace_numerical_parts(event: str) -> str:
    """Replace concrete numeric literals with ``{num}`` placeholders."""
    return re.sub(
        r"\[(\d+(\.\d+)?(?:\+\d+(\.\d+)?|\-\d+(\.\d+)?)*(?:,\d+(\.\d+)?(?:\+\d+(\.\d+)?|\-\d+(\.\d+)?)*?)*)\]",
        lambda match: "[" + ",".join("{num}" for _ in match.group(1).split(",")) + "]",
        event,
    )


def extract_numerical_values(event: str) -> list[float] | None:
    """Extract numeric values from a monitor-state factor."""
    matches = re.findall(
        r"\[(\d+(\.\d+)?(?:\+\d+(\.\d+)?|\-\d+(\.\d+)?)*(?:,\d+(\.\d+)?(?:\+\d+(\.\d+)?|\-\d+(\.\d+)?)*?)*)\]",
        event,
    )
    values: list[float] = []
    for match in matches:
        expressions = match[0].split(",")
        for expression in expressions:
            evaluated = eval(expression)
            values.append(0.01 if evaluated == 0 else float(evaluated))
    return values or None


def extract_events(state: str) -> list[str]:
    """Return the normalized top-level factors of a monitor-state string."""
    return [replace_numerical_parts(part) for part in split_top_level_factors(state)]


def build_one_hot_event_index(states: Iterable[str]) -> dict[str, int]:
    """Construct a stable one-hot event index from a collection of states."""
    ordered_events: OrderedDict[str, int] = OrderedDict()
    for state in states:
        for event in extract_events(state):
            if event not in ordered_events:
                ordered_events[event] = len(ordered_events)
    return dict(ordered_events)


def build_numerical_event_index(states: Iterable[str]) -> dict[str, int]:
    """Construct a stable numerical event index from a collection of states."""
    ordered_events: OrderedDict[str, int] = OrderedDict()
    next_index = 0
    for state in states:
        for event in extract_events(state):
            if event in ordered_events:
                continue
            ordered_events[event] = next_index
            next_index += 1
            placeholder_count = event.count("{num}")
            for extra_index in range(1, placeholder_count):
                ordered_events[event + "£ADDITIONAL£" * extra_index] = next_index
                next_index += 1
    return dict(ordered_events)


def encode_one_hot_monitor_state(state: str, event_index: dict[str, int]) -> np.ndarray:
    """Encode a monitor state as a binary event-presence vector."""
    vector = np.zeros(len(event_index), dtype=np.float32)
    for event in extract_events(state):
        if event in event_index:
            vector[event_index[event]] = 1.0
    return vector


def encode_numerical_monitor_state(state: str, event_index: dict[str, int]) -> np.ndarray:
    """Encode a monitor state as a numerical event vector."""
    vector = np.zeros(len(event_index), dtype=np.float32)
    for event in split_top_level_factors(state):
        normalized_event = replace_numerical_parts(event)
        values = extract_numerical_values(event)
        if normalized_event not in event_index:
            continue
        if values is None:
            vector[event_index[normalized_event]] = 1.0
            continue
        for value_index, value in enumerate(values):
            key = normalized_event + "£ADDITIONAL£" * value_index
            if key in event_index:
                vector[event_index[key]] = value
    return vector
