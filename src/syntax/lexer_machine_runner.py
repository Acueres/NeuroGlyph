import regex as re

from typing import Optional, Iterable
from regex import Pattern
from src.compiler_client.language_spec import (
    LanguageSpec,
    LexerMachine,
    LexerTransition,
)


class LexerMachineRunner:
    """
    Runs a single LexerMachine against input text.

    Behavior:
      - Start at machine.start_state_id
      - For each character:
          pick the first outgoing transition whose predicate matches
          otherwise reject
      - Accept iff final state is marked accepting
    """

    def __init__(self, machine: LexerMachine):
        self._machine = machine

        # state_id -> accepting
        self._accepting: dict[int, bool] = {s.id: s.accepting for s in machine.states}

        # from_state_id -> transitions
        transitions: dict[int, list[LexerTransition]] = {}
        for t in machine.transitions:
            transitions.setdefault(t.from_state_id, []).append(t)
        self._transitions = transitions

        # predicate -> compiled regex
        self._regex_cache: dict[str, Pattern[str]] = {}

    @property
    def machine(self) -> LexerMachine:
        return self._machine

    @staticmethod
    def from_spec(spec: LanguageSpec, token_kind_id: int) -> "LexerMachineRunner":
        for m in spec.lexer_machines:
            if m.token_kind_id == token_kind_id:
                return LexerMachineRunner(m)
        raise KeyError(f"No lexer machine found for token_kind_id={token_kind_id}.")

    def accepts(self, input_text: str) -> bool:
        state = self._machine.start_state_id

        for ch in input_text:
            outs = self._transitions.get(state)
            if not outs:
                return False

            next_state: Optional[int] = None
            for t in outs:
                if self._matches(t.predicate, ch):
                    next_state = t.to_state_id
                    break

            if next_state is None:
                return False

            state = next_state

        return bool(self._accepting.get(state, False))

    def accepts_chars(self, chars: Iterable[str]) -> bool:
        state = self._machine.start_state_id

        for ch in chars:
            if not isinstance(ch, str) or len(ch) != 1:
                raise TypeError(
                    "accepts_chars() expects an iterable of 1-character strings."
                )

            outs = self._transitions.get(state)
            if not outs:
                return False

            next_state: Optional[int] = None
            for t in outs:
                if self._matches(t.predicate, ch):
                    next_state = t.to_state_id
                    break

            if next_state is None:
                return False

            state = next_state

        return bool(self._accepting.get(state, False))

    @property
    def start_state_id(self) -> int:
        return self._machine.start_state_id

    def is_accepting(self, state: int) -> bool:
        return bool(self._accepting.get(state, False))

    def advance_from_state(self, state: int, text: str) -> Optional[int]:
        """
        Consume `text` starting from `state`.
        Returns next state, or None if no transition matches at some character.
        """
        s = state
        for ch in text:
            outs = self._transitions.get(s)
            if not outs:
                return None

            next_state: Optional[int] = None
            for t in outs:
                if self._matches(t.predicate, ch):
                    next_state = t.to_state_id
                    break

            if next_state is None:
                return None

            s = next_state

        return s

    def _matches(self, predicate: str, ch: str) -> bool:
        if predicate == "*":
            return True

        if predicate.startswith("lit:"):
            lit = predicate[len("lit:") :]
            return len(lit) == 1 and lit == ch

        if predicate.startswith("re:"):
            pattern = predicate[len("re:") :]
            rx = self._compile_regex(pattern)
            return rx.search(ch) is not None

        raise ValueError(f"Unknown predicate: {predicate}")

    def _compile_regex(self, pattern: str) -> Pattern[str]:
        rx = self._regex_cache.get(pattern)
        if rx is None:
            rx = re.compile(pattern)
            self._regex_cache[pattern] = rx
        return rx
