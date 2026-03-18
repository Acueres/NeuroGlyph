import zlib

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence
from compiler_client.language_spec import LanguageSpec, LexerMachine
from .fixed_token_trie import TokenTextTrie, FixedTokenTrie
from .lexer_machine_runner import LexerMachineRunner


# ----------------------------
# Token metadata + trivia
# ----------------------------


@dataclass(frozen=True, slots=True)
class TokenMeta:
    token_id: int
    text: str
    core: str
    has_leading_trivia: bool
    has_trailing_trivia: bool
    is_trivia_only: bool
    is_special: bool = False


class TriviaClassifier:
    __slots__ = ("_trivia_chars", "_newline_char", "_line_comment_char")

    def __init__(
        self, whitespace_chars: Iterable[str], newline_char: str, line_comment_char: str
    ):
        # Newline is trivia too.
        self._trivia_chars = frozenset(set(whitespace_chars) | {newline_char})
        self._newline_char = newline_char
        self._line_comment_char = line_comment_char

    def is_trivia_char(self, ch: str) -> bool:
        return ch in self._trivia_chars

    def contains_newline(self, s: str) -> bool:
        for ch in s:
            if ch == self._newline_char:
                return True
        return False

    def is_comment_start(self, s: str) -> bool:
        return s.startswith(self._line_comment_char)

    def strip_leading(self, s: str) -> tuple[str, bool]:
        i = 0
        n = len(s)
        while i < n and self.is_trivia_char(s[i]):
            i += 1
        return s[i:], i > 0

    def strip_trailing(self, s: str) -> tuple[str, bool]:
        j = len(s)
        while j > 0 and self.is_trivia_char(s[j - 1]):
            j -= 1
        return s[:j], j < len(s)

    def strip_both(self, s: str) -> tuple[str, bool, bool, bool]:
        # returns (core, has_lead, has_trail, trivia_only)
        t, has_lead = self.strip_leading(s)
        core, has_trail = self.strip_trailing(t)
        trivia_only = (core == "") and (s != "")
        return core, has_lead, has_trail, trivia_only


@dataclass(slots=True)
class LexerTable:
    """
    Precomputed per-machine transitions for token_ids.
    - step_by_state[state][token_id] = next_state
    - allowed_start_by_state[state] : tokens allowed when candidate has not started (leading trivia ok)
    - allowed_mid_by_state[state]   : tokens allowed after start (leading trivia disallowed)
    """

    token_kind_id: int
    start_state_id: int
    accepting: dict[int, bool]
    step_by_state: dict[int, dict[int, int]]
    allowed_start_by_state: dict[int, tuple[int, ...]]
    allowed_mid_by_state: dict[int, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class FixedTokenTable:
    """
    Wraps a FixedTokenTrie with prefiltered allowed sets for start/mid.
    """

    token_kind_id: int
    literal: str
    trie: FixedTokenTrie
    allowed_start: tuple[tuple[int, ...], ...]
    allowed_mid: tuple[tuple[int, ...], ...]

    def step(self, state: int, token_id: int) -> Optional[int]:
        return self.trie.step(state, token_id)

    @property
    def accept_state(self) -> int:
        return self.trie.accept_state


# ----------------------------
# Candidate types
# ----------------------------


class Candidate:
    def token_kind_id(self) -> int:
        raise NotImplementedError

    def allowed_token_ids(self, at_start: bool) -> tuple[int, ...]:
        raise NotImplementedError

    def can_consume(self, token_id: int, at_start: bool) -> bool:
        raise NotImplementedError

    def consume(self, token_id: int, at_start: bool) -> "CandidateConsume":
        raise NotImplementedError

    def is_complete(self) -> bool:
        raise NotImplementedError

    def can_finish_without_delimiter(self) -> bool:
        # For patterns this can be true when accepting; for fixed tokens it’s only true if complete.
        return self.is_complete()


@dataclass(frozen=True, slots=True)
class CandidateConsume:
    ok: bool
    candidate: Optional[Candidate] = None


@dataclass(frozen=True, slots=True)
class FixedCandidate(Candidate):
    _table: FixedTokenTable
    _state: int

    def token_kind_id(self) -> int:
        return self._table.token_kind_id

    def allowed_token_ids(self, at_start: bool) -> tuple[int, ...]:
        return (
            self._table.allowed_start[self._state]
            if at_start
            else self._table.allowed_mid[self._state]
        )

    def can_consume(self, token_id: int, at_start: bool) -> bool:
        return token_id in (self.allowed_token_ids(at_start))

    def consume(self, token_id: int, at_start: bool) -> CandidateConsume:
        nxt = self._table.step(self._state, token_id)
        if nxt is None:
            return CandidateConsume(ok=False)
        return CandidateConsume(ok=True, candidate=FixedCandidate(self._table, nxt))

    def is_complete(self) -> bool:
        return self._state == self._table.accept_state


@dataclass(frozen=True, slots=True)
class PatternCandidate(Candidate):
    _table: LexerTable
    _state: int

    def token_kind_id(self) -> int:
        return self._table.token_kind_id

    def allowed_token_ids(self, at_start: bool) -> tuple[int, ...]:
        if at_start:
            return self._table.allowed_start_by_state.get(self._state, tuple())
        return self._table.allowed_mid_by_state.get(self._state, tuple())

    def can_consume(self, token_id: int, at_start: bool) -> bool:
        return token_id in self.allowed_token_ids(at_start)

    def consume(self, token_id: int, at_start: bool) -> CandidateConsume:
        per_state = self._table.step_by_state.get(self._state)
        if not per_state:
            return CandidateConsume(ok=False)
        nxt = per_state.get(token_id)
        if nxt is None:
            return CandidateConsume(ok=False)
        return CandidateConsume(ok=True, candidate=PatternCandidate(self._table, nxt))

    def is_complete(self) -> bool:
        return bool(self._table.accepting.get(self._state, False))

    def can_finish_without_delimiter(self) -> bool:
        return self.is_complete()


# ----------------------------
# MaskEngine
# ----------------------------


class MaskEngine:
    """
    Unifies fixed-token tries and lexer machines into one masking orchestrator.

    Usage:
      engine = MaskEngine(spec, tokenizer)
      engine.set_predictions([kind_id1, kind_id2, ...])
      allowed = engine.allowed_token_ids()
      engine.consume(sampled_token_id)
      ...
    """

    def __init__(
        self, spec: LanguageSpec, tokenizer: Any, *, allow_special_tokens: bool = False
    ):
        self._tokenizer = tokenizer
        self._allow_special = allow_special_tokens

        trivia = TriviaClassifier(
            spec.trivia.whitespace_chars,
            spec.trivia.newline_char,
            spec.trivia.line_comment_start,
        )

        # Build TokenMeta and core_map for TokenTextTrie
        token_metas: dict[int, TokenMeta] = {}
        core_map: dict[int, str] = {}
        trivia_only_ids: list[int] = []
        comment_start_ids: set[int] = set()
        comment_body_ids: set[int] = set()
        newline_ids: set[int] = set()

        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

        vocab_size = len(tokenizer)
        for token_id in range(vocab_size):
            is_special = token_id in special_ids
            if is_special and not self._allow_special:
                token_metas[token_id] = TokenMeta(
                    token_id=token_id,
                    text="",
                    core="",
                    has_leading_trivia=False,
                    has_trailing_trivia=False,
                    is_trivia_only=False,
                    is_special=True,
                )
                continue

            token_text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            contains_newline = trivia.contains_newline(token_text)
            if contains_newline:
                newline_ids.add(token_id)

            comment_body_ids.add(token_id)

            core, has_lead, has_trail, trivia_only = trivia.strip_both(token_text)

            meta = TokenMeta(
                token_id=token_id,
                text=token_text,
                core=core,
                has_leading_trivia=has_lead,
                has_trailing_trivia=has_trail,
                is_trivia_only=trivia_only,
                is_special=is_special,
            )
            token_metas[token_id] = meta

            if trivia.is_comment_start(core):
                comment_start_ids.add(token_id)

            if trivia_only:
                trivia_only_ids.append(token_id)
                continue

            if core != "":
                core_map[token_id] = core

        self._token_meta = token_metas
        self._trivia_only_ids = tuple(trivia_only_ids)
        self._comment_start_ids = comment_start_ids
        self._comment_body_ids = comment_body_ids
        self._newline_ids = newline_ids

        token_text_trie = TokenTextTrie(core_map)
        self._token_text_trie = token_text_trie

        # Fixed tokens phase
        fixed_tables: dict[int, FixedTokenTable] = {}
        for ft in spec.fixed_tokens:
            trie = FixedTokenTrie.compile(ft.literal, token_text_trie)
            fixed_tables[ft.id] = self._build_fixed_table(
                ft.id, ft.literal, trie, token_metas
            )
        self._fixed_tables = fixed_tables

        self._virtual_terminator_id = [
            ft.id for ft in spec.fixed_tokens if ft.literal == spec.trivia.newline_char
        ][0]

        # Lexer machines phase
        lexer_tables: dict[int, LexerTable] = self._build_lexer_tables(
            spec.lexer_machines, core_map, token_metas
        )
        self._lexer_tables = lexer_tables

        self._semantic_lexeme_tables: list[FixedTokenTable] = []

        # Runtime state
        self._predicted_kind_ids: tuple[int, ...] = tuple()
        self._active: list[Candidate] = []
        self._committed: bool = False
        self._finishing_candidates: list[Candidate] = []
        self._in_line_comment: bool = False

        # Pattern-based limitation by length
        self._committed_pattern_core_len = 0
        self._committed_pattern_kind_id: int | None = None
        self._max_pattern_chars: dict[int, int] = {
            m.token_kind_id: m.max_lexeme_chars for m in spec.lexer_machines
        }

    # ---------- Public API ----------

    def set_predictions(self, token_kind_ids: Sequence[int]) -> None:
        # Boundary reset
        unique = []
        seen = set()
        for k in token_kind_ids:
            if k not in seen:
                seen.add(k)
                unique.append(int(k))

        self._predicted_kind_ids = tuple(unique)

        self._active = []
        self._committed = False
        self._committed_pattern_core_len = 0
        self._committed_pattern_kind_id = None
        self._finishing_candidates = []
        self._semantic_lexeme_tables = []

        for k in self._predicted_kind_ids:
            cand = self._make_candidate(k)
            if cand is None:
                continue
            self._active.append(cand)

    def allowed_token_ids(self) -> set[int]:
        if not self._active:
            return set()

        if self._in_line_comment:
            return self._comment_body_ids

        at_start = not self._committed

        # Union across candidates
        allowed_set: set[int] = set()
        for c in self._active:
            allowed_set.update(c.allowed_token_ids(at_start))

        # Trivia-only tokens are allowed only at boundary
        if not self._committed:
            allowed_set.update(self._trivia_only_ids)

        if self._committed and self.can_finish_pattern():
            kind = self._committed_pattern_kind_id
            token_limit = (
                self._max_pattern_chars.get(kind) if kind is not None else None
            )

            if (
                token_limit is not None
                and self._committed_pattern_core_len >= token_limit
            ):
                allowed_set = set()
                if self._finishing_candidates:
                    for c in self._finishing_candidates:
                        allowed_set.update(c.allowed_token_ids(at_start=True))
                allowed_set.update(self._trivia_only_ids)
                return allowed_set

            # boundary-start masks for post candidates
            for c in self._finishing_candidates:
                allowed_set.update(c.allowed_token_ids(at_start=True))
            # also allow trivia-only tokens as a valid “stop pattern + emit trivia”
            allowed_set.update(self._trivia_only_ids)

        return allowed_set

    def consume(self, token_id: int) -> None:
        if not self._active:
            raise RuntimeError(
                "MaskEngine.consume() called without active predictions."
            )

        meta = self._token_meta[token_id]
        if meta is None:
            raise KeyError(f"Unknown token id: {token_id}")

        # Trivia-only: allowed only at boundary
        if meta.is_trivia_only:
            if not self._committed:
                if token_id in self._newline_ids and any(
                    c.token_kind_id() == self._virtual_terminator_id
                    for c in self._active
                ):
                    self._active = []
                    self._committed = False
                    self._finishing_candidates = []
                    self._committed_pattern_core_len = 0
                    self._committed_pattern_kind_id = None
                    return
                return

            if self._in_line_comment and meta.token_id in self._newline_ids:
                self._in_line_comment = False

            if meta.token_id in self._comment_start_ids:
                self._in_line_comment = True
                return

            if self.can_finish_pattern():
                self._active = []
                self._committed = False
                self._finishing_candidates = []

                self._committed_pattern_core_len = 0
                self._committed_pattern_kind_id = None
                return

            raise RuntimeError(
                "Trivia-only token emitted while committed to a non-finishable lexeme."
            )

        at_start = not self._committed

        # Prune to candidates that can consume this token
        new_active: list[Candidate] = []
        for c in self._active:
            if c.can_consume(token_id, at_start):
                res = c.consume(token_id, at_start)
                if res.ok and res.candidate is not None:
                    new_active.append(res.candidate)

        if not new_active:
            # stop pattern and start next token
            if (
                self._committed
                and self.can_finish_pattern()
                and len(self._finishing_candidates) > 0
            ):
                post = self._finishing_candidates
                # finish pattern
                self._active = post
                self._committed = False
                self._finishing_candidates = []
                # now consume this token as the first token of the next lexeme
                return self.consume(token_id)

            raise RuntimeError("No predicted candidate can consume the emitted token.")

        self._active = new_active
        self._committed = True

        if at_start:
            self._committed_pattern_kind_id = None
            self._committed_pattern_core_len = 0

            # Remember the pattern
            for c in self._active:
                if isinstance(c, PatternCandidate):
                    self._committed_pattern_kind_id = c.token_kind_id()
                    break

        # If we are in a pattern, accumulate core length
        if self._committed_pattern_kind_id is not None and meta.core:
            self._committed_pattern_core_len += len(meta.core)

        # Fixed tokens have unambiguous completion
        completed_fixed = [
            c for c in self._active if isinstance(c, FixedCandidate) and c.is_complete()
        ]
        if completed_fixed:
            self._active = []
            self._committed = False

            self._committed_pattern_kind_id = None
            self._committed_pattern_core_len = 0

    def can_finish_pattern(self) -> bool:
        """
        True if the current committed candidate set contains an accepting pattern state,
        so the caller may choose to finish the lexeme (epsilon stop) and return to boundary.
        """
        if not self._committed:
            return False
        return any(
            (isinstance(c, PatternCandidate) and c.can_finish_without_delimiter())
            for c in self._active
        )

    def finish_pattern(self) -> None:
        """
        Epsilon-stop for patterns: if you're in an accepting pattern state, return to boundary.
        This is intentionally explicit (caller decides when to stop).
        """
        if not self.can_finish_pattern():
            raise RuntimeError(
                "finish_pattern() called but no active accepting pattern candidate exists."
            )
        self._active = []
        self._committed = False

    def reset(self) -> None:
        self._predicted_kind_ids = tuple()
        self._active = []
        self._committed = False

    def needs_predictions(self) -> bool:
        # Boundary and no active candidates
        return (not self._committed) and (not self._active)

    def needs_post_predictions(self) -> bool:
        return (
            self._committed
            and self.can_finish_pattern()
            and (not self._finishing_candidates)
        )

    def set_post_predictions(self, token_kind_ids: Sequence[int]) -> None:
        # prepare candidates for "after finishing current accepting pattern"
        uniq, seen = [], set()
        for k in token_kind_ids:
            k = int(k)
            if k not in seen:
                seen.add(k)
                uniq.append(k)

        finishing_candidates: list[Candidate] = []
        for k in uniq:
            cand = self._make_candidate(k)
            if cand is not None:
                finishing_candidates.append(cand)

        self._finishing_candidates = finishing_candidates

    def is_trivia_only_token(self, token_id: int) -> bool:
        return self._token_meta[token_id].is_trivia_only

    def compile_lexeme_table(self, lexeme: str) -> FixedTokenTable:
        """
        Compile a FixedTokenTable for an arbitrary lexeme.
        """
        lexeme = str(lexeme)
        # Uses a stable negative id so it never collides with real token kinds.
        token_kind_id = -int(zlib.crc32(lexeme.encode("utf-8")))

        trie = FixedTokenTrie.compile(lexeme, self._token_text_trie)
        return self._build_fixed_table(token_kind_id, lexeme, trie, self._token_meta)

    def add_semantic_lexeme_tables(self, tables: Sequence[FixedTokenTable]) -> None:
        """
        Add semantic fixed-lexeme candidates for the current boundary prediction.
        Call after set_predictions() and only when not committed.
        """
        if self._committed:
            return

        self._semantic_lexeme_tables = list(tables)

        for t in tables:
            self._active.append(FixedCandidate(t, 0))

    # ---------- Internals ----------

    def _make_candidate(self, token_kind_id: int) -> Optional[Candidate]:
        ft = self._fixed_tables.get(token_kind_id)
        if ft is not None:
            return FixedCandidate(ft, 0)
        lt = self._lexer_tables.get(token_kind_id)
        if lt is not None:
            return PatternCandidate(lt, lt.start_state_id)
        return None

    @staticmethod
    def _build_fixed_table(
        token_kind_id: int,
        literal: str,
        trie: FixedTokenTrie,
        metas: Mapping[int, TokenMeta],
    ) -> FixedTokenTable:
        accept = trie.accept_state
        allowed_start: list[tuple[int, ...]] = []
        allowed_mid: list[tuple[int, ...]] = []

        for state in range(0, accept + 1):
            ids = trie.allowed_token_ids(state)
            start_ids: list[int] = []
            mid_ids: list[int] = []
            for tid in ids:
                meta = metas[tid]
                nxt = trie.step(state, tid)
                if nxt is None:
                    continue

                # Trailing trivia allowed only if this token completes the literal
                if meta.has_trailing_trivia and nxt != accept:
                    continue

                # Start allows leading trivia; mid disallows it.
                start_ids.append(tid)
                if not meta.has_leading_trivia:
                    mid_ids.append(tid)

            allowed_start.append(tuple(start_ids))
            allowed_mid.append(tuple(mid_ids))

        return FixedTokenTable(
            token_kind_id=token_kind_id,
            literal=literal,
            trie=trie,
            allowed_start=tuple(allowed_start),
            allowed_mid=tuple(allowed_mid),
        )

    @staticmethod
    def _build_lexer_tables(
        lexer_machines: Sequence[LexerMachine],
        core_map: Mapping[int, str],
        metas: Mapping[int, TokenMeta],
    ) -> dict[int, LexerTable]:
        tables = {}

        for m in lexer_machines:
            runner = LexerMachineRunner(m)

            accepting = {s.id: s.accepting for s in m.states}
            step_by_state: dict[int, dict[int, int]] = {}
            allowed_start: dict[int, list[int]] = {}
            allowed_mid: dict[int, list[int]] = {}

            state_ids = [s.id for s in m.states]

            for tid, core in core_map.items():
                meta = metas[tid]

                for st in state_ids:
                    ns = runner.advance_from_state(st, core)
                    if ns is None:
                        continue

                    # allow trailing trivia only if resulting lexer state is accepting
                    if meta.has_trailing_trivia and not runner.is_accepting(ns):
                        continue

                    step_by_state.setdefault(st, {})[tid] = ns
                    allowed_start.setdefault(st, []).append(tid)

                    # mid-lexeme disallows leading trivia
                    if not meta.has_leading_trivia:
                        allowed_mid.setdefault(st, []).append(tid)

            tables[m.token_kind_id] = LexerTable(
                token_kind_id=m.token_kind_id,
                start_state_id=runner.start_state_id,
                accepting=accepting,
                step_by_state=step_by_state,
                allowed_start_by_state={
                    st: tuple(ids) for st, ids in allowed_start.items()
                },
                allowed_mid_by_state={
                    st: tuple(ids) for st, ids in allowed_mid.items()
                },
            )

        return tables
