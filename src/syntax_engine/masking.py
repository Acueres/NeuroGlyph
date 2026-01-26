from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping, Optional, Sequence
import unicodedata

from model.language_spec import LanguageSpec, FixedToken, TokenInfo


@lru_cache(maxsize=4096)
def _ucategory(ch: str) -> str:
    return unicodedata.category(ch)


class LiteralTrie:
    """Character trie over *literal spellings* (keywords + fixed tokens).

    Nodes store:
      - children transitions
      - terminal masks for tokens ending at this node
      - subtree masks for tokens reachable from this node
    Masks are bitmasks over a dense index of literal token ids.
    """

    __slots__ = (
        "children",
        "terminal_mask",
        "terminal_keyword_mask",
        "subtree_mask",
    )

    def __init__(self) -> None:
        # node 0 is root
        self.children: list[dict[str, int]] = [dict()]
        self.terminal_mask: list[int] = [0]
        self.terminal_keyword_mask: list[int] = [0]
        self.subtree_mask: list[int] = [0]

    def _new_node(self) -> int:
        idx = len(self.children)
        self.children.append(dict())
        self.terminal_mask.append(0)
        self.terminal_keyword_mask.append(0)
        self.subtree_mask.append(0)
        return idx

    def insert(self, text: str, token_bit: int, *, is_keyword: bool) -> None:
        if not text:
            return
        node = 0
        for ch in text:
            nxt = self.children[node].get(ch)
            if nxt is None:
                nxt = self._new_node()
                self.children[node][ch] = nxt
            node = nxt
        self.terminal_mask[node] |= token_bit
        if is_keyword:
            self.terminal_keyword_mask[node] |= token_bit

    def finalize(self) -> None:
        """Compute subtree masks for all nodes."""

        def dfs(n: int) -> int:
            m = self.terminal_mask[n]
            for _, child in self.children[n].items():
                m |= dfs(child)
            self.subtree_mask[n] = m
            return m

        dfs(0)

    def root_start_chars(self, expected_literal_mask: int) -> set[str]:
        """Which first characters can start *some* expected literal."""
        out: set[str] = set()
        for ch, child in self.children[0].items():
            if self.subtree_mask[child] & expected_literal_mask:
                out.add(ch)
        return out

    def continuation_chars(self, node: int, expected_literal_mask: int) -> set[str]:
        """Which next characters can continue *some* expected literal from node."""
        out: set[str] = set()
        for ch, child in self.children[node].items():
            if self.subtree_mask[child] & expected_literal_mask:
                out.add(ch)
        return out


@dataclass(frozen=True)
class FixedTokenTable:
    """Spelling table for fixed tokens/operators/punctuators."""

    by_token_id: Mapping[int, tuple[str, ...]]

    @staticmethod
    def from_fixed_tokens(fixed_tokens: Sequence[FixedToken]) -> "FixedTokenTable":
        tmp: dict[int, list[str]] = {}
        for ft in fixed_tokens:
            tmp.setdefault(ft.token_id, []).append(ft.spelling)
        return FixedTokenTable(by_token_id={k: tuple(v) for k, v in tmp.items()})


@dataclass(frozen=True)
class TokenKindIndex:
    """Lightweight index over TokenInfo for runtime classification."""

    by_id: Mapping[int, TokenInfo]
    identifier_token_ids: frozenset[int]
    integer_token_id: int
    real_token_id: int
    string_token_id: int

    @staticmethod
    def from_tokens(spec: LanguageSpec) -> "TokenKindIndex":
        by_id = {t.id: t for t in spec.tokens}

        ident_ids: set[int] = set()
        for t in spec.tokens:
            name = t.name.lower()
            cat = t.category.lower()
            if name in ("identifier", "ident") or cat == "identifier":
                ident_ids.add(t.id)

        return TokenKindIndex(
            by_id=by_id,
            identifier_token_ids=frozenset(ident_ids),
            integer_token_id=spec.numbers.integer_token_id,
            real_token_id=spec.numbers.real_token_id,
            string_token_id=spec.strings.token_id,
        )


@dataclass(frozen=True)
class CharClassifier:
    whitespace: frozenset[str]
    ident_start_categories: frozenset[str]
    ident_continue_categories: frozenset[str]
    quote_chars: frozenset[str]
    digits: frozenset[str]

    @staticmethod
    def from_spec(spec: LanguageSpec) -> "CharClassifier":
        ws = set(spec.trivia.whitespace_chars)
        ws.add(spec.trivia.newline)

        digits_raw = str(spec.numbers.digits).strip()
        if digits_raw.lower() in {"ascii", "digitset.ascii", "digitset_ascii"}:
            digits = set("0123456789")
        elif digits_raw in {"[0-9]", "0-9"}:
            digits = set("0123456789")
        else:
            digits = {ch for ch in digits_raw if ch}

        return CharClassifier(
            whitespace=frozenset(ws),
            ident_start_categories=frozenset(spec.identifier.start.allowed_categories),
            ident_continue_categories=frozenset(
                spec.identifier.continue_.allowed_categories
            ),
            quote_chars=frozenset(spec.strings.quote_chars),
            digits=frozenset(digits),
        )

    def is_trivia_char(self, ch: str) -> bool:
        return ch in self.whitespace

    def _unicode_category(self, ch: str) -> str:
        return _ucategory(ch)

    def is_ident_start(self, ch: str) -> bool:
        return self._unicode_category(ch) in self.ident_start_categories

    def is_ident_continue(self, ch: str) -> bool:
        return self._unicode_category(ch) in self.ident_continue_categories

    def is_digit(self, ch: str) -> bool:
        return ch in self.digits

    def is_quote(self, ch: str) -> bool:
        return ch in self.quote_chars


@dataclass(frozen=True)
class LiteralBitIndex:
    """Dense bit mapping for literal token ids (keywords + fixed tokens)."""

    token_id_to_bit: Mapping[int, int]
    bit_to_token_id: tuple[int, ...]

    def bit_for(self, token_id: int) -> int:
        return self.token_id_to_bit.get(token_id, 0)


@dataclass(frozen=True)
class SpecArtifacts:
    """Precomputed, language-agnostic structures derived from LanguageSpec."""

    spec_version: str
    spec_hash: str
    token_index: TokenKindIndex
    classifier: CharClassifier
    fixed: FixedTokenTable
    literal_bits: LiteralBitIndex
    literals: LiteralTrie
    allow_leading_dot_real: bool

    @staticmethod
    def from_spec(
        spec: LanguageSpec, *, fallback_hash: str = "NO_HASH"
    ) -> "SpecArtifacts":
        token_index = TokenKindIndex.from_tokens(spec)
        classifier = CharClassifier.from_spec(spec)
        fixed = FixedTokenTable.from_fixed_tokens(spec.fixed_tokens)

        # Build dense bit mapping for all literal token ids.
        literal_ids: set[int] = set()
        for kw in spec.keywords:
            literal_ids.add(kw.token_id)
        for ft in spec.fixed_tokens:
            literal_ids.add(ft.token_id)

        sorted_ids = tuple(sorted(literal_ids))
        token_id_to_bit = {tid: (1 << i) for i, tid in enumerate(sorted_ids)}
        bits = LiteralBitIndex(
            token_id_to_bit=token_id_to_bit, bit_to_token_id=sorted_ids
        )

        trie = LiteralTrie()
        for kw in spec.keywords:
            trie.insert(kw.text, bits.bit_for(kw.token_id), is_keyword=True)
        for ft in spec.fixed_tokens:
            trie.insert(ft.spelling, bits.bit_for(ft.token_id), is_keyword=False)
        trie.finalize()

        return SpecArtifacts(
            spec_version=spec.spec_version,
            spec_hash=spec.spec_hash or fallback_hash,
            token_index=token_index,
            classifier=classifier,
            fixed=fixed,
            literal_bits=bits,
            literals=trie,
            allow_leading_dot_real=bool(spec.numbers.allow_leading_dot_real),
        )


# ----------------------------
# Tokenizer-derived index
# ----------------------------


@dataclass(frozen=True)
class TokenPieceMeta:
    text: str
    lead_len: int
    sig_start: int  # -1 if none
    sig_end: int  # -1 if none
    first_sig_char: Optional[str]
    is_all_trivia: bool
    has_internal_trivia: bool

    @property
    def has_sig(self) -> bool:
        return self.sig_start >= 0

    def sig_text(self) -> str:
        if self.sig_start < 0:
            return ""
        return self.text[self.sig_start : self.sig_end]

    def has_tail_trivia(self) -> bool:
        if self.sig_end < 0:
            return False
        return self.sig_end < len(self.text)


class TokenizerIndex:
    """Predecode tokenizer piece strings and group vocab ids by first significant char."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.meta: list[TokenPieceMeta] = [
            TokenPieceMeta("", 0, -1, -1, None, True, False) for _ in range(vocab_size)
        ]

        self.ids_by_first_sig_char: dict[str, list[int]] = {}
        self.ids_all_trivia: list[int] = []
        self.ids_ident_start: list[int] = []
        self.ids_ident_continue: list[int] = []
        self.ids_digit_start: list[int] = []
        self.ids_by_quote_start: dict[str, list[int]] = {}

        self.always_allow_ids: set[int] = set()  # e.g. special tokens

    @staticmethod
    def _decode_piece(tokenizer: Any, tok_id: int) -> str:
        try:
            return tokenizer.decode(
                [tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        except TypeError:
            return tokenizer.decode([tok_id], skip_special_tokens=False)

    @classmethod
    def build(
        cls, tokenizer: Any, classifier: CharClassifier, *, predecode: bool = True
    ) -> "TokenizerIndex":
        vocab_size = int(
            getattr(tokenizer, "vocab_size", None) or len(tokenizer.get_vocab())
        )
        idx = cls(vocab_size=vocab_size)

        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        idx.always_allow_ids.update(special_ids)

        if not predecode:
            return idx

        for tok_id in range(vocab_size):
            signature = cls._decode_piece(tokenizer, tok_id)
            n = len(signature)

            # Leading trivia
            i = 0
            while i < n and classifier.is_trivia_char(signature[i]):
                i += 1
            lead_len = i

            if i >= n:
                meta = TokenPieceMeta(
                    text=signature,
                    lead_len=lead_len,
                    sig_start=-1,
                    sig_end=-1,
                    first_sig_char=None,
                    is_all_trivia=True,
                    has_internal_trivia=False,
                )
                idx.meta[tok_id] = meta
                idx.ids_all_trivia.append(tok_id)
                continue

            sig_start = i
            while i < n and not classifier.is_trivia_char(signature[i]):
                i += 1
            sig_end = i

            # Tail must be all trivia.
            has_internal_trivia = False
            j = sig_end
            while j < n:
                if not classifier.is_trivia_char(signature[j]):
                    has_internal_trivia = True
                    break
                j += 1

            first_char = signature[sig_start]
            meta = TokenPieceMeta(
                text=signature,
                lead_len=lead_len,
                sig_start=sig_start,
                sig_end=sig_end,
                first_sig_char=first_char,
                is_all_trivia=False,
                has_internal_trivia=has_internal_trivia,
            )
            idx.meta[tok_id] = meta

            idx.ids_by_first_sig_char.setdefault(first_char, []).append(tok_id)

            if classifier.is_ident_start(first_char):
                idx.ids_ident_start.append(tok_id)
            if classifier.is_ident_continue(first_char):
                idx.ids_ident_continue.append(tok_id)
            if classifier.is_digit(first_char):
                idx.ids_digit_start.append(tok_id)
            if classifier.is_quote(first_char):
                idx.ids_by_quote_start.setdefault(first_char, []).append(tok_id)

        return idx


# Runtime state + mask builder

@dataclass(frozen=True)
class ExpectedContext:
    expected_token_ids: frozenset[int]
    can_terminate_statement_here: bool

    @property
    def signature(self) -> tuple[int, bool]:
        return (
            hash(tuple(sorted(self.expected_token_ids))),
            self.can_terminate_statement_here,
        )


@dataclass(frozen=True)
class MaskConfig:
    allow_trivia_only_pieces: bool = True
    include_special_tokens: bool = True


@dataclass(frozen=True)
class LexAtom:
    """One possible lexical micro-state at the cursor.

    mode:
      - 'B': at token boundary (can start a token now)
      - 'L': in the middle of matching a literal (keyword/fixed) at trie node
      - 'I': in the middle of an identifier
    forbid_ident_continue:
      - only meaningful for 'B': if True, next immediate non-trivia char may NOT be
        an identifier-continue char (enforces keyword boundary when a keyword was
        completed without whitespace).
    """

    mode: str
    node: int
    forbid_ident_continue: bool = False


@dataclass(frozen=True)
class CursorLexState:
    atoms: frozenset[LexAtom]

    @staticmethod
    def initial() -> "CursorLexState":
        return CursorLexState(atoms=frozenset({LexAtom("B", 0, False)}))

    @property
    def signature(self) -> int:
        # Stable-ish hash for caching.
        return hash(
            tuple(sorted((a.mode, a.node, a.forbid_ident_continue) for a in self.atoms))
        )


class MaskCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, tuple[int, bool], int], tuple[int, ...]] = {}

    def get(
        self, spec_hash: str, ctx: ExpectedContext, state: CursorLexState
    ) -> Optional[tuple[int, ...]]:
        return self._cache.get((spec_hash, ctx.signature, state.signature))

    def put(
        self,
        spec_hash: str,
        ctx: ExpectedContext,
        state: CursorLexState,
        allowed_ids: Sequence[int],
    ) -> None:
        self._cache[(spec_hash, ctx.signature, state.signature)] = tuple(allowed_ids)


class SubwordMaskEngine:
    """Spec-driven subword token masker.

    Usage pattern:
      state = CursorLexState.initial()
      while generating:
         expected = compiler.PredictNext(text, cursor)
         mask_ids = engine.build_allowed_vocab_ids(expected.ids, state, can_terminate=expected.can_term)
         ... sample tok_id using prefix_allowed_tokens_fn ...
         state = engine.advance_state(state, tok_id, expected.ids)
         ... append tok_id to text ...
    """

    def __init__(
        self,
        artifacts: SpecArtifacts,
        tok_index: TokenizerIndex,
        *,
        config: MaskConfig | None = None,
        cache: MaskCache | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.idx = tok_index
        self.config = config or MaskConfig()
        self.cache = cache or MaskCache()

    def build_allowed_vocab_ids(
        self,
        expected_token_ids: Iterable[int],
        state: CursorLexState | None = None,
        *,
        can_terminate: bool = False,
    ) -> list[int]:
        """Return allowed tokenizer vocab ids for current cursor state."""

        state = state or CursorLexState.initial()
        ctx = ExpectedContext(frozenset(expected_token_ids), can_terminate)

        cached = self.cache.get(self.artifacts.spec_hash, ctx, state)
        if cached is not None:
            return list(cached)

        expected_set = ctx.expected_token_ids
        token_index = self.artifacts.token_index
        classifier = self.artifacts.classifier
        trie = self.artifacts.literals

        wants_ident = any(
            tid in token_index.identifier_token_ids for tid in expected_set
        )
        wants_number = (token_index.integer_token_id in expected_set) or (
            token_index.real_token_id in expected_set
        )
        wants_string = token_index.string_token_id in expected_set

        expected_literal_mask = 0
        for tid in expected_set:
            expected_literal_mask |= self.artifacts.literal_bits.bit_for(tid)

        # Candidate narrowing: union of groups implied by possible first significant char.
        candidate_ids: set[int] = set()

        # We may allow trivia-only pieces only if *some* path is at a boundary.
        if self.config.allow_trivia_only_pieces and self._can_insert_trivia(
            state, expected_literal_mask, wants_ident
        ):
            candidate_ids.update(self.idx.ids_all_trivia)

        # Special tokens (if any).
        if self.config.include_special_tokens:
            candidate_ids.update(self.idx.always_allow_ids)

        # Compute possible first significant characters for this step.
        first_chars: set[str] = set()

        # Epsilon-closure: if we are in 'L' or 'I', we can also be at boundary *now*
        # for the purpose of starting the next token, when completion is possible.
        boundary_atoms = self._boundary_atoms(state, expected_literal_mask)
        has_boundary = len(boundary_atoms) > 0

        if has_boundary:
            # Literal starters from trie root restricted to expected literals.
            if expected_literal_mask:
                first_chars |= trie.root_start_chars(expected_literal_mask)
            # Identifier/number/string starters.
            if wants_ident:
                candidate_ids.update(self.idx.ids_ident_start)
            if wants_number:
                candidate_ids.update(self.idx.ids_digit_start)
                # Leading dot real (if enabled and real token expected)
                if self.artifacts.allow_leading_dot_real and (
                    token_index.real_token_id in expected_set
                ):
                    # Allow '.' to start (caller/compiler should require digit after dot).
                    first_chars.add(".")
            if wants_string:
                for q in classifier.quote_chars:
                    candidate_ids.update(self.idx.ids_by_quote_start.get(q, []))

        # Continuations from literal/identifier states.
        for a in state.atoms:
            if a.mode == "L" and expected_literal_mask:
                first_chars |= trie.continuation_chars(a.node, expected_literal_mask)
            elif a.mode == "I":
                # Identifier continuation can start with any ident-continue char.
                candidate_ids.update(self.idx.ids_ident_continue)

        for ch in first_chars:
            candidate_ids.update(self.idx.ids_by_first_sig_char.get(ch, []))

        allowed_ids: list[int] = []
        for tok_id in candidate_ids:
            if (
                tok_id in self.idx.always_allow_ids
                and self.config.include_special_tokens
            ):
                allowed_ids.append(tok_id)
                continue

            meta = self.idx.meta[tok_id]
            if meta.has_internal_trivia:
                continue

            if meta.is_all_trivia:
                if self.config.allow_trivia_only_pieces and self._can_insert_trivia(
                    state, expected_literal_mask, wants_ident
                ):
                    allowed_ids.append(tok_id)
                continue

            nxt = self._step(
                state,
                meta,
                expected_literal_mask=expected_literal_mask,
                wants_ident=wants_ident,
                wants_number=wants_number,
                wants_string=wants_string,
                expected_set=expected_set,
            )
            if nxt.atoms:
                allowed_ids.append(tok_id)

        allowed_ids.sort()
        self.cache.put(self.artifacts.spec_hash, ctx, state, allowed_ids)
        return allowed_ids

    def advance_state(
        self,
        state: CursorLexState,
        tok_id: int,
        expected_token_ids: Iterable[int],
        *,
        can_terminate: bool = False,
    ) -> CursorLexState:
        """Advance lexical state after a chosen tokenizer piece."""
        expected_set = frozenset(expected_token_ids)
        token_index = self.artifacts.token_index

        wants_ident = any(
            tid in token_index.identifier_token_ids for tid in expected_set
        )
        wants_number = (token_index.integer_token_id in expected_set) or (
            token_index.real_token_id in expected_set
        )
        wants_string = token_index.string_token_id in expected_set

        expected_literal_mask = 0
        for tid in expected_set:
            expected_literal_mask |= self.artifacts.literal_bits.bit_for(tid)

        meta = self.idx.meta[tok_id]
        if meta.has_internal_trivia:
            return CursorLexState.initial()

        return self._step(
            state,
            meta,
            expected_literal_mask=expected_literal_mask,
            wants_ident=wants_ident,
            wants_number=wants_number,
            wants_string=wants_string,
            expected_set=expected_set,
        )

    def _boundary_atoms(
        self, state: CursorLexState, expected_literal_mask: int
    ) -> set[LexAtom]:
        """Atoms that can act as boundary *now* (epsilon-closure)."""
        trie = self.artifacts.literals
        out: set[LexAtom] = set()
        for a in state.atoms:
            if a.mode == "B":
                out.add(a)
            elif a.mode == "I":
                out.add(LexAtom("B", 0, False))
            elif a.mode == "L":
                # Can end a literal at this node if it is terminal for expected literals.
                if trie.terminal_mask[a.node] & expected_literal_mask:
                    is_kw = bool(
                        trie.terminal_keyword_mask[a.node] & expected_literal_mask
                    )
                    out.add(LexAtom("B", 0, is_kw))
        return out

    def _can_insert_trivia(
        self, state: CursorLexState, expected_literal_mask: int, wants_ident: bool
    ) -> bool:
        # Trivia can be inserted if some path can be at boundary now.
        return len(self._boundary_atoms(state, expected_literal_mask)) > 0

    def _step(
        self,
        state: CursorLexState,
        meta: TokenPieceMeta,
        *,
        expected_literal_mask: int,
        wants_ident: bool,
        wants_number: bool,
        wants_string: bool,
        expected_set: frozenset[int],
    ) -> CursorLexState:
        """Transition function: consume one tokenizer piece."""

        classifier = self.artifacts.classifier
        trie = self.artifacts.literals

        # All-trivia piece
        if meta.is_all_trivia:
            if self.config.allow_trivia_only_pieces and self._can_insert_trivia(
                state, expected_literal_mask, wants_ident
            ):
                return CursorLexState(atoms=frozenset({LexAtom("B", 0, False)}))
            return CursorLexState(atoms=frozenset())

        # Split: [leading trivia] + [sig-run] + [tail trivia]
        lead = meta.lead_len
        sig = meta.sig_text()
        tail = meta.has_tail_trivia()

        # Leading trivia consumes any forbid and requires we can be at boundary.
        start_atoms: set[LexAtom]
        if lead > 0:
            if not self._can_insert_trivia(state, expected_literal_mask, wants_ident):
                return CursorLexState(atoms=frozenset())
            start_atoms = {LexAtom("B", 0, False)}
        else:
            start_atoms = set(state.atoms)

        if not sig:
            # Shouldn't happen because meta.is_all_trivia handled above.
            return CursorLexState(atoms=frozenset())

        next_atoms: set[LexAtom] = set()

        # Helper: literal traversal
        def traverse(node: int, text: str) -> Optional[int]:
            for ch in text:
                nxt = trie.children[node].get(ch)
                if nxt is None:
                    return None
                node = nxt
            return node

        first_ch = sig[0]

        for a in start_atoms:
            if a.mode == "B":
                # Enforce keyword-boundary forbid.
                if a.forbid_ident_continue and classifier.is_ident_continue(first_ch):
                    continue

                # 1) Start/continue a literal (keyword/fixed) from trie root.
                if expected_literal_mask:
                    node = traverse(0, sig)
                    if node is not None and (
                        trie.subtree_mask[node] & expected_literal_mask
                    ):
                        if tail:
                            # If the piece has tail trivia, the literal must be complete here.
                            if trie.terminal_mask[node] & expected_literal_mask:
                                next_atoms.add(LexAtom("B", 0, False))
                        else:
                            next_atoms.add(LexAtom("L", node, False))
                            if trie.terminal_mask[node] & expected_literal_mask:
                                is_kw = bool(
                                    trie.terminal_keyword_mask[node]
                                    & expected_literal_mask
                                )
                                next_atoms.add(LexAtom("B", 0, is_kw))

                # 2) Identifier
                if wants_ident and classifier.is_ident_start(first_ch):
                    ok = True
                    for ch in sig[1:]:
                        if not classifier.is_ident_continue(ch):
                            ok = False
                            break
                    if ok:
                        if tail:
                            next_atoms.add(LexAtom("B", 0, False))
                        else:
                            next_atoms.add(LexAtom("I", 0, False))
                            next_atoms.add(LexAtom("B", 0, False))

                # 3) Numbers (basic start-only; multi-piece DFA can be added later)
                if wants_number:
                    # digits-only run
                    if classifier.is_digit(first_ch):
                        ok = all(classifier.is_digit(ch) for ch in sig)
                        if ok:
                            if tail:
                                next_atoms.add(LexAtom("B", 0, False))
                            else:
                                # allow continuing as if identifier-like (digits-only)
                                next_atoms.add(LexAtom("B", 0, False))

                    # leading dot real (only if real is expected and spec allows it)
                    if (
                        first_ch == "."
                        and self.artifacts.allow_leading_dot_real
                        and (self.artifacts.token_index.real_token_id in expected_set)
                    ):
                        # Conservative: only allow '.' (or '.<digits>') in this piece.
                        if sig == "." or (
                            sig.startswith(".")
                            and all(classifier.is_digit(ch) for ch in sig[1:])
                        ):
                            if tail:
                                next_atoms.add(LexAtom("B", 0, False))
                            else:
                                next_atoms.add(LexAtom("B", 0, False))

                # 4) Strings (basic start-only)
                if wants_string and classifier.is_quote(first_ch):
                    # Allow just starting quote or a whole quoted chunk in one piece.
                    if tail:
                        next_atoms.add(LexAtom("B", 0, False))
                    else:
                        next_atoms.add(LexAtom("B", 0, False))

            elif a.mode == "L":
                if not expected_literal_mask:
                    continue
                node = traverse(a.node, sig)
                if node is None:
                    continue
                if not (trie.subtree_mask[node] & expected_literal_mask):
                    continue
                if tail:
                    if trie.terminal_mask[node] & expected_literal_mask:
                        next_atoms.add(LexAtom("B", 0, False))
                else:
                    next_atoms.add(LexAtom("L", node, False))
                    if trie.terminal_mask[node] & expected_literal_mask:
                        is_kw = bool(
                            trie.terminal_keyword_mask[node] & expected_literal_mask
                        )
                        next_atoms.add(LexAtom("B", 0, is_kw))

            elif a.mode == "I":
                # Identifier continuation: all chars must be ident-continue.
                ok = True
                for ch in sig:
                    if not classifier.is_ident_continue(ch):
                        ok = False
                        break
                if not ok:
                    continue

                if tail:
                    next_atoms.add(LexAtom("B", 0, False))
                else:
                    next_atoms.add(LexAtom("I", 0, False))
                    next_atoms.add(LexAtom("B", 0, False))

        return CursorLexState(atoms=frozenset(next_atoms))
