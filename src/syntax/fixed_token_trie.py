from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Protocol


# -----------------------------
# Tokenizer-side trie
# -----------------------------


@dataclass(slots=True)
class TrieNode:
    children: dict[str, "TrieNode"]
    ending_token_ids: list[int]

    def __init__(self) -> None:
        self.children = {}
        self.ending_token_ids = []


class TokenTextTrie:
    """
    Trie over decoded tokenizer token strings.

    Supports:
      match_prefixes(text) -> [(token_id, matched_length), ...]
    where token_text is a prefix of `text`.
    """

    __slots__ = ("_root",)

    def __init__(self, token_texts: Mapping[int, str]) -> None:
        root = TrieNode()
        for tid, s in token_texts.items():
            if not isinstance(tid, int):
                raise TypeError("token_texts keys must be int token ids.")
            if not isinstance(s, str):
                raise TypeError("token_texts values must be decoded token strings.")
            if s == "":
                continue

            node = root
            for ch in s:
                nxt = node.children.get(ch)
                if nxt is None:
                    nxt = TrieNode()
                    node.children[ch] = nxt
                node = nxt
            node.ending_token_ids.append(tid)

        self._root = root

    def match_prefixes(self, text: str) -> list[tuple[int, int]]:
        """
        Returns a list of (token_id, matched_length) for all tokens whose decoded text
        is a prefix of `text`. Order is deterministic given token_text insertion order.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a str.")

        out: list[tuple[int, int]] = []
        node = self._root
        # Walk along `text`; every time we hit a node with ending tokens, collect them.
        for i, ch in enumerate(text):
            nxt = node.children.get(ch)
            if nxt is None:
                break
            node = nxt
            if node.ending_token_ids:
                matched_len = i + 1
                for tid in node.ending_token_ids:
                    out.append((tid, matched_len))
        return out


# -----------------------------
# Fixed-token offset machine
# -----------------------------


class FixedTokenLike(Protocol):
    id: int
    literal: str


@dataclass(frozen=True, slots=True)
class FixedTokenTrie:
    """
    Deterministic machine for forcing generation of a single fixed token literal.

    States are offsets into `literal` (0..len(literal)).
    Transitions are tokenizer token ids whose decoded string matches the next slice.

    Use during generation:
      state = trie.start_state
      while not trie.is_accept(state):
          allowed = trie.allowed_token_ids(state)
          # mask logits to `allowed`
          token_id = sample(...)
          state = trie.step(state, token_id)  # must not be None
    """

    literal: str
    _allowed: tuple[tuple[int, ...], ...]  # per-state allowed token ids
    _next: dict[tuple[int, int], int]  # (state, token_id) -> next_state

    @property
    def start_state(self) -> int:
        return 0

    @property
    def accept_state(self) -> int:
        return len(self.literal)

    def is_accept(self, state: int) -> bool:
        return state == len(self.literal)

    def allowed_token_ids(self, state: int) -> tuple[int, ...]:
        if state < 0 or state >= len(self._allowed):
            raise IndexError(f"Invalid state {state}.")
        return self._allowed[state]

    def step(self, state: int, token_id: int) -> Optional[int]:
        return self._next.get((state, token_id))

    @staticmethod
    def compile(literal: str, token_trie: TokenTextTrie) -> "FixedTokenTrie":
        """
        Compile a FixedTokenTrie for `literal` using a TokenTextTrie for the tokenizer vocab.

        If some offset has no outgoing transitions, the trie will still compile,
        but generation will get stuck there (allowed list empty).
        """
        if not isinstance(literal, str):
            raise TypeError("literal must be a str.")
        if literal == "":
            return FixedTokenTrie(literal="", _allowed=((),), _next={})

        n = len(literal)
        allowed: list[tuple[int, ...]] = [tuple() for _ in range(n + 1)]
        nxt: dict[tuple[int, int], int] = {}

        # Accept state has no outgoing transitions
        allowed[n] = tuple()

        for i in range(n):
            suffix = literal[i:]
            matches = token_trie.match_prefixes(
                suffix
            )  # [(token_id, matched_len), ...]
            if not matches:
                allowed[i] = tuple()
                continue

            # Deduplicate token ids while preserving order.
            seen: set[int] = set()
            ids: list[int] = []
            for tid, mlen in matches:
                if mlen <= 0:
                    continue
                j = i + mlen
                if j > n:
                    continue  # should not happen if match_prefixes is correct, but keep safe
                if tid in seen:
                    continue
                seen.add(tid)
                ids.append(tid)
                nxt[(i, tid)] = j

            allowed[i] = tuple(ids)

        return FixedTokenTrie(literal=literal, _allowed=tuple(allowed), _next=nxt)


# -----------------------------
# Repository for spec fixed tokens
# -----------------------------


class FixedTokenTrieRepository:
    """
    Compiles and stores FixedTokenTrie for many fixed tokens given a tokenizer vocab mapping.
    """

    __slots__ = ("_tries", "_token_trie")

    def __init__(
        self, fixed_tokens: Iterable[FixedTokenLike], token_texts: Mapping[int, str]
    ) -> None:
        self._token_trie = TokenTextTrie(token_texts)
        tries: dict[int, FixedTokenTrie] = {}
        for ft in fixed_tokens:
            tries[int(ft.id)] = FixedTokenTrie.compile(ft.literal, self._token_trie)
        self._tries = tries

    def get(self, fixed_token_id: int) -> FixedTokenTrie:
        try:
            return self._tries[fixed_token_id]
        except KeyError:
            raise KeyError(
                f"No FixedTokenTrie for fixed_token_id={fixed_token_id}"
            ) from None
