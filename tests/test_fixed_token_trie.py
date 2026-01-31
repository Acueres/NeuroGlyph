import pytest

from dataclasses import dataclass
from src.syntax_engine.fixed_token_trie import (
    TokenTextTrie,
    FixedTokenTrie,
    FixedTokenTrieRepository,
)


@dataclass(frozen=True, slots=True)
class FixedTokenTrieFixture:
    token_texts: dict[int, str]

    tid_def: int
    tid_de: int
    tid_d: int
    tid_ef: int
    tid_e: int
    tid_f: int
    tid_space_def: int
    tid_space: int

    tid_return: int
    tid_re: int
    tid_ret: int
    tid_retu: int
    tid_r: int
    tid_t: int
    tid_turn: int

    tid_arrow: int  # "->"
    tid_minus: int  # "-"
    tid_gt: int  # ">"
    tid_ge: int  # ">="
    tid_eq: int  # "="
    tid_eqeq: int  # "=="
    tid_bang: int  # "!"
    tid_ne: int  # "!="

    token_trie: TokenTextTrie
    trie_def: FixedTokenTrie
    trie_space_def: FixedTokenTrie
    trie_return: FixedTokenTrie
    trie_arrow: FixedTokenTrie
    trie_ge: FixedTokenTrie
    trie_eqeq: FixedTokenTrie
    trie_ne: FixedTokenTrie


@pytest.fixture(scope="session")
def fx() -> FixedTokenTrieFixture:
    """
    Typed fixture: a tiny "tokenizer vocab" mapping + precompiled FixedTokenTrie objects.
    """

    token_texts: dict[int, str] = {
        10: "def",
        11: "de",
        12: "d",
        13: "ef",
        14: "e",
        15: "f",
        20: " def",
        21: " ",
        30: "",
        40: "return",
        41: "re",
        42: "ret",
        43: "retu",
        44: "r",
        45: "t",
        46: "turn",
        60: "->",
        61: "-",
        62: ">",
        63: ">=",
        64: "=",
        65: "==",
        66: "!",
        67: "!=",
        99: "xyz",
    }

    token_trie = TokenTextTrie(token_texts)
    trie_def = FixedTokenTrie.compile("def", token_trie)
    trie_space_def = FixedTokenTrie.compile(" def", token_trie)
    trie_return = FixedTokenTrie.compile("return", token_trie)
    trie_arrow = FixedTokenTrie.compile("->", token_trie)
    trie_ge = FixedTokenTrie.compile(">=", token_trie)
    trie_eqeq = FixedTokenTrie.compile("==", token_trie)
    trie_ne = FixedTokenTrie.compile("!=", token_trie)

    return FixedTokenTrieFixture(
        token_texts=token_texts,
        tid_def=10,
        tid_de=11,
        tid_d=12,
        tid_ef=13,
        tid_e=14,
        tid_f=15,
        tid_space_def=20,
        tid_space=21,
        tid_return=40,
        tid_re=41,
        tid_ret=42,
        tid_retu=43,
        tid_r=44,
        tid_t=45,
        tid_turn=46,
        tid_arrow=60,
        tid_minus=61,
        tid_gt=62,
        tid_ge=63,
        tid_eq=64,
        tid_eqeq=65,
        tid_bang=66,
        tid_ne=67,
        token_trie=token_trie,
        trie_def=trie_def,
        trie_space_def=trie_space_def,
        trie_return=trie_return,
        trie_arrow=trie_arrow,
        trie_ge=trie_ge,
        trie_eqeq=trie_eqeq,
        trie_ne=trie_ne,
    )


# ----------------------------
# TokenTextTrie tests
# ----------------------------


def test_token_text_trie_match_prefixes_returns_all_prefix_tokens(
    fx: FixedTokenTrieFixture,
) -> None:
    # For "def", prefixes are: "d", "de", "def"
    matches = fx.token_trie.match_prefixes("def")
    assert matches == [
        (fx.tid_d, 1),
        (fx.tid_de, 2),
        (fx.tid_def, 3),
    ]


def test_token_text_trie_match_prefixes_stops_when_path_breaks(
    fx: FixedTokenTrieFixture,
) -> None:
    # "deX": should match "d" and "de" but then stop at 'X' (no continuation)
    matches = fx.token_trie.match_prefixes("deX")
    assert matches == [
        (fx.tid_d, 1),
        (fx.tid_de, 2),
    ]


def test_token_text_trie_ignores_empty_tokens(fx: FixedTokenTrieFixture) -> None:
    # Empty token id 30 exists but should never be returned for any prefix
    matches = fx.token_trie.match_prefixes("def")
    assert all(tid != 30 for tid, _ in matches)


# ----------------------------
# FixedTokenTrie basic behavior
# ----------------------------


def test_fixed_token_trie_has_offset_states_and_accept_state(
    fx: FixedTokenTrieFixture,
) -> None:
    trie = fx.trie_def
    assert trie.literal == "def"
    assert trie.start_state == 0
    assert trie.accept_state == 3
    assert trie.is_accept(0) is False
    assert trie.is_accept(3) is True


def test_fixed_token_trie_allowed_tokens_state0_includes_full_and_partial_prefixes(
    fx: FixedTokenTrieFixture,
) -> None:
    allowed0 = fx.trie_def.allowed_token_ids(0)

    # Must allow spelling via "def", "de"+"f", "d"+"ef", "d"+"e"+"f"
    # State 0 should allow "d", "de", "def" (NOT "e", "f", "ef")
    assert allowed0 == (fx.tid_d, fx.tid_de, fx.tid_def)


def test_fixed_token_trie_step_advances_offsets(fx: FixedTokenTrieFixture) -> None:
    trie = fx.trie_def

    # 0 --"d"--> 1
    s = trie.step(0, fx.tid_d)
    assert s == 1

    # 0 --"de"--> 2
    s = trie.step(0, fx.tid_de)
    assert s == 2

    # 0 --"def"--> 3 (accept)
    s = trie.step(0, fx.tid_def)
    assert s == 3
    assert trie.is_accept(s) is True


def test_fixed_token_trie_allows_d_then_ef_path(fx: FixedTokenTrieFixture) -> None:
    trie = fx.trie_def

    s1 = trie.step(0, fx.tid_d)
    assert s1 == 1

    # At offset 1, suffix is "ef", so it should allow "e" and "ef"
    allowed1 = trie.allowed_token_ids(s1)
    assert allowed1 == (fx.tid_e, fx.tid_ef)


def test_fixed_token_trie_return_state0_allows_prefixes_and_full(
    fx: FixedTokenTrieFixture,
) -> None:
    trie = fx.trie_return
    assert trie.literal == "return"
    # Prefixes encountered along "return": "r" -> "re" -> "ret" -> "retu" -> "return"
    assert trie.allowed_token_ids(0) == (
        fx.tid_r,
        fx.tid_re,
        fx.tid_ret,
        fx.tid_retu,
        fx.tid_return,
    )


def test_fixed_token_trie_return_allows_re_then_turn(fx: FixedTokenTrieFixture) -> None:
    trie = fx.trie_return

    s2 = trie.step(0, fx.tid_re)
    assert s2 == 2  # "re" length 2

    # Now suffix is "turn": should allow "t" and "turn"
    assert trie.allowed_token_ids(s2) == (fx.tid_t, fx.tid_turn)

    s_end = trie.step(s2, fx.tid_turn)
    assert s_end == trie.accept_state
    assert s_end is not None
    assert trie.is_accept(s_end) is True


def test_fixed_token_trie_arrow_allows_arrow_or_minus_gt(
    fx: FixedTokenTrieFixture,
) -> None:
    trie = fx.trie_arrow
    assert trie.literal == "->"

    # Prefixes: "-" then "->"
    assert trie.allowed_token_ids(0) == (fx.tid_minus, fx.tid_arrow)

    # Path A: single token "->"
    s_end = trie.step(0, fx.tid_arrow)
    assert s_end == trie.accept_state
    assert s_end is not None
    assert trie.is_accept(s_end) is True

    # Path B: "-" then ">"
    s1 = trie.step(0, fx.tid_minus)
    assert s1 == 1
    assert trie.allowed_token_ids(s1) == (fx.tid_gt,)
    s_end2 = trie.step(s1, fx.tid_gt)
    assert s_end2 == trie.accept_state


def test_fixed_token_trie_ge_allows_ge_or_gt_then_eq(fx: FixedTokenTrieFixture) -> None:
    trie = fx.trie_ge
    assert trie.literal == ">="

    # Prefixes: ">" then ">="
    assert trie.allowed_token_ids(0) == (fx.tid_gt, fx.tid_ge)

    # Path A: ">=" in one token
    s_end = trie.step(0, fx.tid_ge)
    assert s_end == trie.accept_state

    # Path B: ">" then "="
    s1 = trie.step(0, fx.tid_gt)
    assert s1 == 1
    assert trie.allowed_token_ids(s1) == (fx.tid_eq,)
    s_end2 = trie.step(s1, fx.tid_eq)
    assert s_end2 == trie.accept_state


def test_fixed_token_trie_eqeq_allows_eqeq_or_eq_then_eq(
    fx: FixedTokenTrieFixture,
) -> None:
    trie = fx.trie_eqeq
    assert trie.literal == "=="

    # Prefixes: "=" then "=="
    assert trie.allowed_token_ids(0) == (fx.tid_eq, fx.tid_eqeq)

    # Path A: "==" one token
    s_end = trie.step(0, fx.tid_eqeq)
    assert s_end == trie.accept_state

    # Path B: "=" then "="
    s1 = trie.step(0, fx.tid_eq)
    assert s1 == 1
    assert trie.allowed_token_ids(s1) == (fx.tid_eq,)
    s_end2 = trie.step(s1, fx.tid_eq)
    assert s_end2 == trie.accept_state


def test_fixed_token_trie_ne_allows_ne_or_bang_then_eq(
    fx: FixedTokenTrieFixture,
) -> None:
    trie = fx.trie_ne
    assert trie.literal == "!="

    # Prefixes: "!" then "!="
    assert trie.allowed_token_ids(0) == (fx.tid_bang, fx.tid_ne)

    # Path A: "!=" one token
    s_end = trie.step(0, fx.tid_ne)
    assert s_end == trie.accept_state

    # Path B: "!" then "="
    s1 = trie.step(0, fx.tid_bang)
    assert s1 == 1
    assert trie.allowed_token_ids(s1) == (fx.tid_eq,)
    s_end2 = trie.step(s1, fx.tid_eq)
    assert s_end2 == trie.accept_state
