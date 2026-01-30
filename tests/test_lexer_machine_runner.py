import pytest

from dataclasses import dataclass
from src.compiler_client.language_spec import (
    LexerMachine,
    LexerState,
    LexerTransition,
)
from src.syntax_engine.lexer_machine_runner import LexerMachineRunner


# =========================
# Typed fixture
# =========================


@dataclass(frozen=True, slots=True)
class LexerMachines:
    string_single_line: LexerMachine
    string_multiline: LexerMachine
    identifier: LexerMachine
    integer: LexerMachine
    real: LexerMachine


def _single_line_string_machine() -> LexerMachine:
    states = (
        LexerState(0, False),
        LexerState(1, False),
        LexerState(2, False),
        LexerState(3, True),
        LexerState(4, False),
        LexerState(5, False),
        LexerState(6, True),
    )

    re_in_single = r"re:^(?!\n|'|\\).$"
    re_in_double = r're:^(?!\n|"|\\).$'
    re_esc_payload = r"re:^(?!\n).$"

    tr = (
        # Start
        LexerTransition(0, "lit:'", 1),
        LexerTransition(0, 'lit:"', 4),
        # Single-quote branch
        LexerTransition(1, "lit:\\", 2),
        LexerTransition(1, "lit:'", 3),
        LexerTransition(1, re_in_single, 1),
        LexerTransition(2, re_esc_payload, 1),
        # Double-quote branch
        LexerTransition(4, "lit:\\", 5),
        LexerTransition(4, 'lit:"', 6),
        LexerTransition(4, re_in_double, 4),
        LexerTransition(5, re_esc_payload, 4),
    )

    return LexerMachine(
        token_kind_id=1001,
        start_state_id=0,
        states=states,
        transitions=tr,
    )


def _multiline_string_machine() -> LexerMachine:
    states = (
        LexerState(0, False),
        LexerState(1, False),
        LexerState(2, False),
        LexerState(3, False),
        LexerState(4, False),
        LexerState(5, False),
        LexerState(6, True),
        LexerState(7, False),
        LexerState(8, False),
        LexerState(9, False),
        LexerState(10, False),
        LexerState(11, False),
        LexerState(12, True),
    )

    tr = (
        # Start triple single: '''
        LexerTransition(0, "lit:'", 1),
        LexerTransition(1, "lit:'", 2),
        LexerTransition(2, "lit:'", 3),
        # InTripleSingle content
        LexerTransition(3, "lit:'", 4),
        LexerTransition(3, "*", 3),
        # Closing detector for single quotes
        LexerTransition(4, "lit:'", 5),
        LexerTransition(4, "*", 3),
        LexerTransition(5, "lit:'", 6),
        LexerTransition(5, "*", 3),
        # Start triple double: """
        LexerTransition(0, 'lit:"', 7),
        LexerTransition(7, 'lit:"', 8),
        LexerTransition(8, 'lit:"', 9),
        # InTripleDouble content
        LexerTransition(9, 'lit:"', 10),
        LexerTransition(9, "*", 9),
        # Closing detector for double quotes
        LexerTransition(10, 'lit:"', 11),
        LexerTransition(10, "*", 9),
        LexerTransition(11, 'lit:"', 12),
        LexerTransition(11, "*", 9),
    )

    return LexerMachine(
        token_kind_id=1002,
        start_state_id=0,
        states=states,
        transitions=tr,
    )


def _identifier_machine() -> LexerMachine:
    start = r"re:^(?:\p{L}|_|@)$"
    cont = r"re:^(?:\p{L}|[0-9]|_)$"

    states = (
        LexerState(0, False),
        LexerState(1, True),
    )

    tr = (
        LexerTransition(0, start, 1),
        LexerTransition(1, cont, 1),
    )

    return LexerMachine(
        token_kind_id=1003,
        start_state_id=0,
        states=states,
        transitions=tr,
    )


def _integer_machine() -> LexerMachine:
    digit = r"re:^[0-9]$"

    states = (
        LexerState(0, False),
        LexerState(1, True),
    )

    tr = (
        LexerTransition(0, digit, 1),
        LexerTransition(1, digit, 1),
    )

    return LexerMachine(
        token_kind_id=1004,
        start_state_id=0,
        states=states,
        transitions=tr,
    )


def _real_machine() -> LexerMachine:
    digit = r"re:^[0-9]$"
    dot = "lit:."

    states = (
        LexerState(0, False),
        LexerState(1, False),
        LexerState(2, False),
        LexerState(3, True),
        LexerState(4, False),
    )

    tr = (
        # Start
        LexerTransition(0, digit, 1),
        LexerTransition(0, dot, 4),
        # Int part
        LexerTransition(1, digit, 1),
        LexerTransition(1, dot, 2),
        # After dot
        LexerTransition(2, digit, 3),
        LexerTransition(4, digit, 3),
        # Frac part
        LexerTransition(3, digit, 3),
    )

    return LexerMachine(
        token_kind_id=1005,
        start_state_id=0,
        states=states,
        transitions=tr,
    )


@pytest.fixture(scope="session")
def lexer_machines() -> LexerMachines:
    return LexerMachines(
        string_single_line=_single_line_string_machine(),
        string_multiline=_multiline_string_machine(),
        identifier=_identifier_machine(),
        integer=_integer_machine(),
        real=_real_machine(),
    )


# =========================
# Runner semantics tests
# =========================


def test_transition_order_is_first_match_priority() -> None:
    m_a = LexerMachine(
        token_kind_id=9000,
        start_state_id=0,
        states=(LexerState(0, False), LexerState(1, False), LexerState(2, True)),
        transitions=(
            LexerTransition(0, "*", 1),
            LexerTransition(0, "lit:a", 2),
        ),
    )
    m_b = LexerMachine(
        token_kind_id=9001,
        start_state_id=0,
        states=(LexerState(0, False), LexerState(1, False), LexerState(2, True)),
        transitions=(
            LexerTransition(0, "lit:a", 2),
            LexerTransition(0, "*", 1),
        ),
    )

    assert LexerMachineRunner(m_a).accepts("a") is False
    assert LexerMachineRunner(m_b).accepts("a") is True


# =========================
# String machines tests
# =========================


def test_single_line_accepts_basic_single_quotes(lexer_machines: LexerMachines) -> None:
    assert (
        LexerMachineRunner(lexer_machines.string_single_line).accepts("'abc'") is True
    )


def test_single_line_accepts_basic_double_quotes(lexer_machines: LexerMachines) -> None:
    assert (
        LexerMachineRunner(lexer_machines.string_single_line).accepts('"abc"') is True
    )


def test_single_line_rejects_newline(lexer_machines: LexerMachines) -> None:
    assert (
        LexerMachineRunner(lexer_machines.string_single_line).accepts("'a\nb'") is False
    )


def test_single_line_accepts_escape_sequences(lexer_machines: LexerMachines) -> None:
    r = LexerMachineRunner(lexer_machines.string_single_line)
    assert r.accepts("'a\\'b'") is True  # escaped quote
    assert r.accepts('"a\\"b"') is True  # escaped quote
    assert r.accepts("'a\\\\b'") is True  # escaped backslash
    assert r.accepts("'a\\nb'") is True  # escape payload is 'n' (not newline)


def test_single_line_accepts_empty_string(lexer_machines: LexerMachines) -> None:
    r = LexerMachineRunner(lexer_machines.string_single_line)
    assert r.accepts("''") is True
    assert r.accepts('""') is True


def test_single_line_rejects_backslash_newline_escape(
    lexer_machines: LexerMachines,
) -> None:
    # Escape payload cannot be an actual newline
    assert (
        LexerMachineRunner(lexer_machines.string_single_line).accepts("'a\\\nb'")
        is False
    )


def test_single_line_rejects_unescaped_quote_in_content(
    lexer_machines: LexerMachines,
) -> None:
    # This would terminate early and leave extra input => must be rejected as one token
    assert (
        LexerMachineRunner(lexer_machines.string_single_line).accepts("'a'b'") is False
    )


def test_single_line_rejects_unterminated(lexer_machines: LexerMachines) -> None:
    r = LexerMachineRunner(lexer_machines.string_single_line)
    assert r.accepts("'abc") is False
    assert r.accepts('"abc') is False


def test_multi_line_accepts_triple_single_quotes_with_newlines(
    lexer_machines: LexerMachines,
) -> None:
    assert (
        LexerMachineRunner(lexer_machines.string_multiline).accepts("'''a\nb'''")
        is True
    )


def test_multi_line_accepts_triple_double_quotes_with_newlines(
    lexer_machines: LexerMachines,
) -> None:
    assert (
        LexerMachineRunner(lexer_machines.string_multiline).accepts('"""a\nb"""')
        is True
    )


def test_multi_line_rejects_unterminated(lexer_machines: LexerMachines) -> None:
    r = LexerMachineRunner(lexer_machines.string_multiline)
    assert r.accepts("'''a\nb''") is False
    assert r.accepts('"""a\nb""') is False


def test_multi_line_handles_quotes_inside_content(
    lexer_machines: LexerMachines,
) -> None:
    r = LexerMachineRunner(lexer_machines.string_multiline)
    assert r.accepts("'''a''x'''") is True
    assert r.accepts("'''a'x'''") is True
    assert r.accepts('"""a""x"""') is True
    assert r.accepts('"""a"x"""') is True


def test_multi_line_accepts_empty(lexer_machines: LexerMachines) -> None:
    r = LexerMachineRunner(lexer_machines.string_multiline)
    assert r.accepts("''''''") is True
    assert r.accepts('""""""') is True


def test_multi_line_rejects_embedded_closing_sequence(
    lexer_machines: LexerMachines,
) -> None:
    # If content contains a full closing triple-quote, the machine closes there,
    # so extra input remains => reject as whole-token match.
    r = LexerMachineRunner(lexer_machines.string_multiline)
    assert r.accepts("'''a'''x'''") is False
    assert r.accepts('"""a"""x"""') is False


# =========================
# Identifier machine tests
# =========================


def test_identifier_accepts_simple_ascii(lexer_machines: LexerMachines) -> None:
    r = LexerMachineRunner(lexer_machines.identifier)
    assert r.accepts("a") is True
    assert r.accepts("abc") is True
    assert r.accepts("a_b") is True
    assert r.accepts("abc123") is True


def test_identifier_accepts_underscore_and_at_start(
    lexer_machines: LexerMachines,
) -> None:
    r = LexerMachineRunner(lexer_machines.identifier)
    assert r.accepts("_") is True
    assert r.accepts("_x") is True
    assert r.accepts("@") is True
    assert r.accepts("@x") is True
    assert r.accepts("@1") is True  # digits allowed in continuation


def test_identifier_accepts_unicode_letters(lexer_machines: LexerMachines) -> None:
    r = LexerMachineRunner(lexer_machines.identifier)
    assert r.accepts("Привет") is True
    assert r.accepts("λx") is True
    assert r.accepts("á") is True  # precomposed letter (category L)


def test_identifier_rejects_starting_with_digit(lexer_machines: LexerMachines) -> None:
    r = LexerMachineRunner(lexer_machines.identifier)
    assert r.accepts("1") is False
    assert r.accepts("1abc") is False


def test_identifier_rejects_invalid_characters(lexer_machines: LexerMachines) -> None:
    r = LexerMachineRunner(lexer_machines.identifier)
    assert r.accepts("a-b") is False
    assert r.accepts("a b") is False
    assert r.accepts("a.") is False
    assert r.accepts("") is False


def test_identifier_rejects_at_in_continuation(lexer_machines: LexerMachines) -> None:
    r = LexerMachineRunner(lexer_machines.identifier)
    assert r.accepts("@@") is False
    assert r.accepts("a@") is False


def test_identifier_rejects_combining_mark_sequence(
    lexer_machines: LexerMachines,
) -> None:
    # 'a' + COMBINING ACUTE ACCENT (category M, not allowed by cont pattern)
    assert LexerMachineRunner(lexer_machines.identifier).accepts("a\u0301") is False


# =========================
# Integer machine tests
# =========================


def test_integer_accepts_digits(lexer_machines: LexerMachines) -> None:
    r = LexerMachineRunner(lexer_machines.integer)
    assert r.accepts("0") is True
    assert r.accepts("123") is True
    assert r.accepts("007") is True


def test_integer_rejects_non_digits(lexer_machines: LexerMachines) -> None:
    r = LexerMachineRunner(lexer_machines.integer)
    assert r.accepts("") is False
    assert r.accepts("1.0") is False
    assert r.accepts("-1") is False
    assert r.accepts("1e2") is False
    assert r.accepts("12_34") is False


def test_integer_rejects_non_ascii_digits(lexer_machines: LexerMachines) -> None:
    # Pattern is ASCII [0-9] only.
    assert (
        LexerMachineRunner(lexer_machines.integer).accepts("١٢٣") is False
    )  # Arabic-Indic digits


# =========================
# Real machine tests
# =========================


def test_real_accepts_decimal_forms(lexer_machines: LexerMachines) -> None:
    r = LexerMachineRunner(lexer_machines.real)
    assert r.accepts("0.0") is True
    assert r.accepts("12.34") is True
    assert r.accepts(".5") is True
    assert r.accepts(".0001") is True
    assert r.accepts("00.1") is True  # leading zeros allowed


def test_real_rejects_integer_only_and_incomplete_decimal(
    lexer_machines: LexerMachines,
) -> None:
    r = LexerMachineRunner(lexer_machines.real)
    assert r.accepts("12") is False  # must have '.' + frac
    assert r.accepts("12.") is False  # must have at least one frac digit
    assert r.accepts(".") is False  # must have at least one digit


def test_real_rejects_scientific_notation_and_multiple_dots(
    lexer_machines: LexerMachines,
) -> None:
    r = LexerMachineRunner(lexer_machines.real)
    assert r.accepts("1e2") is False
    assert r.accepts("1.2e3") is False
    assert r.accepts("1.2.3") is False
    assert r.accepts("..1") is False
