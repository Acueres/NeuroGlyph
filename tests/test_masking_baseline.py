import pytest

from dataclasses import dataclass

from src.model.language_spec import LanguageSpec
from src.syntax_engine.masking import (
    SpecArtifacts,
    TokenizerIndex,
    SubwordMaskEngine,
    CursorLexState,
)


@dataclass(frozen=True)
class PredictReply:
    expected_token_kind_ids: tuple[int, ...]
    can_terminate_statement_here: bool = False


class MockLanguageService:
    def __init__(
        self,
        spec: LanguageSpec,
        *,
        ids: dict[str, int],
    ):
        self._spec = spec
        self._ids = ids

    def get_language_spec(self) -> LanguageSpec:
        return self._spec

    def predict_next(self, text: str, cursor_index: int) -> PredictReply:
        if cursor_index != len(text):
            return PredictReply(())

        sig = "def add(a: int, b: int)"
        if text.endswith(sig):
            return PredictReply((self._ids["Arrow"], self._ids["BraceLeft"]))

        sig_ret = "def add(a: int, b: int) -> int"
        if text.endswith(sig_ret):
            return PredictReply((self._ids["BraceLeft"],))

        if text.endswith("for i in 1"):
            return PredictReply((self._ids["RangeInclusive"], self._ids["Range"]))

        if text.endswith("Outer."):
            return PredictReply((self._ids["Identifier"],))

        return PredictReply(())


class MockTokenizer:
    """Minimal tokenizer stub with a fixed vocab and decode([id]) behavior."""

    def __init__(self, pieces: list[str]):
        self._pieces = pieces
        self.all_special_ids = []

    @property
    def vocab_size(self) -> int:
        return len(self._pieces)

    def get_vocab(self):
        return {s: i for i, s in enumerate(self._pieces)}

    def decode(
        self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    ):
        assert isinstance(ids, list) and len(ids) == 1
        return self._pieces[int(ids[0])]


# Fixtures
@pytest.fixture()
def mock_spec():
    # Pattern tokens
    TOKEN_IDENTIFIER = 1
    TOKEN_LITERAL_INT = 2
    TOKEN_LITERAL_REAL = 3
    TOKEN_LITERAL_STRING = 4

    # Fixed tokens
    TOKEN_BRACE_LEFT = 10
    TOKEN_BRACE_RIGHT = 11
    TOKEN_PAREN_LEFT = 12
    TOKEN_PAREN_RIGHT = 13
    TOKEN_DOT = 14
    TOKEN_EQUAL = 15

    TOKEN_ARROW = 30
    TOKEN_RANGE = 31
    TOKEN_RANGE_INCL = 32

    # Keyword tokens
    TOKEN_DEF = 200

    ids = {
        "Identifier": TOKEN_IDENTIFIER,
        "LiteralInt": TOKEN_LITERAL_INT,
        "LiteralReal": TOKEN_LITERAL_REAL,
        "LiteralString": TOKEN_LITERAL_STRING,
        "BraceLeft": TOKEN_BRACE_LEFT,
        "BraceRight": TOKEN_BRACE_RIGHT,
        "ParenthesisLeft": TOKEN_PAREN_LEFT,
        "ParenthesisRight": TOKEN_PAREN_RIGHT,
        "Dot": TOKEN_DOT,
        "Equal": TOKEN_EQUAL,
        "Arrow": TOKEN_ARROW,
        "Range": TOKEN_RANGE,
        "RangeInclusive": TOKEN_RANGE_INCL,
        "Def": TOKEN_DEF,
    }

    spec_dict = {
        "specVersion": "0.1.0",
        "specHash": "TEST_HASH",
        "tokens": [
            # Patterns / literals
            {
                "id": TOKEN_IDENTIFIER,
                "name": "Identifier",
                "category": "identifier",
                "spellings": [],
                "isLexable": True,
                "isSynthetic": False,
                "isTrivia": False,
                "mayBeVirtual": False,
            },
            {
                "id": TOKEN_LITERAL_INT,
                "name": "LiteralInt",
                "category": "literal",
                "spellings": [],
                "isLexable": True,
                "isSynthetic": False,
                "isTrivia": False,
                "mayBeVirtual": False,
            },
            {
                "id": TOKEN_LITERAL_REAL,
                "name": "LiteralReal",
                "category": "literal",
                "spellings": [],
                "isLexable": True,
                "isSynthetic": False,
                "isTrivia": False,
                "mayBeVirtual": False,
            },
            {
                "id": TOKEN_LITERAL_STRING,
                "name": "LiteralString",
                "category": "literal",
                "spellings": [],
                "isLexable": True,
                "isSynthetic": False,
                "isTrivia": False,
                "mayBeVirtual": False,
            },
            # Fixed tokens
            {
                "id": TOKEN_BRACE_LEFT,
                "name": "BraceLeft",
                "category": "symbol",
                "spellings": ["{"],
                "isLexable": True,
                "isSynthetic": False,
                "isTrivia": False,
                "mayBeVirtual": False,
            },
            {
                "id": TOKEN_BRACE_RIGHT,
                "name": "BraceRight",
                "category": "symbol",
                "spellings": ["}"],
                "isLexable": True,
                "isSynthetic": False,
                "isTrivia": False,
                "mayBeVirtual": False,
            },
            {
                "id": TOKEN_PAREN_LEFT,
                "name": "ParenthesisLeft",
                "category": "symbol",
                "spellings": ["("],
                "isLexable": True,
                "isSynthetic": False,
                "isTrivia": False,
                "mayBeVirtual": False,
            },
            {
                "id": TOKEN_PAREN_RIGHT,
                "name": "ParenthesisRight",
                "category": "symbol",
                "spellings": [")"],
                "isLexable": True,
                "isSynthetic": False,
                "isTrivia": False,
                "mayBeVirtual": False,
            },
            {
                "id": TOKEN_DOT,
                "name": "Dot",
                "category": "symbol",
                "spellings": ["."],
                "isLexable": True,
                "isSynthetic": False,
                "isTrivia": False,
                "mayBeVirtual": False,
            },
            {
                "id": TOKEN_EQUAL,
                "name": "Equal",
                "category": "operator",
                "spellings": ["="],
                "isLexable": True,
                "isSynthetic": False,
                "isTrivia": False,
                "mayBeVirtual": False,
            },
            {
                "id": TOKEN_ARROW,
                "name": "Arrow",
                "category": "operator",
                "spellings": ["->"],
                "isLexable": True,
                "isSynthetic": False,
                "isTrivia": False,
                "mayBeVirtual": False,
            },
            {
                "id": TOKEN_RANGE,
                "name": "Range",
                "category": "operator",
                "spellings": [".."],
                "isLexable": True,
                "isSynthetic": False,
                "isTrivia": False,
                "mayBeVirtual": False,
            },
            {
                "id": TOKEN_RANGE_INCL,
                "name": "RangeInclusive",
                "category": "operator",
                "spellings": ["..="],
                "isLexable": True,
                "isSynthetic": False,
                "isTrivia": False,
                "mayBeVirtual": False,
            },
            # Keywords
            {
                "id": TOKEN_DEF,
                "name": "Def",
                "category": "keyword",
                "spellings": ["def"],
                "isLexable": True,
                "isSynthetic": False,
                "isTrivia": False,
                "mayBeVirtual": False,
            },
        ],
        "keywords": [
            {"text": "def", "tokenId": TOKEN_DEF, "tokenName": "Def"},
        ],
        "fixedTokens": [
            {"tokenId": TOKEN_BRACE_LEFT, "tokenName": "BraceLeft", "spelling": "{"},
            {"tokenId": TOKEN_BRACE_RIGHT, "tokenName": "BraceRight", "spelling": "}"},
            {
                "tokenId": TOKEN_PAREN_LEFT,
                "tokenName": "ParenthesisLeft",
                "spelling": "(",
            },
            {
                "tokenId": TOKEN_PAREN_RIGHT,
                "tokenName": "ParenthesisRight",
                "spelling": ")",
            },
            {"tokenId": TOKEN_DOT, "tokenName": "Dot", "spelling": "."},
            {"tokenId": TOKEN_EQUAL, "tokenName": "Equal", "spelling": "="},
            {"tokenId": TOKEN_ARROW, "tokenName": "Arrow", "spelling": "->"},
            {"tokenId": TOKEN_RANGE, "tokenName": "Range", "spelling": ".."},
            {
                "tokenId": TOKEN_RANGE_INCL,
                "tokenName": "RangeInclusive",
                "spelling": "..=",
            },
        ],
        "identifier": {
            "start": {"allowedCategories": ["Lu", "Ll", "Lt", "Lm", "Lo", "Pc"]},
            "continue": {
                "allowedCategories": ["Lu", "Ll", "Lt", "Lm", "Lo", "Pc", "Nd"]
            },
            "keywordsAreReserved": True,
        },
        "numbers": {
            "digits": "Ascii",
            "allowLeadingDotReal": True,
            "requireDigitAfterDotIfHasLeadingDigits": True,
            "allowTrailingDotReal": False,
            "allowExponent": False,
            "allowUnderscoreSeparators": False,
            "integerTokenId": TOKEN_LITERAL_INT,
            "integerTokenName": "LiteralInt",
            "realTokenId": TOKEN_LITERAL_REAL,
            "realTokenName": "LiteralReal",
        },
        "strings": {
            "quoteChars": ['"', "'"],
            "tripleQuoteEnabled": True,
            "multiLineRequiresTripleQuote": True,
            "escapeMode": "None",
            "allowsNewlineInSingleLineString": False,
            "tokenId": TOKEN_LITERAL_STRING,
            "tokenName": "LiteralString",
        },
        "trivia": {
            "whitespaceChars": [" ", "\t", "\r"],
            "newline": "\n",
            "lineCommentStart": "#",
            "lineCommentEndsAtNewline": True,
        },
        "asi": {
            "enabled": True,
            "virtualTerminatorTokenId": 300,
            "virtualTerminatorTokenName": "Semicolon",
            "newlineTokenId": 301,
            "newlineTokenName": "Newline",
            "dropsNewlineTokens": True,
            "noInsertAfterTokenIds": [],
            "noInsertAfterTokenNames": [],
            "continuationBeforeTokenIds": [],
            "continuationBeforeTokenNames": [],
            "parserExceptions": {
                "allowMissingTerminatorAfterInitializerIfNextTokenOnNewLine": True,
                "optionalTerminatorTokenId": 302,
                "optionalTerminatorTokenName": "OptionalTerminator",
            },
        },
    }

    spec = LanguageSpec.from_dict(spec_dict)
    return spec, ids


@pytest.fixture()
def engine_svc_tok(mock_spec):
    spec, ids = mock_spec

    pieces = [
        "{",
        " {",
        "}",
        " }",
        "->",
        " ->",
        "-",
        " -",
        ">",
        " >",
        ".",
        " .",
        "..",
        " ..",
        "..=",
        "=",
        " =",
        "def",
        " def",
        "d",
        "de",
        "e",
        "ef",
        "f",
        "x",
        " x",
        "Outer",
        " Outer",
        " ",
        "\n",
        "- >",  # internal trivia inside significant run
    ]

    tok = MockTokenizer(pieces)

    art = SpecArtifacts.from_spec(spec)
    tidx = TokenizerIndex.build(tok, art.classifier)
    engine = SubwordMaskEngine(art, tidx)

    svc = MockLanguageService(spec, ids=ids)
    vocab = tok.get_vocab()

    return engine, svc, tok, vocab, ids


# Tests


class TestMaskingAfterFunctionSignature:
    def test_allows_brace_or_arrow_full_or_prefix(
        self,
        engine_svc_tok: tuple[
            SubwordMaskEngine,
            MockLanguageService,
            MockTokenizer,
            dict[str, int],
            dict[str, int],
        ],
    ):
        engine, svc, tok, vocab, ids = engine_svc_tok

        text = "def add(a: int, b: int)"
        reply = svc.predict_next(text, len(text))
        assert set(reply.expected_token_kind_ids) == {ids["Arrow"], ids["BraceLeft"]}

        state = CursorLexState.initial()
        allowed = set(
            engine.build_allowed_vocab_ids(reply.expected_token_kind_ids, state)
        )
        allowed_texts = {tok.decode([i]) for i in allowed}

        # '{' path
        assert "{" in allowed_texts
        assert " {" in allowed_texts

        # '->' path: full literal OR prefix
        assert "->" in allowed_texts
        assert " ->" in allowed_texts
        assert "-" in allowed_texts
        assert " -" in allowed_texts

        # trivia-only pieces are permitted at boundary
        assert " " in allowed_texts
        assert "\n" in allowed_texts

        # Pieces that contain internal trivia in the significant segment must be rejected
        assert "- >" not in allowed_texts

    def test_arrow_continuation_after_dash_disallows_trivia_and_brace(
        self,
        engine_svc_tok: tuple[
            SubwordMaskEngine,
            MockLanguageService,
            MockTokenizer,
            dict[str, int],
            dict[str, int],
        ],
    ):
        engine, svc, tok, vocab, ids = engine_svc_tok

        expected = (ids["Arrow"], ids["BraceLeft"])
        state0 = CursorLexState.initial()

        # Choose '-' as the first piece of '->'
        dash_id = vocab["-"]
        state1 = engine.advance_state(state0, dash_id, expected)

        allowed1 = set(engine.build_allowed_vocab_ids(expected, state1))
        allowed1_texts = {tok.decode([i]) for i in allowed1}

        # Must allow '>' to complete '->'
        assert ">" in allowed1_texts

        # Must not allow whitespace or '{' in the middle of a fixed token
        assert " " not in allowed1_texts
        assert "\n" not in allowed1_texts
        assert "{" not in allowed1_texts
        assert " {" not in allowed1_texts

    def test_after_arrow_then_type_identifier_starters_allowed(
        self,
        engine_svc_tok: tuple[
            SubwordMaskEngine,
            MockLanguageService,
            MockTokenizer,
            dict[str, int],
            dict[str, int],
        ],
    ):
        engine, svc, tok, vocab, ids = engine_svc_tok

        expected0 = (ids["Arrow"], ids["BraceLeft"])
        state = CursorLexState.initial()

        # Emit '->' in two pieces
        state = engine.advance_state(state, vocab["-"], expected0)
        state = engine.advance_state(state, vocab[">"], expected0)

        expected1 = (ids["Identifier"],)
        allowed = set(engine.build_allowed_vocab_ids(expected1, state))
        allowed_texts = {tok.decode([i]) for i in allowed}

        assert "x" in allowed_texts
        assert " x" in allowed_texts
        assert "Outer" in allowed_texts
        assert " Outer" in allowed_texts

        assert "{" not in allowed_texts


class TestMaskingRangeOperators:
    def test_range_operator_allows_dot_double_dot_and_dot_equal(
        self,
        engine_svc_tok: tuple[
            SubwordMaskEngine,
            MockLanguageService,
            MockTokenizer,
            dict[str, int],
            dict[str, int],
        ],
    ):
        engine, svc, tok, vocab, ids = engine_svc_tok

        # After 'for i in 1' ParseRange checks '..=' then '..'
        expected = (ids["RangeInclusive"], ids["Range"])
        state = CursorLexState.initial()

        allowed = set(engine.build_allowed_vocab_ids(expected, state))
        allowed_texts = {tok.decode([i]) for i in allowed}

        assert "." in allowed_texts
        assert " ." in allowed_texts
        assert ".." in allowed_texts
        assert " .." in allowed_texts
        assert "..=" in allowed_texts

        # trivia-only pieces are OK at boundary
        assert " " in allowed_texts

    def test_range_operator_requires_double_dot_before_equal(
        self,
        engine_svc_tok: tuple[
            SubwordMaskEngine,
            MockLanguageService,
            MockTokenizer,
            dict[str, int],
            dict[str, int],
        ],
    ):
        engine, svc, tok, vocab, ids = engine_svc_tok

        expected = (ids["RangeInclusive"], ids["Range"])
        state0 = CursorLexState.initial()

        # Emit a single '.'; now we're inside a literal where the only valid continuation is '.'
        state1 = engine.advance_state(state0, vocab["."], expected)
        allowed1 = set(engine.build_allowed_vocab_ids(expected, state1))
        allowed1_texts = {tok.decode([i]) for i in allowed1}

        assert "." in allowed1_texts
        assert "=" not in allowed1_texts
        assert "..=" not in allowed1_texts
        assert ".." not in allowed1_texts

        # No whitespace is allowed inside '..=' or '..'
        assert " " not in allowed1_texts

    def test_range_operator_after_double_dot_allows_equal_or_boundary(
        self,
        engine_svc_tok: tuple[
            SubwordMaskEngine,
            MockLanguageService,
            MockTokenizer,
            dict[str, int],
            dict[str, int],
        ],
    ):
        engine, svc, tok, vocab, ids = engine_svc_tok

        expected = (ids["RangeInclusive"], ids["Range"])
        state0 = CursorLexState.initial()

        # Emit '..' as one piece. This completes Range, but can also continue to RangeInclusive.
        state1 = engine.advance_state(state0, vocab[".."], expected)
        allowed1 = set(engine.build_allowed_vocab_ids(expected, state1))
        allowed1_texts = {tok.decode([i]) for i in allowed1}

        # '=' must be allowed to complete '..='
        assert "=" in allowed1_texts

        # Since '..' is already a complete token, whitespace is allowed after it.
        assert " " in allowed1_texts


class TestMaskingKeywords:
    def test_keyword_def_allows_single_piece_or_multi_piece_completion(
        self,
        engine_svc_tok: tuple[
            SubwordMaskEngine,
            MockLanguageService,
            MockTokenizer,
            dict[str, int],
            dict[str, int],
        ],
    ):
        engine, svc, tok, vocab, ids = engine_svc_tok

        expected = (ids["Def"],)
        state0 = CursorLexState.initial()

        allowed0 = set(engine.build_allowed_vocab_ids(expected, state0))
        allowed0_texts = {tok.decode([i]) for i in allowed0}

        assert "def" in allowed0_texts
        assert " def" in allowed0_texts
        assert "d" in allowed0_texts
        assert "de" in allowed0_texts

        # Emit 'd' and verify only 'e*' continuations are allowed, not 'f'
        state1 = engine.advance_state(state0, vocab["d"], expected)
        allowed1 = set(engine.build_allowed_vocab_ids(expected, state1))
        allowed1_texts = {tok.decode([i]) for i in allowed1}

        assert "e" in allowed1_texts
        assert "ef" in allowed1_texts
        assert "f" not in allowed1_texts


class TestKnownGaps:
    def test_identifier_context_rejects_reserved_keyword_exact_match(
        self,
        engine_svc_tok: tuple[
            SubwordMaskEngine,
            MockLanguageService,
            MockTokenizer,
            dict[str, int],
            dict[str, int],
        ],
    ):
        engine, svc, tok, vocab, ids = engine_svc_tok

        # After 'Outer.' parser expects an Identifier.
        expected = (ids["Identifier"],)
        state = CursorLexState.initial()
        allowed = set(engine.build_allowed_vocab_ids(expected, state))
        allowed_texts = {tok.decode([i]) for i in allowed}

        # Keyword "def" should not be accepted as Identifier.
        assert "def" not in allowed_texts
