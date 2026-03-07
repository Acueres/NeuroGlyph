import json

from dataclasses import dataclass
from typing import Any, Mapping


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _req(d: Mapping[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required field '{key}' in {ctx}.")
    return d[key]


def _as_dict(v: Any, ctx: str) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise TypeError(f"{ctx} must be an object/dict.")
    return v


def _as_list(v: Any, ctx: str) -> list:
    if not isinstance(v, list):
        raise TypeError(f"{ctx} must be a list.")
    return v


def _as_str(v: Any, ctx: str) -> str:
    if not isinstance(v, str):
        raise TypeError(f"{ctx} must be a string.")
    return v


def _as_int(v: Any, ctx: str) -> int:
    if not _is_int(v):
        raise TypeError(f"{ctx} must be an int.")
    return v


def _as_int_tuple(v: Any, ctx: str) -> tuple[int, ...]:
    arr = _as_list(v, ctx)
    out = []
    for i, item in enumerate(arr):
        out.append(_as_int(item, f"{ctx}[{i}]"))
    return tuple(out)


def _as_str_tuple(v: Any, ctx: str) -> tuple[str, ...]:
    arr = _as_list(v, ctx)
    out = []
    for i, item in enumerate(arr):
        out.append(_as_str(item, f"{ctx}[{i}]"))
    return tuple(out)


# DTOs


@dataclass(frozen=True, slots=True)
class TokenInfo:
    id: int
    name: str

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "TokenInfo":
        ctx = "TokenInfo"
        dd = _as_dict(d, ctx)
        return TokenInfo(
            id=_as_int(_req(dd, "id", ctx), f"{ctx}.id"),
            name=_as_str(_req(dd, "name", ctx), f"{ctx}.name"),
        )


@dataclass(frozen=True, slots=True)
class FixedToken:
    id: int
    name: str
    literal: str

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "FixedToken":
        ctx = "FixedToken"
        dd = _as_dict(d, ctx)
        return FixedToken(
            id=_as_int(_req(dd, "id", ctx), f"{ctx}.id"),
            name=_as_str(_req(dd, "name", ctx), f"{ctx}.name"),
            literal=_as_str(_req(dd, "literal", ctx), f"{ctx}.literal"),
        )


@dataclass(frozen=True, slots=True)
class Trivia:
    whitespace_chars: tuple[str, ...]
    newline_char: str
    line_comment_start: str

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Trivia":
        ctx = "Trivia"
        dd = _as_dict(d, ctx)
        return Trivia(
            whitespace_chars=_as_str_tuple(
                _req(dd, "whitespaceChars", ctx), f"{ctx}.whitespaceChars"
            ),
            newline_char=_as_str(_req(dd, "newlineChar", ctx), f"{ctx}.newlineChar"),
            line_comment_start=_as_str(
                _req(dd, "lineCommentStart", ctx), f"{ctx}.lineCommentStart"
            ),
        )


@dataclass(frozen=True, slots=True)
class LexerState:
    id: int
    accepting: bool

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "LexerState":
        ctx = "LexerState"
        dd = _as_dict(d, ctx)
        acc = _req(dd, "accepting", ctx)
        if not isinstance(acc, bool):
            raise TypeError(f"{ctx}.accepting must be a bool.")
        return LexerState(
            id=_as_int(_req(dd, "id", ctx), f"{ctx}.id"),
            accepting=acc,
        )


@dataclass(frozen=True, slots=True)
class LexerTransition:
    from_state_id: int
    predicate: str
    to_state_id: int

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "LexerTransition":
        ctx = "LexerTransition"
        dd = _as_dict(d, ctx)
        return LexerTransition(
            from_state_id=_as_int(_req(dd, "fromStateId", ctx), f"{ctx}.fromStateId"),
            predicate=_as_str(_req(dd, "predicate", ctx), f"{ctx}.predicate"),
            to_state_id=_as_int(_req(dd, "toStateId", ctx), f"{ctx}.toStateId"),
        )


@dataclass(frozen=True, slots=True)
class LexerMachine:
    token_kind_id: int
    start_state_id: int
    states: tuple[LexerState, ...]
    transitions: tuple[LexerTransition, ...]
    max_lexeme_chars: int

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "LexerMachine":
        ctx = "LexerMachine"
        dd = _as_dict(d, ctx)

        states_raw = _as_list(_req(dd, "states", ctx), f"{ctx}.states")
        trans_raw = _as_list(_req(dd, "transitions", ctx), f"{ctx}.transitions")

        return LexerMachine(
            token_kind_id=_as_int(_req(dd, "tokenKindId", ctx), f"{ctx}.tokenKindId"),
            start_state_id=_as_int(
                _req(dd, "startStateId", ctx), f"{ctx}.startStateId"
            ),
            states=tuple(
                LexerState.from_dict(_as_dict(x, f"{ctx}.states[{i}]"))
                for i, x in enumerate(states_raw)
            ),
            transitions=tuple(
                LexerTransition.from_dict(_as_dict(x, f"{ctx}.transitions[{i}]"))
                for i, x in enumerate(trans_raw)
            ),
            max_lexeme_chars=_as_int(_req(dd, "maxLexemeChars", ctx), f"{ctx}.maxLexemeChars")
        )


@dataclass(frozen=True, slots=True)
class LanguageSpec:
    spec_version: str
    grammar_ebnf: str
    grammar_prompt: str
    tokens: tuple[TokenInfo, ...]
    fixed_tokens: tuple[FixedToken, ...]
    root_tokens: tuple[FixedToken, ...]
    lexer_machines: tuple[LexerMachine, ...]
    trivia: Trivia
    ignored_token_kind_ids: tuple[int, ...]

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "LanguageSpec":
        ctx = "LanguageSpec"
        dd = _as_dict(d, ctx)

        tokens_raw = _as_list(_req(dd, "tokens", ctx), f"{ctx}.tokens")
        fixed_raw = _as_list(_req(dd, "fixedTokens", ctx), f"{ctx}.fixedTokens")
        root_raw = _as_list(_req(dd, "rootTokens", ctx), f"{ctx}.rootTokens")
        machines_raw = _as_list(_req(dd, "lexerMachines", ctx), f"{ctx}.lexerMachines")

        return LanguageSpec(
            spec_version=_as_str(_req(dd, "specVersion", ctx), f"{ctx}.specVersion"),
            grammar_ebnf=_as_str(_req(dd, "grammarEbnf", ctx), f"{ctx}.grammarEbnf"),
            grammar_prompt=_as_str(_req(dd, "grammarPrompt", ctx), f"{ctx}.grammarPrompt"),
            tokens=tuple(
                TokenInfo.from_dict(_as_dict(x, f"{ctx}.tokens[{i}]"))
                for i, x in enumerate(tokens_raw)
            ),
            fixed_tokens=tuple(
                FixedToken.from_dict(_as_dict(x, f"{ctx}.fixedTokens[{i}]"))
                for i, x in enumerate(fixed_raw)
            ),
            root_tokens=tuple(
                FixedToken.from_dict(_as_dict(x, f"{ctx}.rootTokens[{i}]"))
                for i, x in enumerate(root_raw)
            ),
            lexer_machines=tuple(
                LexerMachine.from_dict(_as_dict(x, f"{ctx}.lexerMachines[{i}]"))
                for i, x in enumerate(machines_raw)
            ),
            trivia=Trivia.from_dict(_as_dict(_req(dd, "trivia", ctx), f"{ctx}.trivia")),
            ignored_token_kind_ids=_as_int_tuple(
                _req(dd, "ignoredTokenKindIds", ctx), f"{ctx}.ignoredTokenKindIds"
            ),
        )

    @staticmethod
    def from_json_bytes(data: bytes) -> "LanguageSpec":
        raw = json.loads(data)
        if not isinstance(raw, dict):
            raise TypeError("Language spec JSON must decode to an object/dict.")
        return LanguageSpec.from_dict(raw)
