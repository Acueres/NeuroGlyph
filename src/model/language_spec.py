import json
import re

from dataclasses import dataclass
from typing import Any, Mapping, Optional


_first_cap_re = re.compile(r"(.)([A-Z][a-z]+)")
_all_cap_re = re.compile(r"([a-z0-9])([A-Z])")

_SNAKE_KEY_OVERRIDES = {
    "continue": "continue_",
}


def to_snake(name: str) -> str:
    """
    Convert PascalCase or camelCase to snake_case.
    Leaves existing snake_case mostly as-is.
    """
    if not name:
        return name

    # Normalize separators
    name = name.replace("-", "_")

    if "_" in name and not any(ch.isupper() for ch in name):
        return name

    s1 = _first_cap_re.sub(r"\1_\2", name)
    s2 = _all_cap_re.sub(r"\1_\2", s1)
    return s2.lower()


def normalize_key(key: str) -> str:
    sk = to_snake(key)
    return _SNAKE_KEY_OVERRIDES.get(sk, sk)


def snake_case_keys(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {normalize_key(str(k)): snake_case_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [snake_case_keys(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(snake_case_keys(x) for x in obj)
    return obj


@dataclass(frozen=True)
class IdentifierCharSet:
    allowed_categories: tuple[str, ...]

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "IdentifierCharSet":
        return IdentifierCharSet(
            allowed_categories=tuple(d["allowed_categories"]),
        )


@dataclass(frozen=True)
class IdentifierSpec:
    start: IdentifierCharSet
    continue_: IdentifierCharSet
    keywords_are_reserved: bool

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "IdentifierSpec":
        return IdentifierSpec(
            start=IdentifierCharSet.from_dict(d["start"]),
            continue_=IdentifierCharSet.from_dict(d["continue_"]),
            keywords_are_reserved=bool(d["keywords_are_reserved"]),
        )


@dataclass(frozen=True)
class NumberSpec:
    digits: str
    allow_leading_dot_real: bool
    require_digit_after_dot_if_has_leading_digits: bool
    allow_trailing_dot_real: bool
    allow_exponent: bool
    allow_underscore_separators: bool
    integer_token_id: int
    integer_token_name: str
    real_token_id: int
    real_token_name: str

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "NumberSpec":
        return NumberSpec(
            digits=str(d["digits"]),
            allow_leading_dot_real=bool(d["allow_leading_dot_real"]),
            require_digit_after_dot_if_has_leading_digits=bool(
                d["require_digit_after_dot_if_has_leading_digits"]
            ),
            allow_trailing_dot_real=bool(d["allow_trailing_dot_real"]),
            allow_exponent=bool(d["allow_exponent"]),
            allow_underscore_separators=bool(d["allow_underscore_separators"]),
            integer_token_id=int(d["integer_token_id"]),
            integer_token_name=str(d["integer_token_name"]),
            real_token_id=int(d["real_token_id"]),
            real_token_name=str(d["real_token_name"]),
        )


@dataclass(frozen=True)
class StringSpec:
    quote_chars: tuple[str, ...]
    triple_quote_enabled: bool
    multi_line_requires_triple_quote: bool
    escape_mode: str
    allows_newline_in_single_line_string: bool
    token_id: int
    token_name: str

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "StringSpec":
        return StringSpec(
            quote_chars=tuple(d["quote_chars"]),
            triple_quote_enabled=bool(d["triple_quote_enabled"]),
            multi_line_requires_triple_quote=bool(
                d["multi_line_requires_triple_quote"]
            ),
            escape_mode=str(d["escape_mode"]),
            allows_newline_in_single_line_string=bool(
                d["allows_newline_in_single_line_string"]
            ),
            token_id=int(d["token_id"]),
            token_name=str(d["token_name"]),
        )


@dataclass(frozen=True)
class TriviaSpec:
    whitespace_chars: tuple[str, ...]
    newline: str
    line_comment_start: str
    line_comment_ends_at_newline: bool

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "TriviaSpec":
        return TriviaSpec(
            whitespace_chars=tuple(d["whitespace_chars"]),
            newline=str(d["newline"]),
            line_comment_start=str(d["line_comment_start"]),
            line_comment_ends_at_newline=bool(d["line_comment_ends_at_newline"]),
        )


@dataclass(frozen=True)
class ParserLayoutException:
    allow_missing_terminator_after_initializer_if_next_token_on_new_line: bool
    optional_terminator_token_id: int
    optional_terminator_token_name: str

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ParserLayoutException":
        return ParserLayoutException(
            allow_missing_terminator_after_initializer_if_next_token_on_new_line=bool(
                d[
                    "allow_missing_terminator_after_initializer_if_next_token_on_new_line"
                ]
            ),
            optional_terminator_token_id=int(d["optional_terminator_token_id"]),
            optional_terminator_token_name=(
                str(d["optional_termininator_token_name"])
                if "optional_termininator_token_name" in d  # defensive for typos
                else str(d["optional_terminator_token_name"])
            ),
        )


@dataclass(frozen=True)
class AsiSpec:
    enabled: bool
    virtual_terminator_token_id: int
    virtual_terminator_token_name: str
    newline_token_id: int
    newline_token_name: str
    drops_newline_tokens: bool
    no_insert_after_token_ids: tuple[int, ...]
    no_insert_after_token_names: tuple[str, ...]
    continuation_before_token_ids: tuple[int, ...]
    continuation_before_token_names: tuple[str, ...]
    parser_exceptions: ParserLayoutException

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "AsiSpec":
        return AsiSpec(
            enabled=bool(d["enabled"]),
            virtual_terminator_token_id=int(d["virtual_terminator_token_id"]),
            virtual_terminator_token_name=str(d["virtual_terminator_token_name"]),
            newline_token_id=int(d["newline_token_id"]),
            newline_token_name=str(d["newline_token_name"]),
            drops_newline_tokens=bool(d["drops_newline_tokens"]),
            no_insert_after_token_ids=tuple(
                int(x) for x in d["no_insert_after_token_ids"]
            ),
            no_insert_after_token_names=tuple(
                str(x) for x in d["no_insert_after_token_names"]
            ),
            continuation_before_token_ids=tuple(
                int(x) for x in d["continuation_before_token_ids"]
            ),
            continuation_before_token_names=tuple(
                str(x) for x in d["continuation_before_token_names"]
            ),
            parser_exceptions=ParserLayoutException.from_dict(d["parser_exceptions"]),
        )


@dataclass(frozen=True)
class TokenInfo:
    id: int
    name: str
    category: str
    spellings: tuple[str, ...]
    is_lexable: bool
    is_synthetic: bool
    is_trivia: bool
    may_be_virtual: bool

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "TokenInfo":
        return TokenInfo(
            id=int(d["id"]),
            name=str(d["name"]),
            category=str(d["category"]),
            spellings=tuple(d["spellings"]),
            is_lexable=bool(d["is_lexable"]),
            is_synthetic=bool(d["is_synthetic"]),
            is_trivia=bool(d["is_trivia"]),
            may_be_virtual=bool(d["may_be_virtual"]),
        )


@dataclass(frozen=True)
class KeywordEntry:
    text: str
    token_id: int
    token_name: str

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "KeywordEntry":
        return KeywordEntry(
            text=str(d["text"]),
            token_id=int(d["token_id"]),
            token_name=str(d["token_name"]),
        )


@dataclass(frozen=True)
class FixedToken:
    token_id: int
    token_name: str
    spelling: str

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "FixedToken":
        return FixedToken(
            token_id=int(d["token_id"]),
            token_name=str(d["token_name"]),
            spelling=str(d["spelling"]),
        )


@dataclass(frozen=True)
class LanguageSpec:
    spec_version: str
    spec_hash: Optional[str]
    tokens: tuple[TokenInfo, ...]
    keywords: tuple[KeywordEntry, ...]
    fixed_tokens: tuple[FixedToken, ...]
    identifier: IdentifierSpec
    numbers: NumberSpec
    strings: StringSpec
    trivia: TriviaSpec
    asi: AsiSpec

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "LanguageSpec":
        d = snake_case_keys(raw)

        return LanguageSpec(
            spec_version=str(d["spec_version"]),
            spec_hash=(None if d.get("spec_hash") is None else str(d["spec_hash"])),
            tokens=tuple(TokenInfo.from_dict(x) for x in d["tokens"]),
            keywords=tuple(KeywordEntry.from_dict(x) for x in d["keywords"]),
            fixed_tokens=tuple(FixedToken.from_dict(x) for x in d["fixed_tokens"]),
            identifier=IdentifierSpec.from_dict(d["identifier"]),
            numbers=NumberSpec.from_dict(d["numbers"]),
            strings=StringSpec.from_dict(d["strings"]),
            trivia=TriviaSpec.from_dict(d["trivia"]),
            asi=AsiSpec.from_dict(d["asi"]),
        )

    @staticmethod
    def from_json_bytes(data: bytes) -> "LanguageSpec":
        raw = json.loads(data)
        if not isinstance(raw, dict):
            raise TypeError("Language spec JSON must decode to an object/dict.")
        return LanguageSpec.from_dict(raw)
