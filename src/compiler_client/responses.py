from dataclasses import dataclass
from typing import Any
from .language_spec import LanguageSpec


@dataclass(frozen=True)
class LanguageSpecResponse:
    language: str
    version: str
    hash: str
    spec: LanguageSpec

    @staticmethod
    def from_proto(reply: Any) -> "LanguageSpecResponse":
        """
        `reply` is the generated protobuf `LanguageSpecReply` message:
          - language: str
          - version: str
          - hash: str
          - json: bytes
        """
        spec = LanguageSpec.from_json_bytes(reply.json)
        return LanguageSpecResponse(
            language=str(reply.language),
            version=str(reply.version),
            hash=str(reply.hash),
            spec=spec,
        )


@dataclass(frozen=True)
class PredictResponse:
    expected_token_kind_ids: list[int]
    can_terminate_statement: bool
    can_end_input: bool
    semantic_symbol_context: bool
    root_start: bool


@dataclass(frozen=True)
class SemanticHintsResponse:
    preferred_lexemes: list[str]


@dataclass(frozen=True)
class AnalyzeInputResponse:
    ok: bool
    syntax_errors_number: int
    parse_errors_number: int
    semantic_errors_number: int


@dataclass(frozen=True)
class EvaluateInputResponse:
    ok: bool
    output: str
