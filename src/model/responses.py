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
