import grpc
import proto.compiler_pb2_grpc

from google.protobuf.empty_pb2 import Empty
from typing import Optional
from .requests import PredictRequest
from .responses import (
    LanguageSpecResponse,
    PredictResponse,
    SemanticHintsResponse,
    CheckSyntaxResponse,
)


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def fetch_language_spec(
    target: str,
    root_cert_pem: Optional[str] = None,
    timeout_s: float = 5.0,
) -> LanguageSpecResponse:
    creds = grpc.ssl_channel_credentials(
        root_certificates=_read_bytes(root_cert_pem) if root_cert_pem else None
    )

    options = [
        ("grpc.keepalive_time_ms", 30_000),
        ("grpc.keepalive_timeout_ms", 10_000),
        ("grpc.http2.max_pings_without_data", 0),
    ]

    with grpc.secure_channel(target, creds, options=options) as channel:
        stub = proto.compiler_pb2_grpc.CompilerServiceStub(channel)
        reply = stub.GetLanguageSpec(Empty(), timeout=timeout_s)

    language_spec_response = LanguageSpecResponse.from_proto(reply)
    return language_spec_response


def fetch_expected(
    target: str,
    text: str,
    root_cert_pem: Optional[str] = None,
    timeout_s: float = 5.0,
) -> PredictResponse:
    creds = grpc.ssl_channel_credentials(
        root_certificates=_read_bytes(root_cert_pem) if root_cert_pem else None
    )

    options = [
        ("grpc.keepalive_time_ms", 30_000),
        ("grpc.keepalive_timeout_ms", 10_000),
        ("grpc.http2.max_pings_without_data", 0),
    ]

    with grpc.secure_channel(target, creds, options=options) as channel:
        stub = proto.compiler_pb2_grpc.CompilerServiceStub(channel)
        request = PredictRequest(text=text)
        reply = stub.PredictNext(request.to_proto(), timeout=timeout_s)

    return PredictResponse(
        expected_token_kind_ids=reply.expected_token_kind_ids,
        can_terminate_statement=reply.can_terminate_statement,
        can_end_input=reply.can_end_input,
        semantic_symbol_context=reply.semantic_symbol_context,
        root_start=reply.root_start,
    )


def fetch_semantic_hints(
    target: str,
    root_cert_pem: Optional[str] = None,
    timeout_s: float = 5.0,
) -> SemanticHintsResponse:

    creds = grpc.ssl_channel_credentials(
        root_certificates=_read_bytes(root_cert_pem) if root_cert_pem else None
    )

    options = [
        ("grpc.keepalive_time_ms", 30_000),
        ("grpc.keepalive_timeout_ms", 10_000),
        ("grpc.http2.max_pings_without_data", 0),
    ]

    with grpc.secure_channel(target, creds, options=options) as channel:
        stub = proto.compiler_pb2_grpc.CompilerServiceStub(channel)
        reply = stub.GetSemanticHints(Empty(), timeout=timeout_s)

    return SemanticHintsResponse(preferred_lexemes=reply.preferred_lexemes)


def fetch_parse_ok(
    target: str,
    text: str,
    root_cert_pem: Optional[str] = None,
    timeout_s: float = 5.0,
) -> CheckSyntaxResponse:
    creds = grpc.ssl_channel_credentials(
        root_certificates=_read_bytes(root_cert_pem) if root_cert_pem else None
    )

    options = [
        ("grpc.keepalive_time_ms", 30_000),
        ("grpc.keepalive_timeout_ms", 10_000),
        ("grpc.http2.max_pings_without_data", 0),
    ]

    with grpc.secure_channel(target, creds, options=options) as channel:
        stub = proto.compiler_pb2_grpc.CompilerServiceStub(channel)
        request = PredictRequest(text=text)
        reply = stub.CheckSyntax(request.to_proto(), timeout=timeout_s)

    response = CheckSyntaxResponse(
        ok=reply.ok,
        syntax_errors_number=reply.syntax_errors_number,
        parse_errors_number=reply.parse_errors_number,
    )

    return response
