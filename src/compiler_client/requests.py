import proto.compiler_pb2

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictRequest:
    text: str

    def to_proto(self) -> proto.compiler_pb2.PredictRequest:
        return proto.compiler_pb2.PredictRequest(text=self.text)