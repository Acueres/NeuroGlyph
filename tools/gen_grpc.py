import re

from importlib.resources import files, as_file
from grpc_tools import protoc
from pathlib import Path

OUT_DIR = Path("./src/proto")


def _ensure_pkg():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "__init__.py").touch(exist_ok=True)


def _fix_relative_imports():
    pat = re.compile(r"^import (\w+_pb2) as (\w+)\s*$", re.MULTILINE)

    for py in OUT_DIR.glob("*.py"):
        txt = py.read_text(encoding="utf-8")
        new = pat.sub(r"from . import \1 as \2", txt)
        if new != txt:
            py.write_text(new, encoding="utf-8")


def main() -> None:
    _ensure_pkg()

    bundled = files("grpc_tools") / "_proto"
    with as_file(bundled) as bundled_path:
        args = [
            "grpc_tools.protoc",
            "-I",
            ".",
            "-I",
            str(bundled_path),
            f"--python_out={OUT_DIR}",
            f"--grpc_python_out={OUT_DIR}",
            "./compiler.proto",
        ]

        rc = protoc.main(args)
        if rc != 0:
            raise SystemExit(rc)

    _fix_relative_imports()
