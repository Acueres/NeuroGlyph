from typing import Callable, Iterable, Sequence

from syntax.mask_engine import MaskEngine, FixedTokenTable


class SemanticHintsCache:
    def __init__(
        self,
        engine: MaskEngine,
        preferred_type_lexemes: Sequence[str],
    ) -> None:
        self._engine = engine
        self._preferred_type_lexemes = preferred_type_lexemes
        self._lexeme_tables: dict[str, FixedTokenTable] = {}

    def ensure_tables(self, lexemes: Iterable[str]) -> list[FixedTokenTable]:
        out: list[FixedTokenTable] = []
        for lex in lexemes:
            lex = str(lex).strip()
            if not lex:
                continue
            t = self._lexeme_tables.get(lex)
            if t is None:
                t = self._engine.compile_lexeme_table(lex)
                self._lexeme_tables[lex] = t
            out.append(t)
        return out

    def apply_type_hints(self) -> None:
        tables = self.ensure_tables(self._preferred_type_lexemes)
        if tables:
            self._engine.add_semantic_lexeme_tables(tables)
