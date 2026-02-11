import textwrap

from typing import Optional, Sequence
from compiler_client.language_spec import LanguageSpec, FixedToken, Trivia


def _wrap_csv(items: Sequence[str], width: int = 88) -> str:
    if not items:
        return "(none)"
    line = ", ".join(items)
    return "\n".join(
        textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False)
    )


def select_prompt_fixed_tokens(
    fixed_tokens: tuple[FixedToken, ...], trivia: Trivia
) -> tuple[list[str], list[str]]:
    lits: list[str] = []
    seen = set()
    for t in fixed_tokens:
        seen.add(t.literal)
        lits.append(t.literal)

    # Single-char allowlist
    allow_single = set()
    lcs = trivia.line_comment_start

    allow_single.add(lcs)

    keywords = sorted([x for x in lits if x.isidentifier()])
    symbols = sorted(
        [
            x
            for x in lits
            if (not x.isidentifier()) and (len(x) > 1 or x in allow_single)
        ],
        key=lambda s: (-len(s), s),
    )
    return keywords, symbols


def build_system_prompt(
    spec: LanguageSpec,
    *,
    language_name: Optional[str] = None,
    assistant_name: str = "NeuroGlyph",
    code_only: bool = True,
    include_style: bool = True,
    indent_spaces: int = 2,
    max_line_width: int = 88,
    ebnf_grammar: bool = False,
) -> str:
    """
    Construct a language-agnostic system prompt from a LanguageSpec.

    Expected attributes on `spec`:
      - grammar_ebnf: str
      - fixed_tokens: iterable with .literal (and optionally .name)
      - trivia: line comment start
    """
    if ebnf_grammar:
        grammar = spec.grammar_ebnf
    else:
        grammar = spec.grammar_prompt

    fixed_tokens = spec.fixed_tokens

    trivia = spec.trivia
    line_comment_start = trivia.line_comment_start

    # Classify fixed tokens
    literals: list[str] = [t.literal for t in fixed_tokens]

    # De-dup while preserving order
    seen = set()
    uniq_literals: list[str] = []
    for lit in literals:
        if lit not in seen:
            seen.add(lit)
            uniq_literals.append(lit)

    keywords, symbols = select_prompt_fixed_tokens(spec.fixed_tokens, spec.trivia)

    # Header / policy
    lang = language_name or "the target language"
    parts: list[str] = []
    parts.append(f"You are {assistant_name}, a code generator for {lang}.")

    if code_only:
        parts.append(
            "Output only valid code unless the user explicitly asks for an explanation."
        )

    # Fixed tokens
    parts.append("Fixed tokens (must be used exactly as written):")
    parts.append(f"Keywords: { _wrap_csv(keywords, width=max_line_width) }")
    parts.append(f"Symbols/operators: { _wrap_csv(symbols, width=max_line_width) }")

    # Trivia note
    if line_comment_start is not None:
        parts.append(f"Line comments start with: {line_comment_start!r}")

    # Grammar
    parts.append("Grammar info:")
    parts.append(grammar.strip())

    # Style
    if include_style:
        parts.append("Style:")
        parts.append(
            f"- Use consistent indentation ({indent_spaces} spaces) inside blocks."
        )
        parts.append(
            "- Prefer minimal whitespace (avoid spaces just inside parentheses/brackets)."
        )
        parts.append(
            "- Avoid trailing filler, meta commentary, or repeated comments after the code is complete."
        )

    # Termination
    parts.append(
        "When you have finished the requested code, stop output (end the turn)."
    )

    return "\n".join(parts)
