"""Load and validate chat benchmark labels."""

import re
import unicodedata
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def words(text: str) -> str:
    """Ignore punctuation, ligatures, and line wrapping in PDF extraction."""
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("\u00ad", "")
    return " ".join(re.findall(r"\w+", normalized))


class Evidence(BaseModel):
    document: NonEmpty
    pdf_page: int = Field(ge=1)
    quote: NonEmpty

    @model_validator(mode="after")
    def validate_anchor(self) -> "Evidence":
        if Path(self.document).name != self.document or not self.document.lower().endswith(".pdf"):
            raise ValueError("Evidence document must be a PDF filename.")
        if not words(self.quote):
            raise ValueError("Evidence quote must contain searchable text.")
        return self

    def matches(self, normalized_text: str) -> bool:
        return f" {words(self.quote)} " in f" {normalized_text} "


class Fact(BaseModel):
    fact: NonEmpty
    evidence: list[Evidence] = Field(min_length=1)


class PriorTurn(BaseModel):
    role: Literal["user"]
    content: NonEmpty


class Case(BaseModel):
    id: NonEmpty
    type: Literal["direct", "follow_up", "comparison", "unanswerable"]
    prior_turns: list[NonEmpty | PriorTurn]
    question: NonEmpty
    expected_behavior: Literal["answer", "abstain"]
    required_facts: list[Fact]
    abstention_reason: NonEmpty | None = None

    @model_validator(mode="after")
    def validate_expectations(self) -> "Case":
        if self.expected_behavior == "answer" and not self.required_facts:
            raise ValueError("An answerable case needs at least one required fact.")
        if self.expected_behavior == "abstain" and (
            self.required_facts or not self.abstention_reason
        ):
            raise ValueError("An abstention needs a reason and an empty required_facts list.")
        if self.type == "follow_up" and not self.prior_turns:
            raise ValueError("A follow-up case needs a prior user turn.")
        return self

    @property
    def questions(self) -> list[str]:
        return [
            *(turn if isinstance(turn, str) else turn.content for turn in self.prior_turns),
            self.question,
        ]


def read_cases(path: Path) -> list[Case]:
    cases = []
    seen = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = Case.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
        if case.id in seen:
            raise ValueError(f"{path}:{line_no}: duplicate case ID {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases


def fixture_paths(cases: list[Case], directory: Path) -> list[Path]:
    names = sorted({
        anchor.document
        for case in cases
        for fact in case.required_facts
        for anchor in fact.evidence
    })
    if not names:
        raise ValueError("The benchmark needs evidence labels to identify its fixture PDFs.")
    paths = [directory / name for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("Missing fixture PDFs:\n  " + "\n  ".join(missing))
    return paths
