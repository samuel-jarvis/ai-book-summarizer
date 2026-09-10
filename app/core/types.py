from typing import Any

from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator


class PydanticJSONB[MetadataModel: BaseModel](TypeDecorator[MetadataModel]):
    """Validate JSONB on writes and restore the configured model on reads.

    Accepts a model or dictionary. Assign a replacement model to persist edits;
    SQLAlchemy does not track in-place changes inside Pydantic models.
    """

    impl = JSONB
    cache_ok = True

    def __init__(self, pydantic_model: type[MetadataModel]):
        super().__init__(none_as_null=True)
        self.pydantic_model = pydantic_model

    def process_bind_param(
        self, value: MetadataModel | dict[str, Any] | None, dialect: Dialect
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        # Revalidate instances too, including models modified after construction.
        if isinstance(value, self.pydantic_model):
            value = value.model_dump(warnings=False)
        return self.pydantic_model.model_validate(value).model_dump(mode="json")

    def process_result_value(
        self, value: dict[str, Any] | None, dialect: Dialect
    ) -> MetadataModel | None:
        if value is None:
            return None
        return self.pydantic_model.model_validate(value)
