"""enable pgvector extension

Revision ID: eacc12e8ecd1
Revises: 31cee32c9fbe
Create Date: 2026-09-10 03:35:38.550083

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'eacc12e8ecd1'
down_revision: Union[str, Sequence[str], None] = '31cee32c9fbe'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP EXTENSION IF EXISTS vector")
