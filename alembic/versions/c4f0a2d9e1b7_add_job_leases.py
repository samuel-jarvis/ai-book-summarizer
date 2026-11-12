"""add job leases and error codes

Background jobs (document summary, document embedding, chat runs, legacy
summaries) gain a lease that the running worker renews, an attempt counter for
capped automatic re-runs, and a safe error code. A job past its lease belongs
to a dead worker and may be swept. See app/services/jobs.

Revision ID: c4f0a2d9e1b7
Revises: a84365bf9fb4
Create Date: 2026-11-12 09:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4f0a2d9e1b7"
down_revision: str | Sequence[str] | None = "a84365bf9fb4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # documents: one lease per background job
    op.add_column(
        "documents",
        sa.Column(
            "summary_attempts", sa.Integer(), server_default="0", nullable=False
        ),
    )
    op.add_column(
        "documents",
        sa.Column("summary_lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("summary_error_code", sa.String(length=50), nullable=True),
    )
    op.add_column("documents", sa.Column("embedding_job_id", sa.Uuid(), nullable=True))
    op.add_column(
        "documents",
        sa.Column(
            "embedding_attempts", sa.Integer(), server_default="0", nullable=False
        ),
    )
    op.add_column(
        "documents",
        sa.Column(
            "embedding_lease_expires_at", sa.DateTime(timezone=True), nullable=True
        ),
    )
    op.add_column(
        "documents",
        sa.Column("embedding_error_code", sa.String(length=50), nullable=True),
    )
    # Enum columns store member names, so the predicates use 'STARTED', not 'started'.
    op.create_index(
        "ix_documents_summary_lease",
        "documents",
        ["summary_lease_expires_at"],
        unique=False,
        postgresql_where=sa.text("summary_status = 'STARTED'"),
    )
    op.create_index(
        "ix_documents_embedding_lease",
        "documents",
        ["embedding_lease_expires_at"],
        unique=False,
        postgresql_where=sa.text("embedding_status = 'STARTED'"),
    )

    # conversation_runs: heartbeat while RUNNING
    op.add_column(
        "conversation_runs",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_conversation_runs_lease",
        "conversation_runs",
        ["lease_expires_at"],
        unique=False,
        postgresql_where=sa.text("status = 'RUNNING'"),
    )

    # summaries (legacy): job fence, attempts and lease
    op.add_column("summaries", sa.Column("job_id", sa.Uuid(), nullable=True))
    op.add_column(
        "summaries",
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
    )
    op.add_column(
        "summaries",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_summaries_lease",
        "summaries",
        ["lease_expires_at"],
        unique=False,
        postgresql_where=sa.text("status = 'PROCESSING'"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_summaries_lease", table_name="summaries")
    op.drop_column("summaries", "lease_expires_at")
    op.drop_column("summaries", "attempts")
    op.drop_column("summaries", "job_id")

    op.drop_index("ix_conversation_runs_lease", table_name="conversation_runs")
    op.drop_column("conversation_runs", "lease_expires_at")

    op.drop_index("ix_documents_embedding_lease", table_name="documents")
    op.drop_index("ix_documents_summary_lease", table_name="documents")
    op.drop_column("documents", "embedding_error_code")
    op.drop_column("documents", "embedding_lease_expires_at")
    op.drop_column("documents", "embedding_attempts")
    op.drop_column("documents", "embedding_job_id")
    op.drop_column("documents", "summary_error_code")
    op.drop_column("documents", "summary_lease_expires_at")
    op.drop_column("documents", "summary_attempts")
