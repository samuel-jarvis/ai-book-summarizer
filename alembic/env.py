from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

import app.models  # noqa: F401 -- register all tables before autogeneration
from alembic import context
from app.core.config import settings
from app.core.database import Base
from app.core.types import PydanticJSONB

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# The app forces DATABASE_URL onto asyncpg (see config.py), which the sync
# engine below cannot drive — swap in psycopg, which is installed for exactly
# this. The `%` doubling is required because ConfigParser interpolates values.
config.set_main_option(
    "sqlalchemy.url",
    settings.DATABASE_URL.replace("+asyncpg", "+psycopg").replace("%", "%%"),
)


def render_item(type_, obj, autogen_context):
    """Persist the database type without coupling migrations to app schemas."""
    if type_ == "type" and isinstance(obj, PydanticJSONB):
        autogen_context.imports.add("from sqlalchemy.dialects import postgresql")
        return "postgresql.JSONB(none_as_null=True)"
    return False


def run_migrations_offline() -> None:
    """Generates SQL scripts without a live DB connection (e.g. for review/CI)."""

    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_item=render_item,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connects to the DB and applies migrations directly — the common path."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_item=render_item,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
