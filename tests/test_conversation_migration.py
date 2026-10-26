"""PostgreSQL integration checks; set CHAT_TEST_DATABASE_URL to opt in.

Each test uses a unique schema and leaves application tables untouched.
"""

import importlib.util
import os
import unittest
from pathlib import Path
from uuid import uuid4

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Column, MetaData, Table, Uuid, create_engine, inspect, text
from sqlalchemy.schema import CreateSchema, DropSchema

import app.models  # noqa: F401
from app.core.database import Base

spec = importlib.util.spec_from_file_location(
    "conversation_migration", Path("alembic/versions/a84365bf9fb4_add_conversations.py")
)
MIGRATION = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MIGRATION)
TABLES = {"conversations", "conversation_messages", "conversation_runs"}


@unittest.skipUnless(os.getenv("CHAT_TEST_DATABASE_URL"), "PostgreSQL URL not supplied")
class ConversationMigrationTests(unittest.TestCase):
    def test_upgrade_matches_models_and_downgrade_removes_tables(self):
        url = os.environ["CHAT_TEST_DATABASE_URL"].replace("+asyncpg", "+psycopg")
        engine = create_engine(url)
        schema = "chat_test_" + uuid4().hex
        try:
            with engine.begin() as connection:
                connection.execute(CreateSchema(schema))
                connection.execute(text(f'SET LOCAL search_path TO "{schema}", public'))
                # Minimal referenced tables are enough to validate the new migration.
                connection.execute(text("CREATE TABLE projects (id uuid PRIMARY KEY)"))
                context = MigrationContext.configure(connection)
                with Operations.context(context):
                    MIGRATION.upgrade()
                expected = MetaData(naming_convention=Base.metadata.naming_convention)
                Table("projects", expected, Column("id", Uuid, primary_key=True))
                for name in sorted(TABLES):
                    Base.metadata.tables[name].to_metadata(expected)
                context = MigrationContext.configure(
                    connection,
                    opts={
                        "include_object": lambda obj, name, type_, reflected, compare_to: (
                            type_ != "table" or name in TABLES
                        ),
                        "compare_type": True,
                        "compare_server_default": True,
                    },
                )
                self.assertEqual(compare_metadata(context, expected), [])
                constraints = inspect(connection).get_check_constraints(
                    "conversation_runs"
                )
                self.assertTrue(any("COMPLETED" in c["sqltext"] for c in constraints))
                with Operations.context(context):
                    MIGRATION.downgrade()
                self.assertEqual(
                    inspect(connection).get_table_names(schema=schema), ["projects"]
                )
        finally:
            with engine.begin() as connection:
                connection.execute(DropSchema(schema, cascade=True, if_exists=True))
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
