from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Document


class DocumentRemovalService:
    """Explicitly remove source documents and every row that depends on them.

    This destructive testing utility temporarily removes only the schema's
    DELETE immutability triggers, inside the same SQLite transaction as the
    purge. Any error rolls back both the data changes and trigger changes.
    UPDATE protections remain enabled throughout.
    """

    _IMMUTABLE_TABLES = (
        "documents", "books", "chapters", "verses", "chunks",
        "translation_versions", "translation_segments", "evaluation_versions",
        "evaluation_results", "human_reviews", "published_outputs",
    )

    def __init__(self, session: Session):
        self.session = session

    def remove_document(self, document_id: UUID) -> bool:
        """Remove one document and dependent translation/review/evaluation data."""
        with self.session.begin():
            exists = self.session.scalar(select(Document.id).where(Document.id == document_id))
            if exists is None:
                return False
            self._remove_documents([document_id])
        return True

    def clear_documents(self) -> int:
        """Remove all source documents while retaining projects and their settings."""
        with self.session.begin():
            ids = list(self.session.scalars(select(Document.id).order_by(Document.created_at, Document.id)))
            if ids:
                self._remove_documents(ids)
        return len(ids)

    def _remove_documents(self, document_ids: list[UUID]) -> None:
        connection = self.session.connection()
        placeholders = ", ".join("?" for _ in document_ids)
        raw_ids = [value.hex for value in document_ids]

        # Dropping and restoring triggers inside this transaction ensures the
        # immutable-source safeguard is never left disabled after a failure.
        table_placeholders = ", ".join("?" for _ in self._IMMUTABLE_TABLES)
        triggers = connection.exec_driver_sql(
            "SELECT name, sql FROM sqlite_master WHERE type='trigger' "
            f"AND tbl_name IN ({table_placeholders}) "
            "AND (name LIKE 'trg_%_no_delete' OR name = 'trg_documents_source_no_delete')",
            self._IMMUTABLE_TABLES,
        ).all()
        for name, _sql in triggers:
            quoted_name = '"' + name.replace('"', '""') + '"'
            connection.exec_driver_sql(f"DROP TRIGGER {quoted_name}")

        def execute(sql: str, parameters: tuple = ()):
            return connection.exec_driver_sql(sql, parameters)

        # Resolve source units first so their referencing rows can be removed
        # before the source hierarchy itself.
        chunk_ids = [row[0] for row in execute(
            f"SELECT id FROM chunks WHERE document_id IN ({placeholders})", tuple(raw_ids)
        )]
        translation_ids = [row[0] for row in execute(
            f"SELECT id FROM translations WHERE document_id IN ({placeholders})", tuple(raw_ids)
        )]
        translation_version_ids: list[str] = []
        evaluation_ids: list[str] = []
        evaluation_version_ids: list[str] = []
        if translation_ids:
            ids_sql = ", ".join("?" for _ in translation_ids)
            params = tuple(translation_ids)
            translation_version_ids = [row[0] for row in execute(
                f"SELECT id FROM translation_versions WHERE translation_id IN ({ids_sql})", params
            )]
            evaluation_ids = [row[0] for row in execute(
                f"SELECT id FROM evaluations WHERE translation_id IN ({ids_sql})", params
            )]
        if evaluation_ids:
            ids_sql = ", ".join("?" for _ in evaluation_ids)
            evaluation_version_ids = [row[0] for row in execute(
                f"SELECT id FROM evaluation_versions WHERE evaluation_id IN ({ids_sql})",
                tuple(evaluation_ids),
            )]

        review_pass_ids: list[str] = []
        if translation_version_ids:
            placeholders_versions = ", ".join("?" for _ in translation_version_ids)
            review_pass_ids = [row[0] for row in execute(
                f"SELECT id FROM review_passes WHERE translation_version_id IN ({placeholders_versions})",
                tuple(translation_version_ids),
            )]
        # Remove cross-cutting records that reference any of the affected IDs.
        clauses, params = [], []
        if translation_version_ids:
            clauses.append(f"resulting_translation_version_id IN ({', '.join('?' for _ in translation_version_ids)})")
            params.extend(translation_version_ids)
        if chunk_ids:
            clauses.append(f"chunk_id IN ({', '.join('?' for _ in chunk_ids)})")
            params.extend(chunk_ids)
        if review_pass_ids:
            clauses.append(f"review_pass_id IN ({', '.join('?' for _ in review_pass_ids)})")
            params.extend(review_pass_ids)
        if clauses:
            execute(f"DELETE FROM human_reviews WHERE {' OR '.join(clauses)}", tuple(params))
        if review_pass_ids:
            execute(
                f"DELETE FROM review_passes WHERE id IN ({', '.join('?' for _ in review_pass_ids)})",
                tuple(review_pass_ids),
            )

        clauses, params = [], []
        if translation_version_ids:
            clauses.append(
                f"translation_version_id IN ({', '.join('?' for _ in translation_version_ids)})"
            )
            params.extend(translation_version_ids)
        if evaluation_version_ids:
            clauses.append(f"evaluation_version_id IN ({', '.join('?' for _ in evaluation_version_ids)})")
            params.extend(evaluation_version_ids)
        if clauses:
            execute(f"DELETE FROM experiment_runs WHERE {' OR '.join(clauses)}", tuple(params))

        clauses, params = [], []
        if translation_version_ids:
            clauses.append(f"translation_version_id IN ({', '.join('?' for _ in translation_version_ids)})")
            params.extend(translation_version_ids)
        if raw_ids:
            clauses.append(f"document_id IN ({placeholders})")
            params.extend(raw_ids)
        if clauses:
            execute(f"DELETE FROM published_outputs WHERE {' OR '.join(clauses)}", tuple(params))

        if translation_version_ids:
            ids_sql = ", ".join("?" for _ in translation_version_ids)
            execute(f"DELETE FROM translation_memories WHERE origin_translation_version_id IN ({ids_sql})",
                    tuple(translation_version_ids))
            execute(f"DELETE FROM translation_segments WHERE translation_version_id IN ({ids_sql})",
                    tuple(translation_version_ids))
        if evaluation_version_ids:
            ids_sql = ", ".join("?" for _ in evaluation_version_ids)
            execute(f"DELETE FROM evaluation_results WHERE evaluation_version_id IN ({ids_sql})",
                    tuple(evaluation_version_ids))
            execute(f"DELETE FROM evaluation_versions WHERE id IN ({ids_sql})",
                    tuple(evaluation_version_ids))
        if evaluation_ids:
            ids_sql = ", ".join("?" for _ in evaluation_ids)
            execute(f"DELETE FROM evaluations WHERE id IN ({ids_sql})", tuple(evaluation_ids))

        # A child version may reference an earlier parent version. Repeatedly
        # delete leaves to honor the self-referential RESTRICT foreign key.
        remaining = set(translation_version_ids)
        while remaining:
            ids_sql = ", ".join("?" for _ in remaining)
            leaves = [row[0] for row in execute(
                "SELECT candidate.id FROM translation_versions AS candidate "
                f"WHERE candidate.id IN ({ids_sql}) AND NOT EXISTS ("
                "SELECT 1 FROM translation_versions AS child "
                "WHERE child.parent_version_id = candidate.id)",
                tuple(remaining),
            )]
            if not leaves:
                raise RuntimeError("Cannot purge translation versions: parent-version cycle detected")
            leaf_sql = ", ".join("?" for _ in leaves)
            execute(f"DELETE FROM translation_versions WHERE id IN ({leaf_sql})", tuple(leaves))
            remaining.difference_update(leaves)
        if translation_ids:
            ids_sql = ", ".join("?" for _ in translation_ids)
            execute(f"DELETE FROM translations WHERE id IN ({ids_sql})", tuple(translation_ids))

        if chunk_ids:
            ids_sql = ", ".join("?" for _ in chunk_ids)
            execute(f"DELETE FROM chunks WHERE id IN ({ids_sql})", tuple(chunk_ids))
        # Scripture hierarchy is deleted from leaves up to the document.
        execute(f"DELETE FROM verses WHERE chapter_id IN (SELECT id FROM chapters WHERE book_id IN "
                f"(SELECT id FROM books WHERE document_id IN ({placeholders})))", tuple(raw_ids))
        execute(f"DELETE FROM chapters WHERE book_id IN (SELECT id FROM books WHERE document_id IN ({placeholders}))",
                tuple(raw_ids))
        execute(f"DELETE FROM books WHERE document_id IN ({placeholders})", tuple(raw_ids))
        execute(f"DELETE FROM documents WHERE id IN ({placeholders})", tuple(raw_ids))

        restored_names: set[str] = set()
        for name, sql in triggers:
            if sql:
                connection.exec_driver_sql(sql)
                restored_names.add(name)
        # Older local databases may be missing delete guards. Restore the
        # canonical protections as part of every successful purge as well.
        for table in self._IMMUTABLE_TABLES:
            if table == "documents":
                name = "trg_documents_source_no_delete"
            else:
                name = f"trg_{table}_no_delete"
            if name not in restored_names:
                connection.exec_driver_sql(self._delete_trigger_sql(table))

    @staticmethod
    def _delete_trigger_sql(table: str) -> str:
        if table == "documents":
            return (
                "CREATE TRIGGER IF NOT EXISTS trg_documents_source_no_delete BEFORE DELETE ON documents "
                "BEGIN SELECT RAISE(ABORT, 'source snapshots are immutable'); END"
            )
        return (
            f"CREATE TRIGGER IF NOT EXISTS trg_{table}_no_delete BEFORE DELETE ON {table} "
            "BEGIN SELECT RAISE(ABORT, 'record is immutable; append a new version or event'); END"
        )

