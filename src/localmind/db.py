from __future__ import annotations

import sqlite3
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import default_db_path, ensure_parent
from .models import FileRecord, IndexStats, TextChunk, utc_now_iso


class LocalMindDB:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else default_db_path()
        ensure_parent(self.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        from .search_filters import epoch
        self.conn.create_function("localmind_epoch", 1, epoch)
        from .identifiers import indexed_identifier
        self.conn.create_function("localmind_identifier", 3, indexed_identifier, deterministic=True)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.init_schema()

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                ext TEXT NOT NULL,
                size INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                file_hash TEXT,
                scan_status TEXT NOT NULL,
                parse_status TEXT NOT NULL,
                ocr_status TEXT NOT NULL,
                needs_ocr INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                indexed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                chunk_text TEXT NOT NULL,
                page_no INTEGER,
                sheet_name TEXT,
                source_type TEXT NOT NULL,
                start_offset INTEGER,
                end_offset INTEGER,
                ocr_confidence REAL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ocr_results (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                page_no INTEGER,
                text TEXT,
                confidence REAL,
                status TEXT NOT NULL,
                error TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS vectors (
                chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                vector_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                vector_dim INTEGER NOT NULL,
                embedding BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scan_jobs (
                id INTEGER PRIMARY KEY,
                root_path TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                files_seen INTEGER NOT NULL DEFAULT 0,
                files_indexed INTEGER NOT NULL DEFAULT 0,
                errors INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS watched_folders (
                path TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 1,
                added_at TEXT NOT NULL,
                last_scan_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_files_ext ON files(ext);
            CREATE INDEX IF NOT EXISTS idx_files_modified ON files(modified_at);
            CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
            """
        )
        if "text_index_version" not in {row[1] for row in self.conn.execute("PRAGMA table_info(files)")}:
            self.conn.execute("ALTER TABLE files ADD COLUMN text_index_version INTEGER NOT NULL DEFAULT 0")
        self._init_fts()
        from .chats import init_chat_schema

        init_chat_schema(self.conn)
        self._init_index_revisions()
        self.conn.commit()

    def _init_index_revisions(self) -> None:
        self.conn.execute("""CREATE TABLE IF NOT EXISTS index_revisions (
            id INTEGER PRIMARY KEY CHECK(id=1),
            revision INTEGER NOT NULL,
            vector_revision INTEGER NOT NULL
        )""")
        self.conn.execute("INSERT OR IGNORE INTO index_revisions VALUES(1,0,0)")
        # Triggers share the writer's transaction, including cascade deletes and rollback.
        for table in ("files", "chunks", "vectors", "chat_sources", "chat_messages",
                      "chat_attachments", "chat_units", "chat_attachment_locations"):
            vector_increment = ",vector_revision=vector_revision+1" if table == "vectors" else ""
            for action in ("INSERT", "UPDATE", "DELETE"):
                self.conn.execute(f"""CREATE TRIGGER IF NOT EXISTS lm_revision_{table}_{action.lower()}
                    AFTER {action} ON {table} BEGIN
                    UPDATE index_revisions SET revision=revision+1{vector_increment} WHERE id=1;
                    END""")
        for action, condition in (
            ("INSERT", "NEW.key='embedding_model'"),
            ("DELETE", "OLD.key='embedding_model'"),
            ("UPDATE", "(OLD.key='embedding_model' OR NEW.key='embedding_model') "
                       "AND (OLD.key IS NOT NEW.key OR OLD.value IS NOT NEW.value)"),
        ):
            self.conn.execute(f"""CREATE TRIGGER IF NOT EXISTS lm_revision_model_{action.lower()}
                AFTER {action} ON settings WHEN {condition} BEGIN
                UPDATE index_revisions SET revision=revision+1,vector_revision=vector_revision+1 WHERE id=1;
                END""")

    def index_revisions(self) -> tuple[int, int]:
        row = self.conn.execute("SELECT revision,vector_revision FROM index_revisions WHERE id=1").fetchone()
        if row is None:
            raise sqlite3.DatabaseError("Missing index revision record")
        return int(row[0]), int(row[1])

    @contextmanager
    def _savepoint(self, name):
        self.conn.execute(f"SAVEPOINT {name}")
        try:
            yield
        except BaseException:
            self.conn.execute(f"ROLLBACK TO {name}")
            raise
        finally:
            self.conn.execute(f"RELEASE {name}")

    @contextmanager
    def search_snapshot(self):
        with self._savepoint("localmind_search_snapshot"):
            # The first main-database read pins the WAL snapshot before cache checks.
            yield self.index_revisions()

    def _init_fts(self) -> None:
        if self._table_exists("chunk_fts"):
            return
        try:
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE chunk_fts USING fts5(
                    file_name,
                    file_path,
                    chunk_text,
                    source_type UNINDEXED,
                    tokenize='trigram'
                )
                """
            )
            self.set_setting("fts_tokenizer", "trigram")
        except sqlite3.OperationalError:
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE chunk_fts USING fts5(
                    file_name,
                    file_path,
                    chunk_text,
                    source_type UNINDEXED,
                    tokenize='unicode61'
                )
                """
            )
            self.set_setting("fts_tokenizer", "unicode61")

    def _table_exists(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None

    def set_setting(self, key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT INTO settings(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, value),
        )
        self.conn.commit()

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def upsert_file(self, record: FileRecord) -> int:
        self.conn.execute(
            """
            INSERT INTO files(
                path, name, ext, size, created_at, modified_at, file_hash,
                scan_status, parse_status, ocr_status, needs_ocr, error, indexed_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                name=excluded.name,
                ext=excluded.ext,
                size=excluded.size,
                created_at=excluded.created_at,
                modified_at=excluded.modified_at,
                file_hash=excluded.file_hash,
                scan_status=excluded.scan_status,
                parse_status=excluded.parse_status,
                ocr_status=excluded.ocr_status,
                needs_ocr=excluded.needs_ocr,
                error=excluded.error
            """,
            (
                record.path,
                record.name,
                record.ext,
                record.size,
                record.created_at,
                record.modified_at,
                record.file_hash,
                record.scan_status,
                record.parse_status,
                record.ocr_status,
                int(record.needs_ocr),
                record.error,
                utc_now_iso(),
            ),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT id FROM files WHERE path=?", (record.path,)).fetchone()
        file_id = int(row["id"])
        record.id = file_id
        return file_id

    def get_file(self, file_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone()

    def get_file_by_path(self, path: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()

    def update_file_status(
        self,
        file_id: int,
        *,
        parse_status: str | None = None,
        ocr_status: str | None = None,
        error: str | None = None,
    ) -> None:
        current = self.get_file(file_id)
        if not current:
            return
        self.conn.execute(
            """
            UPDATE files
            SET parse_status=?, ocr_status=?, error=?, indexed_at=?
            WHERE id=?
            """,
            (
                parse_status or current["parse_status"],
                ocr_status or current["ocr_status"],
                error,
                utc_now_iso(),
                file_id,
            ),
        )
        self.conn.commit()

    def delete_file_index(self, file_id: int) -> None:
        self._delete_file_index_no_commit(file_id)
        self.conn.commit()

    def _delete_file_index_no_commit(self, file_id: int, *, preserve_ocr_results=False) -> None:
        chunk_rows = self.conn.execute(
            "SELECT id FROM chunks WHERE file_id=?", (file_id,)
        ).fetchall()
        for row in chunk_rows:
            self.conn.execute("DELETE FROM chunk_fts WHERE rowid=?", (row["id"],))
        self.conn.execute("DELETE FROM vectors WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id=?)", (file_id,))
        self.conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
        if not preserve_ocr_results:
            self.conn.execute("DELETE FROM ocr_results WHERE file_id=?", (file_id,))

    def delete_file_by_path(self, path: str | Path) -> bool:
        row = self.get_file_by_path(str(path))
        if not row:
            return False
        with self.conn:
            self._delete_file_index_no_commit(int(row["id"]))
            self.conn.execute("DELETE FROM files WHERE id=?", (int(row["id"]),))
        return True

    def update_file_metadata(self, record: FileRecord) -> None:
        self.conn.execute(
            """
            UPDATE files
            SET name=?, ext=?, size=?, created_at=?, modified_at=?, file_hash=?,
                scan_status=?, needs_ocr=?
            WHERE path=?
            """,
            (
                record.name,
                record.ext,
                record.size,
                record.created_at,
                record.modified_at,
                record.file_hash,
                record.scan_status,
                int(record.needs_ocr),
                record.path,
            ),
        )
        self.conn.commit()

    def insert_chunks(self, file_id: int, chunks: Iterable[TextChunk]) -> list[int]:
        file_row = self.get_file(file_id)
        if not file_row:
            raise ValueError(f"Unknown file_id {file_id}")
        inserted: list[int] = []
        for chunk in chunks:
            text = chunk.chunk_text.strip()
            if not text:
                continue
            created_at = chunk.created_at or utc_now_iso()
            cursor = self.conn.execute(
                """
                INSERT INTO chunks(
                    file_id, chunk_text, page_no, sheet_name, source_type,
                    start_offset, end_offset, ocr_confidence, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    text,
                    chunk.page_no,
                    chunk.sheet_name,
                    chunk.source_type,
                    chunk.start_offset,
                    chunk.end_offset,
                    chunk.ocr_confidence,
                    created_at,
                ),
            )
            chunk_id = int(cursor.lastrowid)
            chunk.id = chunk_id
            chunk.file_id = file_id
            inserted.append(chunk_id)
            self.conn.execute(
                """
                INSERT INTO chunk_fts(rowid, file_name, file_path, chunk_text, source_type)
                VALUES(?, ?, ?, ?, ?)
                """,
                (chunk_id, file_row["name"], file_row["path"], text, chunk.source_type),
            )
        self.conn.commit()
        return inserted

    def insert_ocr_result(
        self,
        file_id: int,
        *,
        page_no: int | None,
        text: str,
        confidence: float | None,
        status: str,
        error: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO ocr_results(file_id, page_no, text, confidence, status, error, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (file_id, page_no, text, confidence, status, error, utc_now_iso()),
        )
        self.conn.commit()

    def insert_vectors(
        self,
        chunk_ids: list[int],
        embeddings: np.ndarray,
        *,
        model_name: str,
    ) -> None:
        if len(chunk_ids) != len(embeddings):
            raise ValueError("chunk_ids and embeddings length mismatch")
        for vector_id, (chunk_id, vector) in enumerate(zip(chunk_ids, embeddings), start=1):
            vec = np.asarray(vector, dtype="float32")
            self.conn.execute(
                """
                INSERT INTO vectors(chunk_id, vector_id, model_name, vector_dim, embedding)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    vector_id=excluded.vector_id,
                    model_name=excluded.model_name,
                    vector_dim=excluded.vector_dim,
                    embedding=excluded.embedding
                """,
                (chunk_id, vector_id, model_name, int(vec.shape[0]), vec.tobytes()),
            )
        self.conn.commit()

    def replace_file_index(
        self,
        record: FileRecord,
        chunks: list[TextChunk],
        *,
        parse_status: str,
        ocr_status: str,
        embeddings: np.ndarray | None = None,
        model_name: str | None = None,
        preserve_ocr_results: bool = False,
    ) -> tuple[int, list[int]]:
        if embeddings is not None and len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        indexed_at = utc_now_iso()
        inserted: list[int] = []
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO files(
                    path, name, ext, size, created_at, modified_at, file_hash,
                    scan_status, parse_status, ocr_status, needs_ocr, error, indexed_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(path) DO UPDATE SET
                    name=excluded.name,
                    ext=excluded.ext,
                    size=excluded.size,
                    created_at=excluded.created_at,
                    modified_at=excluded.modified_at,
                    file_hash=excluded.file_hash,
                    scan_status=excluded.scan_status,
                    parse_status=excluded.parse_status,
                    ocr_status=excluded.ocr_status,
                    needs_ocr=excluded.needs_ocr,
                    error=NULL,
                    indexed_at=excluded.indexed_at
                """,
                (
                    record.path,
                    record.name,
                    record.ext,
                    record.size,
                    record.created_at,
                    record.modified_at,
                    record.file_hash,
                    record.scan_status,
                    parse_status,
                    ocr_status,
                    int(record.needs_ocr),
                    indexed_at,
                ),
            )
            row = self.conn.execute("SELECT id FROM files WHERE path=?", (record.path,)).fetchone()
            file_id = int(row["id"])
            self._delete_file_index_no_commit(file_id, preserve_ocr_results=preserve_ocr_results)
            from .chunker import TEXT_INDEX_VERSION
            self.conn.execute("UPDATE files SET text_index_version=? WHERE id=?", (TEXT_INDEX_VERSION, file_id))

            for index, chunk in enumerate(chunks):
                text = chunk.chunk_text.strip()
                if not text:
                    continue
                cursor = self.conn.execute(
                    """
                    INSERT INTO chunks(
                        file_id, chunk_text, page_no, sheet_name, source_type,
                        start_offset, end_offset, ocr_confidence, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        text,
                        chunk.page_no,
                        chunk.sheet_name,
                        chunk.source_type,
                        chunk.start_offset,
                        chunk.end_offset,
                        chunk.ocr_confidence,
                        chunk.created_at or indexed_at,
                    ),
                )
                chunk_id = int(cursor.lastrowid)
                chunk.id = chunk_id
                chunk.file_id = file_id
                inserted.append(chunk_id)
                self.conn.execute(
                    """
                    INSERT INTO chunk_fts(rowid, file_name, file_path, chunk_text, source_type)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (chunk_id, record.name, record.path, text, chunk.source_type),
                )
                if embeddings is not None and model_name:
                    vector = np.asarray(embeddings[index], dtype="float32")
                    self.conn.execute(
                        """
                        INSERT INTO vectors(chunk_id, vector_id, model_name, vector_dim, embedding)
                        VALUES(?, ?, ?, ?, ?)
                        """,
                        (chunk_id, len(inserted), model_name, int(vector.shape[0]), vector.tobytes()),
                    )
        record.id = file_id
        return file_id, inserted

    def load_vectors(self, model_name: str | None = None) -> tuple[list[int], np.ndarray]:
        sql = "SELECT chunk_id, vector_dim, embedding FROM vectors"
        params: tuple[str, ...] = ()
        if model_name:
            sql += " WHERE model_name=?"
            params = (model_name,)
        rows = self.conn.execute(sql, params).fetchall()
        if not rows:
            return [], np.empty((0, 0), dtype="float32")
        dim = int(rows[0]["vector_dim"])
        ids = [int(row["chunk_id"]) for row in rows]
        vectors = np.vstack(
            [np.frombuffer(row["embedding"], dtype="float32", count=dim) for row in rows]
        )
        return ids, vectors.astype("float32", copy=False)

    def vector_models(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT model_name, COUNT(*) AS n FROM vectors GROUP BY model_name ORDER BY n DESC"
        ).fetchall()
        return {str(row["model_name"]): int(row["n"]) for row in rows}

    def file_has_vectors(self, file_id: int, model_name: str) -> bool:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS chunk_count,
                   SUM(CASE WHEN vectors.model_name=? THEN 1 ELSE 0 END) AS vector_count
            FROM chunks
            LEFT JOIN vectors ON vectors.chunk_id = chunks.id
            WHERE chunks.file_id=?
            """,
            (model_name, file_id),
        ).fetchone()
        return bool(row and row["chunk_count"] and row["chunk_count"] == row["vector_count"])

    def clear_vectors(self) -> None:
        self.conn.execute("DELETE FROM vectors")
        self.conn.commit()

    def fetch_chunks_by_ids(self, chunk_ids: Iterable[int]) -> dict[int, sqlite3.Row]:
        ids = list(dict.fromkeys(int(item) for item in chunk_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""
            SELECT
                chunks.*,
                files.name,
                files.path,
                files.ext,
                files.modified_at
            FROM chunks
            JOIN files ON files.id = chunks.file_id
            WHERE chunks.id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        return {int(row["id"]): row for row in rows}

    def set_search_scope(self, ids: set[int]) -> None:
        with self._savepoint("localmind_search_scope"):
            self.conn.execute("CREATE TEMP TABLE IF NOT EXISTS search_scope(id INTEGER PRIMARY KEY)")
            self.conn.execute("DELETE FROM search_scope")
            self.conn.executemany("INSERT INTO search_scope(id) VALUES(?)", ((cid,) for cid in ids))

    @staticmethod
    def _candidate_target_sql(target_mode):
        if target_mode not in {"files", "messages"}:
            raise ValueError("Invalid candidate target mode")
        linked = "WHEN a.file_id IS NOT NULL THEN 'file:'||a.file_id" if target_mode == "files" else ""
        return f"""CASE WHEN u.message_id IS NULL THEN 'file:'||chunks.file_id
            {linked} ELSE 'chat:'||u.message_id||':'||coalesce(u.attachment_id,0) END"""

    def candidate_groups(self, *, target_mode="files") -> dict[int, str]:
        target = self._candidate_target_sql(target_mode)
        return {int(row[0]): row[1] for row in self.conn.execute(f"""
            SELECT chunks.id,{target} FROM chunks
            LEFT JOIN chat_units u ON u.chunk_id=chunks.id
            LEFT JOIN chat_attachments a ON a.id=u.attachment_id""")}

    def search_fts(self, fts_query: str, limit: int = 50, *, restricted: bool = False,
                   target_mode: str | None = None, identifier: str = "") -> list[sqlite3.Row]:
        # IN(rowid) makes FTS5 repeat MATCH for every scope ID. Correlating
        # membership keeps MATCH as the driver and uses the scope primary key.
        scope = ("AND EXISTS (SELECT 1 FROM search_scope WHERE search_scope.id=chunk_fts.rowid)"
                 if restricted else "")
        exact = ("AND ((source_type != 'chat' AND localmind_identifier(file_name,'',?)) "
                 "OR localmind_identifier(chunk_text,source_type,?))") if identifier else ""
        params = (fts_query, identifier, identifier, limit) if identifier else (fts_query, limit)
        if target_mode is not None:
            target = self._candidate_target_sql(target_mode)
            # Materialize BM25 before the window function; FTS auxiliary functions
            # must run in the MATCH query, not in the outer ranking context.
            return self.conn.execute(f"""
                WITH matches AS MATERIALIZED (
                    SELECT chunk_fts.rowid AS chunk_id,bm25(chunk_fts,2.0,1.5,6.0,0.1) AS rank
                    FROM chunk_fts WHERE chunk_fts MATCH ?
                    {scope} {exact}
                ), diversified AS (
                    SELECT matches.*,ROW_NUMBER() OVER (
                        PARTITION BY {target} ORDER BY matches.rank,matches.chunk_id) AS target_rank
                    FROM matches JOIN chunks ON chunks.id=matches.chunk_id
                    LEFT JOIN chat_units u ON u.chunk_id=chunks.id
                    LEFT JOIN chat_attachments a ON a.id=u.attachment_id
                )
                SELECT chunk_id,rank FROM diversified WHERE target_rank<=4
                ORDER BY rank,chunk_id LIMIT ?""", params).fetchall()
        return self.conn.execute(
            f"""
            SELECT
                chunk_fts.rowid AS chunk_id,
                bm25(chunk_fts, 2.0, 1.5, 6.0, 0.1) AS rank
            FROM chunk_fts
            WHERE chunk_fts MATCH ?
            {scope} {exact}
            ORDER BY rank
            LIMIT ?
            """,
            params,
        ).fetchall()

    def search_like(self, text: str, limit: int = 50) -> list[sqlite3.Row]:
        pattern = f"%{text}%"
        return self.conn.execute(
            """
            SELECT chunks.id AS chunk_id, -1.0 AS rank
            FROM chunks
            JOIN files ON files.id = chunks.file_id
            WHERE chunks.chunk_text LIKE ?
               OR files.name LIKE ?
               OR files.path LIKE ?
            LIMIT ?
            """,
            (pattern, pattern, pattern, limit),
        ).fetchall()

    def search_like_terms(self, terms: Iterable[str], limit: int = 100, *, restricted: bool = False,
                          target_mode: str | None = None) -> list[sqlite3.Row]:
        cleaned = list(dict.fromkeys(term.strip() for term in terms if term.strip()))[:16]
        if not cleaned:
            return []
        scores: list[str] = []
        params: list[object] = []
        for term in cleaned:
            # Literal substring matching: user '%' and '_' are not SQL wildcards.
            # Chinese-only terms need no repeated lower() over the full document.
            columns = [f"lower({column})" if term.lower() != term.upper() else column
                       for column in ("chunks.chunk_text", "files.name", "files.path")]
            scores.append(f"""CASE WHEN instr({columns[0]}, ?) > 0 OR instr({columns[1]}, ?) > 0
                THEN 2 WHEN instr({columns[2]}, ?) > 0 THEN 1 ELSE 0 END""")
            params.extend([term.lower()] * 3)
        params.append(limit)
        target = self._candidate_target_sql(target_mode) if target_mode else "chunks.file_id"
        joins = ("LEFT JOIN chat_units u ON u.chunk_id=chunks.id "
                 "LEFT JOIN chat_attachments a ON a.id=u.attachment_id") if target_mode else ""
        return self.conn.execute(
            f"""
            WITH scored AS MATERIALIZED (
                SELECT chunks.id AS chunk_id, {target} AS target_id, ({' + '.join(scores)}) AS match_score
                FROM chunks JOIN files ON files.id = chunks.file_id
                {joins}
                {'WHERE chunks.id IN (SELECT id FROM search_scope)' if restricted else ''}
            ), diversified AS (
                SELECT chunk_id, match_score,
                    ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY match_score DESC, chunk_id) AS file_rank
                FROM scored WHERE match_score > 0
            )
            SELECT chunk_id FROM diversified WHERE file_rank <= 4
            ORDER BY match_score DESC, chunk_id LIMIT ?
            """,
            params,
        ).fetchall()

    def add_watched_folder(self, path: str | Path) -> None:
        normalized = str(Path(path))
        self.conn.execute(
            """
            INSERT INTO watched_folders(path, enabled, added_at)
            VALUES(?, 1, ?)
            ON CONFLICT(path) DO UPDATE SET enabled=1
            """,
            (normalized, utc_now_iso()),
        )
        self.conn.commit()

    def remove_watched_folder(self, path: str | Path) -> None:
        self.conn.execute("DELETE FROM watched_folders WHERE path=?", (str(Path(path)),))
        self.conn.commit()

    def list_watched_folders(self, *, enabled_only: bool = True) -> list[str]:
        sql = "SELECT path FROM watched_folders"
        if enabled_only:
            sql += " WHERE enabled=1"
        sql += " ORDER BY added_at, path"
        return [str(row["path"]) for row in self.conn.execute(sql).fetchall()]

    def mark_folder_scanned(self, path: str | Path) -> None:
        self.conn.execute(
            "UPDATE watched_folders SET last_scan_at=? WHERE path=?",
            (utc_now_iso(), str(Path(path))),
        )
        self.conn.commit()

    def remove_missing_files(self, root: str | Path, seen_paths: Iterable[str]) -> int:
        root_key = os.path.normcase(os.path.abspath(str(root)))
        seen = {os.path.normcase(os.path.abspath(path)) for path in seen_paths}
        removed = 0
        rows = self.conn.execute("SELECT id, path FROM files").fetchall()
        with self.conn:
            for row in rows:
                if str(row["path"]).startswith("localmind-chat://"):
                    continue
                path_key = os.path.normcase(os.path.abspath(str(row["path"])))
                try:
                    under_root = os.path.commonpath([root_key, path_key]) == root_key
                except ValueError:
                    under_root = False
                if under_root and path_key not in seen:
                    self._delete_file_index_no_commit(int(row["id"]))
                    self.conn.execute("DELETE FROM files WHERE id=?", (int(row["id"]),))
                    removed += 1
        return removed

    def clear_index(self) -> None:
        self.conn.execute("DELETE FROM chat_sources")
        self.conn.execute("DELETE FROM chat_sync_issues")
        self.conn.execute("DELETE FROM settings WHERE key LIKE 'weflow_checkpoint:%'")
        self.conn.execute("DELETE FROM settings WHERE key LIKE 'weflow_batch:%'")
        self.conn.execute("DELETE FROM settings WHERE key LIKE 'weflow_scope:%'")
        self.conn.execute("DELETE FROM chunk_fts")
        self.conn.execute("DELETE FROM vectors")
        self.conn.execute("DELETE FROM chunks")
        self.conn.execute("DELETE FROM ocr_results")
        self.conn.execute("DELETE FROM files")
        self.conn.execute("DELETE FROM scan_jobs")
        self.conn.commit()

    def create_scan_job(self, root_path: str) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO scan_jobs(root_path, status, started_at)
            VALUES(?, 'running', ?)
            """,
            (root_path, utc_now_iso()),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def finish_scan_job(
        self,
        job_id: int,
        *,
        status: str,
        files_seen: int,
        files_indexed: int,
        errors: int,
    ) -> None:
        self.conn.execute(
            """
            UPDATE scan_jobs
            SET status=?, finished_at=?, files_seen=?, files_indexed=?, errors=?
            WHERE id=?
            """,
            (status, utc_now_iso(), files_seen, files_indexed, errors, job_id),
        )
        self.conn.commit()

    def stats(self) -> IndexStats:
        files_total = self.conn.execute("SELECT COUNT(*) AS n FROM files WHERE path NOT LIKE 'localmind-chat://%'").fetchone()["n"]
        chunks_total = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        parsed_files = self.conn.execute(
            "SELECT COUNT(*) AS n FROM files WHERE parse_status='parsed' AND path NOT LIKE 'localmind-chat://%'"
        ).fetchone()["n"]
        ocr_done = self.conn.execute(
            "SELECT COUNT(*) AS n FROM files WHERE ocr_status='done'"
        ).fetchone()["n"]
        ocr_pending = self.conn.execute(
            "SELECT COUNT(*) AS n FROM files WHERE ocr_status='pending'"
        ).fetchone()["n"]
        vectors_total = self.conn.execute("SELECT COUNT(*) AS n FROM vectors").fetchone()["n"]
        watched_folders = self.conn.execute(
            "SELECT COUNT(*) AS n FROM watched_folders WHERE enabled=1"
        ).fetchone()["n"]
        last_scan = self.conn.execute(
            "SELECT finished_at FROM scan_jobs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return IndexStats(
            files_total=int(files_total),
            chunks_total=int(chunks_total),
            parsed_files=int(parsed_files),
            ocr_done=int(ocr_done),
            ocr_pending=int(ocr_pending),
            vectors_total=int(vectors_total),
            watched_folders=int(watched_folders),
            db_size_bytes=int(db_size),
            last_scan_at=last_scan["finished_at"] if last_scan else None,
        )
