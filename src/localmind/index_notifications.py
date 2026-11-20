"""Read committed search revisions without a schema-writing DB constructor."""
from __future__ import annotations

import sqlite3
from pathlib import Path


class IndexRevisionProbe:
    def __init__(self, db_path: str | Path) -> None:
        self.uri = Path(db_path).resolve().as_uri() + "?mode=ro"
        self._conn: sqlite3.Connection | None = None

    def read(self) -> int:
        try:
            if self._conn is None:
                self._conn = sqlite3.connect(self.uri, uri=True, timeout=0, isolation_level=None)
            cursor = self._conn.execute("SELECT revision FROM index_revisions WHERE id=1")
            try:
                row = cursor.fetchone()
            finally:
                # Do not keep a WAL read snapshot between timer ticks.
                cursor.close()
            if row is None:
                raise sqlite3.DatabaseError("Missing index revision record")
            return int(row[0])
        except sqlite3.Error:
            self.close()
            raise

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
