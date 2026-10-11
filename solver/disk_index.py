"""Exact session indexes with a fixed SQLite page cache, no approximate keys.

Explicit database files live in managed temporary directories, removed on
close. They contain only derived session metadata; durable checkpoints remain JSON.
Keys are the project's immutable tuples/strings, serialized in full, without
hash-only comparisons, eviction, or lost deduplication history.
"""
import json
import sqlite3
import threading
import tempfile
from pathlib import Path


class DiskIndex:
    def __init__(self):
        self._connection = None
        self._directory = None
        self._lock = threading.RLock()
        self._count = 0
        self._closed = False

    def _db(self):
        if self._closed:
            raise RuntimeError('session index is closed')
        if self._connection is None:
            # An empty SQLite filename can remain memory-backed on macOS.
            # Use an explicit main database file so cache_size can truly
            # bound resident pages; this directory is removed on close.
            self._directory = tempfile.TemporaryDirectory(prefix='pqcp-session-index-')
            connection = sqlite3.connect(str(Path(self._directory.name)/'index.sqlite3'),
                check_same_thread=False, isolation_level=None, cached_statements=8)
            connection.execute('PRAGMA journal_mode=OFF')
            connection.execute('PRAGMA synchronous=OFF')
            connection.execute('PRAGMA cache_size=-512')
            connection.execute('PRAGMA mmap_size=0')
            connection.execute('PRAGMA temp_store=FILE')
            connection.execute('CREATE TABLE entries (identity TEXT PRIMARY KEY, value TEXT) WITHOUT ROWID')
            self._connection = connection
        return self._connection

    @staticmethod
    def _key(key):
        hash(key)  # Preserve the old requirement of immutable/hashable keys.
        return json.dumps(key, ensure_ascii=True, separators=(',', ':'))

    def add(self, key):
        """Return True on the first occurrence, including after cache spill."""
        with self._lock:
            cursor = self._db().execute('INSERT OR IGNORE INTO entries VALUES (?, NULL)', (self._key(key),))
            self._count += cursor.rowcount
            return bool(cursor.rowcount)

    def get(self, key, default=None):
        with self._lock:
            if self._connection is None:
                return default
            row = self._connection.execute('SELECT value FROM entries WHERE identity=?', (self._key(key),)).fetchone()
            return json.loads(row[0]) if row and row[0] is not None else default

    def __setitem__(self, key, value):
        with self._lock:
            db = self._db(); identity = self._key(key)
            cursor = db.execute('INSERT OR IGNORE INTO entries VALUES (?, ?)', (identity, json.dumps(value)))
            self._count += cursor.rowcount
            if not cursor.rowcount:
                db.execute('UPDATE entries SET value=? WHERE identity=?', (json.dumps(value), identity))

    def values(self):
        """Stream values; callers must not mutate this index during iteration."""
        if self._connection is not None:
            for row in self._connection.execute('SELECT value FROM entries ORDER BY value'):
                yield json.loads(row[0])

    def clear(self):
        with self._lock:
            if self._connection is not None:
                self._connection.execute('DELETE FROM entries')
            self._count = 0

    def __len__(self):
        return self._count

    def close(self):
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if self._directory is not None:
                self._directory.cleanup()
                self._directory = None
            self._closed = True
