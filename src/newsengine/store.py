"""Step 7 (storage): DuckDB-backed store of enriched articles.

Schema additions over the minimal version
-----------------------------------------
* ``tickers``    VARCHAR  – comma-joined list for indexed text search
* ``event_type`` VARCHAR  – enables O(1) filter pushdown to DuckDB
* Separate indexes on (published, event_type, tickers) allow the API
  endpoints to filter without deserialising every JSON payload.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import duckdb

from .models import Article

_DDL = """
CREATE TABLE IF NOT EXISTS articles (
    id          VARCHAR PRIMARY KEY,
    published   VARCHAR NOT NULL,
    source      VARCHAR NOT NULL,
    event_type  VARCHAR NOT NULL DEFAULT 'other',
    tickers     VARCHAR NOT NULL DEFAULT '',
    payload     VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_published   ON articles (published DESC);
CREATE INDEX IF NOT EXISTS idx_event_type  ON articles (event_type);
CREATE INDEX IF NOT EXISTS idx_tickers     ON articles (tickers);
"""


class Store:
    def __init__(self, path: str = "news.duckdb"):
        self.con = duckdb.connect(path)
        for stmt in _DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self.con.execute(stmt)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def upsert(self, articles: list[Article]) -> None:
        """Insert or replace articles (idempotent on article.id)."""
        for a in articles:
            d = a.to_dict()
            self.con.execute(
                """INSERT OR REPLACE INTO articles
                   (id, published, source, event_type, tickers, payload)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    a.id,
                    d["published"],
                    a.source,
                    a.event_type,
                    ",".join(a.tickers),
                    json.dumps(d),
                ],
            )

    # ------------------------------------------------------------------
    # Read – raw dicts
    # ------------------------------------------------------------------
    def all(self) -> list[dict]:
        rows = self.con.execute(
            "SELECT payload FROM articles ORDER BY published DESC"
        ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def query(
        self,
        *,
        ticker: str | None = None,
        event_type: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Filtered query – pushes predicates to DuckDB for efficiency."""
        clauses, params = [], []
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if ticker:
            # tickers column stores comma-separated values; use LIKE for
            # simple containment (works for the demo universe size).
            clauses.append("tickers LIKE ?")
            params.append(f"%{ticker.upper()}%")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.con.execute(
            f"SELECT payload FROM articles {where} ORDER BY published DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [json.loads(r[0]) for r in rows]

    # ------------------------------------------------------------------
    # Read – Article objects
    # ------------------------------------------------------------------
    @staticmethod
    def _to_article(d: dict) -> Article:
        d = {k: v for k, v in d.items() if k != "id"}
        d["published"] = datetime.fromisoformat(d["published"]).astimezone(timezone.utc)
        return Article(**d)

    def articles(self) -> list[Article]:
        return [self._to_article(d) for d in self.all()]

    def query_articles(
        self,
        *,
        ticker: str | None = None,
        event_type: str | None = None,
        limit: int = 500,
    ) -> list[Article]:
        return [self._to_article(d) for d in self.query(ticker=ticker, event_type=event_type, limit=limit)]
