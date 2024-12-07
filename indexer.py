import sqlite3
from typing import List

import sqlite_vec
from sqlite_vec import serialize_float32

from embedder import OpenAIEmbedder
from models import Row


class SQLiteVecIndexer:
    def __init__(self, db_path: str, using_virtual_table: bool):
        self.db_path = db_path

        self.index_type = "virtual_table" if using_virtual_table else "normal_table"
        self.virtual_table = "vec_items"

    def insert_rows(self, rows: List[Row]) -> None:
        db = sqlite3.connect(self.db_path)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
    
        embedder = OpenAIEmbedder(model="text-embedding-3-small")
        rows = embedder.embed_rows(rows)

        match self.index_type:
            case "virtual_table":
                self._insert_rows_virtual_table(db, rows)
            case "normal_table":
                self._insert_rows_normal_table(db, rows)

    def if_not_exists_create_virtual_table(self, db: sqlite3.Connection, embedding_dim: int = 1536) -> None:
        try:
            db.execute(f"SELECT * FROM {self.virtual_table} LIMIT 1").fetchone()
        except sqlite3.OperationalError:
            db.execute(
                f"""CREATE VIRTUAL TABLE {self.virtual_table} USING vec0(
                    row_id integer primary key,
                    embedding float[{embedding_dim}]
                )"""
            )

    def _insert_rows_virtual_table(self, db: sqlite3.Connection, rows: List[Row]) -> None:
        self.if_not_exists_create_virtual_table(db)

        place_holder = ",".join(["?" for _ in range(len(rows[0].values))])

        for row in rows:
            row_id = row.values[0]

            db.execute(f"INSERT INTO goods VALUES ({place_holder})", row.values)

            db.execute(
                f"INSERT INTO {self.virtual_table} VALUES (?, ?)",
                (row_id, serialize_float32(row.embeddings)),
            )

        db.commit()

    def _insert_rows_normal_table(self, db: sqlite3.Connection, rows: List[Row]) -> None:
        place_holder = ",".join(["?" for _ in range(len(rows[0].values) + 1)])

        for row in rows:
            db.execute(f"INSERT INTO goods VALUES ({place_holder})", (*row.values, serialize_float32(row.embeddings)))
        db.commit()
