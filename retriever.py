import sqlite3
from typing import List

import sqlite_vec
from sqlite_vec import serialize_float32

from embedder import OpenAIEmbedder
from models import Row


class SQLiteVecRetriever:
    def __init__(self, db_path: str, using_virtual_table: bool):
        self.db_path = db_path
        self.search_type = "virtual_table" if using_virtual_table else "normal_table"

    def retrieve(self, query: str, limit: int = 3) -> List[Row]:
        db = sqlite3.connect(self.db_path)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)

        embedder = OpenAIEmbedder("text-embedding-3-small")
        query_embedding = embedder.embed_texts([query])[0]

        match self.search_type:
            case "virtual_table":
                return self._retreive_virtual_table(db, query_embedding, limit)
            case "normal_table":
                return self._retreive_normal_table(db, query_embedding, limit)

    def _retreive_virtual_table(self, db, query_embedding, limit) -> List[Row]:
        # In virtual table, we get the relevant row_id(s) by matching the query embedding with the embeddings of all rows
        # in the virtual table and then using the row_id(s) to fetch the relevant rows from the actual table
        relevant_rows = db.execute(
            f"""
                SELECT
                    row_id,
                    distance
                FROM 
                    vec_items
                WHERE 
                    embedding MATCH ?
                LIMIT ?
            """, (serialize_float32(query_embedding), limit),
        ).fetchall()
        row_distance_map = {row_id: distance for row_id, distance in relevant_rows}

        placeholder = ",".join(["?" for _ in range(len(relevant_rows))])
        sql_query = f"""
                SELECT 
                    row_id,
                    Schedule,
                    Serial_No,
                    Chapter_Heading_Subheading_Tariff_Item,
                    Description_of_Goods,
                    CGST_Rate,
                    SGST_UTGST_Rate,
                    IGST_Rate,
                    Compensation_Cess
                FROM 
                    goods 
                WHERE 
                    rowid IN ({placeholder})
            """
        
        cursor = db.execute(sql_query, list(row_distance_map.keys()))

        results: List[Row] = []
        for row in cursor.fetchall():
            results.append((Row(values=row)))

        # Sort the results by distance because the results are not in order of distance
        # due to the usage of --WHERE row_id IN (...)-- clause
        # Sorting is done with the help of row_distance_map -> row.values[0] is the row_id
        results.sort(key=lambda row: row_distance_map[row.values[0]])
        return results

    def _retreive_normal_table(self, db, query_embedding, limit) -> List[Row]:
        sql_query = """
                SELECT
                    row_id,
                    Schedule,
                    Serial_No,
                    Chapter_Heading_Subheading_Tariff_Item,
                    Description_of_Goods,
                    CGST_Rate,
                    SGST_UTGST_Rate,
                    IGST_Rate,
                    Compensation_Cess,
                    vec_distance_L2(embeddings, ?) as distance
                FROM 
                    goods
                ORDER BY 
                    distance
                LIMIT ?;
                """
        cursor = db.execute(
            sql_query, (serialize_float32(query_embedding), limit)
        )

        # The sorting is done in the SQL query itself due the direct usage of ORDER BY clause
        # Therefore omitting the distance from the results
        results: List[Row] = []
        for row in cursor.fetchall():
            values = row[:-1]
            results.append(Row(values=values))
        
        return results
    

if __name__ == "__main__":
    query = "Give me about milk, cream, powder for babies"
    import time

    retriever_virtual = SQLiteVecRetriever("./virtual_tables.db", using_virtual_table=True)
    start_time = time.time()
    x = retriever_virtual.retrieve(
        query=query,
        limit=100,
    )
    end_time = time.time()
    print("Time taken using virtual table: ", end_time - start_time)


    retriever_normal = SQLiteVecRetriever("./normal_tables.db", using_virtual_table=False)
    start_time = time.time()
    x = retriever_normal.retrieve(
        query=query,
        limit=100,
    )
    end_time = time.time()
    print("Time taken using normal table (Manually): ", end_time - start_time)
