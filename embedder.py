from typing import List

from openai import OpenAI

from models import Row


# Embedding
class OpenAIEmbedder:
    def __init__(self, model: str):
        self.client = OpenAI(
            api_key=""
        )
        self.model = model

    def embed_rows(self, rows: List[Row]) -> List[float]:
        embeddings = self.client.embeddings.create(
            input=[row.values[4] for row in rows], model=self.model
        )

        for i, row in enumerate(rows):
            row.embeddings = embeddings.data[i].embedding

        return rows

    def embed_texts(self, texts: List[str]) -> List[float]:
        embedding_response = self.client.embeddings.create(
            input=texts, model=self.model
        )

        embeddings = []
        for embedding in embedding_response.data:
            embeddings.append(embedding.embedding)

        return embeddings
