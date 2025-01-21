from base_db import SearchEngine
from txtai import Embeddings
import pandas as pd


class VectorIndex(SearchEngine):
    def index_data(self):
        self.engine = Embeddings(
            path="Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True
        )
        self.engine.index(self.data)

    def search(self, query: str, k: int = 10) -> pd.DataFrame:
        if self.engine is None:
            raise NotImplementedError("Vector Search Engine is not built yet.")
        _dict = {"doc_id": [], "text": [], "vector_score": [], "vector_rank": []}
        rank = 1
        for idx, score in self.engine.search(query, k):
            _dict["doc_id"].append(idx)
            _dict["text"].append(self.data[idx])
            _dict["vector_score"].append(score)
            _dict["vector_rank"].append(rank)
            rank += 1
        return pd.DataFrame(_dict)


def build_vector_index(data_entries: list[str]) -> VectorIndex:
    return VectorIndex(data_entries)
