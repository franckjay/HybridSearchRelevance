from base_db import SearchEngine
from txtai.scoring import ScoringFactory
import pandas as pd


class LexicalIndex(SearchEngine):
    def index_data(self) -> None:
        self.engine = ScoringFactory.create({"method": "bm25", "terms": True})
        self.engine.index((x, text, None) for x, text in enumerate(self.data))

    def search(self, query: str, k: int = 10) -> pd.DataFrame:
        if self.engine is None:
            raise NotImplementedError("Classic Search Engine is not built yet.")
        _dict = {"doc_id": [], "text": [], "bm25_score": [], "bm25_rank": []}
        rank = 1
        for idx, score in self.engine.search(query, k):
            _dict["doc_id"].append(idx)
            _dict["text"].append(self.data[idx])
            _dict["bm25_score"].append(score)
            _dict["bm25_rank"].append(rank)
            rank += 1
        return pd.DataFrame(_dict)


def build_bm25_index(data_entries: list[str]) -> LexicalIndex:
    return LexicalIndex(data_entries)
