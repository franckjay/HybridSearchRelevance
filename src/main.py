import logging
import pandas as pd
from data.search_entries import journal
from utils.vector_database import build_vector_index
from utils.bm25_db import build_bm25_index

# Build the search indices on program start
VECTOR_INDEX = build_vector_index(journal)
LEXICAL_INDEX = build_bm25_index(journal)


def main(query_str: str, get_k_results: int) -> None:

    logging.info("Job done!")
    vector_results = VECTOR_INDEX.search(query_str, get_k_results)
    bm25_results = LEXICAL_INDEX.search(query_str, get_k_results)
    combined_results = pd.merge(
        vector_results, bm25_results, on=["doc_id", "text"], how="outer"
    ).fillna(0.0)

    max_rank = combined_results.count()[0]
    combined_results["rrf_score"] = 1.0 / (
        max_rank + combined_results["vector_rank"]
    ) + 1.0 / (max_rank + combined_results["bm25_rank"])
    combined_results = combined_results.sort_values(by="rrf_score", ascending=False)

    return combined_results["text"][:get_k_results]


if __name__ == "__main__":
    # TODO: run argparse here
    logging.info("Starting job.")
    top_k_results = 10
    main("trail", top_k_results)
