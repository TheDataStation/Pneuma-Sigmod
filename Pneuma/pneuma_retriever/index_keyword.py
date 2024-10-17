import sys

sys.path.append("..")

import bm25s
import time
import Stemmer
from benchmark_generator.context.utils.jsonl import read_jsonl

stemmer = Stemmer.Stemmer("english")


def indexing_keyword(
    stemmer,
    schema_contents: list[dict[str, str]],
    row_contents: list[dict[str, str]],
    contexts: list[dict[str, str]],
):
    corpus_json = []
    tables = sorted({content["table"] for content in schema_contents})
    for table in tables:
        table_schema_contents = [content for content in schema_contents if content["table"] == table]
        table_row_contents = [content for content in row_contents if content["table"] == table]

        for idx, schema_content in enumerate(table_schema_contents):
            corpus_json.append(
                {
                    "text": schema_content["summary"],
                    "metadata": {"table": f"{table}_SEP_contents_SEP_schema-{idx}"},
                }
            )
        
        for idx, row_content in enumerate(table_row_contents):
            corpus_json.append(
                {
                    "text": row_content["summary"],
                    "metadata": {"table": f"{table}_SEP_contents_SEP_row-{idx}"},
                }
            )
        
        if contexts is not None:
            table_contexts = [
                context for context in contexts if context["table"] == table
            ]
            for context_idx, context in enumerate(table_contexts):
                corpus_json.append(
                    {
                        "text": context["context"],
                        "metadata": {"table": f"{table}_SEP_contexts-{context_idx}"},
                    }
                )

    corpus_text = [doc["text"] for doc in corpus_json]
    corpus_tokens = bm25s.tokenize(
        corpus_text, stopwords="en", stemmer=stemmer, show_progress=False
    )

    retriever = bm25s.BM25(corpus=corpus_json)
    retriever.index(corpus_tokens, show_progress=True)
    retriever.save(f"indices/keyword-index-{dataset}-pneuma-summarizer")


def get_information(dataset: str):
    """
    Return the contents and contexts of a dataset
    """
    schema_contents = read_jsonl(
        f"../pneuma_summarizer/summaries/narrations/{dataset}_splitted.jsonl"
    )
    row_contents = read_jsonl(
        f"../pneuma_summarizer/summaries/rows/{dataset}_merged.jsonl"
    )
    contexts = read_jsonl(
        f"../data_src/benchmarks/context/{dataset}/contexts_{dataset}_merged.jsonl"
    )
    return [schema_contents, row_contents, contexts]


if __name__ == "__main__":
    start = time.time()
    dataset = "chembl"
    schema_contents, row_contents, contexts = get_information(dataset)
    indexing_keyword(stemmer, schema_contents, row_contents, contexts)
    end = time.time()
    print(f"Indexing time: {end-start} seconds")

    start = time.time()
    dataset = "adventure"
    schema_contents, row_contents, contexts = get_information(dataset)
    indexing_keyword(stemmer, schema_contents, row_contents, contexts)
    end = time.time()
    print(f"Indexing time: {end-start} seconds")

    start = time.time()
    dataset = "public"
    schema_contents, row_contents, contexts = get_information(dataset)
    indexing_keyword(stemmer, schema_contents, row_contents, contexts)
    end = time.time()
    print(f"Indexing time: {end-start} seconds")

    start = time.time()
    dataset = "chicago"
    schema_contents, row_contents, contexts = get_information(dataset)
    indexing_keyword(stemmer, schema_contents, row_contents, contexts)
    end = time.time()
    print(f"Indexing time: {end-start} seconds")

    start = time.time()
    dataset = "fetaqa"
    schema_contents, row_contents, contexts = get_information(dataset)
    indexing_keyword(stemmer, schema_contents, row_contents, contexts)
    end = time.time()
    print(f"Indexing time: {end-start} seconds")
