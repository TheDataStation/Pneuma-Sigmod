import os
import time
import sys
import chromadb

sys.path.append("..")

from transformers import set_seed
from chromadb.api.client import Client
from sentence_transformers import SentenceTransformer
from sentence_transformers.SentenceTransformer import SentenceTransformer
from benchmark_generator.context.utils.jsonl import read_jsonl


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_seed(42, deterministic=True)


embedding_model = SentenceTransformer("../models/bge-base", local_files_only=True)


def indexing_vector(
    client: Client,
    embedding_model: SentenceTransformer,
    schema_contents: list[dict[str, str]],
    row_contents: list[dict[str, str]],
    contexts: list[dict[str, str]] = None,
    collection_name="benchmark",
    reindex=False,
):
    documents = []
    ids = []

    if not reindex:
        try:
            collection = client.get_collection(collection_name)
            return collection
        except:
            pass
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:random_seed": 42,
            "hnsw:M": 48,
        },
    )

    tables = sorted({content["table"] for content in schema_contents})
    for table in tables:
        table_schema_contents = [
            content for content in schema_contents if content["table"] == table
        ]
        table_row_contents = [
            content for content in row_contents if content["table"] == table
        ]

        for idx, schema_content in enumerate(table_schema_contents):
            documents.append(schema_content["summary"])
            ids.append(f"{table}_SEP_contents_SEP_schema-{idx}")

        for idx, row_content in enumerate(table_row_contents):
            documents.append(row_content["summary"])
            ids.append(f"{table}_SEP_contents_SEP_row-{idx}")

        if contexts is not None:
            table_contexts = [
                context for context in contexts if context["table"] == table
            ]
            for context_idx, context in enumerate(table_contexts):
                documents.append(context["context"])
                ids.append(f"{table}_SEP_contexts-{context_idx}")

    for i in range(0, len(documents), 30000):
        embeddings = embedding_model.encode(
            documents[i : i + 30000],
            batch_size=100,
            show_progress_bar=True,
            device="cuda",
        )

        collection.add(
            embeddings=[embed.tolist() for embed in embeddings],
            documents=documents[i : i + 30000],
            ids=ids[i : i + 30000],
        )
    return collection


def start_indexing(dataset, schema_contents, row_contents, contexts):
    print(f"Indexing dataset: {dataset}")
    start = time.time()
    client = chromadb.PersistentClient(f"indices/index-{dataset}-pneuma-summarizer")
    indexing_vector(client, embedding_model, schema_contents, row_contents, contexts)
    end = time.time()
    print(f"Indexing time: {end-start} seconds")


def get_information(dataset: str):
    """
    Return the contents, contexts, and context benchmarks of a dataset
    """
    schema_contents = read_jsonl(
        f"../pneuma_summarizer/summaries/narrations/{dataset}_splitted.jsonl"
    )
    row_contents = read_jsonl(f"../pneuma_summarizer/summaries/rows/{dataset}_merged.jsonl")
    contexts = read_jsonl(
        f"../data_src/benchmarks/context/{dataset}/contexts_{dataset}_merged.jsonl"
    )
    return [schema_contents, row_contents, contexts]


if __name__ == "__main__":
    dataset = "chembl"
    schema_contents, row_contents, contexts = get_information(dataset)
    start_indexing(dataset, schema_contents, row_contents, contexts)

    dataset = "adventure"
    schema_contents, row_contents, contexts = get_information(dataset)
    start_indexing(dataset, schema_contents, row_contents, contexts)

    dataset = "public"
    schema_contents, row_contents, contexts = get_information(dataset)
    start_indexing(dataset, schema_contents, row_contents, contexts)

    dataset = "chicago"
    schema_contents, row_contents, contexts = get_information(dataset)
    start_indexing(dataset, schema_contents, row_contents, contexts)

    dataset = "fetaqa"
    schema_contents, row_contents, contexts = get_information(dataset)
    start_indexing(dataset, schema_contents, row_contents, contexts)
