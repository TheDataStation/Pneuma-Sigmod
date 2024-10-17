import setproctitle

setproctitle.setproctitle("python3.12")
import os
import time
import sys
import Stemmer
import numpy as np

sys.path.append("..")


from transformers import set_seed
from sentence_transformers import SentenceTransformer
from sentence_transformers.SentenceTransformer import SentenceTransformer
from benchmark_generator.context.utils.jsonl import read_jsonl


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_seed(42, deterministic=True)


embedding_model = SentenceTransformer("../models/bge-base", local_files_only=True)
stemmer = Stemmer.Stemmer("english")


def get_question_key(benchmark_type: str, use_rephrased_questions: bool):
    if benchmark_type == "content":
        if not use_rephrased_questions:
            print("Processing BC1")
            question_key = "question_from_sql_1"
        else:
            print("Processing BC2")
            question_key = "question"
    else:
        if not use_rephrased_questions:
            print("Processing BX1")
            question_key = "question_bx1"
        else:
            print("Processing BX2")
            question_key = "question_bx2"
    return question_key


def produce_embeddings(
    dataset: str,
    benchmark: list[dict[str, str]],
    benchmark_type: str,
    embedding_model: SentenceTransformer,
    use_rephrased_questions=False,
):
    start = time.time()
    print(f"Producing embeddings for benchmark questions of {dataset} dataset")

    questions = []
    question_key = get_question_key(benchmark_type, use_rephrased_questions)
    for data in benchmark:
        questions.append(data[question_key])
    embed_questions = embedding_model.encode(
        questions, batch_size=64, show_progress_bar=True
    )
    np.savetxt(
        f"embeddings/embed-{dataset}-questions-{benchmark_type}-{use_rephrased_questions}.txt",
        embed_questions,
    )

    end = time.time()
    print(f"Embedding production time: {end - start} seconds")


if __name__ == "__main__":
    dataset = "public"
    content_benchmark = read_jsonl(
        "../data_src/benchmarks/content/pneuma_public_bi_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        "../data_src/benchmarks/context/public/bx_public.jsonl"
    )
    produce_embeddings(dataset, content_benchmark, "content", embedding_model, False)
    produce_embeddings(dataset, content_benchmark, "content", embedding_model, True)
    produce_embeddings(dataset, context_benchmark, "context", embedding_model, False)
    produce_embeddings(dataset, context_benchmark, "context", embedding_model, True)

    dataset = "chembl"
    content_benchmark = read_jsonl(
        "../data_src/benchmarks/content/pneuma_chembl_10K_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        "../data_src/benchmarks/context/chembl/bx_chembl.jsonl"
    )
    produce_embeddings(dataset, content_benchmark, "content", embedding_model, False)
    produce_embeddings(dataset, content_benchmark, "content", embedding_model, True)
    produce_embeddings(dataset, context_benchmark, "context", embedding_model, False)
    produce_embeddings(dataset, context_benchmark, "context", embedding_model, True)

    dataset = "adventure"
    content_benchmark = read_jsonl(
        "../data_src/benchmarks/content/pneuma_adventure_works_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        "../data_src/benchmarks/context/adventure/bx_adventure.jsonl"
    )
    produce_embeddings(dataset, content_benchmark, "content", embedding_model, False)
    produce_embeddings(dataset, content_benchmark, "content", embedding_model, True)
    produce_embeddings(dataset, context_benchmark, "context", embedding_model, False)
    produce_embeddings(dataset, context_benchmark, "context", embedding_model, True)

    dataset = "chicago"
    content_benchmark = read_jsonl(
        "../data_src/benchmarks/content/pneuma_chicago_10K_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        "../data_src/benchmarks/context/chicago/bx_chicago.jsonl"
    )
    produce_embeddings(dataset, content_benchmark, "content", embedding_model, False)
    produce_embeddings(dataset, content_benchmark, "content", embedding_model, True)
    produce_embeddings(dataset, context_benchmark, "context", embedding_model, False)
    produce_embeddings(dataset, context_benchmark, "context", embedding_model, True)

    dataset = "fetaqa"
    content_benchmark = read_jsonl(
        "../data_src/benchmarks/content/pneuma_fetaqa_questions_annotated.jsonl"
    )
    context_benchmark = read_jsonl(
        "../data_src/benchmarks/context/fetaqa/bx_fetaqa.jsonl"
    )
    produce_embeddings(dataset, content_benchmark, "content", embedding_model, False)
    produce_embeddings(dataset, content_benchmark, "content", embedding_model, True)
    produce_embeddings(dataset, context_benchmark, "context", embedding_model, False)
    produce_embeddings(dataset, context_benchmark, "context", embedding_model, True)
