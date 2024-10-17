import os
import gc

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import torch
import pandas as pd

sys.path.append("..")

from tqdm import tqdm
from collections import defaultdict
from benchmark_generator.context.utils.pipeline_initializer import initialize_pipeline
from benchmark_generator.context.utils.prompting_interface import prompt_pipeline, prompt_pipeline_robust
from benchmark_generator.context.utils.jsonl import write_jsonl, read_jsonl

pipe = initialize_pipeline("../models/qwen", torch.bfloat16, context_length=32768)
# Specific setting for batching
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
pipe.tokenizer.padding_side = "left"


def get_col_description_prompt(columns: str, column: str):
    return f"""A table has the following columns:
/*
{columns}
*/
Describe very briefly what the {column} column represents. If not possible, simply state "No description.\""""

def get_special_indices(texts: list[str], batch_size: int):
    # Step 1: Sort the conversations (indices) in decreasing order
    sorted_indices = sorted(range(len(texts)), key=lambda x: len(texts[x]), reverse=True)

    # Step 2: Interleave the indices (longest, shortest, second longest, second shortest, ...)
    final_indices = []
    i, j = 0, len(sorted_indices) - 1

    while i <= j:
        if i == j:
            final_indices.append(sorted_indices[i])
            break

        final_indices.append(sorted_indices[i])
        i += 1

        for _ in range(batch_size - 1):
            if i <= j:
                final_indices.append(sorted_indices[j])
                j -= 1
            else:
                break
    return final_indices

def is_fit_in_memory(conversations, batch_size: int):
    special_indices = get_special_indices(conversations, batch_size)
    adjusted_conversations = [conversations[i] for i in special_indices]

    conv_low_idx = len(adjusted_conversations)//2 - batch_size // 2
    conv_high_idx = conv_low_idx + batch_size
    output = prompt_pipeline(
        pipe,
        adjusted_conversations[conv_low_idx:conv_high_idx],
        batch_size=batch_size,
        context_length=32768,
        max_new_tokens=1,
        temperature=None,
        top_p=None,
        top_k=None
    )

    torch.cuda.empty_cache()
    gc.collect()

    if output[0][0]["content"] == "":
        del output
        return False
    else:
        del output
        return True

def get_optimal_batch_size(conversations):
    print(f"Looking for an optimal batch size")
    max_batch_size = 50  # Change to a higher value if you have more capacity to explore batch size
    min_batch_size = 1
    while min_batch_size < max_batch_size:
        mid_batch_size = (min_batch_size + max_batch_size) // 2
        print(f"Current mid batch size: {mid_batch_size}")
        if is_fit_in_memory(conversations, mid_batch_size):
            min_batch_size = mid_batch_size + 1
        else:
            max_batch_size = mid_batch_size - 1
    optimal_batch_size = min_batch_size
    print(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size

def generate_llm_narration_summaries(src_path: str, descriptions_path: str):
    tables = sorted([file[:-4] for file in os.listdir(src_path)])
    try:
        summaries = read_jsonl(descriptions_path)
    except:
        summaries: list[dict[str, str]] = []

    conversations = []
    conv_tables = []
    conv_cols = []
    for table in tqdm(tables):
        # Skip already summarized tables
        if table in [summary["table"] for summary in summaries]:
            continue
        df = pd.read_csv(f"{src_path}/{table}.csv", nrows=0)
        cols = df.columns
        for col in cols:
            prompt = get_col_description_prompt(" | ".join(cols), col)
            conversations.append([{"role": "user", "content": prompt}])
            conv_tables.append(table)
            conv_cols.append(col)

    optimal_batch_size = get_optimal_batch_size(conversations)
    sorted_indices = get_special_indices(conversations, optimal_batch_size)

    conversations = [conversations[i] for i in sorted_indices]
    conv_tables = [conv_tables[i] for i in sorted_indices]
    conv_cols = [conv_cols[i] for i in sorted_indices]

    if len(conversations) > 0:
        outputs = []
        max_batch_size = optimal_batch_size
        same_batch_size_counter = 0
        for i in tqdm(range(0, len(conversations), max_batch_size)):
            llm_output = prompt_pipeline_robust(
                pipe,
                conversations[i:i+max_batch_size],
                batch_size=optimal_batch_size,
                context_length=32768,
                max_new_tokens=400,
                temperature=None,
                top_p=None,
                top_k=None,
            )
            outputs += llm_output[0]

            if llm_output[1] == optimal_batch_size:
                same_batch_size_counter += 1
                if same_batch_size_counter % 10 == 0:
                    optimal_batch_size = min(optimal_batch_size+2, max_batch_size)
            else:
                optimal_batch_size = llm_output[1]
                same_batch_size_counter = 0

        col_narrations: dict[str,list[str]] = defaultdict(list)
        for output_idx, output in enumerate(outputs):
            col_narrations[conv_tables[output_idx]] += [f"{conv_cols[output_idx]}: {output[-1]["content"]}"]

        for table in tables:
            summaries.append({
                "id": f"{table}_SEP_contents_SEP_schema",
                "table": table,
                "summary": " | ".join(col_narrations[table]),
            })
            write_jsonl(summaries, descriptions_path)
    summaries = sorted(summaries, key=lambda x: x["table"])
    write_jsonl(summaries, descriptions_path)

if __name__ == "__main__":
    start = time.time()
    src_path = "../data_src/tables/pneuma_chembl_10K"
    descriptions_path = "chembl.jsonl"
    generate_llm_narration_summaries(src_path, descriptions_path)
    end = time.time()
    print(f"Total time: {end - start} seconds")

    start = time.time()
    src_path = "../data_src/tables/pneuma_adventure_works"
    descriptions_path = "adventure.jsonl"
    generate_llm_narration_summaries(src_path, descriptions_path)
    end = time.time()
    print(f"Total time: {end - start} seconds")

    start = time.time()
    src_path = "../data_src/tables/pneuma_fetaqa"
    descriptions_path = "fetaqa.jsonl"
    generate_llm_narration_summaries(src_path, descriptions_path)
    end = time.time()
    print(f"Total time: {end - start} seconds")

    start = time.time()
    src_path = "../data_src/tables/pneuma_chicago_10K"
    descriptions_path = "chicago.jsonl"
    generate_llm_narration_summaries(src_path, descriptions_path)
    end = time.time()
    print(f"Total time: {end - start} seconds")

    start = time.time()
    src_path = "../data_src/tables/pneuma_public_bi"
    descriptions_path = "public.jsonl"
    generate_llm_narration_summaries(src_path, descriptions_path)
    end = time.time()
    print(f"Total time: {end - start} seconds")
