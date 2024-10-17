import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import torch
import pandas as pd

sys.path.append("..")

from tqdm import tqdm
from benchmark_generator.context.utils.pipeline_initializer import initialize_pipeline
from benchmark_generator.context.utils.prompting_interface import prompt_pipeline
from benchmark_generator.context.utils.jsonl import write_jsonl, read_jsonl

pipe = initialize_pipeline("../models/llama", torch.bfloat16)
# Specific setting for Llama-3-8B-Instruct for batching
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
pipe.tokenizer.padding_side = "left"


def get_col_description_prompt(columns: str, column: str):
    return f"""A table has the following columns:
/*
{columns}
*/
Describe briefly what the {column} column represents. If not possible, simply state "No description.\""""


def generate_llm_narration_summaries(src_path: str, descriptions_path: str):
    tables = sorted([file[:-4] for file in os.listdir(src_path)])
    try:
        summaries = read_jsonl(descriptions_path)
    except:
        summaries: list[dict[str, str]] = []

    bar = tqdm(tables)
    for table in bar:
        # Skip already summarized tables
        if table in [summary["table"] for summary in summaries]:
            continue
        bar.set_description(f"Processing table {table}")
        df = pd.read_csv(f"{src_path}/{table}.csv", on_bad_lines="skip")
        cols = df.columns
        conversations = []
        for col in cols:
            prompt = get_col_description_prompt(" | ".join(cols), col)
            conversations.append([{"role": "user", "content": prompt}])
        
        if len(conversations) > 0:
            outputs = prompt_pipeline(
                pipe,
                conversations,
                batch_size=2,
                context_length=8192,
                max_new_tokens=400,
                temperature=None,
                top_p=None,
            )

            col_narrations: list[str] = []
            for output_idx, output in enumerate(outputs):
                col_narrations.append(f"{cols[output_idx]}: {output[-1]["content"]}")
            
            summaries.append({
                "id": f"{table}_SEP_contents_SEP_schema",
                "table": table,
                "summary": " | ".join(col_narrations),
            })
            write_jsonl(summaries, descriptions_path)


if __name__ == "__main__":
    src_path = "../data_src/tables/pneuma_chembl_10K"
    descriptions_path = "summaries/narrations/chembl.jsonl"
    generate_llm_narration_summaries(src_path, descriptions_path)

    src_path = "../data_src/tables/pneuma_adventure_works"
    descriptions_path = "summaries/narrations/adventure.jsonl"
    generate_llm_narration_summaries(src_path, descriptions_path)

    src_path = "../data_src/tables/pneuma_public_bi"
    descriptions_path = "summaries/narrations/public.jsonl"
    generate_llm_narration_summaries(src_path, descriptions_path)

    src_path = "../data_src/tables/pneuma_chicago_10K"
    descriptions_path = "summaries/narrations/chicago.jsonl"
    generate_llm_narration_summaries(src_path, descriptions_path)

    src_path = "../data_src/tables/pneuma_fetaqa"
    descriptions_path = "summaries/narrations/fetaqa.jsonl"
    generate_llm_narration_summaries(src_path, descriptions_path)
