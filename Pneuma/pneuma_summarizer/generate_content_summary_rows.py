import sys

sys.path.append("..")


import math
import os
import pandas as pd

from tqdm import tqdm
from benchmark_generator.context.utils.jsonl import write_jsonl


def generate_row_summaries(table_path: str, summary_path: str):
    content_summaries = []
    tables = sorted(os.listdir(table_path))
    for table_idx, table in enumerate(tqdm(tables)):
        df = pd.read_csv(f"{table_path}/{table}", on_bad_lines="skip")
        sample_size = math.ceil(min(len(df), 5))

        selected_df = df.sample(n=sample_size, random_state=table_idx).reset_index(
            drop=True
        )
        for df_idx, row in selected_df.iterrows():
            formatted_row = " | ".join([f"{col}: {val}" for col, val in row.items()])
            content_summaries.append(
                {
                    "id": f"{table[:-4]}_SEP_contents_SEP_row-{df_idx}",
                    "table": table[:-4],
                    "summary": formatted_row,
                }
            )
    write_jsonl(content_summaries, summary_path)


if __name__ == "__main__":
    name = "chembl"
    table_path = "../data_src/tables/pneuma_chembl_10K"
    summary_path = f"summaries/rows/{name}.jsonl"
    generate_row_summaries(table_path, summary_path)

    name = "adventure"
    table_path = "../data_src/tables/pneuma_adventure_works"
    summary_path = f"summaries/rows/{name}.jsonl"
    generate_row_summaries(table_path, summary_path)

    name = "public"
    table_path = "../data_src/tables/pneuma_public_bi"
    summary_path = f"summaries/rows/{name}.jsonl"
    generate_row_summaries(table_path, summary_path)

    name = "chicago"
    table_path = "../data_src/tables/pneuma_chicago_10K"
    summary_path = f"summaries/rows/{name}.jsonl"
    generate_row_summaries(table_path, summary_path)

    name = "fetaqa"
    table_path = "../data_src/tables/pneuma_fetaqa"
    summary_path = f"summaries/rows/{name}.jsonl"
    generate_row_summaries(table_path, summary_path)
