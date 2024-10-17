import os
import sys
import pandas as pd

sys.path.append("..")

from tqdm import tqdm
from benchmark_generator.context.utils.jsonl import write_jsonl


def generate_std_summaries(tables_path: str, summaries_name: str):
    content_summaries: list[dict[str, str]] = []
    tables = [
        table[:-4]
        for table in sorted(os.listdir(tables_path))
        if table.endswith(".csv")
    ]

    for table in tqdm(tables, "processing tables"):
        df = pd.read_csv(f"{tables_path}/{table}.csv")
        summary = {
            "id": f"{table}_SEP_contents_SEP_schema",
            "table": table,
            "summary": " | ".join(df.columns),
        }
        content_summaries.append(summary)

    write_jsonl(content_summaries, f"summaries/standard/{summaries_name}")


if __name__ == "__main__":
    tables_path = "../data_src/tables/pneuma_chembl_10K"
    summaries_name = "chembl.jsonl"
    generate_std_summaries(tables_path, summaries_name)

    tables_path = "../data_src/tables/pneuma_adventure_works"
    summaries_name = "adventure.jsonl"
    generate_std_summaries(tables_path, summaries_name)

    tables_path = "../data_src/tables/pneuma_public_bi"
    summaries_name = "public.jsonl"
    generate_std_summaries(tables_path, summaries_name)

    tables_path = "../data_src/tables/pneuma_chicago_10K"
    summaries_name = "chicago.jsonl"
    generate_std_summaries(tables_path, summaries_name)

    tables_path = "../data_src/tables/pneuma_fetaqa"
    summaries_name = "fetaqa.jsonl"
    generate_std_summaries(tables_path, summaries_name)
