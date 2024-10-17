# Pneuma-Summarizer

We provide the scripts to generate all summaries used in our experiments, all of which has this naming convention: `generate_content_summary_[].py`. You may download all the generated summaries (except for the DBReader variant) using `summaries/downloader.ipynb`. Alternatively, you can generate the summaries manually by running the following scripts:

```bash
pip install -r requirements.txt
pip install -r ../benchmark_generator/context/requirements.txt

nohup python -u generate_content_summary_dbreader.py >> generate_content_summary_dbreader.out &
nohup python -u generate_content_summary_llm.py >> generate_content_summary_llm.out &
nohup python -u generate_content_summary_rows.py >> generate_content_summary_rows.out &
nohup python -u generate_content_summary_std.py >> generate_content_summary_std.out &
```

Then, adjust the summaries for vector search to account for the limited context window of an embedding model.

Note: Ensure that you have already downloaded the necessary models using `../models/downloader.ipynb`.
