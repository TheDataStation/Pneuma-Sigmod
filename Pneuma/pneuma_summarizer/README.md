# Pneuma-Summarizer

We provide the scripts to generate all summaries used in our experiments, all of which has this naming convention: `[batch_]generate_content_summary_[].py`. You may download the generated row and schema summaries using `summaries/downloader.ipynb`. Alternatively, you can generate the summaries manually by running the following scripts:

```bash
pip install -r requirements.txt
pip install -r ../benchmark_generator/context/requirements.txt

nohup python -u batch_generate_content_summary_llm_narrations.py >> generate_content_summary_llm.out &
nohup python -u generate_content_summary_rows.py >> generate_content_summary_rows.out &
```

Then, adjust the summaries for vector search to account for the limited context window of an embedding model. For this, use the `adjust_summaries_for_vector_search.py` script.

Note: Ensure that you have already downloaded the necessary models using `../models/downloader.ipynb`.
