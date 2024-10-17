# Context Benchmark Generator

We provide the script to generate the context benchmarks, including the contexts and the questions. You may directly download the generated files using `../../data_src/benchmarks/context/downloader.ipynb`, but you can also generate them manually by doing the following steps:

1. Download the LLM using `../../models/downloader.ipynb`.
2. Install the requirements with `pip install -r requirements.txt`.
3. Run the generator with `nohup python -u generate_benchmark.py >> generate_benchmark.out &`.
4. Perform processing for BX2 questions using `final_processing.ipynb`.
5. Further rephrase the BX2 questions using `extra_rephrase.py`, then process it with `final_processing.ipynb`.
