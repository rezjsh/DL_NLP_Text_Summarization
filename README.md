## DL_NLP_Text_Summarization

End‑to‑end text summarization toolkit with multiple extractive and abstractive methods, a simple Flask UI, and evaluation utilities.

### Features
- **Extractive**: TF‑IDF, TextRank, LSA, BERT (embedding‑based)
- **Abstractive**: T5 (via Hugging Face `transformers`)
- **Evaluation**: ROUGE and BLEU (optional, auto‑skips if unavailable)
- **App**: Minimal Flask web UI in `app.py`
- **Logging**: Structured logs to `logs/running_logs.log`

### Repository layout
- `Text_Summarization/src/` – library code (summarizers, factory, utils)
- `Text_Summarization/main.py` – example driver + evaluation to JSON
- `app.py` – Flask web app
- `Text_Summarization/config/config.yaml` and `Text_Summarization/params.yaml` – configs
- `Text_Summarization/environment.yml` – conda environment (recommended)

---

### Quickstart

1) Create and activate the conda environment

```bash
conda env create -f Text_Summarization/environment.yml
conda activate text-summarization
```

2) Install the package in editable mode (already referenced in the env file; safe to re‑run)

```bash
pip install -e .
```

3) (First run) Download required NLTK data

The scripts call a helper to download NLTK resources automatically. If you prefer manual install:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
try:
    nltk.download('punkt_tab')  # fixes LookupError in some environments
except Exception:
    pass
```

---

### Usage

#### A) As a library

```python
from Text_Summarization.src.components.summarizer import Summarizer

text = """
Deep learning has transformed natural language processing. Summarization condenses long text into key points.
"""

summarizer = Summarizer(language='english')

# Extractive examples
print(summarizer.summarize_text(text, method='tfidf', num_sentences=3))
print(summarizer.summarize_text(text, method='textrank', num_sentences=3))
print(summarizer.summarize_text(text, method='lsa', num_sentences=3))
print(summarizer.summarize_text(text, method='bert_extractive', num_sentences=3))

# Abstractive (requires transformers + torch)
print(summarizer.summarize_text(text, method='t5', max_length=60, min_length=20))
```

Supported methods: `tfidf`, `textrank`, `lsa`, `bert_extractive`, `t5`.

#### B) Run the example/evaluation script

```bash
python Text_Summarization/main.py
```

This will:
- Download NLTK resources on first run
- Generate summaries with multiple methods
- Evaluate (if `rouge_score` and `nltk` BLEU are available)
- Save metrics to `evaluation_results.json`

#### C) Launch the Flask app

```bash
python app.py
# open http://localhost:5000
```

Select a method, paste text, choose output length, and submit.

---

### Configuration

- `Text_Summarization/config/config.yaml` – general settings (extend as needed)
- `Text_Summarization/params.yaml` – algorithm parameters (optional)

The summarizers are created via `SummarizerFactory` and accept method‑specific kwargs:
- Extractive: `num_sentences`
- Abstractive (T5): `max_length`, `min_length`

---

### Requirements and environment

Recommended: use the provided conda env `Text_Summarization/environment.yml`:
- Python 3.10
- Core libs: `numpy<2`, `scikit-learn`, `networkx`, `nltk`, `pyyaml`, `tqdm`
- DL stack: `transformers`, `torch`, `keras<3`, `tf-keras`
- Utils: `python-box`, `rouge_score`, `flask`, `huggingface-hub[hf_xet]`

Notes:
- GPU is optional; if available, PyTorch will use it automatically for BERT/T5.
- Hugging Face models are downloaded on first use and cached in your user cache.

---

### Troubleshooting

- **NLTK LookupError (punkt/punkt_tab/stopwords)**: Ensure the downloads complete. See Quickstart step 3.
- **CUDA not available**: Install a CUDA‑enabled PyTorch per the official selector, or run CPU‑only.
- **Model download/auth errors**: Check internet connectivity and any private model access tokens in Hugging Face. You can pre‑download models using `transformers` CLI or Python API.
- **Large inputs truncated (T5)**: Increase `max_length` and `min_length`, but note model limits; very long inputs are truncated by tokenizers.
- **Import errors for evaluation**: If `rouge_score` or BLEU is missing, evaluation is skipped automatically.

---

### Development

```bash
conda activate text-summarization
pip install -e .
pytest -q   # if you add tests
```

Package metadata is in `setup.py` (package root is `Text_Summarization`).

---

### License

See `LICENSE`.

