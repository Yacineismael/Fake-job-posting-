# Fake Job Posting Detector

A Streamlit web application that detects fraudulent job postings using three complementary AI approaches.

## Demo

Enter a job posting (title, company profile, description, requirements, benefits) and get instant predictions from three models with a majority-vote ensemble verdict.

## Approaches

| Model | Type | Accuracy | AUC-ROC |
|-------|------|----------|---------|
| **BiLSTM** (trained from scratch) | Classic deep learning | 94.91% | 0.9617 |
| **BART-large-mnli** (HuggingFace) | Zero-shot NLI | 94.67%* | 0.552 |
| **GPT-OSS-120B** (OpenRouter API) | LLM with explanation | 62.5% | N/A |

> *BART's high accuracy is misleading — the dataset is ~95% real postings (class imbalance). Its F1=0.00 on fake postings shows it detects none without fine-tuning.

## Dataset

- **Source**: [Employment Scam Aegean Dataset (EMSCAD)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Size**: 17 880 job postings (~4.8% fraudulent)
- **File**: `fake_job_postings.csv`

## Project Structure

```
├── app.py                  # Streamlit application
├── main.py                 # Training script (BiLSTM)
├── fakejob.ipynb           # Exploratory analysis & model comparison notebook
├── lstm_model.keras        # Trained BiLSTM model weights
├── keras_tokenizer.pkl     # Fitted Keras tokenizer
├── fake_job_postings.csv   # Raw dataset
├── requirements.txt        # Python dependencies
├── class_distribution.png  # Dataset class balance chart
├── lstm_training.png       # BiLSTM training curves
├── lstm_cm.png             # BiLSTM confusion matrix
├── model_comparison.png    # Side-by-side model comparison
└── model_comparison.csv    # Numeric comparison results
```

## Installation

```bash
git clone https://github.com/Yacineismael/Fake-job-posting-.git
cd Fake-job-posting-
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

For the OpenRouter (GPT) tab, add your free API key from [openrouter.ai](https://openrouter.ai) in the sidebar.

## Model Details

### BiLSTM (Classic AI)
- Architecture: Embedding → Bidirectional LSTM → Dense
- Input: concatenation of all text fields, lowercased and HTML-stripped
- Max sequence length: 300 tokens
- Training set: 14 304 postings / Test set: 3 576 postings

### BART Zero-Shot (HuggingFace)
- Model: `facebook/bart-large-mnli`
- Labels: `"legitimate job posting"` vs `"fake fraudulent job posting"`
- No training required — pure NLI inference

### GPT-OSS-120B (OpenRouter API)
- Returns a structured JSON verdict with a natural-language explanation
- Prompt-engineered to flag known red flags (vague description, unrealistic salary, requests for personal info, poor grammar)

## Requirements

- Python 3.9+
- TensorFlow 2.x
- PyTorch (for BART inference)
- Streamlit
- Transformers (HuggingFace)

See `requirements.txt` for the full list.

## Author

**Yacine Ismail** — Mastère Data Intelligence Artificielle, NEXA
