# BERT Sentiment Analysis

**Name:** Raj Kanade | **Roll No:** 26 | **PRN:** 12413760 | **Batch:** B3

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rey-1e/Deep-Learning/blob/main/26_Assignment_8/BERT.ipynb)

## Aim
Implement a pre-trained BERT model for binary sentiment analysis using the Rotten Tomatoes movie-review dataset.

## Dataset
The **Rotten Tomatoes** dataset contains movie reviews classified into two categories:

- `0` – Negative
- `1` – Positive

For this assignment:
- 2000 training samples
- 500 testing samples

## Model
The project uses the pre-trained:

`bert-base-uncased`

BERT is fine-tuned for binary sentiment classification.

## Methodology

```text
Rotten Tomatoes Dataset
        ↓
Text Tokenization
        ↓
Pre-trained BERT
        ↓
Fine-tuning
        ↓
Sentiment Prediction
        ↓
Model Evaluation
```

## Project File Structure

```text
26_Assignment_8/
│
├── BERT.ipynb
├── bert_sentiment_analysis.py
├── README.md
├── requirements.txt
└── Assignment-8.pdf
```

### File Description

| File | Description |
|---|---|
| `BERT.ipynb` | Interactive Jupyter Notebook runnable on Google Colab with GPU support |
| `bert_sentiment_analysis.py` | Main Python code for loading the dataset, tokenization, BERT fine-tuning, evaluation, and sentiment prediction |
| `README.md` | Project documentation and instructions |
| `requirements.txt` | Required Python libraries |
| `Assignment-8.pdf` | Detailed laboratory report |

## Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

## Requirements

Install the required libraries:

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install transformers datasets accelerate scikit-learn torch
```

## How to Run

1. Open the project folder.
2. Install the required libraries.
3. Run `bert_sentiment_analysis.py`.
4. The Rotten Tomatoes dataset will be loaded automatically.
5. BERT will be fine-tuned on the training data.
6. The model will be evaluated on the test data.
7. New movie reviews will be classified as **Positive** or **Negative**.

## Example

**Input:**

```text
This movie was absolutely fantastic.
```

**Output:**

```text
Sentiment: Positive
```

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- BERT

## Conclusion

The project demonstrates how a pre-trained BERT model can be fine-tuned for sentiment analysis and used to classify movie reviews into positive and negative sentiments.
