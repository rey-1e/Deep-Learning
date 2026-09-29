# ================================================================
# BERT FOR SENTIMENT ANALYSIS
# Dataset: Rotten Tomatoes Movie Reviews
# Model: bert-base-uncased
# Name: Raj Kanade | Roll No: 26 | PRN: 12413760 | Batch: B3
# ================================================================


# ================================================================
# 1. INSTALL REQUIRED LIBRARIES
# ================================================================

# Run this cell/command once if using Google Colab or Jupyter

!pip -q install -U transformers datasets accelerate scikit-learn matplotlib seaborn


# ================================================================
# 2. IMPORT LIBRARIES
# ================================================================

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ================================================================
# 3. SET RANDOM SEED AND DEVICE
# ================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Check whether GPU is available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE INFORMATION")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ================================================================
# 4. LOAD ROTTEN TOMATOES DATASET
# ================================================================

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

dataset = load_dataset(
    "cornell-movie-review-data/rotten_tomatoes"
)

print(dataset)


# ================================================================
# 5. CREATE SMALL SAMPLE
# ================================================================

# Using a smaller sample makes the assignment faster to execute.
# Increase these values if you have a GPU.

TRAIN_SIZE = 2000
TEST_SIZE = 500

# Shuffle the training data
train_ds = dataset["train"].shuffle(
    seed=SEED
).select(
    range(TRAIN_SIZE)
)

# Shuffle the test data
test_ds = dataset["test"].shuffle(
    seed=SEED
).select(
    range(TEST_SIZE)
)

print("\nTraining examples:", len(train_ds))
print("Testing examples :", len(test_ds))


# ================================================================
# 6. DISPLAY SAMPLE REVIEWS
# ================================================================

print("\n" + "=" * 60)
print("SAMPLE REVIEWS")
print("=" * 60)

for i in range(5):

    print("\nReview:")
    print(train_ds[i]["text"])

    print("Label:", train_ds[i]["label"])

    if train_ds[i]["label"] == 0:
        print("Sentiment: NEGATIVE")
    else:
        print("Sentiment: POSITIVE")


# ================================================================
# 7. CHECK CLASS DISTRIBUTION
# ================================================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

labels = train_ds["label"]

unique_labels, counts = np.unique(
    labels,
    return_counts=True
)

print("Negative reviews:", counts[0])
print("Positive reviews:", counts[1])

plt.figure(figsize=(7, 5))

plt.bar(
    ["Negative", "Positive"],
    counts
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.show()


# ================================================================
# 8. LOAD PRE-TRAINED BERT TOKENIZER
# ================================================================

print("\n" + "=" * 60)
print("LOADING BERT TOKENIZER")
print("=" * 60)

MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded successfully.")


# ================================================================
# 9. TOKENIZE DATASET
# ================================================================

print("\n" + "=" * 60)
print("TOKENIZING DATASET")
print("=" * 60)


def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256
    )


# Tokenize training data
tokenized_train = train_ds.map(
    tokenize_function,
    batched=True
)

# Tokenize test data
tokenized_test = test_ds.map(
    tokenize_function,
    batched=True
)

print("Tokenization completed.")


# ================================================================
# 10. TEST TOKENIZER
# ================================================================

sample_text = "This movie was absolutely fantastic!"

print("\nOriginal Text:")
print(sample_text)

print("\nTokens:")
print(tokenizer.tokenize(sample_text))

print("\nToken IDs:")
print(tokenizer(sample_text)["input_ids"])


# ================================================================
# 11. DATA COLLATOR
# ================================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

print("\nData collator created successfully.")


# ================================================================
# 12. LOAD PRE-TRAINED BERT MODEL
# ================================================================

print("\n" + "=" * 60)
print("LOADING PRE-TRAINED BERT MODEL")
print("=" * 60)

id2label = {
    0: "NEGATIVE",
    1: "POSITIVE"
}

label2id = {
    "NEGATIVE": 0,
    "POSITIVE": 1
}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

print("BERT model loaded successfully.")

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "Trainable parameters:",
    trainable_parameters
)


# ================================================================
# 13. DEFINE EVALUATION METRICS
# ================================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    # Convert logits into predicted class
    predictions = np.argmax(
        logits,
        axis=-1
    )

    # Calculate precision, recall and F1
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0
        )
    )

    # Calculate accuracy
    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ================================================================
# 14. DEFINE TRAINING PARAMETERS
# ================================================================

print("\n" + "=" * 60)
print("SETTING TRAINING PARAMETERS")
print("=" * 60)

training_args = TrainingArguments(

    # Directory for results
    output_dir="./bert_rotten_tomatoes_results",

    # Number of epochs
    num_train_epochs=1,

    # Training batch size
    per_device_train_batch_size=8,

    # Evaluation batch size
    per_device_eval_batch_size=16,

    # Learning rate
    learning_rate=2e-5,

    # Weight decay
    weight_decay=0.01,

    # Evaluate after every epoch
    eval_strategy="epoch",

    # Save after every epoch
    save_strategy="epoch",

    # Load best model at the end
    load_best_model_at_end=True,

    # Select best model using F1
    metric_for_best_model="f1",

    greater_is_better=True,

    # Logging
    logging_steps=50,

    # Disable external logging
    report_to="none",

    # Use FP16 if GPU is available
    fp16=torch.cuda.is_available(),

    # Reproducibility
    seed=SEED
)


# ================================================================
# 15. CREATE TRAINER
# ================================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_test,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)

print("Trainer created successfully.")


# ================================================================
# 16. TRAIN / FINE-TUNE BERT
# ================================================================

print("\n" + "=" * 60)
print("STARTING BERT TRAINING")
print("=" * 60)

trainer.train()


# ================================================================
# 17. EVALUATE MODEL
# ================================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

metrics = trainer.evaluate()

for key, value in metrics.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ================================================================
# 18. GENERATE PREDICTIONS
# ================================================================

print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

pred_output = trainer.predict(
    tokenized_test
)

predictions = np.argmax(
    pred_output.predictions,
    axis=-1
)

true_labels = pred_output.label_ids

print("Predictions generated successfully.")


# ================================================================
# 19. CLASSIFICATION REPORT
# ================================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(

    true_labels,

    predictions,

    target_names=[
        "NEGATIVE",
        "POSITIVE"
    ],

    digits=4
)

print(report)


# ================================================================
# 20. CONFUSION MATRIX
# ================================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    true_labels,
    predictions
)

print(cm)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "NEGATIVE",
        "POSITIVE"
    ],
    yticklabels=[
        "NEGATIVE",
        "POSITIVE"
    ]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(
    "BERT Sentiment Classification - Confusion Matrix"
)

plt.show()


# ================================================================
# 21. PREDICT SENTIMENT OF NEW SENTENCES
# ================================================================

print("\n" + "=" * 60)
print("TESTING ON NEW SENTENCES")
print("=" * 60)


def predict_sentiment(texts):

    # Put model into evaluation mode
    model.eval()

    # Tokenize input sentences
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    # Move inputs to model device
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    # Disable gradient calculation
    with torch.no_grad():

        outputs = model(**inputs)

        # Convert logits into probabilities
        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        # Select highest probability class
        predictions = torch.argmax(
            probabilities,
            dim=-1
        )

    results = []

    for text, pred, probs in zip(
        texts,
        predictions,
        probabilities
    ):

        results.append({

            "text": text,

            "sentiment": id2label[
                pred.item()
            ],

            "confidence": float(
                probs[pred].item()
            )
        })

    return results


# New sentences
sample_texts = [

    "This movie was absolutely fantastic. I loved every minute of it!",

    "The movie was boring and disappointing.",

    "The acting was excellent and the story was very entertaining.",

    "I did not enjoy this movie at all.",

    "A wonderful film with great performances.",

    "The plot was terrible and the movie was a complete waste of time."
]


# Generate predictions
results = predict_sentiment(
    sample_texts
)


# Display predictions
for result in results:

    print("\nText:")
    print(result["text"])

    print(
        "Sentiment:",
        result["sentiment"]
    )

    print(
        "Confidence:",
        f"{result['confidence']:.2%}"
    )


# ================================================================
# 22. SAVE FINE-TUNED MODEL
# ================================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

SAVE_PATH = "./bert_rotten_tomatoes_model"

# Save trained model
trainer.save_model(
    SAVE_PATH
)

# Save tokenizer
tokenizer.save_pretrained(
    SAVE_PATH
)

print(
    "Model saved successfully at:",
    SAVE_PATH
)


# ================================================================
# 23. LOAD SAVED MODEL
# ================================================================

print("\n" + "=" * 60)
print("LOADING SAVED MODEL")
print("=" * 60)

loaded_tokenizer = AutoTokenizer.from_pretrained(
    SAVE_PATH
)

loaded_model = AutoModelForSequenceClassification.from_pretrained(
    SAVE_PATH
)

print("Saved model loaded successfully.")


# ================================================================
# 24. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 60)
print("ASSIGNMENT COMPLETED")
print("=" * 60)

print("Dataset : Rotten Tomatoes")
print("Model   : BERT (bert-base-uncased)")
print("Task    : Binary Sentiment Classification")
print("Classes : NEGATIVE / POSITIVE")

print("\nEvaluation Metrics:")

if "eval_accuracy" in metrics:
    print(
        "Accuracy :",
        f"{metrics['eval_accuracy']:.4f}"
    )

if "eval_precision" in metrics:
    print(
        "Precision:",
        f"{metrics['eval_precision']:.4f}"
    )

if "eval_recall" in metrics:
    print(
        "Recall   :",
        f"{metrics['eval_recall']:.4f}"
    )

if "eval_f1" in metrics:
    print(
        "F1 Score :",
        f"{metrics['eval_f1']:.4f}"
    )

